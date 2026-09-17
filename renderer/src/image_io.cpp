#include "image_io.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace pt {

namespace {

void put_bytes(std::vector<unsigned char>& out, const void* p, size_t n) {
    const unsigned char* b = static_cast<const unsigned char*>(p);
    out.insert(out.end(), b, b + n);
}
// EXR is little-endian; all supported build targets (x86-64, ARM64) are too.
void put_i32(std::vector<unsigned char>& out, int32_t v) { put_bytes(out, &v, 4); }
void put_u64(std::vector<unsigned char>& out, uint64_t v) { put_bytes(out, &v, 8); }
void put_f32(std::vector<unsigned char>& out, float v) { put_bytes(out, &v, 4); }
void put_str(std::vector<unsigned char>& out, const std::string& s) { put_bytes(out, s.c_str(), s.size() + 1); }

void put_attr_header(std::vector<unsigned char>& out, const char* name, const char* type, int32_t size) {
    put_str(out, name);
    put_str(out, type);
    put_i32(out, size);
}

void write_file(const std::string& path, const std::vector<unsigned char>& bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    f.write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize)bytes.size());
}

void put_u32_be(std::vector<unsigned char>& out, uint32_t v) {
    unsigned char b[4] = {(unsigned char)(v >> 24), (unsigned char)(v >> 16), (unsigned char)(v >> 8), (unsigned char)v};
    put_bytes(out, b, 4);
}

uint32_t crc32(const unsigned char* data, size_t n, uint32_t crc = 0) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        init = true;
    }
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

void put_png_chunk(std::vector<unsigned char>& out, const char* type, const std::vector<unsigned char>& data) {
    put_u32_be(out, (uint32_t)data.size());
    size_t start = out.size();
    put_bytes(out, type, 4);
    put_bytes(out, data.data(), data.size());
    put_u32_be(out, crc32(out.data() + start, out.size() - start));
}

}  // namespace

void write_exr(const std::string& path, const Image& img, const std::vector<std::string>& channel_names) {
    if ((int)channel_names.size() != img.channels) throw std::runtime_error("write_exr: channel name count mismatch");
    // EXR stores channels sorted by name.
    std::vector<int> order(img.channels);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return channel_names[a] < channel_names[b]; });

    std::vector<unsigned char> out;
    const int32_t magic = 20000630;
    put_i32(out, magic);
    put_i32(out, 2);  // version 2, single-part scanline

    int32_t chlist_size = 1;
    for (const auto& n : channel_names) chlist_size += (int32_t)n.size() + 1 + 16;
    put_attr_header(out, "channels", "chlist", chlist_size);
    for (int c : order) {
        put_str(out, channel_names[c]);
        put_i32(out, 2);  // FLOAT
        unsigned char linear_and_reserved[4] = {0, 0, 0, 0};
        put_bytes(out, linear_and_reserved, 4);
        put_i32(out, 1);
        put_i32(out, 1);
    }
    out.push_back(0);

    put_attr_header(out, "compression", "compression", 1);
    out.push_back(0);  // NO_COMPRESSION
    for (const char* win : {"dataWindow", "displayWindow"}) {
        put_attr_header(out, win, "box2i", 16);
        put_i32(out, 0);
        put_i32(out, 0);
        put_i32(out, img.width - 1);
        put_i32(out, img.height - 1);
    }
    put_attr_header(out, "lineOrder", "lineOrder", 1);
    out.push_back(0);  // INCREASING_Y
    put_attr_header(out, "pixelAspectRatio", "float", 4);
    put_f32(out, 1.0f);
    put_attr_header(out, "screenWindowCenter", "v2f", 8);
    put_f32(out, 0.0f);
    put_f32(out, 0.0f);
    put_attr_header(out, "screenWindowWidth", "float", 4);
    put_f32(out, 1.0f);
    out.push_back(0);  // end of header

    int32_t line_bytes = img.channels * img.width * 4;
    uint64_t offset = out.size() + (uint64_t)img.height * 8;
    for (int y = 0; y < img.height; ++y) {
        put_u64(out, offset);
        offset += 8 + (uint64_t)line_bytes;
    }
    out.reserve(out.size() + (size_t)img.height * (8 + line_bytes));
    for (int y = 0; y < img.height; ++y) {
        put_i32(out, y);
        put_i32(out, line_bytes);
        for (int c : order)
            for (int x = 0; x < img.width; ++x) put_f32(out, img.at(x, y, c));
    }
    write_file(path, out);
}

void write_exr_rgb(const std::string& path, const Image& img) {
    if (img.channels == 1) write_exr(path, img, {"Z"});
    else write_exr(path, img, {"R", "G", "B"});
}

std::vector<unsigned char> to_rgb8(const Image& img, PngMode mode) {
    float lo = 0.0f, hi = 1.0f;
    if (mode == PngMode::Normalize) {
        lo = 1e30f;
        hi = -1e30f;
        for (float v : img.data) {
            if (!std::isfinite(v) || v >= 1000.0f) continue;  // skip escaped-ray depth
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        if (hi <= lo) hi = lo + 1.0f;
    }
    std::vector<unsigned char> rgb((size_t)img.width * img.height * 3);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            for (int c = 0; c < 3; ++c) {
                float v = img.at(x, y, img.channels == 1 ? 0 : c);
                if (mode == PngMode::Normalize) v = v >= 1000.0f ? 1.0f : (v - lo) / (hi - lo);
                v = std::isfinite(v) ? std::min(std::max(v, 0.0f), 1.0f) : 0.0f;
                if (mode == PngMode::Gamma) v = std::pow(v, 1.0f / 2.2f);
                rgb[((size_t)y * img.width + x) * 3 + c] = (unsigned char)std::lround(v * 255.0f);
            }
        }
    }
    return rgb;
}

void write_png_rgb8(const std::string& path, int w, int h, const std::vector<unsigned char>& rgb) {
    // Raw scanlines with filter byte 0.
    std::vector<unsigned char> raw;
    raw.reserve((size_t)h * (w * 3 + 1));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);
        put_bytes(raw, rgb.data() + (size_t)y * w * 3, (size_t)w * 3);
    }
    // zlib stream made of uncompressed ("stored") deflate blocks.
    std::vector<unsigned char> z = {0x78, 0x01};
    size_t pos = 0;
    do {
        size_t n = std::min<size_t>(65535, raw.size() - pos);
        bool last = pos + n == raw.size();
        z.push_back(last ? 1 : 0);
        uint16_t len = (uint16_t)n, nlen = (uint16_t)~len;
        put_bytes(z, &len, 2);
        put_bytes(z, &nlen, 2);
        put_bytes(z, raw.data() + pos, n);
        pos += n;
    } while (pos < raw.size());
    uint32_t a = 1, b = 0;
    for (unsigned char c : raw) {
        a = (a + c) % 65521;
        b = (b + a) % 65521;
    }
    put_u32_be(z, (b << 16) | a);

    std::vector<unsigned char> out = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    std::vector<unsigned char> ihdr;
    put_u32_be(ihdr, (uint32_t)w);
    put_u32_be(ihdr, (uint32_t)h);
    unsigned char rest[5] = {8, 2, 0, 0, 0};  // 8-bit, RGB, deflate, no filter, no interlace
    put_bytes(ihdr, rest, 5);
    put_png_chunk(out, "IHDR", ihdr);
    put_png_chunk(out, "IDAT", z);
    put_png_chunk(out, "IEND", {});
    write_file(path, out);
}

void write_png(const std::string& path, const Image& img, PngMode mode) {
    write_png_rgb8(path, img.width, img.height, to_rgb8(img, mode));
}

void write_ppm(const std::string& path, const Image& img) {
    std::vector<unsigned char> rgb = to_rgb8(img, PngMode::Gamma);
    std::string header = "P6\n" + std::to_string(img.width) + " " + std::to_string(img.height) + "\n255\n";
    std::vector<unsigned char> out(header.begin(), header.end());
    out.insert(out.end(), rgb.begin(), rgb.end());
    write_file(path, out);
}

}  // namespace pt
