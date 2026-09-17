// Dependency-free image writers: uncompressed OpenEXR (float32), PNG, PPM.
#pragma once

#include <string>
#include <vector>

#include "framebuffer.h"

namespace pt {

// `channel_names` lists the name of each interleaved channel in `img`
// (e.g. {"R","G","B"} or {"Z"}). Throws on I/O failure.
void write_exr(const std::string& path, const Image& img, const std::vector<std::string>& channel_names);
void write_exr_rgb(const std::string& path, const Image& img);

enum class PngMode {
    Gamma,     // clamp to [0,1] then gamma 2.2 (display of linear radiance)
    Linear,    // clamp to [0,1] (normal/albedo buffers)
    Normalize, // min/max normalize (depth, disparity, stats)
};
void write_png(const std::string& path, const Image& img, PngMode mode);
void write_ppm(const std::string& path, const Image& img);  // gamma-corrected

// 8-bit helpers used for previews.
std::vector<unsigned char> to_rgb8(const Image& img, PngMode mode);
void write_png_rgb8(const std::string& path, int w, int h, const std::vector<unsigned char>& rgb);

}  // namespace pt
