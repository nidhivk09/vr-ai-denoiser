// pathtracer: command line front end for the CPU/CUDA path tracer.
// Run `pathtracer --help` for options.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "image_io.h"
#include "render_backend.h"

using namespace pt;
namespace fs = std::filesystem;

namespace {

struct Options {
    std::string scene = "builtin:cornell";
    std::string out = "outputs/render";
    std::string device;  // empty = auto
    int width = 512;
    int height = 512;
    RenderParams params;
    uint64_t seed = 1;
    int frame = 0;
    int threads = 0;
    int block_x = 16, block_y = 16;
    std::string rng = "curand";
    bool stereo = false;
    float ipd = 0.063f;
    bool png = true;
    bool ppm = false;
    bool save = true;
    int frames = 1;
    int warmup = 0;
    std::string csv;
    std::string bench_blocks;
    bool quiet = false;
};

void usage() {
    std::cout << R"(usage: pathtracer [options]

scene / image
  --scene SPEC         scene file (.scene) or builtin:cornell | builtin:cornell_glossy
  --width W --height H image size per eye (default 512x512)
  --spp N              samples per pixel (default 8)
  --max-depth N        maximum path length (default 8)
  --seed N             random seed (default 1)
  --frame N            frame index, changes the noise pattern (default 0)
  --no-nee             disable next-event estimation (pure BSDF sampling)
  --no-rr              disable Russian roulette
  --clamp X            clamp indirect sample contributions to X (0 = off, default)

device
  --device cpu|cuda    default: cuda if available, else cpu
  --threads N          CPU threads (default: all cores; 1 = single-threaded reference)
  --block BX BY        CUDA thread block size (default 16 16)
  --rng curand|pcg     CUDA random number generator (default curand)
  --info               print device info and exit

stereo
  --stereo             render left and right eyes (parallel-axis rig)
  --ipd X              interpupillary distance in scene units (default 0.063)

output
  --out DIR            output directory (default outputs/render)
  --no-png             skip PNG previews
  --ppm                also write color as PPM
  --no-save            do not write images (timing runs)
  --stats              record per-pixel work counters (BVH nodes, prim tests, bounces)

timing
  --frames N           render N frames (images of the last frame are saved)
  --warmup N           untimed warm-up frames (default 0)
  --csv FILE           append per-frame timings to FILE
  --bench-blocks LIST  CUDA block-size sweep, e.g. 8,16,32 (square blocks); uses
                       --frames/--warmup and writes rows to --csv
)";
}

std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) v.push_back(std::stoi(tok));
    return v;
}

Options parse_args(int argc, char** argv) {
    Options o;
    auto need = [&](int& i, const std::string& flag) -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { usage(); std::exit(0); }
        else if (a == "--scene") o.scene = need(i, a);
        else if (a == "--out") o.out = need(i, a);
        else if (a == "--device") o.device = need(i, a);
        else if (a == "--width") o.width = std::stoi(need(i, a));
        else if (a == "--height") o.height = std::stoi(need(i, a));
        else if (a == "--spp") o.params.spp = std::stoi(need(i, a));
        else if (a == "--max-depth") o.params.max_depth = std::stoi(need(i, a));
        else if (a == "--seed") o.seed = std::stoull(need(i, a));
        else if (a == "--frame") o.frame = std::stoi(need(i, a));
        else if (a == "--no-nee") o.params.nee = false;
        else if (a == "--no-rr") o.params.russian_roulette = false;
        else if (a == "--clamp") o.params.clamp_indirect = std::stof(need(i, a));
        else if (a == "--threads") o.threads = std::stoi(need(i, a));
        else if (a == "--block") { o.block_x = std::stoi(need(i, a)); o.block_y = std::stoi(need(i, a)); }
        else if (a == "--rng") o.rng = need(i, a);
        else if (a == "--stereo") o.stereo = true;
        else if (a == "--ipd") o.ipd = std::stof(need(i, a));
        else if (a == "--no-png") o.png = false;
        else if (a == "--ppm") o.ppm = true;
        else if (a == "--no-save") o.save = false;
        else if (a == "--stats") o.params.collect_stats = true;
        else if (a == "--frames") o.frames = std::stoi(need(i, a));
        else if (a == "--warmup") o.warmup = std::stoi(need(i, a));
        else if (a == "--csv") o.csv = need(i, a);
        else if (a == "--bench-blocks") o.bench_blocks = need(i, a);
        else if (a == "--quiet") o.quiet = true;
        else if (a == "--info") {
            std::cout << "CUDA backend: " << (cuda_backend_available() ? "available" : "unavailable") << "\n"
                      << "device: " << cuda_device_info() << "\n";
            std::exit(0);
        } else throw std::runtime_error("unknown option " + a + " (see --help)");
    }
    if (o.width <= 0 || o.height <= 0 || o.params.spp <= 0 || o.frames <= 0)
        throw std::runtime_error("width, height, spp and frames must be positive");
    if (o.rng != "curand" && o.rng != "pcg") throw std::runtime_error("--rng must be curand or pcg");
    return o;
}

std::string json_escape(const std::string& s) {
    std::string r;
    for (char c : s) {
        if (c == '"' || c == '\\') r += '\\';
        r += c;
    }
    return r;
}

std::string vec_json(const Vec3& v) {
    std::ostringstream ss;
    ss << "[" << v.x << ", " << v.y << ", " << v.z << "]";
    return ss.str();
}

std::ofstream open_csv(const std::string& path, const std::string& header) {
    if (!fs::path(path).parent_path().empty()) fs::create_directories(fs::path(path).parent_path());
    bool exists = fs::exists(path) && fs::file_size(path) > 0;
    std::ofstream f(path, std::ios::app);
    if (!f) throw std::runtime_error("cannot open " + path);
    if (!exists) f << header << "\n";
    return f;
}

void save_eye(const Options& o, const std::string& prefix, const FrameBuffers& fb, float ipd, float focal_px) {
    fs::path dir(o.out);
    write_exr_rgb((dir / (prefix + "color.exr")).string(), fb.color);
    write_exr_rgb((dir / (prefix + "normal.exr")).string(), fb.normal);
    write_exr_rgb((dir / (prefix + "albedo.exr")).string(), fb.albedo);
    write_exr((dir / (prefix + "depth.exr")).string(), fb.depth, {"Z"});
    if (o.png) {
        write_png((dir / (prefix + "color.png")).string(), fb.color, PngMode::Gamma);
        write_png((dir / (prefix + "normal.png")).string(), fb.normal, PngMode::Linear);
        write_png((dir / (prefix + "albedo.png")).string(), fb.albedo, PngMode::Gamma);
        write_png((dir / (prefix + "depth.png")).string(), fb.depth, PngMode::Normalize);
    }
    if (o.ppm) write_ppm((dir / (prefix + "color.ppm")).string(), fb.color);
    if (o.stereo) {
        Image disp;
        disp.resize(fb.depth.width, fb.depth.height, 1);
        for (size_t i = 0; i < disp.data.size(); ++i) disp.data[i] = disparity_px(ipd, focal_px, fb.depth.data[i]);
        write_exr((dir / (prefix + "disparity.exr")).string(), disp, {"Y"});
        if (o.png) write_png((dir / (prefix + "disparity.png")).string(), disp, PngMode::Normalize);
    }
    if (!fb.stats.empty()) {
        Image st;
        st.resize(fb.color.width, fb.color.height, 4);
        for (size_t i = 0; i < fb.stats.size(); ++i) {
            st.data[4 * i] = (float)fb.stats[i].bounces;
            st.data[4 * i + 1] = (float)fb.stats[i].nodes_visited;
            st.data[4 * i + 2] = (float)fb.stats[i].prims_tested;
            st.data[4 * i + 3] = (float)fb.stats[i].shadow_rays;
        }
        write_exr((dir / (prefix + "stats.exr")).string(), st, {"bounces", "nodes", "prims", "shadow"});
    }
}

void write_stereo_previews(const Options& o, const FrameBuffers& left, const FrameBuffers& right) {
    int w = left.color.width, h = left.color.height;
    auto l = to_rgb8(left.color, PngMode::Gamma);
    auto r = to_rgb8(right.color, PngMode::Gamma);
    // Red-cyan anaglyph: red from the left eye, green+blue from the right eye.
    std::vector<unsigned char> ana(l.size());
    for (size_t i = 0; i < l.size(); i += 3) {
        ana[i] = l[i];
        ana[i + 1] = r[i + 1];
        ana[i + 2] = r[i + 2];
    }
    write_png_rgb8((fs::path(o.out) / "anaglyph.png").string(), w, h, ana);
    std::vector<unsigned char> sbs((size_t)(2 * w) * h * 3);
    for (int y = 0; y < h; ++y) {
        std::copy(l.begin() + (size_t)y * w * 3, l.begin() + (size_t)(y + 1) * w * 3, sbs.begin() + (size_t)y * 2 * w * 3);
        std::copy(r.begin() + (size_t)y * w * 3, r.begin() + (size_t)(y + 1) * w * 3,
                  sbs.begin() + (size_t)y * 2 * w * 3 + (size_t)w * 3);
    }
    write_png_rgb8((fs::path(o.out) / "side_by_side.png").string(), 2 * w, h, sbs);
}

struct Timing {
    double mean = 0, stddev = 0, min = 0;
};

Timing summarize(const std::vector<double>& ms) {
    Timing t;
    if (ms.empty()) return t;
    double sum = 0;
    t.min = ms[0];
    for (double v : ms) { sum += v; t.min = std::min(t.min, v); }
    t.mean = sum / ms.size();
    double var = 0;
    for (double v : ms) var += (v - t.mean) * (v - t.mean);
    t.stddev = ms.size() > 1 ? std::sqrt(var / (ms.size() - 1)) : 0.0;
    return t;
}

int run(int argc, char** argv) {
    Options o = parse_args(argc, argv);
    Scene scene = Scene::load(o.scene);

    BackendConfig cfg;
    if (o.device.empty()) cfg.kind = cuda_backend_available() ? BackendKind::Cuda : BackendKind::Cpu;
    else if (o.device == "cuda") cfg.kind = BackendKind::Cuda;
    else if (o.device == "cpu") cfg.kind = BackendKind::Cpu;
    else throw std::runtime_error("--device must be cpu or cuda");
    cfg.width = o.width;
    cfg.height = o.height;
    cfg.params = o.params;
    cfg.seed = o.seed;
    cfg.threads = o.threads;
    cfg.block_x = o.block_x;
    cfg.block_y = o.block_y;
    cfg.rng = o.rng == "pcg" ? RngKind::Pcg : RngKind::Curand;
    const char* device_name = cfg.kind == BackendKind::Cuda ? "cuda" : "cpu";

    std::unique_ptr<RenderBackend> backend = make_backend(scene, cfg);
    if (!o.quiet) {
        std::cout << "scene '" << scene.name << "': " << scene.prims.size() << " primitives, " << scene.nodes.size()
                  << " BVH nodes (" << scene.bvh_bytes() << " bytes), " << scene.lights.size() << " light(s)\n"
                  << "backend: " << backend->description() << "\n"
                  << "image: " << o.width << "x" << o.height << (o.stereo ? " x2 eyes" : "") << ", spp "
                  << o.params.spp << ", max depth " << o.params.max_depth << "\n";
    }

    std::vector<int> eyes = o.stereo ? std::vector<int>{EYE_LEFT, EYE_RIGHT} : std::vector<int>{EYE_MONO};
    StereoRig rig = make_stereo_rig(scene.camera, o.width, o.height, o.ipd);
    Camera mono = make_camera(scene.camera, o.width, o.height);
    auto camera_for = [&](int eye) -> const Camera& {
        return eye == EYE_LEFT ? rig.left : (eye == EYE_RIGHT ? rig.right : mono);
    };
    const double samples = (double)o.width * o.height * o.params.spp;

    // ---- CUDA block-size sweep -------------------------------------------------
    if (!o.bench_blocks.empty()) {
        if (cfg.kind != BackendKind::Cuda) throw std::runtime_error("--bench-blocks requires --device cuda");
        std::string csv = o.csv.empty() ? "benchmarks/results/block_size_sweep.csv" : o.csv;
        std::ofstream f = open_csv(csv, "scene,device,width,height,spp,block_x,block_y,frames,mean_ms,std_ms,min_ms,msamples_per_sec");
        std::cout << "block   mean ms   std ms   min ms   Msamples/s\n";
        for (int b : parse_int_list(o.bench_blocks)) {
            backend->set_block_size(b, b);
            for (int i = 0; i < o.warmup; ++i) backend->render(mono, EYE_MONO, i);
            std::vector<double> ms;
            for (int i = 0; i < o.frames; ++i) {
                backend->render(mono, EYE_MONO, o.frame + i);
                ms.push_back(backend->last_render_ms);
            }
            Timing t = summarize(ms);
            double mps = samples / (t.mean / 1000.0) / 1e6;
            printf("%2dx%-2d  %8.2f  %7.2f  %7.2f  %10.2f\n", b, b, t.mean, t.stddev, t.min, mps);
            f << scene.name << "," << device_name << "," << o.width << "," << o.height << "," << o.params.spp << ","
              << b << "," << b << "," << o.frames << "," << t.mean << "," << t.stddev << "," << t.min << "," << mps << "\n";
        }
        std::cout << "wrote " << csv << "\n";
        return 0;
    }

    // ---- regular render (optionally timed over several frames) ------------------
    for (int i = 0; i < o.warmup; ++i)
        for (int eye : eyes) backend->render(camera_for(eye), eye, o.frame + i);

    std::ofstream csv;
    if (!o.csv.empty())
        csv = open_csv(o.csv, "scene,device,backend,width,height,spp,max_depth,block_x,block_y,rng,eye,frame,render_ms");

    std::vector<double> frame_ms;
    for (int f = 0; f < o.frames; ++f) {
        double total = 0;
        for (int eye : eyes) {
            backend->render(camera_for(eye), eye, o.frame + f);
            total += backend->last_render_ms;
            if (csv.is_open())
                csv << scene.name << "," << device_name << ",\"" << backend->description() << "\"," << o.width << ","
                    << o.height << "," << o.params.spp << "," << o.params.max_depth << "," << o.block_x << ","
                    << o.block_y << "," << (cfg.kind == BackendKind::Cuda ? o.rng : "pcg") << "," << eye_name(eye)
                    << "," << o.frame + f << "," << backend->last_render_ms << "\n";
            if (!o.quiet)
                printf("frame %d %-5s %9.2f ms  (%.2f Msamples/s)\n", o.frame + f, eye_name(eye),
                       backend->last_render_ms, samples / (backend->last_render_ms / 1000.0) / 1e6);
        }
        frame_ms.push_back(total);
    }
    Timing t = summarize(frame_ms);
    if (!o.quiet && o.frames > 1)
        printf("mean frame time %.2f ms (std %.2f, min %.2f) for %zu eye(s)\n", t.mean, t.stddev, t.min, eyes.size());

    if (!o.save) return 0;
    fs::create_directories(o.out);
    std::vector<FrameBuffers> fbs(eyes.size());
    for (size_t i = 0; i < eyes.size(); ++i) {
        backend->download(eyes[i], fbs[i]);
        std::string prefix = o.stereo ? std::string(eye_name(eyes[i])) + "_" : "";
        save_eye(o, prefix, fbs[i], o.ipd, mono.focal_px);
    }
    if (o.stereo && o.png) write_stereo_previews(o, fbs[0], fbs[1]);

    std::ofstream meta(fs::path(o.out) / "meta.json");
    meta << "{\n"
         << "  \"scene\": \"" << json_escape(scene.name) << "\",\n"
         << "  \"scene_spec\": \"" << json_escape(o.scene) << "\",\n"
         << "  \"device\": \"" << device_name << "\",\n"
         << "  \"backend\": \"" << json_escape(backend->description()) << "\",\n"
         << "  \"width\": " << o.width << ",\n"
         << "  \"height\": " << o.height << ",\n"
         << "  \"spp\": " << o.params.spp << ",\n"
         << "  \"max_depth\": " << o.params.max_depth << ",\n"
         << "  \"nee\": " << (o.params.nee ? "true" : "false") << ",\n"
         << "  \"clamp_indirect\": " << o.params.clamp_indirect << ",\n"
         << "  \"seed\": " << o.seed << ",\n"
         << "  \"frame\": " << o.frame << ",\n"
         << "  \"stereo\": " << (o.stereo ? "true" : "false") << ",\n"
         << "  \"eyes\": [";
    for (size_t i = 0; i < eyes.size(); ++i) meta << (i ? ", " : "") << "\"" << eye_name(eyes[i]) << "\"";
    meta << "],\n"
         << "  \"ipd\": " << o.ipd << ",\n"
         << "  \"focal_px\": " << mono.focal_px << ",\n"
         << "  \"vfov_deg\": " << scene.camera.vfov_deg << ",\n"
         << "  \"lookfrom\": " << vec_json(scene.camera.lookfrom) << ",\n"
         << "  \"lookat\": " << vec_json(scene.camera.lookat) << ",\n"
         << "  \"num_primitives\": " << scene.prims.size() << ",\n"
         << "  \"num_bvh_nodes\": " << scene.nodes.size() << ",\n"
         << "  \"bvh_node_bytes\": " << sizeof(BvhNode) << ",\n"
         << "  \"primitive_bytes\": " << sizeof(Primitive) << ",\n"
         << "  \"stats\": " << (o.params.collect_stats ? "true" : "false") << ",\n"
         << "  \"frames\": " << o.frames << ",\n"
         << "  \"mean_frame_ms\": " << t.mean << ",\n"
         << "  \"gpu_memory_gb\": " << (cfg.kind == BackendKind::Cuda ? cuda_device_memory_gb() : 0.0) << "\n"
         << "}\n";
    if (!o.quiet) std::cout << "wrote " << o.out << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
