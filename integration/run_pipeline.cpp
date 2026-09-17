// Phase 8: end-to-end stereo pipeline
//   render left (CUDA) -> render right (CUDA) -> denoise both (LibTorch) -> output
// with per-stage timing. Run `run_pipeline --help`.
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "image_io.h"
#include "inference_pipeline.h"
#include "profiling.h"
#include "render_backend.h"

using namespace pt;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

constexpr double kVrBudgetMs = 1000.0 / 90.0;

struct Options {
    std::string scene = "builtin:cornell";
    std::string model;  // empty: no denoising
    std::string config = "";
    std::string out = "outputs/pipeline";
    std::string csv;
    std::string device;
    int width = 512, height = 512;
    RenderParams params;
    uint64_t seed = 1;
    int frames = 50;
    int warmup = 5;
    int block = 16;
    std::string rng = "curand";
    float ipd = 0.063f;
    bool half = false;
    bool readback = false;
    bool save = true;
    int threads = 0;
};

void usage() {
    std::cout << R"(usage: run_pipeline [options]
  --scene SPEC        scene file or builtin:cornell (default)
  --model FILE        TorchScript denoiser (omit to measure path tracing only)
  --half              the model was exported with --half
  --config NAME       label written to the CSV (e.g. lowspp_stereo)
  --width W --height H --spp N --max-depth N --clamp X --seed N --ipd X
  --device cuda|cpu   renderer + denoiser device (default cuda if available)
  --block B           CUDA block size (default 16)   --rng curand|pcg
  --threads N         CPU renderer threads
  --frames N          timed frames (default 50)   --warmup N (default 5)
  --readback          copy denoised frames to host every frame (adds transfer time)
  --csv FILE          append per-frame timings
  --out DIR           where the last frame is written (default outputs/pipeline)
  --no-save           do not write images
)";
}

Options parse(int argc, char** argv) {
    Options o;
    auto need = [&](int& i) -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { usage(); std::exit(0); }
        else if (a == "--scene") o.scene = need(i);
        else if (a == "--model") o.model = need(i);
        else if (a == "--config") o.config = need(i);
        else if (a == "--out") o.out = need(i);
        else if (a == "--csv") o.csv = need(i);
        else if (a == "--device") o.device = need(i);
        else if (a == "--width") o.width = std::stoi(need(i));
        else if (a == "--height") o.height = std::stoi(need(i));
        else if (a == "--spp") o.params.spp = std::stoi(need(i));
        else if (a == "--max-depth") o.params.max_depth = std::stoi(need(i));
        else if (a == "--clamp") o.params.clamp_indirect = std::stof(need(i));
        else if (a == "--seed") o.seed = std::stoull(need(i));
        else if (a == "--frames") o.frames = std::stoi(need(i));
        else if (a == "--warmup") o.warmup = std::stoi(need(i));
        else if (a == "--block") o.block = std::stoi(need(i));
        else if (a == "--rng") o.rng = need(i);
        else if (a == "--threads") o.threads = std::stoi(need(i));
        else if (a == "--ipd") o.ipd = std::stof(need(i));
        else if (a == "--half") o.half = true;
        else if (a == "--readback") o.readback = true;
        else if (a == "--no-save") o.save = false;
        else throw std::runtime_error("unknown option " + a);
    }
    if (o.config.empty()) o.config = o.model.empty() ? "pathtrace_only" : "denoised";
    return o;
}

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct Stat {
    std::vector<double> v;
    double mean() const {
        double s = 0;
        for (double x : v) s += x;
        return v.empty() ? 0 : s / v.size();
    }
    double pct(double p) const {
        if (v.empty()) return 0;
        std::vector<double> s = v;
        std::sort(s.begin(), s.end());
        return s[std::min(s.size() - 1, (size_t)std::llround(p * (s.size() - 1)))];
    }
};

Image wrap_host(const torch::Tensor& t, int w, int h) {
    Image img;
    img.resize(w, h, 3);
    torch::Tensor c = t.to(torch::kCPU).contiguous();
    std::copy(c.data_ptr<float>(), c.data_ptr<float>() + img.data.size(), img.data.begin());
    return img;
}

int run(int argc, char** argv) {
    Options o = parse(argc, argv);
    Scene scene = Scene::load(o.scene);

    BackendConfig cfg;
    bool use_cuda = o.device.empty() ? cuda_backend_available() : o.device == "cuda";
    cfg.kind = use_cuda ? BackendKind::Cuda : BackendKind::Cpu;
    cfg.width = o.width;
    cfg.height = o.height;
    cfg.params = o.params;
    cfg.seed = o.seed;
    cfg.threads = o.threads;
    cfg.block_x = cfg.block_y = o.block;
    cfg.rng = o.rng == "pcg" ? RngKind::Pcg : RngKind::Curand;
    std::unique_ptr<RenderBackend> backend = make_backend(scene, cfg);

    std::unique_ptr<NeuralDenoiser> denoiser;
    if (!o.model.empty()) denoiser.reset(new NeuralDenoiser(o.model, use_cuda, o.half));

    auto out_opts = torch::TensorOptions().dtype(torch::kFloat32)
                        .device(use_cuda ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU));
    torch::Tensor out_left = torch::zeros({o.height, o.width, 3}, out_opts);
    torch::Tensor out_right = torch::zeros({o.height, o.width, 3}, out_opts);
    torch::Tensor host_left = torch::empty({o.height, o.width, 3}, torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(use_cuda));
    torch::Tensor host_right = torch::empty_like(host_left);

    StereoRig rig = make_stereo_rig(scene.camera, o.width, o.height, o.ipd);
    std::vector<EyeBuffers> eyes = {
        {backend->color_ptr(EYE_LEFT), backend->normal_ptr(EYE_LEFT), backend->albedo_ptr(EYE_LEFT), out_left.data_ptr<float>()},
        {backend->color_ptr(EYE_RIGHT), backend->normal_ptr(EYE_RIGHT), backend->albedo_ptr(EYE_RIGHT), out_right.data_ptr<float>()},
    };

    std::cout << "config '" << o.config << "' | scene " << scene.name << " | " << o.width << "x" << o.height
              << " per eye | spp " << o.params.spp << "\nrenderer: " << backend->description()
              << "\ndenoiser: " << (denoiser ? o.model + (o.half ? " (fp16)" : "") : "none") << "\n";

    std::ofstream csv;
    if (!o.csv.empty()) {
        if (!fs::path(o.csv).parent_path().empty()) fs::create_directories(fs::path(o.csv).parent_path());
        bool exists = fs::exists(o.csv) && fs::file_size(o.csv) > 0;
        csv.open(o.csv, std::ios::app);
        if (!exists)
            csv << "config,scene,device,width,height,spp,frame,render_left_ms,render_right_ms,denoise_ms,readback_ms,total_ms\n";
    }

    Stat rl, rr, dn, rb, total;
    for (int f = -o.warmup; f < o.frames; ++f) {
        PT_NVTX_RANGE("frame");
        auto t_frame = Clock::now();
        backend->render(rig.left, EYE_LEFT, f);
        double t_rl = backend->last_render_ms;
        backend->render(rig.right, EYE_RIGHT, f);
        double t_rr = backend->last_render_ms;
        double t_dn = 0.0;
        if (denoiser) {
            denoiser->denoise(eyes, o.width, o.height, backend->on_device());
            t_dn = denoiser->last_ms;
        }
        double t_rb = 0.0;
        if (o.readback) {
            PT_NVTX_RANGE("readback");
            auto t0 = Clock::now();
            torch::Tensor src_l = denoiser ? out_left : torch::from_blob(eyes[0].color, {o.height, o.width, 3}, out_opts);
            torch::Tensor src_r = denoiser ? out_right : torch::from_blob(eyes[1].color, {o.height, o.width, 3}, out_opts);
            host_left.copy_(src_l);
            host_right.copy_(src_r);
            t_rb = ms_since(t0);
        }
        double t_total = ms_since(t_frame);
        if (f < 0) continue;
        rl.v.push_back(t_rl);
        rr.v.push_back(t_rr);
        dn.v.push_back(t_dn);
        rb.v.push_back(t_rb);
        total.v.push_back(t_total);
        if (csv.is_open())
            csv << o.config << "," << scene.name << "," << (use_cuda ? "cuda" : "cpu") << "," << o.width << ","
                << o.height << "," << o.params.spp << "," << f << "," << t_rl << "," << t_rr << "," << t_dn << ","
                << t_rb << "," << t_total << "\n";
    }

    printf("\nper-stage timing over %d frames (mean / p95, ms)\n", o.frames);
    printf("  render left   %8.2f / %8.2f\n", rl.mean(), rl.pct(0.95));
    printf("  render right  %8.2f / %8.2f\n", rr.mean(), rr.pct(0.95));
    printf("  denoise (2x)  %8.2f / %8.2f\n", dn.mean(), dn.pct(0.95));
    if (o.readback) printf("  readback      %8.2f / %8.2f\n", rb.mean(), rb.pct(0.95));
    printf("  total frame   %8.2f / %8.2f   -> %.1f FPS (VR budget %.1f ms: %s)\n", total.mean(), total.pct(0.95),
           1000.0 / total.mean(), kVrBudgetMs, total.mean() <= kVrBudgetMs ? "MET" : "missed");

    if (o.save) {
        fs::create_directories(o.out);
        std::vector<std::pair<std::string, int>> sides = {{"left", EYE_LEFT}, {"right", EYE_RIGHT}};
        std::vector<std::vector<unsigned char>> previews;
        for (auto& [name, eye] : sides) {
            FrameBuffers fb;
            backend->download(eye, fb);
            write_exr_rgb((fs::path(o.out) / (name + "_color.exr")).string(), fb.color);
            write_exr_rgb((fs::path(o.out) / (name + "_normal.exr")).string(), fb.normal);
            write_exr_rgb((fs::path(o.out) / (name + "_albedo.exr")).string(), fb.albedo);
            write_png((fs::path(o.out) / (name + "_color.png")).string(), fb.color, PngMode::Gamma);
            if (denoiser) {
                Image den = wrap_host(eye == EYE_LEFT ? out_left : out_right, o.width, o.height);
                write_exr_rgb((fs::path(o.out) / (name + "_denoised.exr")).string(), den);
                write_png((fs::path(o.out) / (name + "_denoised.png")).string(), den, PngMode::Gamma);
                previews.push_back(to_rgb8(den, PngMode::Gamma));
            } else {
                previews.push_back(to_rgb8(fb.color, PngMode::Gamma));
            }
        }
        std::vector<unsigned char> ana(previews[0].size());
        for (size_t i = 0; i < ana.size(); i += 3) {
            ana[i] = previews[0][i];
            ana[i + 1] = previews[1][i + 1];
            ana[i + 2] = previews[1][i + 2];
        }
        write_png_rgb8((fs::path(o.out) / "anaglyph.png").string(), o.width, o.height, ana);
        std::ofstream timing(fs::path(o.out) / "timing.json");
        timing << "{\"config\": \"" << o.config << "\", \"frames\": " << o.frames
               << ", \"render_left_ms\": " << rl.mean() << ", \"render_right_ms\": " << rr.mean()
               << ", \"denoise_ms\": " << dn.mean() << ", \"readback_ms\": " << rb.mean()
               << ", \"total_ms\": " << total.mean() << ", \"total_p95_ms\": " << total.pct(0.95) << "}\n";
        std::cout << "wrote " << o.out << "\n";
    }
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
