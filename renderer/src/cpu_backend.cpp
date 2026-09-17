// Multithreaded CPU backend. With --threads 1 this is the single-threaded
// Phase 1 reference path tracer.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

#include "render_backend.h"

namespace pt {

namespace {

class CpuBackend : public RenderBackend {
public:
    CpuBackend(const Scene& scene, const BackendConfig& cfg) : scene_(scene), cfg_(cfg) {
        view_ = scene_.host_view();
        for (auto& fb : slots_) fb.resize(cfg.width, cfg.height, cfg.params.collect_stats);
        threads_ = cfg.threads > 0 ? cfg.threads : (int)std::max(1u, std::thread::hardware_concurrency());
    }

    void render(const Camera& cam, int eye, int frame) override {
        auto t0 = std::chrono::steady_clock::now();
        FrameBuffers& fb = slots_[eye_slot(eye)];
        const int w = cfg_.width, h = cfg_.height;
        const bool stats = cfg_.params.collect_stats;
        std::atomic<int> next_row{0};

        auto worker = [&]() {
            int y;
            while ((y = next_row.fetch_add(1)) < h) {
                for (int x = 0; x < w; ++x) {
                    int idx = y * w + x;
                    PcgRng rng;
                    rng.seed(pixel_seed(cfg_.seed, (uint32_t)frame, (uint32_t)eye, (uint32_t)idx), (uint64_t)idx);
                    PathStats st{};
                    render_pixel(view_, cam, x, y, cfg_.params, rng, &fb.color.data[3 * idx],
                                 &fb.normal.data[3 * idx], &fb.albedo.data[3 * idx], &fb.depth.data[idx],
                                 stats ? &st : nullptr);
                    if (stats) fb.stats[idx] = st;
                }
            }
        };
        if (threads_ == 1) {
            worker();
        } else {
            std::vector<std::thread> pool;
            for (int i = 0; i < threads_; ++i) pool.emplace_back(worker);
            for (auto& t : pool) t.join();
        }
        last_render_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    void download(int eye, FrameBuffers& fb) override { fb = slots_[eye_slot(eye)]; }

    float* color_ptr(int eye) override { return slots_[eye_slot(eye)].color.data.data(); }
    float* normal_ptr(int eye) override { return slots_[eye_slot(eye)].normal.data.data(); }
    float* albedo_ptr(int eye) override { return slots_[eye_slot(eye)].albedo.data.data(); }
    bool on_device() const override { return false; }

    void set_params(const RenderParams& p) override {
        bool resize = p.collect_stats != cfg_.params.collect_stats;
        cfg_.params = p;
        if (resize)
            for (auto& fb : slots_) fb.resize(cfg_.width, cfg_.height, p.collect_stats);
    }

    std::string description() const override {
        return "cpu (" + std::to_string(threads_) + " thread" + (threads_ == 1 ? "" : "s") + ")";
    }

private:
    const Scene& scene_;
    BackendConfig cfg_;
    SceneView view_;
    FrameBuffers slots_[2];
    int threads_ = 1;
};

}  // namespace

std::unique_ptr<RenderBackend> make_cpu_backend(const Scene& scene, const BackendConfig& cfg) {
    return std::unique_ptr<RenderBackend>(new CpuBackend(scene, cfg));
}

std::unique_ptr<RenderBackend> make_backend(const Scene& scene, const BackendConfig& cfg) {
    return cfg.kind == BackendKind::Cuda ? make_cuda_backend(scene, cfg) : make_cpu_backend(scene, cfg);
}

#ifndef PT_WITH_CUDA
std::unique_ptr<RenderBackend> make_cuda_backend(const Scene&, const BackendConfig&) {
    throw std::runtime_error("this binary was built without CUDA support (reconfigure CMake with a CUDA toolkit)");
}
bool cuda_backend_available() { return false; }
std::string cuda_device_info() { return "none (built without CUDA)"; }
double cuda_device_memory_gb() { return 0.0; }
#endif

}  // namespace pt
