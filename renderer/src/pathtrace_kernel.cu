// CUDA backend: one GPU thread per pixel runs render_pixel() from the shared
// integrator. Per-pixel randomness comes from curand's device API (one
// curandState per pixel, seeded once with curand_init) or from PCG.
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

#include "profiling.h"
#include "render_backend.h"

namespace pt {

namespace {

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(err));
}

template <class T>
T* device_alloc(size_t count, const char* what) {
    void* p = nullptr;
    check(cudaMalloc(&p, count * sizeof(T)), what);
    return static_cast<T*>(p);
}

template <class T>
T* device_upload(const T* host, size_t count, const char* what) {
    if (count == 0) return nullptr;
    T* p = device_alloc<T>(count, what);
    check(cudaMemcpy(p, host, count * sizeof(T), cudaMemcpyHostToDevice), what);
    return p;
}

// Adapter so the templated integrator can draw numbers from curand.
struct CurandRng {
    curandState* state;
    __device__ float uniform() {
        float u = curand_uniform(state);  // (0, 1]
        return u >= 1.0f ? 0.99999994f : u;
    }
};

__global__ void curand_init_kernel(curandState* states, int n, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curand_init(seed + (unsigned int)i, 0, 0, &states[i]);
}

__global__ void render_kernel_curand(SceneView s, Camera cam, RenderParams p, int w, int h,
                                     curandState* states, float* color, float* normal,
                                     float* albedo, float* depth, PathStats* stats) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    CurandRng rng{&states[idx]};
    PathStats st = {0, 0, 0, 0};
    render_pixel(s, cam, x, y, p, rng, color + 3 * idx, normal + 3 * idx, albedo + 3 * idx,
                 depth + idx, stats ? &st : nullptr);
    if (stats) stats[idx] = st;
}

__global__ void render_kernel_pcg(SceneView s, Camera cam, RenderParams p, int w, int h,
                                  uint64_t seed, int frame, int eye, float* color, float* normal,
                                  float* albedo, float* depth, PathStats* stats) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    PcgRng rng;
    rng.seed(pixel_seed(seed, (uint32_t)frame, (uint32_t)eye, (uint32_t)idx), (uint64_t)idx);
    PathStats st = {0, 0, 0, 0};
    render_pixel(s, cam, x, y, p, rng, color + 3 * idx, normal + 3 * idx, albedo + 3 * idx,
                 depth + idx, stats ? &st : nullptr);
    if (stats) stats[idx] = st;
}

struct DeviceSlot {
    float* color = nullptr;
    float* normal = nullptr;
    float* albedo = nullptr;
    float* depth = nullptr;
    PathStats* stats = nullptr;
    curandState* rng = nullptr;
};

class CudaBackend : public RenderBackend {
public:
    CudaBackend(const Scene& scene, const BackendConfig& cfg) : cfg_(cfg) {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
            throw std::runtime_error("no CUDA device available");
        check(cudaSetDevice(0), "cudaSetDevice");

        prims_ = device_upload(scene.prims.data(), scene.prims.size(), "upload primitives");
        nodes_ = device_upload(scene.nodes.data(), scene.nodes.size(), "upload BVH");
        mats_ = device_upload(scene.materials.data(), scene.materials.size(), "upload materials");
        lights_ = device_upload(scene.lights.data(), scene.lights.size(), "upload lights");
        view_ = scene.host_view();
        view_.prims = prims_;
        view_.nodes = nodes_;
        view_.materials = mats_;
        view_.lights = lights_;

        size_t n = (size_t)cfg.width * cfg.height;
        for (int s = 0; s < 2; ++s) {
            DeviceSlot& d = slots_[s];
            d.color = device_alloc<float>(3 * n, "alloc color");
            d.normal = device_alloc<float>(3 * n, "alloc normal");
            d.albedo = device_alloc<float>(3 * n, "alloc albedo");
            d.depth = device_alloc<float>(n, "alloc depth");
            if (cfg.params.collect_stats) d.stats = device_alloc<PathStats>(n, "alloc stats");
            if (cfg.rng == RngKind::Curand) {
                d.rng = device_alloc<curandState>(n, "alloc curand states");
                int block = 256;
                int blocks = (int)((n + block - 1) / block);
                unsigned int seed = (unsigned int)(cfg.seed * 7919u + 1000003u * (unsigned int)s);
                curand_init_kernel<<<blocks, block>>>(d.rng, (int)n, seed);
                check(cudaGetLastError(), "curand_init kernel launch");
            }
        }
        check(cudaDeviceSynchronize(), "curand_init");
    }

    ~CudaBackend() override {
        for (DeviceSlot& d : slots_) {
            cudaFree(d.color);
            cudaFree(d.normal);
            cudaFree(d.albedo);
            cudaFree(d.depth);
            cudaFree(d.stats);
            cudaFree(d.rng);
        }
        cudaFree(prims_);
        cudaFree(nodes_);
        cudaFree(mats_);
        cudaFree(lights_);
    }

    void render(const Camera& cam, int eye, int frame) override {
        PT_NVTX_RANGE(eye == EYE_RIGHT ? "pathtrace_right" : "pathtrace_left");
        DeviceSlot& d = slots_[eye_slot(eye)];
        const int w = cfg_.width, h = cfg_.height;
        dim3 threads(cfg_.block_x, cfg_.block_y);
        dim3 blocks((w + cfg_.block_x - 1) / cfg_.block_x, (h + cfg_.block_y - 1) / cfg_.block_y);
        PathStats* stats = cfg_.params.collect_stats ? d.stats : nullptr;

        auto t0 = std::chrono::steady_clock::now();
        if (cfg_.rng == RngKind::Curand) {
            render_kernel_curand<<<blocks, threads>>>(view_, cam, cfg_.params, w, h, d.rng, d.color, d.normal,
                                                      d.albedo, d.depth, stats);
        } else {
            render_kernel_pcg<<<blocks, threads>>>(view_, cam, cfg_.params, w, h, cfg_.seed, frame, eye, d.color,
                                                   d.normal, d.albedo, d.depth, stats);
        }
        check(cudaGetLastError(), "render kernel launch");
        check(cudaDeviceSynchronize(), "render kernel");
        last_render_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    void download(int eye, FrameBuffers& fb) override {
        DeviceSlot& d = slots_[eye_slot(eye)];
        size_t n = (size_t)cfg_.width * cfg_.height;
        fb.resize(cfg_.width, cfg_.height, cfg_.params.collect_stats);
        check(cudaMemcpy(fb.color.data.data(), d.color, 3 * n * sizeof(float), cudaMemcpyDeviceToHost), "download");
        check(cudaMemcpy(fb.normal.data.data(), d.normal, 3 * n * sizeof(float), cudaMemcpyDeviceToHost), "download");
        check(cudaMemcpy(fb.albedo.data.data(), d.albedo, 3 * n * sizeof(float), cudaMemcpyDeviceToHost), "download");
        check(cudaMemcpy(fb.depth.data.data(), d.depth, n * sizeof(float), cudaMemcpyDeviceToHost), "download");
        if (cfg_.params.collect_stats)
            check(cudaMemcpy(fb.stats.data(), d.stats, n * sizeof(PathStats), cudaMemcpyDeviceToHost), "download");
    }

    float* color_ptr(int eye) override { return slots_[eye_slot(eye)].color; }
    float* normal_ptr(int eye) override { return slots_[eye_slot(eye)].normal; }
    float* albedo_ptr(int eye) override { return slots_[eye_slot(eye)].albedo; }
    bool on_device() const override { return true; }

    void set_params(const RenderParams& p) override {
        if (p.collect_stats && !cfg_.params.collect_stats) {
            size_t n = (size_t)cfg_.width * cfg_.height;
            for (DeviceSlot& d : slots_)
                if (!d.stats) d.stats = device_alloc<PathStats>(n, "alloc stats");
        }
        cfg_.params = p;
    }

    void set_block_size(int bx, int by) override {
        if (bx * by > 1024) throw std::runtime_error("CUDA block size must satisfy bx*by <= 1024");
        cfg_.block_x = bx;
        cfg_.block_y = by;
    }

    std::string description() const override {
        return "cuda (" + cuda_device_info() + ", block " + std::to_string(cfg_.block_x) + "x" +
               std::to_string(cfg_.block_y) + ", rng " + (cfg_.rng == RngKind::Curand ? "curand" : "pcg") + ")";
    }

private:
    BackendConfig cfg_;
    SceneView view_{};
    Primitive* prims_ = nullptr;
    BvhNode* nodes_ = nullptr;
    Material* mats_ = nullptr;
    int* lights_ = nullptr;
    DeviceSlot slots_[2];
};

}  // namespace

std::unique_ptr<RenderBackend> make_cuda_backend(const Scene& scene, const BackendConfig& cfg) {
    return std::unique_ptr<RenderBackend>(new CudaBackend(scene, cfg));
}

bool cuda_backend_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string cuda_device_info() {
    if (!cuda_backend_available()) return "no CUDA device";
    cudaDeviceProperties props;
    cudaGetDeviceProperties(&props, 0);
    return std::string(props.name) + ", compute " + std::to_string(props.major) + "." + std::to_string(props.minor);
}

double cuda_device_memory_gb() {
    if (!cuda_backend_available()) return 0.0;
    cudaDeviceProperties props;
    cudaGetDeviceProperties(&props, 0);
    return (double)props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
}

}  // namespace pt
