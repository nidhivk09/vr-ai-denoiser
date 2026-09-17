// Backend-agnostic renderer interface. The CUDA backend keeps its buffers in
// device memory so the integration pipeline can hand them to LibTorch without
// a host round trip.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "core/pathtrace.h"
#include "core/stereo.h"
#include "framebuffer.h"
#include "scene_loader.h"

namespace pt {

enum class BackendKind { Cpu, Cuda };
enum class RngKind { Pcg, Curand };

struct BackendConfig {
    BackendKind kind = BackendKind::Cuda;
    int width = 512;
    int height = 512;
    RenderParams params;
    uint64_t seed = 1;
    int threads = 0;           // CPU: 0 = hardware concurrency
    int block_x = 16;          // CUDA thread block size
    int block_y = 16;
    RngKind rng = RngKind::Curand;  // CUDA only; CPU always uses PCG
};

// Each backend owns two buffer slots: slot 0 (mono / left eye), slot 1 (right eye).
inline int eye_slot(int eye) { return eye == EYE_RIGHT ? 1 : 0; }

class RenderBackend {
public:
    virtual ~RenderBackend() = default;

    // Renders `cam` into the slot of `eye`. Blocks until finished.
    virtual void render(const Camera& cam, int eye, int frame) = 0;
    // Copies the slot of `eye` into host buffers.
    virtual void download(int eye, FrameBuffers& fb) = 0;

    // Interleaved HWC float32 buffers owned by the backend (device memory if on_device()).
    virtual float* color_ptr(int eye) = 0;
    virtual float* normal_ptr(int eye) = 0;
    virtual float* albedo_ptr(int eye) = 0;
    virtual bool on_device() const = 0;

    virtual void set_params(const RenderParams& p) = 0;
    virtual void set_block_size(int bx, int by) { (void)bx; (void)by; }
    virtual std::string description() const = 0;

    // Wall time of the last render() call in milliseconds (kernel + sync).
    double last_render_ms = 0.0;
};

std::unique_ptr<RenderBackend> make_cpu_backend(const Scene& scene, const BackendConfig& cfg);
// Throws std::runtime_error if the binary was built without CUDA.
std::unique_ptr<RenderBackend> make_cuda_backend(const Scene& scene, const BackendConfig& cfg);
std::unique_ptr<RenderBackend> make_backend(const Scene& scene, const BackendConfig& cfg);

bool cuda_backend_available();
std::string cuda_device_info();
double cuda_device_memory_gb();

}  // namespace pt
