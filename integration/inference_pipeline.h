// Phase 8: LibTorch inference on renderer buffers.
//
// The CUDA renderer's color/normal/albedo buffers are wrapped as CUDA tensors
// with torch::from_blob (no copy); the network runs on the GPU and writes the
// result into output buffers that also stay on the GPU.
#pragma once

#include <torch/script.h>

#include <string>
#include <vector>

namespace pt {

// Interleaved HWC float32 buffers for one eye (device memory when on_device).
struct EyeBuffers {
    float* color;
    float* normal;
    float* albedo;
    float* output;  // receives the denoised color
    float* depth = nullptr;  // planar depth, HW (1 channel); only read for fusion models
};

class NeuralDenoiser {
public:
    // `use_cuda`: run the model on GPU 0. `half`: the model was exported with
    // --half (inputs are converted to FP16). Inputs are padded to `pad_multiple`.
    NeuralDenoiser(const std::string& model_path, bool use_cuda, bool half, int pad_multiple = 16);

    // Denoises all eyes in one batched forward pass. `on_device` says whether
    // the buffers are CUDA memory (true) or host memory (false). Blocks until done.
    void denoise(const std::vector<EyeBuffers>& eyes, int width, int height, bool on_device);

    // Cross-eye fusion models (exported from a Denoiser(fusion=True)) take a 10th
    // input channel, the disparity in pixels = ipd * focal_px / depth, and need
    // the eyes ordered {left, right}. Call once with ipd * focal_px to enable.
    void enable_fusion(float ipd_times_focal_px) { disparity_scale_ = ipd_times_focal_px; }
    bool fusion() const { return disparity_scale_ > 0.0f; }

    bool on_cuda() const { return device_.is_cuda(); }
    double last_ms = 0.0;

private:
    torch::jit::script::Module module_;
    torch::Device device_;
    bool half_;
    int pad_multiple_;
    float disparity_scale_ = 0.0f;  // > 0: fusion model
};

}  // namespace pt
