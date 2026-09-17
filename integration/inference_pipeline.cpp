#include "inference_pipeline.h"

#include <torch/torch.h>
#ifdef PT_WITH_CUDA
#include <torch/cuda.h>
#endif

#include <chrono>
#include <stdexcept>

#include "profiling.h"

namespace pt {

namespace {

void synchronize(const torch::Device& device) {
#ifdef PT_WITH_CUDA
    if (device.is_cuda()) torch::cuda::synchronize(device.index());
#else
    (void)device;
#endif
}

}  // namespace

NeuralDenoiser::NeuralDenoiser(const std::string& model_path, bool use_cuda, bool half, int pad_multiple)
    : device_(use_cuda ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU)),
      half_(half),
      pad_multiple_(pad_multiple) {
    if (use_cuda && !torch::cuda::is_available())
        throw std::runtime_error("LibTorch reports no CUDA device (is this a CPU-only libtorch build?)");
    if (half && !use_cuda) throw std::runtime_error("FP16 models require CUDA");
    try {
        module_ = torch::jit::load(model_path, device_);
    } catch (const c10::Error& e) {
        throw std::runtime_error("failed to load TorchScript model " + model_path + ": " + e.what_without_backtrace());
    }
    module_.eval();
}

void NeuralDenoiser::denoise(const std::vector<EyeBuffers>& eyes, int width, int height, bool on_device) {
    PT_NVTX_RANGE("denoise");
    auto t0 = std::chrono::steady_clock::now();
    c10::InferenceMode guard;

    torch::Device buf_device = on_device ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(buf_device);

    std::vector<torch::Tensor> inputs;
    inputs.reserve(eyes.size());
    for (const EyeBuffers& e : eyes) {
        // Zero-copy views of the renderer's buffers.
        torch::Tensor c = torch::from_blob(e.color, {height, width, 3}, opts);
        torch::Tensor n = torch::from_blob(e.normal, {height, width, 3}, opts);
        torch::Tensor a = torch::from_blob(e.albedo, {height, width, 3}, opts);
        inputs.push_back(torch::cat({c, n, a}, 2).permute({2, 0, 1}));
    }
    torch::Tensor x = torch::stack(inputs, 0).to(device_);  // (N, 9, H, W); no-op when already there

    int64_t ph = (pad_multiple_ - height % pad_multiple_) % pad_multiple_;
    int64_t pw = (pad_multiple_ - width % pad_multiple_) % pad_multiple_;
    if (ph || pw) {
        namespace F = torch::nn::functional;
        x = F::pad(x, F::PadFuncOptions({0, pw, 0, ph}).mode(torch::kReplicate));
    }
    if (half_) x = x.to(torch::kHalf);

    torch::Tensor y = module_.forward({x}).toTensor();
    y = y.slice(2, 0, height).slice(3, 0, width).to(buf_device, torch::kFloat32).permute({0, 2, 3, 1});

    for (size_t i = 0; i < eyes.size(); ++i)
        torch::from_blob(eyes[i].output, {height, width, 3}, opts).copy_(y[(int64_t)i]);
    synchronize(device_);
    last_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace pt
