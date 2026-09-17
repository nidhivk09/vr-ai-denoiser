# Stereo-Consistent Neural Denoising for Real-Time VR Path Tracing

A CUDA path tracer renders stereo (left/right eye) images of procedural Cornell-box
scenes at low sample counts. A small U-Net denoiser cleans them up, and it is
trained with a **stereo-consistency loss** so the two eyes agree with each other.
LibTorch runs the denoiser inside the C++ pipeline, and the benchmarks measure
frame time against the 90 FPS VR budget (11.1 ms), image quality (PSNR/SSIM),
stereo disagreement, and GPU workload divergence and memory throughput.

The work follows the project build spec phase by phase. `scripts/run_phase.py`
runs each phase and prints the files to inspect afterwards.

---

## 1. Requirements (on the NVIDIA machine)

| What | Version | Notes |
|---|---|---|
| NVIDIA GPU | compute capability 6.0+ | any GTX 10xx / RTX card |
| NVIDIA driver + CUDA Toolkit | CUDA 11.8 or 12.x | `nvcc --version` must work |
| CMake | 3.18+ (3.24+ recommended) | 3.24+ auto-detects your GPU architecture |
| C++ compiler | GCC 9+ / Visual Studio 2022 | C++17 for the renderer; C++20 for the LibTorch part |
| Python | 3.10+ | |
| PyTorch | with CUDA | install from https://pytorch.org/get-started/locally/ |
| LibTorch | same CUDA major version as the toolkit | Phase 8 only (see below) |
| Nsight Systems / Nsight Compute | any recent | Phase 9 profiling (optional) |

Linux (or WSL2) is the easiest platform, but Windows works too.

## 2. Setup

```bash
git clone https://github.com/nidhivk09/vr-ai-denoiser.git
cd vr-ai-denoiser

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
# install CUDA PyTorch first, e.g. for CUDA 12.6:
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
```

### Build (renderer + LibTorch pipeline)

**Linux / WSL2.** The pip PyTorch install already ships LibTorch, so the easiest option is:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build -j
./build/renderer/pathtracer --info        # should print your GPU
```

**Windows (Developer PowerShell for VS 2022).** Download LibTorch (Windows, C++,
matching CUDA, *Release*) from pytorch.org and unzip it, e.g. to `C:\libtorch`:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH=C:/libtorch
cmake --build build --config Release -j
.\build\renderer\Release\pathtracer.exe --info
```

If CMake cannot find LibTorch, it still builds the renderer and prints a warning.
You can then use the Python fallback pipeline (section 5, Phase 8). If the Python
scripts cannot find the binaries, set `PATHTRACER_BIN` and `RUN_PIPELINE_BIN` to
their full paths.

## 3. Running the phases

```bash
python scripts/run_phase.py 1 --quick     # first: a fast smoke test of any phase
python scripts/run_phase.py 1             # full settings
python scripts/run_phase.py 2 3 4
python scripts/run_phase.py 5             # dataset (longest render step)
python scripts/run_phase.py 6 7           # training (longest overall)
python scripts/run_phase.py 8 9 --gpu-bandwidth 336
```

`--gpu-bandwidth` is the rated memory bandwidth of the GPU in GB/s, from its spec
sheet. It is only used to express the memory-throughput estimate as a percentage.

| Phase | What runs | Inspect | Rough time* |
|---|---|---|---|
| 1 | CPU reference render at spp 1 and 500 | `outputs/phase1/*/color.png` | 1 min |
| 2 | CPU vs GPU agreement + speed-up; CUDA block-size sweep | `benchmarks/results/cpu_vs_gpu*.{csv,json}`, `block_size_sweep.csv` | 5 min |
| 3 | color / normal / albedo / depth buffers | `outputs/phase3/*.png` | <1 min |
| 4 | stereo render, anaglyph, disparity | `outputs/phase4/anaglyph.png` | <1 min |
| 5 | 24 scenes x (spp 1, 4, 8 + 2048 GT) x 2 eyes, 512², patches | `dataset/patches/stats.json`, `spotcheck.png` | 30-90 min |
| 6 | baseline U-Net training + test evaluation | `benchmarks/results/eval_baseline/` | 1-3 h |
| 7 | λ sweep (0, 0.1, 0.5, 1, 2, 5) + baseline vs. stereo evaluation | `benchmarks/results/lambda_sweep.csv`, `eval/` | 4-12 h |
| 8 | TorchScript export, C++ pipeline, C++ vs Python check | `outputs/phase8/` | 5 min |
| 9 | frame-time/quality bench, divergence, Nsight, plots | `benchmarks/results/figures/` | 30 min |

\*These are guesses for a mid-range laptop GPU. Phases 6 and 7 dominate the total. To
shorten them, use fewer epochs (`--epochs 60`) or a shorter sweep (`--lambdas 0,0.5,1,2`).
Training uses a one-cycle learning-rate schedule, so `--epochs` sets the run length
(add `--patience N` for early stopping). On an 8 GB GPU, keep `--batch-size 16` (the default); raise
it on bigger GPUs.

**What to send back after each phase:** the files in the "Inspect" column, plus the
console output. For Phase 9, send the whole `benchmarks/results/` folder (it is small
apart from `frames/`).

## 4. Repository layout

```
renderer/
  src/core/            shared CPU+CUDA code (__host__ __device__)
    vec3.h ray.h rng.h   math, AABB, PCG RNG
    scene.h              materials, sphere/quad primitives, BVH node, SceneView
    bvh.h                BVH traversal (closest hit, shadow rays) + work counters
    camera.h stereo.h    pinhole camera, parallel-axis stereo rig, disparity
    pathtrace.h          the integrator (NEE, Russian roulette, feature buffers)
  src/pathtrace_kernel.cu  CUDA backend: kernels, curand, device buffers
  src/cpu_backend.cpp      multithreaded CPU backend (--threads 1 = Phase 1 reference)
  src/bvh.cpp scene_loader.* image_io.* framebuffer.h render_backend.h main.cpp
  scenes/*.scene           built-in scenes as text files (format in scene_loader.h)
dataset/     generate_dataset.py, extract_patches.py, layout.py (channel layout)
denoiser/    model.py, dataset.py, losses.py, train.py, evaluate.py,
             sweep_lambda.py, export_model.py
integration/ inference_pipeline.{h,cpp}, run_pipeline.cpp (LibTorch),
             python_pipeline.py (fallback), compare_with_python.py
benchmarks/  cpu_vs_gpu.py, frame_time_bench.py, quality_metrics.py,
             divergence_analysis.py, nsight_profile_runs.sh, make_plots.py
common/      EXR I/O, paths, image helpers
scripts/     run_phase.py
paper/       outline.md
```

This layout differs from the spec in one way. Instead of separate `camera.cu`,
`scene.cu`, etc., the renderer keeps **one shared implementation** in
`renderer/src/core/*.h`, compiled both by the host compiler (CPU backend) and by
`nvcc` (inside `pathtrace_kernel.cu`). That's why the Phase 2 CPU-vs-GPU comparison
tests the same algorithm on both devices, rather than two versions that could drift.

## 5. Useful commands

```bash
# renderer
pathtracer --help
pathtracer --scene builtin:cornell_glossy --spp 16 --stereo --out outputs/test
pathtracer --device cuda --bench-blocks 8,16,32 --frames 20 --spp 4
pathtracer --stats --spp 1 --out outputs/stats     # per-pixel work counters

# training / evaluation
python -m denoiser.train --mode mono --run-name baseline --amp
python -m denoiser.train --mode stereo --lambda-cons 1.0 --run-name stereo_l1 --amp
python -m denoiser.evaluate --models baseline=denoiser/checkpoints/baseline/best.pt \
                                     stereo=denoiser/checkpoints/sweep_l1/best.pt

# pipeline
python -m denoiser.export_model --checkpoint denoiser/checkpoints/baseline/best.pt --device cuda
build/integration/run_pipeline --scene dataset/scenes/scene_003.scene \
    --model denoiser/checkpoints/baseline/best_ts.pt --spp 4 --clamp 8 --frames 60
# fallback without LibTorch:
python -m integration.python_pipeline --scene dataset/scenes/scene_003.scene \
    --checkpoint denoiser/checkpoints/baseline/best.pt --spp 4
python -m benchmarks.frame_time_bench --python \
    --baseline-model denoiser/checkpoints/baseline/best.pt \
    --stereo-model denoiser/checkpoints/sweep_l1/best.pt

# profiling
bash benchmarks/nsight_profile_runs.sh denoiser/checkpoints/baseline/best_ts.pt
```

## 6. Design notes and definitions

- **Units and stereo geometry.** The Cornell box is 1 unit wide (read it as 1 m),
  so the default IPD is 0.063. Disparity is `d = ipd · focal_px / Z`, where `Z` is
  the planar depth of the pixel-center ray, and `x_right = x_left − d`.
- **Integrator.** It traces iterative paths with next-event estimation on the area
  lights and Russian roulette after 3 bounces. `--clamp X` limits indirect
  contributions to suppress fireflies from glass caustics. The dataset uses
  `--clamp 8` for both the noisy and the ground-truth renders, so the targets stay
  consistent (this introduces a small, deliberate bias).
- **Buffers.** `color` is linear HDR. `normal` is the world-space normal mapped to
  [0,1]. `albedo` is the first-hit base color (1 for glass). All of them average
  the spp jittered samples.
- **Denoiser.** It predicts a residual in `log1p` space and is trained with L1 in
  that space. The consistency loss compares the left prediction with the right
  prediction warped into the left view (and vice versa), masked by a left-right
  disparity check. Inference is identical for both models, so consistency costs
  nothing at run time.
- **Metrics.** Metrics use display-referred images (clamp to [0,1], then gamma
  2.2). *Stereo disagreement* is the mean |left − warp(right)| over mutually
  visible pixels. The ground truth's own disagreement (from view-dependent
  shading and resampling) is the floor.
- **Divergence.** This comes from renderer instrumentation (`--stats`): per-pixel
  BVH nodes visited plus primitives tested. For each B×B thread block, the idle
  capacity is `Σ(max − w) / Σ max`, i.e. the lockstep model of threads waiting on
  the slowest one. *Memory throughput* is (nodes·32 B + primitive tests·sizeof(Primitive))
  per second. Nsight Systems and Nsight Compute provide the timeline and
  hardware-level cross-checks, and `nvidia-smi`'s memory-utilization counter is
  sampled as well. The report should say which numbers come from which source.
- **TorchScript.** `torch.jit.trace` prints deprecation warnings on PyTorch 2.9+.
  They are harmless: the export verifies the traced model against eager mode.

## 7. Troubleshooting

- `CUDA backend: OFF` during CMake means `nvcc` wasn't found. Add the CUDA
  `bin` folder to PATH, or pass `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`.
- `Unsupported gpu architecture` means you should pass your GPU's architecture,
  e.g. `-DCMAKE_CUDA_ARCHITECTURES=86` for RTX 30xx or `89` for RTX 40xx.
- A LibTorch C++20 error means your compiler is too old. Use GCC 10+ or VS 2022.
- If LibTorch reports that the CUDA toolkit or NVTX is missing, make sure the
  toolkit version matches LibTorch's CUDA version.
- `LibTorch reports no CUDA device` means you downloaded or linked a CPU-only LibTorch.
- On Windows, missing DLLs when starting `run_pipeline.exe` mean the post-build step
  didn't copy them. Copy `C:\libtorch\lib\*.dll` next to the executable.
- If training runs out of memory, lower `--batch-size` or use `--base 24`.
- If the Python scripts can't find the binaries, set `PATHTRACER_BIN` and `RUN_PIPELINE_BIN`.

## 8. Verification status

The code was developed on a Mac without an NVIDIA GPU:

- **Tested there:** the CPU renderer (Phases 1, 3, 4), dataset generation and
  patches, training and evaluation for both modes, TorchScript export, the C++
  LibTorch pipeline with the CPU backend (its output matched Python inference
  exactly), the Python fallback pipeline, the benchmark scripts on CPU, and the plots.
- **Not compiled or run yet:** the CUDA backend (`renderer/src/pathtrace_kernel.cu`),
  the CUDA code paths of the LibTorch pipeline, and the Nsight script.

Building on the NVIDIA machine (section 2) is the first compile of the CUDA code, and
Phase 2 is its first run. Please send back any compiler errors or output.
