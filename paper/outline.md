# Stereo-Consistent Neural Denoising for Real-Time VR Path Tracing — paper outline

Each section lists what goes in it and which generated file backs it
(paths are relative to `benchmarks/results/` unless stated otherwise).

## 1. Abstract
One paragraph (use the draft abstract prepared earlier): low-spp CUDA path tracing
+ U-Net denoiser for stereo VR; contribution = stereo-consistency loss + GPU
architecture analysis; headline numbers = frame time vs. 11.1 ms budget
(`frame_time_summary.csv`), PSNR/SSIM gain (`eval/summary.csv`), stereo
disagreement reduction baseline → stereo (`eval/summary.csv`).

## 2. Introduction
- VR needs 90 FPS (11.1 ms / frame) *per stereo pair*; path tracing is physically
  correct but needs hundreds of samples per pixel.
- Low-spp + neural denoising closes the gap, but denoising each eye independently
  can produce *different* residual artefacts in the two eyes → binocular rivalry,
  shimmering, discomfort.
- Gap: (1) stereo consistency is not a training objective in existing denoisers,
  (2) foveation is not combined with it (future work here), (3) few works
  benchmark the full pipeline against the VR frame budget with an
  architecture-level explanation.
- Contributions:
  1. A stereo-consistency loss using disparity-based warping with occlusion masking.
  2. A CUDA path tracer + LibTorch pipeline benchmarked against the VR budget.
  3. An analysis of path-tracing workload divergence and memory throughput vs.
     sample count, contrasted with the regular compute of the denoiser.

## 3. Related Work
(a) Neural denoising for path tracing (KPCN, OIDN/U-Net style, recurrent/temporal
denoisers, NVIDIA OptiX denoiser). (b) Foveated rendering for VR. (c) GPU ray
tracing architecture: ray/traversal divergence, packet/stream tracing, memory
bandwidth as a bottleneck. Use the papers gathered earlier.

## 4. Method
- **Figure 1 — pipeline diagram**: CUDA path tracer (left/right eye, color +
  normal + albedo + depth) → U-Net (shared weights, per eye) → stereo frame;
  training-time branch: disparity warp + consistency loss.
- Renderer outputs; normal encoding; planar depth; disparity `d = ipd·f/Z`,
  `x_R = x_L − d` (renderer/src/core/stereo.h).
- U-Net: 9→3 channels, 3 down-sampling levels, base 32, residual prediction in
  log space, zero-initialised head (denoiser/model.py).
- Loss: `L = L1(ŷ_L, y_L) + L1(ŷ_R, y_R) + λ·L_cons`, L1 in log1p space;
  `L_cons` = masked L1 between ŷ_L and the right prediction warped into the left
  view (bilinear, `grid_sample`), symmetric; visibility mask from a left-right
  disparity check (denoiser/losses.py). Note: consistency costs nothing at
  inference — the same network runs per eye.
- Dataset: procedural Cornell-box variants, splits by scene, patch extraction
  with identical left/right windows, flip augmentation with eye swap
  (dataset/, denoiser/dataset.py). **Table 1**: `dataset/patches/stats.json`
  + `dataset/raw/manifest.json` (scenes, resolution, spp levels, patches per split).

## 5. Implementation
- Shared `__host__ __device__` integrator (one implementation, CPU and CUDA);
  iterative paths, next-event estimation, Russian roulette, optional indirect
  clamp; flattened 32-byte BVH nodes, median split, stack traversal.
- One CUDA thread per pixel; curand per-pixel states (`curand_init`); PCG as an
  alternative. **Figure 2**: `figures/block_size_sweep.png`.
- Phase 2 validation: **Table 2** `cpu_vs_gpu.csv` (speed-up) and
  `cpu_vs_gpu_agreement.json` (PSNR/SSIM CPU vs GPU); `figures/cpu_vs_gpu_images.png`.
- LibTorch integration: zero-copy `torch::from_blob` on CUDA buffers, batched
  two-eye inference, optional FP16; NVTX ranges for Nsight.
  Validation: `outputs/phase8/python_comparison.json`.

## 6. Evaluation
- Setup: GPU model, driver/CUDA/LibTorch versions, resolution, spp.
- **Figure 3** frame time vs. budget: `figures/frame_time_vs_budget.png`;
  **Table 3**: `frame_time_summary.csv`.
- **Figure 4** quality per configuration: `figures/quality_by_config.png`,
  `quality_by_config.csv`; per noise level: `figures/denoiser_eval_by_spp.png`.
- **Figure 5** qualitative comparison: `eval/<scene>_sppXXXX.png`
  (noisy / baseline / stereo / GT and |L − warp(R)| maps).
- **Headline — stereo disagreement**: baseline vs. stereo model
  (`eval/summary.csv`), with the ground-truth disagreement as the floor
  (view-dependent shading + resampling).
- **Figure 6** λ sweep: `figures/lambda_sweep.png` (quality vs. consistency trade-off).
- **Figure 7** training curves: `figures/training_curves.png`.
- **Figure 8** divergence vs. spp and memory throughput: `figures/divergence_vs_spp.png`,
  `divergence.csv`, heat map `figures/work_heatmap_spp1.png`. Discuss why
  relative divergence falls with spp (per-pixel work averages over more samples).
- **Figure 9** irregular vs. regular compute: `figures/stage_breakdown.png` and the
  Nsight Systems timeline (`nsight/pipeline_spp4*`), Nsight Compute reports.

## 7. Limitations & Future Work
- Simple procedural scenes (few primitives; BVH is shallow) — divergence and
  bandwidth numbers will differ for production scenes.
- Divergence/bandwidth are derived from renderer instrumentation (lockstep
  model) and cross-checked with Nsight / nvidia-smi, not measured directly
  per SIMD group.
- No temporal accumulation / reprojection; single frames only.
- No HMD or eye-tracking validation; no user study.
- Foveated shading not integrated (future work).
- (If used) Python fallback pipeline instead of LibTorch.

## 8. Conclusion
Restate: VR-budget result, consistency improvement at zero inference cost, and the
architecture insight (irregular path tracing vs. regular inference).
