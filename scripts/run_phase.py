"""Runs the build-spec phases one at a time, each ending in something to inspect.

    python scripts/run_phase.py 1          # Phase 1 only
    python scripts/run_phase.py 2 3 4      # several phases
    python scripts/run_phase.py all        # everything, in order
    python scripts/run_phase.py 5 --quick  # small/fast settings (smoke test)

Build the C++ code first (see README). Phases 2, 8 and 9 need an NVIDIA GPU.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import CHECKPOINT_DIR, RESULTS_DIR, find_pipeline, find_renderer  # noqa: E402

PY = sys.executable
OUT = ROOT / "outputs"


def sh(*cmd, check=True):
    cmd = [str(c) for c in cmd]
    print("\n>>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=check)


def look_at(*paths):
    print("\n==> inspect:")
    for p in paths:
        print("   ", p)


def phase1(a):
    pt = find_renderer()
    sh(pt, "--device", "cpu", "--threads", "1", "--spp", "1", "--width", 256, "--height", 256, "--out", OUT / "phase1/spp1")
    sh(pt, "--device", "cpu", "--spp", 64 if a.quick else 500, "--width", 256, "--height", 256, "--out", OUT / "phase1/spp500")
    look_at(OUT / "phase1/spp1/color.png", OUT / "phase1/spp500/color.png",
            "(expect: Cornell box, soft shadows, red/green colour bleeding onto the white surfaces)")


def phase2(a):
    pt = find_renderer()
    sh(pt, "--info")
    sh(PY, "-m", "benchmarks.cpu_vs_gpu", "--agree-spp", 128 if a.quick else 1024)
    for spp in ([4] if a.quick else [1, 4, 64]):
        sh(pt, "--device", "cuda", "--scene", "builtin:cornell_glossy", "--spp", spp, "--width", 512, "--height", 512,
           "--bench-blocks", "8,16,32", "--frames", 5 if a.quick else 20, "--warmup", 3,
           "--csv", RESULTS_DIR / "block_size_sweep.csv")
    look_at(RESULTS_DIR / "cpu_vs_gpu.csv", RESULTS_DIR / "cpu_vs_gpu_agreement.json",
            RESULTS_DIR / "figures/cpu_vs_gpu_images.png", RESULTS_DIR / "block_size_sweep.csv")


def phase3(a):
    pt = find_renderer()
    sh(pt, "--scene", "builtin:cornell_glossy", "--spp", 64 if a.quick else 256, "--clamp", 8, "--out", OUT / "phase3")
    look_at(*(OUT / f"phase3/{b}.png" for b in ("color", "normal", "albedo", "depth")),
            "(normal: smooth gradients on spheres; albedo: flat colours, no lighting)")


def phase4(a):
    pt = find_renderer()
    sh(pt, "--scene", "builtin:cornell_glossy", "--spp", 64 if a.quick else 256, "--clamp", 8, "--stereo",
       "--out", OUT / "phase4")
    look_at(OUT / "phase4/side_by_side.png", OUT / "phase4/anaglyph.png", OUT / "phase4/left_disparity.png",
            "(anaglyph: near objects show larger red/cyan offset than the back wall)")


def phase5(a):
    if a.quick:
        sh(PY, "-m", "dataset.generate_dataset", "--num-scenes", 8, "--width", 128, "--height", 128, "--gt-spp", 256)
        sh(PY, "-m", "dataset.extract_patches", "--patch-size", 64, "--random-per-image", 16)
    else:
        sh(PY, "-m", "dataset.generate_dataset", "--num-scenes", a.scenes, "--width", 512, "--height", 512,
           "--gt-spp", a.gt_spp, "--previews")
        sh(PY, "-m", "dataset.extract_patches", "--patch-size", 128, "--random-per-image", 48)
    look_at(ROOT / "dataset/patches/stats.json", ROOT / "dataset/patches/spotcheck.png")


def _train_args(a):
    if a.quick:
        return ["--epochs", 3, "--batch-size", 8, "--base", 16]
    return ["--epochs", a.epochs, "--batch-size", a.batch_size, "--amp"]


def phase6(a):
    sh(PY, "-m", "denoiser.train", "--mode", "mono", "--run-name", "baseline", *_train_args(a))
    sh(PY, "-m", "denoiser.evaluate", "--models", f"baseline={CHECKPOINT_DIR / 'baseline/best.pt'}",
       "--out", RESULTS_DIR / "eval_baseline")
    look_at(CHECKPOINT_DIR / "baseline/log.csv", RESULTS_DIR / "eval_baseline/summary.csv",
            RESULTS_DIR / "eval_baseline/*.png", "(denoised PSNR/SSIM should clearly beat 'noisy')")


def phase7(a):
    lambdas = "0,1" if a.quick else a.lambdas
    sh(PY, "-m", "denoiser.sweep_lambda", "--lambdas", lambdas, "--", *_train_args(a))
    stereo = CHECKPOINT_DIR / f"sweep_l{a.stereo_lambda:g}/best.pt"
    if not stereo.exists():
        stereo = CHECKPOINT_DIR / f"sweep_l{lambdas.split(',')[-1]}/best.pt"
    sh(PY, "-m", "denoiser.evaluate", "--models", f"baseline={CHECKPOINT_DIR / 'baseline/best.pt'}",
       f"stereo={stereo}", "--out", RESULTS_DIR / "eval")
    look_at(RESULTS_DIR / "lambda_sweep.csv", RESULTS_DIR / "eval/summary.csv", RESULTS_DIR / "eval/*.png",
            "(headline: 'stereo' disagreement should be lower than 'baseline')")


def _stereo_ckpt(a):
    p = CHECKPOINT_DIR / f"sweep_l{a.stereo_lambda:g}/best.pt"
    return p if p.exists() else sorted(CHECKPOINT_DIR.glob("sweep_l*/best.pt"))[-1]


def phase8(a):
    import json
    manifest = json.loads((ROOT / "dataset/raw/manifest.json").read_text())
    test_scene = next(s for s in manifest["scenes"] if s["split"] == "test")
    half = [] if a.fp32 else ["--half"]
    for ckpt in (CHECKPOINT_DIR / "baseline/best.pt", _stereo_ckpt(a)):
        sh(PY, "-m", "denoiser.export_model", "--checkpoint", ckpt, "--device", "cuda",
           "--height", manifest["height"], "--width", manifest["width"], "--out", ckpt.with_name("best_ts.pt"))
        if half:
            sh(PY, "-m", "denoiser.export_model", "--checkpoint", ckpt, "--device", "cuda", "--half",
               "--height", manifest["height"], "--width", manifest["width"], "--out", ckpt.with_name("best_ts_fp16.pt"))
    sh(find_pipeline(), "--scene", ROOT / test_scene["scene_file"], "--model", CHECKPOINT_DIR / "baseline/best_ts.pt",
       "--width", manifest["width"], "--height", manifest["height"], "--spp", 4, "--clamp", manifest["clamp"],
       "--frames", 30, "--out", OUT / "phase8")
    sh(PY, "-m", "integration.compare_with_python", "--pipeline-out", OUT / "phase8",
       "--checkpoint", CHECKPOINT_DIR / "baseline/best.pt")
    look_at(OUT / "phase8/left_denoised.png", OUT / "phase8/anaglyph.png", OUT / "phase8/timing.json",
            OUT / "phase8/python_comparison.json")


def phase9(a):
    fp16 = not a.fp32
    suffix = "best_ts_fp16.pt" if fp16 else "best_ts.pt"
    base = CHECKPOINT_DIR / "baseline" / suffix
    stereo = _stereo_ckpt(a).with_name(suffix)
    sh(PY, "-m", "benchmarks.frame_time_bench", "--baseline-model", base, "--stereo-model", stereo,
       *(["--half"] if fp16 else []), "--frames", 10 if a.quick else 60, "--high-spp", 256 if a.quick else 1024,
       *(["--max-scenes", 1] if a.quick else []))
    sh(PY, "-m", "benchmarks.divergence_analysis", "--spp", "1,4,16" if a.quick else "1,2,4,8,16,64,256",
       "--gpu-bandwidth-gbps", a.gpu_bandwidth)
    if shutil.which("bash") and shutil.which("nsys") and os.name != "nt":
        sh("bash", "benchmarks/nsight_profile_runs.sh", base.with_name("best_ts.pt"), check=False)
    else:
        print("\n(skipping Nsight: run benchmarks/nsight_profile_runs.sh manually where nsys is installed)")
    sh(PY, "-m", "benchmarks.make_plots")
    look_at(RESULTS_DIR / "frame_time_summary.csv", RESULTS_DIR / "quality_by_config.csv",
            RESULTS_DIR / "divergence.csv", RESULTS_DIR / "figures/")


PHASES = {1: phase1, 2: phase2, 3: phase3, 4: phase4, 5: phase5, 6: phase6, 7: phase7, 8: phase8, 9: phase9}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phases", nargs="+", help="phase numbers 1-9 or 'all'")
    ap.add_argument("--quick", action="store_true", help="small, fast settings for a smoke test")
    ap.add_argument("--scenes", type=int, default=24)
    ap.add_argument("--gt-spp", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lambdas", default="0,0.1,0.5,1,2,5")
    ap.add_argument("--stereo-lambda", type=float, default=1.0, help="which sweep model is 'the' stereo model")
    ap.add_argument("--fp32", action="store_true", help="benchmark FP32 models instead of FP16")
    ap.add_argument("--gpu-bandwidth", type=float, default=0.0,
                    help="GPU memory bandwidth in GB/s from the spec sheet (e.g. RTX 3060 Laptop: 336)")
    a = ap.parse_args()
    phases = list(PHASES) if a.phases == ["all"] else [int(p) for p in a.phases]
    for p in phases:
        print(f"\n{'=' * 70}\nPHASE {p}\n{'=' * 70}")
        PHASES[p](a)


if __name__ == "__main__":
    main()
