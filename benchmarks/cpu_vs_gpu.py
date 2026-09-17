"""Phase 2 validation: CPU reference vs CUDA renderer.

1. Image agreement: renders the same scene at high spp on CPU and GPU (the
   noise patterns differ, so the images converge to the same answer rather than
   being bit-identical) and reports PSNR/SSIM between them.
2. Speed-up: times single-threaded CPU, multi-threaded CPU and CUDA at the
   same (lower) spp.

Writes benchmarks/results/cpu_vs_gpu.csv, cpu_vs_gpu_agreement.json and
figures/cpu_vs_gpu_images.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.quality_metrics import psnr, ssim  # noqa: E402
from common.exr import read_exr  # noqa: E402
from common.imageops import labeled_grid, tonemap  # noqa: E402
from common.paths import RESULTS_DIR, find_renderer  # noqa: E402


def run(renderer, *extra):
    subprocess.run([str(renderer), *map(str, extra)], check=True)


def timed(renderer, device, spp, args, threads=None) -> float:
    csv_path = RESULTS_DIR / "_tmp_timing.csv"
    csv_path.unlink(missing_ok=True)
    extra = ["--device", device, "--spp", spp, "--width", args.width, "--height", args.height,
             "--scene", args.scene, "--frames", args.frames, "--warmup", 1, "--no-save", "--quiet",
             "--csv", csv_path]
    if threads:
        extra += ["--threads", threads]
    run(args.renderer, *extra)
    with open(csv_path) as f:
        ms = [float(r["render_ms"]) for r in csv.DictReader(f)]
    csv_path.unlink()
    return sum(ms) / len(ms)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="builtin:cornell")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--agree-spp", type=int, default=1024)
    ap.add_argument("--timing-spp", type=int, default=16)
    ap.add_argument("--frames", type=int, default=3)
    args = ap.parse_args()
    args.renderer = find_renderer()
    out = RESULTS_DIR
    (out / "figures").mkdir(parents=True, exist_ok=True)

    imgs = {}
    for device, seed in (("cpu", 11), ("cuda", 22)):
        d = out / "cpu_vs_gpu" / device
        run(args.renderer, "--device", device, "--scene", args.scene, "--spp", args.agree_spp, "--seed", seed,
            "--width", args.width, "--height", args.height, "--out", d, "--quiet")
        imgs[device] = torch.from_numpy(read_exr(d / "color.exr")).permute(2, 0, 1).unsqueeze(0)
    agreement = {"spp": args.agree_spp, "psnr_cpu_vs_gpu": psnr(imgs["cuda"], imgs["cpu"]),
                 "ssim_cpu_vs_gpu": ssim(imgs["cuda"], imgs["cpu"]),
                 "mean_abs_linear_diff": (imgs["cuda"] - imgs["cpu"]).abs().mean().item()}
    (out / "cpu_vs_gpu_agreement.json").write_text(json.dumps(agreement, indent=2))
    diff = (imgs["cuda"] - imgs["cpu"]).abs()[0].permute(1, 2, 0).numpy() * 10
    labeled_grid([[tonemap(imgs["cpu"][0].permute(1, 2, 0).numpy()),
                   tonemap(imgs["cuda"][0].permute(1, 2, 0).numpy()), diff.clip(0, 1)]],
                 ["CPU", "CUDA", "|diff| x10"]).save(out / "figures" / "cpu_vs_gpu_images.png")
    print(json.dumps(agreement, indent=2))

    cores = os.cpu_count() or 1
    timings = {
        "cpu_1_thread": timed(args.renderer, "cpu", args.timing_spp, args, threads=1),
        f"cpu_{cores}_threads": timed(args.renderer, "cpu", args.timing_spp, args, threads=cores),
        "cuda": timed(args.renderer, "cuda", args.timing_spp, args),
    }
    base = timings["cpu_1_thread"]
    samples = args.width * args.height * args.timing_spp
    with open(out / "cpu_vs_gpu.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "width", "height", "spp", "ms_per_frame", "msamples_per_sec", "speedup_vs_1_thread"])
        for k, ms in timings.items():
            w.writerow([k, args.width, args.height, args.timing_spp, f"{ms:.3f}", f"{samples / ms * 1e-3:.3f}",
                        f"{base / ms:.2f}"])
            print(f"{k:>16}: {ms:10.2f} ms  speed-up x{base / ms:.1f}")
    print(f"wrote {out / 'cpu_vs_gpu.csv'}")


if __name__ == "__main__":
    main()
