"""Phase 9: frame-time and quality benchmark of the four configurations

  (a) highspp_pathtrace  brute-force high-spp path tracing
  (b) lowspp_pathtrace   low-spp path tracing, no denoising (speed floor)
  (c) lowspp_baseline    low-spp + single-image denoiser
  (d) lowspp_stereo      low-spp + stereo-consistent denoiser

on the held-out test scenes, using the C++ pipeline (or --python fallback).
Outputs in benchmarks/results/: frame_times.csv (every frame),
frame_time_summary.csv, quality_by_config.csv, quality_per_scene.csv.

    python -m benchmarks.frame_time_bench \
        --baseline-model denoiser/checkpoints/baseline/best_ts.pt \
        --stereo-model denoiser/checkpoints/stereo_l0.5/best_ts.pt
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.quality_metrics import stereo_metrics  # noqa: E402
from common.exr import read_exr  # noqa: E402
from common.paths import RAW_DIR, REPO_ROOT, RESULTS_DIR, find_pipeline  # noqa: E402
from dataset import layout  # noqa: E402

VR_BUDGET_MS = 1000.0 / 90.0


def _load(path: Path) -> torch.Tensor:
    return torch.from_numpy(read_exr(path)).permute(2, 0, 1).unsqueeze(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-model", type=Path, help="TorchScript (C++) or .pt checkpoint (--python)")
    ap.add_argument("--stereo-model", type=Path)
    ap.add_argument("--low-spp", type=int, default=4)
    ap.add_argument("--high-spp", type=int, default=1024)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--high-frames", type=int, default=3)
    ap.add_argument("--max-scenes", type=int, default=0, help="limit number of test scenes (0 = all)")
    ap.add_argument("--half", action="store_true", help="models were exported with --half")
    ap.add_argument("--readback", action="store_true", help="include per-frame GPU->CPU copy")
    ap.add_argument("--python", action="store_true", help="use integration/python_pipeline.py (needs .pt checkpoints)")
    ap.add_argument("--device", default="", help="cpu or cuda")
    args = ap.parse_args()

    manifest = layout.load_manifest(RAW_DIR)
    scenes = [s for s in manifest["scenes"] if s["split"] == "test"]
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    configs = [("highspp_pathtrace", args.high_spp, None, args.high_frames),
               ("lowspp_pathtrace", args.low_spp, None, args.frames)]
    if args.baseline_model:
        configs.append(("lowspp_baseline", args.low_spp, args.baseline_model, args.frames))
    if args.stereo_model:
        configs.append(("lowspp_stereo", args.low_spp, args.stereo_model, args.frames))

    frame_csv = RESULTS_DIR / "frame_times.csv"
    frame_csv.unlink(missing_ok=True)
    frames_dir = RESULTS_DIR / "frames"
    quality_rows = []
    for s in scenes:
        scene_file = REPO_ROOT / s["scene_file"]
        for name, spp, model, frames in configs:
            out = frames_dir / name / s["id"]
            common = ["--scene", str(scene_file), "--width", str(manifest["width"]), "--height",
                      str(manifest["height"]), "--spp", str(spp), "--clamp", str(manifest["clamp"]),
                      "--ipd", str(manifest["ipd"]), "--frames", str(frames), "--config", name,
                      "--csv", str(frame_csv), "--out", str(out)]
            if args.python:
                cmd = [sys.executable, "-m", "integration.python_pipeline", *common,
                       "--warmup", "1" if spp >= 256 else "2"]
                if model:
                    cmd += ["--checkpoint", str(model)]
                if args.device:
                    cmd += ["--render-device", args.device, "--device", args.device]
            else:
                cmd = [str(find_pipeline()), *common, "--warmup", "1" if spp >= 256 else "5"]
                if model:
                    cmd += ["--model", str(model)] + (["--half"] if args.half else [])
                if args.readback:
                    cmd.append("--readback")
                if args.device:
                    cmd += ["--device", args.device]
            print(f">> {s['id']} {name}", flush=True)
            subprocess.run(cmd, check=True, cwd=REPO_ROOT)

            stem = "denoised" if model else "color"
            gt = RAW_DIR / s["id"] / "gt"
            m = stereo_metrics(_load(out / f"left_{stem}.exr"), _load(out / f"right_{stem}.exr"),
                               _load(gt / "left_color.exr"), _load(gt / "right_color.exr"),
                               _load(gt / "left_disparity.exr"), _load(gt / "right_disparity.exr"))
            quality_rows.append({"config": name, "scene": s["id"], "spp": spp, **m})

    with open(frame_csv) as f:
        frames = list(csv.DictReader(f))
    by = defaultdict(list)
    for r in frames:
        by[r["config"]].append(r)
    summary = []
    for name, *_ in configs:
        rows = by[name]
        tot = np.array([float(r["total_ms"]) for r in rows])
        row = {"config": name, "spp": rows[0]["spp"], "frames": len(rows)}
        for k in ("render_left_ms", "render_right_ms", "denoise_ms", "readback_ms"):
            row[k] = float(np.mean([float(r[k]) for r in rows]))
        row.update({"total_ms": float(tot.mean()), "total_p95_ms": float(np.percentile(tot, 95)),
                    "fps": 1000.0 / float(tot.mean()), "meets_vr_budget": bool(tot.mean() <= VR_BUDGET_MS)})
        summary.append(row)
    with open(RESULTS_DIR / "frame_time_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    with open(RESULTS_DIR / "quality_per_scene.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(quality_rows[0]))
        w.writeheader()
        w.writerows(quality_rows)
    qagg = []
    for name, spp, *_ in configs:
        rs = [q for q in quality_rows if q["config"] == name]
        qagg.append({"config": name, "spp": spp, **{k: float(np.mean([q[k] for q in rs]))
                                                   for k in ("psnr", "ssim", "disagreement")}})
    with open(RESULTS_DIR / "quality_by_config.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(qagg[0]))
        w.writeheader()
        w.writerows(qagg)

    print(f"\n{'config':>20} {'total ms':>9} {'p95':>8} {'fps':>7} {'VR?':>5} {'PSNR':>7} {'SSIM':>7} {'disagree':>9}")
    for srow, q in zip(summary, qagg):
        print(f"{srow['config']:>20} {srow['total_ms']:9.2f} {srow['total_p95_ms']:8.2f} {srow['fps']:7.1f} "
              f"{'yes' if srow['meets_vr_budget'] else 'no':>5} {q['psnr']:7.2f} {q['ssim']:7.4f} {q['disagreement']:9.4f}")


if __name__ == "__main__":
    main()
