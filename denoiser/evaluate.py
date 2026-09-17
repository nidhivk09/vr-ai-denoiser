"""Phase 6/7 validation: full-resolution evaluation on held-out test scenes.

    python -m denoiser.evaluate --models baseline=denoiser/checkpoints/baseline/best.pt \
        stereo=denoiser/checkpoints/stereo_l0.5/best.pt

For each test scene and noise level, compares the noisy input, each model's
output and the ground truth (PSNR / SSIM / stereo disagreement) and saves
side-by-side comparison images. Writes <out>/per_image.csv and <out>/summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.quality_metrics import display, stereo_metrics  # noqa: E402
from common.imageops import labeled_grid  # noqa: E402
from common.paths import RAW_DIR, RESULTS_DIR  # noqa: E402
from dataset import layout  # noqa: E402
from denoiser.dataset import load_scene_pair, scenes_in_split  # noqa: E402
from denoiser.losses import visibility_masks, warp_right_to_left  # noqa: E402
from denoiser.model import denoise_full, load_checkpoint  # noqa: E402
from denoiser.train import pick_device  # noqa: E402


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _img(t: torch.Tensor):
    return t[0].permute(1, 2, 0).cpu().numpy()


def _error_map(left, right, ld, rd):
    lm, _ = visibility_masks(ld, rd)
    r_in_l, _ = warp_right_to_left(display(right), ld)
    err = (display(left) - r_in_l).abs().mean(dim=1, keepdim=True) * lm
    return torch.clamp(err * 5.0, 0, 1).expand(-1, 3, -1, -1)  # x5 for visibility


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="name=checkpoint.pt entries")
    ap.add_argument("--split", default="test")
    ap.add_argument("--spp", default="", help="comma-separated noise levels (default: all in manifest)")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "eval")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.device)
    manifest = layout.load_manifest(RAW_DIR)
    spps = [int(s) for s in args.spp.split(",")] if args.spp else manifest["noisy_spp"]
    models = {}
    for entry in args.models:
        name, path = entry.split("=", 1)
        models[name], _ = load_checkpoint(path, device)
    scenes = scenes_in_split(args.split)
    warm = torch.zeros(2, 9, manifest["height"], manifest["width"], device=device)
    for model in models.values():  # warm-up so timings exclude kernel selection / allocation
        denoise_full(model, warm)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"evaluating {list(models)} on {len(scenes)} {args.split} scenes, spp {spps}, device {device}")

    rows = []
    for sid in scenes:
        for spp in spps:
            s = {k: v.to(device) for k, v in load_scene_pair(sid, spp).items()}
            gl, gr, ld, rd = s["left_target"], s["right_target"], s["left_disp"], s["right_disp"]
            outputs = {"noisy": (s["left_input"][:, :3], s["right_input"][:, :3], 0.0)}
            for name, model in models.items():
                _sync(device)
                t0 = time.perf_counter()
                pred = denoise_full(model, torch.cat([s["left_input"], s["right_input"]], dim=0))
                _sync(device)
                outputs[name] = (pred[0:1], pred[1:2], (time.perf_counter() - t0) * 1000.0)
            outputs["ground_truth"] = (gl, gr, 0.0)

            for method, (l, r, ms) in outputs.items():
                m = stereo_metrics(l, r, gl, gr, ld, rd, args.lpips)
                rows.append({"scene": sid, "spp": spp, "method": method, **m, "denoise_ms_both_eyes": ms})

            if not args.no_images:
                grid = [[_img(display(l)), _img(display(r)), _img(_error_map(l, r, ld, rd))]
                        for (l, r, _) in outputs.values()]
                labeled_grid(grid, ["left", "right", "|L - warp(R)| x5"], list(outputs)).save(
                    args.out / f"{sid}_spp{spp:04d}.png")

    keys = [k for k in rows[0] if k not in ("scene", "spp", "method")]
    with open(args.out / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["spp"])].append(r)
    summary = []
    for (method, spp), items in groups.items():
        agg = {"method": method, "spp": spp, "n": len(items)}
        for k in keys:
            vals = [it[k] for it in items if it[k] is not None]
            agg[k] = sum(vals) / len(vals) if vals else None
        summary.append(agg)
    with open(args.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    print(f"\n{'method':>16} {'spp':>4} {'PSNR':>7} {'SSIM':>7} {'disagree':>9} {'ms(2 eyes)':>10}")
    for a in summary:
        print(f"{a['method']:>16} {a['spp']:>4} {a['psnr']:7.2f} {a['ssim']:7.4f} {a['disagreement']:9.4f} "
              f"{a['denoise_ms_both_eyes']:10.1f}")
    print(f"\nwrote {args.out / 'summary.csv'} and comparison images")


if __name__ == "__main__":
    main()
