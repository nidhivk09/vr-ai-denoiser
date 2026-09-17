"""Phase 8 fallback: Python end-to-end pipeline (renderer subprocess + PyTorch).

Use this if the LibTorch C++ build is not available. The renderer is invoked
as a subprocess per frame; the reported render time is the renderer's own
kernel timing (process start-up and file I/O are excluded), and denoising is
timed around the PyTorch forward pass. Because buffers travel through files,
this does not measure the zero-copy GPU hand-off of run_pipeline.cpp - report
that as a limitation if you use these numbers.

    python -m integration.python_pipeline --scene dataset/scenes/scene_003.scene \
        --checkpoint denoiser/checkpoints/baseline/best.pt --spp 4 --frames 10
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.exr import read_exr, write_exr  # noqa: E402
from common.imageops import anaglyph, save_png, tonemap  # noqa: E402
from common.paths import find_renderer  # noqa: E402
from denoiser.model import denoise_full, load_checkpoint  # noqa: E402
from denoiser.train import pick_device  # noqa: E402

VR_BUDGET_MS = 1000.0 / 90.0


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="builtin:cornell")
    ap.add_argument("--checkpoint", type=Path, default=None, help="omit for path tracing only")
    ap.add_argument("--config", default=None)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--spp", type=int, default=4)
    ap.add_argument("--clamp", type=float, default=8.0)
    ap.add_argument("--ipd", type=float, default=0.063)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--render-device", default="", help="cpu or cuda (renderer)")
    ap.add_argument("--device", default="auto", help="denoiser device")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/python_pipeline"))
    args = ap.parse_args()

    config = args.config or ("denoised" if args.checkpoint else "pathtrace_only")
    device = pick_device(args.device)
    model = load_checkpoint(args.checkpoint, device)[0] if args.checkpoint else None
    renderer = find_renderer()
    scene_name = Path(args.scene).stem if not args.scene.startswith("builtin:") else args.scene.split(":", 1)[1]

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in range(-args.warmup, args.frames):
            timing_csv = tmp / "t.csv"
            timing_csv.unlink(missing_ok=True)
            cmd = [str(renderer), "--scene", args.scene, "--stereo", "--out", str(tmp / "frame"), "--no-png",
                   "--width", str(args.width), "--height", str(args.height), "--spp", str(args.spp),
                   "--clamp", str(args.clamp), "--ipd", str(args.ipd), "--frame", str(max(f, 0)),
                   "--csv", str(timing_csv), "--quiet"]
            if args.render_device:
                cmd += ["--device", args.render_device]
            subprocess.run(cmd, check=True)
            with open(timing_csv) as fh:
                eye_ms = {r["eye"]: float(r["render_ms"]) for r in csv.DictReader(fh)}

            eyes = {}
            for eye in ("left", "right"):
                parts = [read_exr(tmp / "frame" / f"{eye}_{b}.exr") for b in ("color", "normal", "albedo")]
                eyes[eye] = torch.from_numpy(np.concatenate(parts, -1).transpose(2, 0, 1)).unsqueeze(0)

            denoise_ms = 0.0
            outputs = {e: t[:, :3] for e, t in eyes.items()}
            if model is not None:
                x = torch.cat([eyes["left"], eyes["right"]], 0).to(device)  # host->device (untimed, see docstring)
                _sync(device)
                t0 = time.perf_counter()
                y = denoise_full(model, x)
                _sync(device)
                denoise_ms = (time.perf_counter() - t0) * 1000.0
                outputs = {"left": y[0:1].cpu(), "right": y[1:2].cpu()}
            if f < 0:
                continue
            total = eye_ms["left"] + eye_ms["right"] + denoise_ms
            rows.append({"config": config, "scene": scene_name, "device": str(device), "width": args.width,
                         "height": args.height, "spp": args.spp, "frame": f, "render_left_ms": eye_ms["left"],
                         "render_right_ms": eye_ms["right"], "denoise_ms": denoise_ms, "readback_ms": 0.0,
                         "total_ms": total})

        args.out.mkdir(parents=True, exist_ok=True)
        imgs = {}
        for eye in ("left", "right"):
            noisy = eyes[eye][0, :3].permute(1, 2, 0).numpy()
            write_exr(args.out / f"{eye}_color.exr", noisy)
            out = outputs[eye][0].permute(1, 2, 0).numpy()
            if model is not None:
                write_exr(args.out / f"{eye}_denoised.exr", out)
            imgs[eye] = tonemap(out)
            save_png(args.out / f"{eye}_{'denoised' if model is not None else 'color'}.png", imgs[eye])
        save_png(args.out / "anaglyph.png", anaglyph(imgs["left"], imgs["right"]))

    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("render_left_ms", "render_right_ms", "denoise_ms", "total_ms")}
    print(f"config {config}: render L {mean['render_left_ms']:.2f} ms, R {mean['render_right_ms']:.2f} ms, "
          f"denoise {mean['denoise_ms']:.2f} ms, total {mean['total_ms']:.2f} ms "
          f"({1000 / mean['total_ms']:.1f} FPS; VR budget {VR_BUDGET_MS:.1f} ms "
          f"{'MET' if mean['total_ms'] <= VR_BUDGET_MS else 'missed'})")
    (args.out / "timing.json").write_text(json.dumps({"config": config, **mean}, indent=2))
    if args.csv:
        new = not args.csv.exists() or args.csv.stat().st_size == 0
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
