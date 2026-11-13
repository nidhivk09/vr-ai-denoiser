"""Phase 8 validation: check that the C++ pipeline's denoised output matches
PyTorch inference on the same noisy buffers.

    python -m integration.compare_with_python --pipeline-out outputs/pipeline \
        --checkpoint denoiser/checkpoints/baseline/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.exr import read_exr  # noqa: E402
from denoiser.model import denoise_full, load_checkpoint, pair_input  # noqa: E402
from denoiser.train import pick_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline-out", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    report = {}

    def load(eye: str, buffers: tuple[str, ...]) -> torch.Tensor:
        x = np.concatenate([read_exr(args.pipeline_out / f"{eye}_{b}.exr") for b in buffers], -1)
        return torch.from_numpy(x.transpose(2, 0, 1))[None]

    inputs = {eye: load(eye, ("color", "normal", "albedo")) for eye in ("left", "right")}
    # Only fusion models read the disparity; older pipeline outputs do not have the file.
    disp = {eye: load(eye, ("disparity",)) if model.fusion else None for eye in ("left", "right")}
    with torch.no_grad():
        y = denoise_full(model, pair_input(model, inputs["left"], inputs["right"], disp["left"], disp["right"]).to(device))
    for i, eye in enumerate(("left", "right")):
        py = y[i].permute(1, 2, 0).cpu().numpy()
        cpp = read_exr(args.pipeline_out / f"{eye}_denoised.exr")
        diff = np.abs(cpp - py)
        report[eye] = {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
                       "mean_value": float(py.mean())}
    ok = all(v["mean_abs_diff"] < 1e-3 * max(v["mean_value"], 1e-3) + 1e-4 for v in report.values())
    report["match"] = ok
    print(json.dumps(report, indent=2))
    print("C++ and Python outputs MATCH" if ok else "WARNING: C++ and Python outputs differ")
    (args.pipeline_out / "python_comparison.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
