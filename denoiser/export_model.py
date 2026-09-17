"""Phase 8: export a trained checkpoint to TorchScript for the C++ pipeline.

    python -m denoiser.export_model --checkpoint denoiser/checkpoints/baseline/best.pt \
        --out denoiser/checkpoints/baseline/denoiser_ts.pt --device cuda

The exported module takes a raw (N, 9, H, W) float tensor (noisy color, normal,
albedo straight from the renderer) and returns (N, 3, H, W) linear HDR color.
H and W must be multiples of 2**levels; the C++ pipeline pads to 16.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from denoiser.model import load_checkpoint  # noqa: E402
from denoiser.train import pick_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--half", action="store_true", help="export FP16 weights (CUDA only)")
    ap.add_argument("--onnx", action="store_true", help="also write an ONNX file")
    args = ap.parse_args()

    device = pick_device(args.device)
    if device.type == "mps":
        device = torch.device("cpu")  # C++ LibTorch has no MPS path; trace on CPU
    model, ckpt = load_checkpoint(args.checkpoint, device)
    dtype = torch.float16 if args.half else torch.float32
    if args.half:
        if device.type != "cuda":
            raise SystemExit("--half requires --device cuda")
        model = model.half()
    out = args.out or args.checkpoint.with_name(args.checkpoint.stem + ("_ts_fp16.pt" if args.half else "_ts.pt"))

    mult = 2 ** model.levels
    h = (args.height + mult - 1) // mult * mult
    w = (args.width + mult - 1) // mult * mult
    example = torch.rand(2, 9, h, w, device=device, dtype=dtype)
    example[:, 0:3] *= 4.0
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
        traced = torch.jit.freeze(traced.eval())
        # Verify against eager mode, including a different resolution.
        for shape in [(2, 9, h, w), (1, 9, h // 2, w // 2 + mult)]:
            x = torch.rand(*shape, device=device, dtype=dtype) * 2.0
            err = (traced(x).float() - model(x).float()).abs().max().item()
            print(f"traced vs eager max abs diff at {tuple(shape)}: {err:.2e}")
            if err > 1e-2:
                raise SystemExit("traced model does not match eager model")
    out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out))
    meta = {"checkpoint": str(args.checkpoint), "run": ckpt.get("run"), "mode": ckpt.get("mode"),
            "lambda_cons": ckpt.get("lambda_cons"), "pad_multiple": mult, "half": args.half,
            "traced_on": str(device), **ckpt["model_config"]}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"saved TorchScript model to {out}")

    if args.onnx:
        onnx_path = out.with_suffix(".onnx")
        torch.onnx.export(model, example, str(onnx_path), input_names=["buffers"], output_names=["color"],
                          dynamic_axes={"buffers": {0: "n", 2: "h", 3: "w"}, "color": {0: "n", 2: "h", 3: "w"}},
                          opset_version=17)
        print(f"saved ONNX model to {onnx_path}")


if __name__ == "__main__":
    main()
