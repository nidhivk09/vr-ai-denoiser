"""Where the cross-eye exchange's frame time goes (CUDA, FP16, both eyes).

Times traced FP16 models with random weights (speed does not depend on the
weights, so nothing needs training) at the dataset resolution, batch = one stereo
pair, the way the C++ pipeline runs them:

  baseline / fusion / blend / xchg           as trained (channel counts 9, 13, 2c+1)
  ... aligned                                conv inputs zero-padded to multiples of 8
                                             (Denoiser.align_channels; same output)
  xchg levels a,b aligned                    exchange only at some U-Net levels
                                             (would need retraining: --exchange-levels)

and profiles the CUDA kernels of xchg with and without alignment. The C++
pipeline (run_exchange.py timing) stays the reported number; this explains it.

    python -m benchmarks.bench_exchange            -> benchmarks/results/exchange_bench.csv, exchange_bench_kernels.txt
"""
from __future__ import annotations

import argparse
import csv
import time
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import RAW_DIR, RESULTS_DIR  # noqa: E402
from denoiser.model import Denoiser  # noqa: E402

CONFIGS = [  # name, Denoiser kwargs, aligned
    ("baseline", dict(fusion=False), False),
    ("baseline", dict(fusion=False), True),
    ("fusion", dict(fusion=True), False),
    ("fusion", dict(fusion=True), True),
    ("blend", dict(fusion=True, blend=True), False),
    ("blend", dict(fusion=True, blend=True), True),
    ("xchg", dict(fusion=True, exchange=True), False),
    ("xchg", dict(fusion=True, exchange=True), True),
    ("xchg levels 1,2,3", dict(fusion=True, exchange=True, exchange_levels=[1, 2, 3]), True),
    ("xchg levels 2,3", dict(fusion=True, exchange=True, exchange_levels=[2, 3]), True),
    ("xchg levels 3", dict(fusion=True, exchange=True, exchange_levels=[3]), True),
    ("xchg levels 0", dict(fusion=True, exchange=True, exchange_levels=[0]), True),
]


def build(kwargs: dict, aligned: bool, h: int, w: int, device: torch.device, dtype: torch.dtype):
    torch.manual_seed(0)
    model = Denoiser(32, 3, **kwargs).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(std=0.02)
    if aligned:
        model.align_channels()
    model = model.to(device=device, dtype=dtype)
    x = torch.rand(2, model.in_channels, h, w, device=device, dtype=dtype)
    if model.fusion:
        x[:, 9:10] = 4.0 + 8.0 * torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, 1, 1, w)
    with torch.no_grad():
        traced = torch.jit.freeze(torch.jit.trace(model, x).eval())
    n_ops = sum(1 for node in traced.graph.nodes() if node.kind().startswith("aten::"))
    return traced, x, n_ops


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def time_ms(fn, x, iters: int, warmup: int) -> list[float]:
    """Per-call wall time with the device synchronised (CUDA events on CUDA)."""
    device = x.device
    for _ in range(warmup):
        fn(x)
    _sync(device)
    out = []
    for _ in range(iters):
        if device.type == "cuda":
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn(x)
            b.record()
            torch.cuda.synchronize()
            out.append(a.elapsed_time(b))
        else:
            t0 = time.perf_counter()
            fn(x)
            _sync(device)
            out.append((time.perf_counter() - t0) * 1000.0)
    return out


@torch.no_grad()
def kernel_table(fn, x, rows: int = 25) -> str:
    from torch.profiler import ProfilerActivity, profile
    for _ in range(20):
        fn(x)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            fn(x)
        torch.cuda.synchronize()
    return prof.key_averages().table(sort_by="cuda_time_total", row_limit=rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "exchange_bench.csv")
    ap.add_argument("--device", default="auto", help="cuda (the target), mps or cpu (indicative only: no cuDNN)")
    ap.add_argument("--width", type=int, default=0, help="per-eye size (default: the dataset's, else 256)")
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--fp32", action="store_true", help="time in FP32 instead of FP16 (FP16 is what the pipeline runs)")
    args = ap.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(args.device)
    dtype = torch.float32 if args.fp32 or device.type == "cpu" else torch.float16
    torch.backends.cudnn.benchmark = True
    manifest_path = RAW_DIR / "manifest.json"
    if args.width and args.height:
        h, w = args.height, args.width
    elif manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        h, w = manifest["height"], manifest["width"]
    else:  # no dataset (e.g. a fresh Colab): the resolution of the GPU runs
        h, w = 256, 256
    gpu = torch.cuda.get_device_name() if device.type == "cuda" else device.type
    print(f"{gpu} | {w}x{h} per eye, {str(dtype).split('.')[-1]}, traced, both eyes per call")
    if device.type != "cuda":
        print("note: not CUDA - op overhead shows, but not the FP16 tensor-core effect of alignment")

    rows = []
    for name, kwargs, aligned in CONFIGS:
        traced, x, n_ops = build(kwargs, aligned, h, w, device, dtype)
        t = time_ms(traced, x, args.iters, args.warmup)
        t.sort()
        row = {"device": gpu, "dtype": str(dtype).split(".")[-1], "model": name, "aligned": aligned, "ops": n_ops, "median_ms": statistics.median(t),
               "p95_ms": t[int(0.95 * (len(t) - 1))], "min_ms": t[0]}
        rows.append(row)
        print(f"{name:>20} {'aligned' if aligned else 'as trained':>10} {n_ops:4d} ops  "
              f"median {row['median_ms']:.3f} ms  p95 {row['p95_ms']:.3f} ms")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    if device.type != "cuda":
        print(f"wrote {args.out}")
        return
    tables = []
    for aligned in (False, True):
        traced, x, _ = build(dict(fusion=True, exchange=True), aligned, h, w, device, dtype)
        tables.append(f"== xchg, {'aligned' if aligned else 'as trained'}\n{kernel_table(traced, x)}")
    kpath = args.out.with_name(args.out.stem + "_kernels.txt")
    kpath.write_text("\n\n".join(tables))
    print(f"wrote {args.out} and {kpath}")


if __name__ == "__main__":
    main()
