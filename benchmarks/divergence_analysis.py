"""Phase 9: workload divergence and memory-throughput analysis of the path tracer.

The renderer's --stats mode counts, per pixel, the BVH nodes visited,
primitives tested, path segments and shadow rays. From these this script
derives:

  divergence (block B)  For each BxB thread block, threads that finish early
                        sit idle while the slowest pixel finishes its work
                        (lockstep execution). divergence = sum(max - w) / sum(max)
                        over blocks, occupancy = 1 - divergence.
  path-length divergence  same, using path segments instead of traversal work.
  coefficient of variation of per-pixel work.
  memory throughput     bytes of BVH nodes + primitives read per second
                        (visited * sizeof(node) + tested * sizeof(primitive)) / time,
                        and its ratio to the GPU's rated memory bandwidth.

These are computed from instrumentation, not read from Nsight; Nsight
Systems/Compute runs (nsight_profile_runs.sh) provide the timeline and
hardware-level cross-checks. If `nvidia-smi` is present, its memory
utilization counter (share of time the memory bus was busy) is sampled too.

    python -m benchmarks.divergence_analysis --scene dataset/scenes/scene_003.scene \
        --spp 1,2,4,8,16,64,256 --gpu-bandwidth-gbps 224
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.exr import read_exr_channels  # noqa: E402
from common.imageops import save_png  # noqa: E402
from common.paths import RESULTS_DIR, find_renderer  # noqa: E402


def block_divergence(work: np.ndarray, block: int) -> float:
    h, w = work.shape
    hh, ww = h // block * block, w // block * block
    blocks = work[:hh, :ww].reshape(hh // block, block, ww // block, block).transpose(0, 2, 1, 3).reshape(-1, block * block)
    mx = blocks.max(axis=1)
    capacity = mx.sum() * block * block
    used = blocks.sum()
    return float(1.0 - used / capacity) if capacity > 0 else 0.0


def colormap(x: np.ndarray) -> np.ndarray:
    """Perceptual-ish sequential map (dark blue -> yellow) without matplotlib."""
    stops = np.array([[0.05, 0.03, 0.2], [0.25, 0.1, 0.5], [0.7, 0.2, 0.4], [0.98, 0.55, 0.1], [0.99, 0.95, 0.4]])
    t = np.clip(x, 0, 1) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    f = (t - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


class SmiSampler:
    """Samples nvidia-smi utilization.gpu / utilization.memory in the background."""

    def __init__(self):
        self.proc = None
        if shutil.which("nvidia-smi"):
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory", "--format=csv,noheader,nounits",
                 "-lms", "100"], stdout=subprocess.PIPE, text=True)

    def stop(self) -> tuple[float | None, float | None]:
        if not self.proc:
            return None, None
        self.proc.terminate()
        out, _ = self.proc.communicate(timeout=5)
        vals = [tuple(float(v) for v in line.split(",")) for line in out.strip().splitlines() if "," in line]
        busy = [v for v in vals if v[0] > 0] or vals
        if not busy:
            return None, None
        return float(np.mean([v[0] for v in busy])), float(np.mean([v[1] for v in busy]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="builtin:cornell_glossy")
    ap.add_argument("--spp", default="1,2,4,8,16,64,256")
    ap.add_argument("--blocks", default="8,16,32")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frames", type=int, default=10, help="timed frames per spp (without --stats)")
    ap.add_argument("--gpu-bandwidth-gbps", type=float, default=0.0,
                    help="rated memory bandwidth of the GPU in GB/s (from the spec sheet), for utilization")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    renderer = find_renderer()
    blocks = [int(b) for b in args.blocks.split(",")]
    work_dir = args.out / "divergence"
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    rows = []
    for spp in [int(s) for s in args.spp.split(",")]:
        d = work_dir / f"spp_{spp:04d}"
        common = ["--scene", args.scene, "--spp", str(spp), "--width", str(args.width),
                  "--height", str(args.height), "--device", args.device, "--quiet"]
        subprocess.run([str(renderer), *common, "--stats", "--out", str(d), "--no-png"], check=True)
        meta = json.loads((d / "meta.json").read_text())
        st = read_exr_channels(d / "stats.exr")
        traversal = st["nodes"] + st["prims"]
        segments = st["bounces"]

        timing_csv = d / "timing.csv"
        timing_csv.unlink(missing_ok=True)
        sampler = SmiSampler()
        t0 = time.perf_counter()
        subprocess.run([str(renderer), *common, "--frames", str(args.frames), "--warmup", "2", "--no-save",
                        "--csv", str(timing_csv)], check=True)
        wall = time.perf_counter() - t0
        gpu_util, mem_util = sampler.stop()
        with open(timing_csv) as f:
            frame_ms = float(np.mean([float(r["render_ms"]) for r in csv.DictReader(f)]))

        bytes_read = (st["nodes"].sum() * meta["bvh_node_bytes"] + st["prims"].sum() * meta["primitive_bytes"])
        gbps = bytes_read / (frame_ms / 1000.0) / 1e9
        rays = st["bounces"].sum() + st["shadow"].sum()
        row = {
            "scene": meta["scene"], "spp": spp, "width": args.width, "height": args.height,
            "frame_ms": frame_ms, "mrays_per_sec": rays / (frame_ms / 1000.0) / 1e6,
            "work_per_sample_mean": float(traversal.mean() / spp),
            "work_cv": float(traversal.std() / max(traversal.mean(), 1e-9)),
            "segments_per_sample": float(segments.mean() / spp),
            "traversal_gb_per_sec": gbps,
            "bandwidth_utilization": gbps / args.gpu_bandwidth_gbps if args.gpu_bandwidth_gbps > 0 else "",
            "smi_gpu_util_pct": "" if gpu_util is None else gpu_util,
            "smi_mem_util_pct": "" if mem_util is None else mem_util,
            "timing_wall_sec": wall,
        }
        for b in blocks:
            row[f"divergence_b{b}"] = block_divergence(traversal, b)
            row[f"path_divergence_b{b}"] = block_divergence(segments, b)
        rows.append(row)
        print(f"spp {spp:5d}: {frame_ms:9.2f} ms, divergence(16x16) {row.get('divergence_b16', float('nan')):.3f}, "
              f"work CV {row['work_cv']:.3f}, traversal {gbps:.3f} GB/s", flush=True)

        if spp == 1 or spp == min(int(s) for s in args.spp.split(",")):
            norm = traversal / np.percentile(traversal, 99.5)
            save_png(args.out / "figures" / f"work_heatmap_spp{spp}.png", colormap(norm))

    dest = args.out / "divergence.csv"
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
