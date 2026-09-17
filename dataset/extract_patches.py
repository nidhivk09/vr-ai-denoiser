"""Phase 5: crop the full-resolution stereo renders into training patches.

Left and right patches are cropped at the same pixel window so the stereo
consistency loss can warp between them. Splits come from the manifest (by
scene), so no scene contributes patches to more than one split.

Writes dataset/patches/{train,val,test}.npy (float16, N x 28 x P x P),
{split}_index.csv, stats.json and spotcheck.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.imageops import labeled_grid, normalize, tonemap  # noqa: E402
from common.paths import PATCH_DIR, RAW_DIR  # noqa: E402
from dataset import layout  # noqa: E402


def crop_positions(rng: np.random.Generator, h: int, w: int, size: int, count: int, stride: int) -> list[tuple[int, int]]:
    """A regular grid (with overlap) plus random crops."""
    pos = []
    if stride > 0:
        for y in range(0, h - size + 1, stride):
            for x in range(0, w - size + 1, stride):
                pos.append((y, x))
    for _ in range(count):
        pos.append((int(rng.integers(0, h - size + 1)), int(rng.integers(0, w - size + 1))))
    return pos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=PATCH_DIR)
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--random-per-image", type=int, default=16, help="random crops per (scene, spp)")
    ap.add_argument("--stride", type=int, default=0, help="also take a grid of crops with this stride (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = layout.load_manifest(args.raw)
    size = args.patch_size
    if size > min(manifest["width"], manifest["height"]):
        raise SystemExit(f"patch size {size} is larger than the renders ({manifest['width']}x{manifest['height']})")
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    stats = {"patch_size": size, "channels": 2 * layout.EYE_CHANNELS, "splits": {}}
    spot_rows = []
    for split in ("train", "val", "test"):
        scenes = [s["id"] for s in manifest["scenes"] if s["split"] == split]
        jobs = []
        for sid in scenes:
            for spp in manifest["noisy_spp"]:
                for (y, x) in crop_positions(rng, manifest["height"], manifest["width"], size,
                                             args.random_per_image, args.stride):
                    jobs.append((sid, spp, y, x))
        n = len(jobs)
        arr = np.lib.format.open_memmap(args.out / f"{split}.npy", mode="w+", dtype=np.float16,
                                        shape=(n, 2 * layout.EYE_CHANNELS, size, size))
        with open(args.out / f"{split}_index.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["patch", "scene", "spp", "y", "x"])
            cache_key, cache = None, None
            for i, (sid, spp, y, x) in enumerate(jobs):
                if cache_key != (sid, spp):
                    cache = layout.load_pair(args.raw / sid, spp)
                    cache_key = (sid, spp)
                arr[i] = cache[:, y:y + size, x:x + size]
                wr.writerow([i, sid, spp, y, x])
                if split == "train" and len(spot_rows) < 6 and i % max(1, n // 6) == 0:
                    p = cache[:, y:y + size, x:x + size].transpose(1, 2, 0)
                    L, R = 0, layout.EYE_CHANNELS
                    spot_rows.append([
                        tonemap(p[..., L:L + 3]), tonemap(p[..., L + 9:L + 12]), p[..., L + 3:L + 6],
                        tonemap(p[..., L + 6:L + 9]), normalize(p[..., L + 13:L + 14]),
                        tonemap(p[..., R:R + 3]), tonemap(p[..., R + 9:R + 12]),
                    ])
        arr.flush()
        del arr
        stats["splits"][split] = {"scenes": len(scenes), "patches": n, "scene_ids": scenes}
        print(f"{split:5s}: {len(scenes):3d} scenes -> {n:6d} patches")

    stats["total_patches"] = sum(v["patches"] for v in stats["splits"].values())
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    if spot_rows:
        grid = labeled_grid(spot_rows, ["L noisy", "L clean", "L normal", "L albedo", "L disparity", "R noisy", "R clean"])
        grid.save(args.out / "spotcheck.png")
        print(f"spot-check image: {args.out / 'spotcheck.png'}")
    gb = sum(v["patches"] for v in stats["splits"].values()) * 2 * layout.EYE_CHANNELS * size * size * 2 / 1e9
    print(f"total {stats['total_patches']} patches ({gb:.2f} GB on disk)")


if __name__ == "__main__":
    main()
