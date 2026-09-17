"""Phase 5: generate the noisy/clean stereo dataset by calling the renderer.

For every procedural scene variant this renders both eyes at each low sample
count (the noisy inputs, with their normal/albedo/depth buffers) and once at a
high sample count (the ground truth). Splits are assigned per scene.

Output layout:
    dataset/scenes/scene_007.scene
    dataset/raw/scene_007/spp_0004/{left,right}_{color,normal,albedo,depth,disparity}.exr
    dataset/raw/scene_007/gt/...
    dataset/raw/manifest.json

Example (full dataset on an NVIDIA GPU):
    python -m dataset.generate_dataset --num-scenes 24 --width 512 --height 512 --gt-spp 2048
Quick CPU smoke test:
    python -m dataset.generate_dataset --num-scenes 4 --width 96 --height 96 --gt-spp 64 --device cpu
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import DATASET_DIR, RAW_DIR, find_renderer  # noqa: E402

SCENE_DIR = DATASET_DIR / "scenes"

WALL_PALETTES = [
    ((0.65, 0.05, 0.05), (0.12, 0.45, 0.15)),  # classic red / green
    ((0.10, 0.20, 0.60), (0.70, 0.60, 0.10)),  # blue / yellow
    ((0.75, 0.35, 0.05), (0.05, 0.45, 0.45)),  # orange / teal
    ((0.45, 0.10, 0.55), (0.20, 0.55, 0.10)),  # purple / green
    ((0.73, 0.73, 0.73), (0.60, 0.10, 0.10)),  # white / red
    ((0.05, 0.40, 0.55), (0.65, 0.20, 0.35)),  # cyan / pink
]


@dataclass
class Placed:
    x: float
    z: float
    radius: float


def _fmt(*vals: float) -> str:
    return " ".join(f"{v:.4f}" for v in vals)


def _random_color(rng: random.Random, lo: float = 0.1, hi: float = 0.85) -> tuple[float, float, float]:
    return tuple(rng.uniform(lo, hi) for _ in range(3))


def make_scene(index: int, seed: int) -> str:
    """Returns the .scene text for procedural Cornell-box variant `index`."""
    rng = random.Random(seed * 1000003 + index)
    left, right = rng.choice(WALL_PALETTES)
    if rng.random() < 0.5:
        left, right = right, left
    white = rng.uniform(0.6, 0.8)
    tint = tuple(white * rng.uniform(0.9, 1.0) for _ in range(3))

    lines = [f"# procedural scene {index} (seed {seed})"]
    cam_from = (rng.uniform(-0.15, 0.15), rng.uniform(0.35, 0.65), rng.uniform(1.15, 1.6))
    cam_at = (cam_from[0] * 0.3, rng.uniform(0.4, 0.55), 0.0)
    lines.append(f"camera {_fmt(*cam_from)}   {_fmt(*cam_at)}   0 1 0   {rng.uniform(38, 45):.2f}")
    lines.append("background 0 0 0")
    lines.append(f"material wall lambertian {_fmt(*tint)}")
    lines.append(f"material leftwall lambertian {_fmt(*left)}")
    lines.append(f"material rightwall lambertian {_fmt(*right)}")

    # One or two ceiling lights (quad normal points down).
    n_lights = 1 if rng.random() < 0.75 else 2
    for li in range(n_lights):
        intensity = rng.uniform(8.0, 18.0) / n_lights
        warm = rng.uniform(0.85, 1.0)
        lines.append(f"material light{li} light {_fmt(intensity, intensity * warm, intensity * warm * rng.uniform(0.85, 1.0))}")

    lines += [
        "quad -0.5 0 0   1 0 0   0 0 -1   wall",
        "quad -0.5 1 0   1 0 0   0 0 -1   wall",
        "quad -0.5 0 -1  1 0 0   0 1 0    wall",
        "quad -0.5 0 0   0 0 -1  0 1 0    leftwall",
        "quad  0.5 0 0   0 0 -1  0 1 0    rightwall",
    ]
    for li in range(n_lights):
        sx, sz = rng.uniform(0.12, 0.3), rng.uniform(0.12, 0.28)
        cx = rng.uniform(-0.25, 0.25) if n_lights == 1 else (-0.22 if li == 0 else 0.22) + rng.uniform(-0.05, 0.05)
        cz = rng.uniform(-0.7, -0.3)
        lines.append(f"quad {_fmt(cx - sx / 2, 0.999, cz + sz / 2)}   0 0 {-sz:.4f}   {sx:.4f} 0 0   light{li}")

    placed: list[Placed] = []
    n_objects = rng.randint(2, 4)
    for oi in range(n_objects):
        kind = rng.choices(["lambertian", "metal", "glass"], weights=[0.6, 0.25, 0.15])[0]
        mat = f"obj{oi}"
        if kind == "lambertian":
            lines.append(f"material {mat} lambertian {_fmt(*_random_color(rng))}")
        elif kind == "metal":
            lines.append(f"material {mat} metal {_fmt(*_random_color(rng, 0.5, 0.95))} {rng.uniform(0.05, 0.4):.3f}")
        else:
            lines.append(f"material {mat} dielectric {rng.uniform(1.3, 1.7):.3f}")
        shape = "sphere" if (kind == "glass" or rng.random() < 0.6) else "box"
        for _attempt in range(50):
            if shape == "sphere":
                r = rng.uniform(0.07, 0.2)
                footprint = r
            else:
                sx, sz, sy = rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.55)
                footprint = 0.5 * (sx * sx + sz * sz) ** 0.5
            x = rng.uniform(-0.5 + footprint + 0.01, 0.5 - footprint - 0.01)
            z = rng.uniform(-1.0 + footprint + 0.01, -0.15 - footprint)
            if all((x - p.x) ** 2 + (z - p.z) ** 2 > (footprint + p.radius + 0.02) ** 2 for p in placed):
                break
        else:
            continue
        placed.append(Placed(x, z, footprint))
        if shape == "sphere":
            y = r if rng.random() < 0.8 else rng.uniform(r + 0.1, 0.6)
            lines.append(f"sphere {_fmt(x, y, z)} {r:.4f} {mat}")
        else:
            lines.append(f"box {_fmt(x - sx / 2, 0.0, z - sz / 2)} {_fmt(x + sx / 2, sy, z + sz / 2)} {mat} {rng.uniform(-45, 45):.2f}")
    return "\n".join(lines) + "\n"


def assign_splits(scene_ids: list[str], val_frac: float, test_frac: float, seed: int) -> dict[str, str]:
    ids = sorted(scene_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    if n_test + n_val >= n:
        raise ValueError(f"need more scenes than {n_test + n_val} for the requested splits")
    splits = {}
    for i, sid in enumerate(ids):
        splits[sid] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return splits


def render(renderer: Path, scene_file: Path, out_dir: Path, spp: int, seed: int, args) -> float:
    if (out_dir / "meta.json").exists() and not args.overwrite:
        return 0.0
    cmd = [str(renderer), "--scene", str(scene_file), "--out", str(out_dir), "--spp", str(spp),
           "--width", str(args.width), "--height", str(args.height), "--seed", str(seed),
           "--max-depth", str(args.max_depth), "--clamp", str(args.clamp), "--stereo",
           "--ipd", str(args.ipd), "--quiet"]
    if args.device:
        cmd += ["--device", args.device]
    if not args.previews:
        cmd.append("--no-png")
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-scenes", type=int, default=24)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--noisy-spp", default="1,4,8", help="comma-separated low sample counts")
    ap.add_argument("--gt-spp", type=int, default=2048)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--clamp", type=float, default=8.0, help="indirect clamp (firefly suppression), 0 = off")
    ap.add_argument("--ipd", type=float, default=0.063)
    ap.add_argument("--device", default="", help="cpu or cuda (default: renderer's choice)")
    ap.add_argument("--seed", type=int, default=2024, help="scene generation seed")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--previews", action="store_true", help="also write PNG previews")
    ap.add_argument("--overwrite", action="store_true", help="re-render existing outputs")
    args = ap.parse_args()

    renderer = find_renderer()
    noisy_spp = [int(s) for s in args.noisy_spp.split(",")]
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    scene_ids = [f"scene_{i:03d}" for i in range(args.num_scenes)]
    splits = assign_splits(scene_ids, args.val_frac, args.test_frac, args.seed)
    print(f"renderer: {renderer}")
    print(f"{len(scene_ids)} scenes, {args.width}x{args.height}, noisy spp {noisy_spp}, gt spp {args.gt_spp}")

    t_start = time.perf_counter()
    for i, sid in enumerate(scene_ids):
        scene_file = SCENE_DIR / f"{sid}.scene"
        scene_file.write_text(make_scene(i, args.seed))
        scene_dir = RAW_DIR / sid
        t_gt = render(renderer, scene_file, scene_dir / "gt", args.gt_spp, 7, args)
        t_noisy = 0.0
        for spp in noisy_spp:
            t_noisy += render(renderer, scene_file, scene_dir / f"spp_{spp:04d}", spp, 100 + spp, args)
        elapsed = time.perf_counter() - t_start
        eta = elapsed / (i + 1) * (len(scene_ids) - i - 1)
        print(f"[{i + 1:3d}/{len(scene_ids)}] {sid} ({splits[sid]:5s}) gt {t_gt:6.1f}s noisy {t_noisy:5.1f}s "
              f"| elapsed {elapsed / 60:5.1f} min, eta {eta / 60:5.1f} min", flush=True)

    manifest = {
        "width": args.width,
        "height": args.height,
        "noisy_spp": noisy_spp,
        "gt_spp": args.gt_spp,
        "clamp": args.clamp,
        "ipd": args.ipd,
        "max_depth": args.max_depth,
        "seed": args.seed,
        "scenes": [{"id": sid, "split": splits[sid], "scene_file": f"dataset/scenes/{sid}.scene"} for sid in scene_ids],
    }
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    counts = {s: sum(1 for v in splits.values() if v == s) for s in ("train", "val", "test")}
    print(f"done in {(time.perf_counter() - t_start) / 60:.1f} min; splits {counts}; wrote {RAW_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
