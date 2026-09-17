"""Channel layout of the packed patch arrays (shared by extraction and training).

Each patch stores both eyes: array shape (2 * EYE_CHANNELS, H, W), float16.
Per eye (offset 0 = left, EYE_CHANNELS = right):
    0:3   noisy color (linear HDR)
    3:6   normal (encoded to [0,1])
    6:9   albedo
    9:12  clean color (ground truth, linear HDR)
    12    planar depth
    13    disparity in pixels (positive; x_right = x_left - d)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.exr import read_exr

EYE_CHANNELS = 14
COLOR = slice(0, 3)
NORMAL = slice(3, 6)
ALBEDO = slice(6, 9)
INPUT = slice(0, 9)
TARGET = slice(9, 12)
DEPTH = 12
DISPARITY = 13
EYES = ("left", "right")


def load_eye(raw_scene_dir: Path, spp: int, eye: str) -> np.ndarray:
    """Loads one eye of one (scene, spp) render as a (EYE_CHANNELS, H, W) float32 array."""
    noisy = raw_scene_dir / f"spp_{spp:04d}"
    gt = raw_scene_dir / "gt"
    parts = [
        read_exr(noisy / f"{eye}_color.exr"),
        read_exr(noisy / f"{eye}_normal.exr"),
        read_exr(noisy / f"{eye}_albedo.exr"),
        read_exr(gt / f"{eye}_color.exr"),
        read_exr(gt / f"{eye}_depth.exr"),
        read_exr(gt / f"{eye}_disparity.exr"),
    ]
    return np.concatenate(parts, axis=-1).transpose(2, 0, 1).copy()


def load_pair(raw_scene_dir: Path, spp: int) -> np.ndarray:
    """(2 * EYE_CHANNELS, H, W): left eye followed by right eye."""
    return np.concatenate([load_eye(raw_scene_dir, spp, e) for e in EYES], axis=0)


def load_manifest(raw_dir: Path) -> dict:
    path = raw_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run python -m dataset.generate_dataset first")
    return json.loads(path.read_text())
