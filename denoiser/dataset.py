"""PyTorch datasets for the packed patches and for full-resolution test scenes."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import PATCH_DIR, RAW_DIR  # noqa: E402
from dataset import layout  # noqa: E402

E = layout.EYE_CHANNELS


def _flip_pair(x: np.ndarray, hflip: bool, vflip: bool) -> np.ndarray:
    """Augments a (2E,H,W) stereo patch. A horizontal flip mirrors the scene,
    so the eyes swap roles (left <-> right) to keep the disparity convention.
    Normal channels are mirrored accordingly (encoded n -> 1 - n)."""
    if vflip:
        x = x[:, ::-1, :]
        x[3 + 1] = 1.0 - x[3 + 1]
        x[E + 3 + 1] = 1.0 - x[E + 3 + 1]
    if hflip:
        x = x[:, :, ::-1]
        x = np.concatenate([x[E:], x[:E]], axis=0)
        x[3 + 0] = 1.0 - x[3 + 0]
        x[E + 3 + 0] = 1.0 - x[E + 3 + 0]
    return np.ascontiguousarray(x)


class PatchDataset(Dataset):
    """mode="mono": each eye is an independent sample -> {"input", "target"}.
    mode="stereo": one sample per stereo patch -> left/right inputs, targets, disparities."""

    def __init__(self, split: str, mode: str = "mono", patch_dir: Path = PATCH_DIR,
                 augment: bool = False, in_memory: bool = True):
        path = Path(patch_dir) / f"{split}.npy"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run python -m dataset.extract_patches first")
        self.data = np.load(path, mmap_mode=None if in_memory else "r")
        if mode not in ("mono", "stereo"):
            raise ValueError(mode)
        self.mode = mode
        self.augment = augment

    def __len__(self) -> int:
        n = len(self.data)
        return 2 * n if self.mode == "mono" else n

    def _patch(self, i: int) -> np.ndarray:
        x = self.data[i].astype(np.float32)
        if self.augment:
            x = _flip_pair(x, bool(np.random.rand() < 0.5), bool(np.random.rand() < 0.5))
        return x

    def __getitem__(self, idx: int) -> dict:
        if self.mode == "mono":
            x = self._patch(idx // 2)
            o = (idx % 2) * E
            return {"input": torch.from_numpy(x[o:o + 9].copy()),
                    "target": torch.from_numpy(x[o + 9:o + 12].copy())}
        x = self._patch(idx)
        t = torch.from_numpy(x)
        return {
            "left_input": t[0:9], "left_target": t[9:12], "left_disp": t[13:14],
            "right_input": t[E:E + 9], "right_target": t[E + 9:E + 12], "right_disp": t[E + 13:E + 14],
        }


def load_scene_pair(scene_id: str, spp: int, raw_dir: Path = RAW_DIR) -> dict[str, torch.Tensor]:
    """Full-resolution stereo pair as (1,C,H,W) tensors."""
    x = torch.from_numpy(layout.load_pair(Path(raw_dir) / scene_id, spp)).unsqueeze(0)
    return {
        "left_input": x[:, 0:9], "left_target": x[:, 9:12], "left_disp": x[:, 13:14],
        "right_input": x[:, E:E + 9], "right_target": x[:, E + 9:E + 12], "right_disp": x[:, E + 13:E + 14],
    }


def scenes_in_split(split: str, raw_dir: Path = RAW_DIR) -> list[str]:
    return [s["id"] for s in layout.load_manifest(Path(raw_dir))["scenes"] if s["split"] == split]
