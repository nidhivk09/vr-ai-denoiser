"""Small image helpers shared by the dataset, denoiser and benchmark scripts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def tonemap(hdr: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Linear HDR -> display [0,1]: clamp then gamma (same as the renderer's PNGs)."""
    return np.clip(hdr, 0.0, 1.0) ** (1.0 / gamma)


def to_uint8(img01: np.ndarray) -> np.ndarray:
    img = np.asarray(img01, dtype=np.float32)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def normalize(img: np.ndarray, far: float = 1000.0) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    valid = img[img < far]
    lo, hi = (float(valid.min()), float(valid.max())) if valid.size else (0.0, 1.0)
    out = (img - lo) / max(hi - lo, 1e-6)
    out[img >= far] = 1.0
    return out


def save_png(path: str | Path, img01: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(img01)).save(path)


def anaglyph(left01: np.ndarray, right01: np.ndarray) -> np.ndarray:
    out = right01.copy()
    out[..., 0] = left01[..., 0]
    return out


def labeled_grid(rows: list[list[np.ndarray]], labels: list[str] | None = None,
                 row_labels: list[str] | None = None, pad: int = 4) -> Image.Image:
    """Tiles [0,1] images (all the same size) into a grid with optional labels."""
    h, w = rows[0][0].shape[:2]
    top = 16 if labels else 0
    left = 90 if row_labels else 0
    ncols = max(len(r) for r in rows)
    canvas = Image.new("RGB", (left + ncols * (w + pad), top + len(rows) * (h + pad)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for j, lab in enumerate(labels or []):
        draw.text((left + j * (w + pad) + 2, 2), lab, fill=(0, 0, 0))
    for i, row in enumerate(rows):
        if row_labels:
            draw.text((2, top + i * (h + pad) + h // 2 - 6), row_labels[i], fill=(0, 0, 0))
        for j, img in enumerate(row):
            canvas.paste(Image.fromarray(to_uint8(img)), (left + j * (w + pad), top + i * (h + pad)))
    return canvas
