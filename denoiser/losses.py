"""Losses and the stereo warp.

Stereo geometry (parallel-axis rig, see renderer/src/core/stereo.h): a point
seen at left-image column x appears in the right image at x - d_left(x), where
d = ipd * focal_px / depth is stored (positive) in the disparity maps.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def log_space(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.clamp(x, min=0.0))


def hdr_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 in log1p space - behaves well on HDR radiance (bright lights don't dominate)."""
    return F.l1_loss(log_space(pred), log_space(target))


def _horizontal_warp(img: torch.Tensor, shift: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples img at (x + shift(x), y) with bilinear filtering.

    img: (N,C,H,W); shift: (N,1,H,W) in pixels. Returns (warped, in_bounds mask).
    """
    n, _, h, w = img.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=img.device, dtype=img.dtype),
        torch.arange(w, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    src_x = xs.unsqueeze(0) + shift[:, 0]
    gx = (src_x + 0.5) / w * 2.0 - 1.0  # align_corners=False convention
    gy = ((ys + 0.5) / h * 2.0 - 1.0).unsqueeze(0).expand(n, -1, -1)
    grid = torch.stack([gx, gy], dim=-1)
    warped = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
    in_bounds = ((src_x >= 0.0) & (src_x <= w - 1.0)).unsqueeze(1).to(img.dtype)
    return warped, in_bounds


def warp_right_to_left(right: torch.Tensor, left_disp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-eye image resampled into the left view: left(x) <- right(x - d_left(x))."""
    return _horizontal_warp(right, -left_disp)


def warp_left_to_right(left: torch.Tensor, right_disp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-eye image resampled into the right view: right(x) <- left(x + d_right(x))."""
    return _horizontal_warp(left, right_disp)


def visibility_masks(left_disp: torch.Tensor, right_disp: torch.Tensor, tol: float = 1.0):
    """Left-right consistency check. Returns (left_mask, right_mask): 1 where the
    pixel is visible in the other eye (not occluded, inside the image)."""
    rd_in_left, inb_l = warp_right_to_left(right_disp, left_disp)
    ld_in_right, inb_r = warp_left_to_right(left_disp, right_disp)
    left_mask = inb_l * ((rd_in_left - left_disp).abs() < tol).to(left_disp.dtype)
    right_mask = inb_r * ((ld_in_right - right_disp).abs() < tol).to(right_disp.dtype)
    return left_mask, right_mask


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() * x.shape[1]).clamp(min=1.0)


def stereo_consistency_loss(left_pred: torch.Tensor, right_pred: torch.Tensor,
                            left_disp: torch.Tensor, right_disp: torch.Tensor,
                            symmetric: bool = True) -> torch.Tensor:
    """L1(left_pred, warp(right_pred)) over mutually visible pixels (log space).
    With symmetric=True the right->left term is added and the two are averaged."""
    left_mask, right_mask = visibility_masks(left_disp, right_disp)
    lp, rp = log_space(left_pred), log_space(right_pred)
    r_in_l, _ = warp_right_to_left(rp, left_disp)
    loss = _masked_mean((lp - r_in_l).abs(), left_mask)
    if symmetric:
        l_in_r, _ = warp_left_to_right(lp, right_disp)
        loss = 0.5 * (loss + _masked_mean((rp - l_in_r).abs(), right_mask))
    return loss


@torch.no_grad()
def stereo_disagreement(left: torch.Tensor, right: torch.Tensor,
                        left_disp: torch.Tensor, right_disp: torch.Tensor) -> torch.Tensor:
    """Evaluation metric: mean |left - warp(right)| on display-referred images
    (clamp to [0,1], gamma 2.2), over mutually visible left-eye pixels. Lower is
    more consistent. Returns one value per batch element."""
    def display(x):
        return torch.clamp(x, 0.0, 1.0) ** (1.0 / 2.2)

    left_mask, _ = visibility_masks(left_disp, right_disp)
    r_in_l, _ = warp_right_to_left(display(right), left_disp)
    diff = (display(left) - r_in_l).abs().mean(dim=1, keepdim=True) * left_mask
    return diff.sum(dim=(1, 2, 3)) / left_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
