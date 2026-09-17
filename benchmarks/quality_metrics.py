"""Image quality and stereo consistency metrics.

All metrics are computed on display-referred images (linear HDR clamped to
[0,1], gamma 2.2), matching what a viewer sees.

  PSNR, SSIM          - against the high-spp ground truth
  LPIPS               - optional (pip install lpips)
  stereo disagreement - mean |left - warp(right)| over mutually visible pixels

CLI: compare rendered/denoised stereo outputs against a ground-truth render:
    python -m benchmarks.quality_metrics --pred outputs/pipeline --gt dataset/raw/scene_000/gt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.exr import read_exr  # noqa: E402
from denoiser.losses import stereo_disagreement  # noqa: E402


def display(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0) ** (1.0 / 2.2)


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(display(pred), display(target)).item()
    return 99.0 if mse <= 1e-10 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def _gaussian_window(size: int = 11, sigma: float = 1.5, channels: int = 3, device=None) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()
    return w


@torch.no_grad()
def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Standard SSIM (Wang et al. 2004), Gaussian 11x11 window, averaged over channels."""
    x, y = display(pred).float(), display(target).float()
    c = x.shape[1]
    w = _gaussian_window(channels=c, device=x.device)
    mu_x = F.conv2d(x, w, groups=c)
    mu_y = F.conv2d(y, w, groups=c)
    sxx = F.conv2d(x * x, w, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, groups=c) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.mean().item()


_lpips_model = None


@torch.no_grad()
def lpips_distance(pred: torch.Tensor, target: torch.Tensor) -> float | None:
    """LPIPS (AlexNet) if the `lpips` package is installed, else None."""
    global _lpips_model
    try:
        import lpips  # type: ignore
    except ImportError:
        return None
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex", verbose=False)
    m = _lpips_model.to(pred.device)
    return m(display(pred) * 2 - 1, display(target) * 2 - 1).mean().item()


def stereo_metrics(left: torch.Tensor, right: torch.Tensor, gt_left: torch.Tensor, gt_right: torch.Tensor,
                   left_disp: torch.Tensor, right_disp: torch.Tensor, with_lpips: bool = False) -> dict:
    """Per-eye averaged quality plus stereo disagreement for one stereo frame."""
    out = {
        "psnr": 0.5 * (psnr(left, gt_left) + psnr(right, gt_right)),
        "ssim": 0.5 * (ssim(left, gt_left) + ssim(right, gt_right)),
        "disagreement": stereo_disagreement(left, right, left_disp, right_disp).mean().item(),
    }
    if with_lpips:
        a, b = lpips_distance(left, gt_left), lpips_distance(right, gt_right)
        out["lpips"] = None if a is None else 0.5 * (a + b)
    return out


def _load(path: Path) -> torch.Tensor:
    return torch.from_numpy(read_exr(path)).permute(2, 0, 1).unsqueeze(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True, help="dir with left_/right_ color EXRs (or --pred-name)")
    ap.add_argument("--pred-name", default="color", help="file stem after left_/right_ (e.g. denoised)")
    ap.add_argument("--gt", type=Path, required=True, help="ground-truth render dir (with disparity maps)")
    ap.add_argument("--lpips", action="store_true")
    args = ap.parse_args()
    m = stereo_metrics(
        _load(args.pred / f"left_{args.pred_name}.exr"), _load(args.pred / f"right_{args.pred_name}.exr"),
        _load(args.gt / "left_color.exr"), _load(args.gt / "right_color.exr"),
        _load(args.gt / "left_disparity.exr"), _load(args.gt / "right_disparity.exr"), args.lpips)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
