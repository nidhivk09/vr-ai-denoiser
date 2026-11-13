"""U-Net denoiser.

Input: 9 channels per eye - noisy linear HDR color (3), normal encoded to
[0,1] (3), albedo (3). Output: denoised linear HDR color (3).

Preprocessing is part of the module (log transform of color, normal
re-centering) so the exported TorchScript model takes raw renderer buffers.
Height and width must be divisible by 2**levels (use pad_to_multiple).

Cross-eye fusion (Denoiser(fusion=True)): the input gains a 10th channel, the
eye's disparity in pixels, and the batch must hold stereo pairs ordered
[left eyes..., right eyes...]. Inside the module each eye's noisy color is
reprojected into the other eye and given to the U-Net as extra input, with a
mask of where that reprojection is valid. Both eyes then denoise from the
same pool of samples wherever a surface is visible in both.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 9, out_ch: int = 3, base: int = 32, levels: int = 3):
        super().__init__()
        chans = [base * (2 ** i) for i in range(levels + 1)]
        self.levels = levels
        self.enc = nn.ModuleList()
        c = in_ch
        for i in range(levels):
            self.enc.append(ConvBlock(c, chans[i]))
            c = chans[i]
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(chans[levels - 1], chans[levels])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in reversed(range(levels)):
            self.up.append(nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2))
            self.dec.append(ConvBlock(2 * chans[i], chans[i]))
        self.head = nn.Conv2d(chans[0], out_ch, 1)
        # Zero init: the residual Denoiser starts out as the identity (returns the noisy input).
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.enc:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([x, skips.pop()], dim=1))
        return self.head(x)


def _shift_columns(img: torch.Tensor, shift: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples img at (x + shift(x), y), bilinear. Same convention as
    losses._horizontal_warp, but sizes are taken from tensors rather than Python
    ints so a traced module keeps working at other resolutions and batch sizes.
    img: (N,C,H,W) float32; shift: (N,1,H,W) float32 pixels. Returns (warped, in_bounds)."""
    ones = torch.ones_like(shift)
    xs = ones.cumsum(dim=3) - 1.0
    ys = ones.cumsum(dim=2) - 1.0
    w = ones.sum(dim=3, keepdim=True)
    h = ones.sum(dim=2, keepdim=True)
    src_x = xs + shift
    grid = torch.cat([(src_x + 0.5) / w * 2.0 - 1.0, (ys + 0.5) / h * 2.0 - 1.0], dim=1).permute(0, 2, 3, 1)
    warped = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
    in_bounds = ((src_x >= 0.0) & (src_x <= w - 1.0)).to(img.dtype)
    return warped, in_bounds


def cross_eye_features(color: torch.Tensor, disp: torch.Tensor, tol: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """For a batch ordered [left eyes..., right eyes...]: the other eye's color
    reprojected into each eye, (2B,3,H,W), and a (2B,1,H,W) mask that is 1 where
    that pixel is really the same surface point (inside the other image and
    passing the left-right disparity check) and 0 elsewhere. The reprojected
    color is zeroed where the mask is 0.

    Runs in float32 whatever the input type: half precision cannot hold pixel
    coordinates of a 512-wide image to better than half a pixel."""
    dtype = color.dtype
    cl, cr = torch.chunk(color.float(), 2, dim=0)
    dl, dr = torch.chunk(disp.float(), 2, dim=0)
    # x_right = x_left - d: the left eye looks up the right image at x - d_left,
    # the right eye looks up the left image at x + d_right.
    from_right, inb_l = _shift_columns(torch.cat([cr, dr], dim=1), -dl)
    from_left, inb_r = _shift_columns(torch.cat([cl, dl], dim=1), dr)
    mask_l = inb_l * ((from_right[:, 3:4] - dl).abs() < tol).to(inb_l.dtype)
    mask_r = inb_r * ((from_left[:, 3:4] - dr).abs() < tol).to(inb_r.dtype)
    other = torch.cat([from_right[:, 0:3] * mask_l, from_left[:, 0:3] * mask_r], dim=0)
    mask = torch.cat([mask_l, mask_r], dim=0)
    return other.to(dtype), mask.to(dtype)


FIRST_CONV = "unet.enc.0.body.0.weight"


class Denoiser(nn.Module):
    """Raw buffers in, linear HDR color out. Predicts a residual in log space.
    fusion=True adds the cross-eye input (see the module docstring)."""

    def __init__(self, base: int = 32, levels: int = 3, fusion: bool = False):
        super().__init__()
        self.base = base
        self.levels = levels
        self.fusion = fusion
        self.unet = UNet(13 if fusion else 9, 3, base, levels)

    @property
    def in_channels(self) -> int:
        return 10 if self.fusion else 9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        color = torch.log1p(torch.clamp(x[:, 0:3], min=0.0))
        normal = x[:, 3:6] * 2.0 - 1.0
        albedo = x[:, 6:9]
        feats = [color, normal, albedo]
        if self.fusion:
            feats.extend(cross_eye_features(color, x[:, 9:10]))
        y = color + self.unet(torch.cat(feats, dim=1))
        return torch.expm1(torch.clamp(y, min=0.0))

    def config(self) -> dict:
        return {"base": self.base, "levels": self.levels, "fusion": self.fusion}

    def load_weights(self, state: dict) -> None:
        """load_state_dict that also accepts a non-fusion checkpoint for a fusion
        model: the first convolution gets zero weights for the new input
        channels, so the model starts out computing exactly what the checkpoint did."""
        own = self.state_dict()[FIRST_CONV]
        src = state[FIRST_CONV]
        if src.shape[1] < own.shape[1]:
            padded = torch.zeros_like(own)
            padded[:, :src.shape[1]] = src
            state = {**state, FIRST_CONV: padded}
        self.load_state_dict(state)


def pair_input(model: Denoiser, left_input: torch.Tensor, right_input: torch.Tensor,
               left_disp: torch.Tensor, right_disp: torch.Tensor) -> torch.Tensor:
    """The batch a model expects for stereo pairs: [lefts..., rights...], with the
    disparity appended as a 10th channel for fusion models."""
    if model.fusion:
        left_input = torch.cat([left_input, left_disp], dim=1)
        right_input = torch.cat([right_input, right_disp], dim=1)
    return torch.cat([left_input, right_input], dim=0)


def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Replicate-pads (N,C,H,W) on the bottom/right. Returns the padded tensor and original (H, W)."""
    h, w = x.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return x, (h, w)


@torch.no_grad()
def denoise_full(model: Denoiser, x: torch.Tensor) -> torch.Tensor:
    """Denoises (N,C,H,W) of any size by padding to a valid multiple (C = model.in_channels)."""
    xp, (h, w) = pad_to_multiple(x, 2 ** model.levels)
    return model(xp)[..., :h, :w]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_checkpoint(path, device="cpu") -> tuple[Denoiser, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = Denoiser(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
