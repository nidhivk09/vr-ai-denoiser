"""U-Net denoiser.

Input: 9 channels per eye - noisy linear HDR color (3), normal encoded to
[0,1] (3), albedo (3). Output: denoised linear HDR color (3).

Preprocessing is part of the module (log transform of color, normal
re-centering) so the exported TorchScript model takes raw renderer buffers.
Height and width must be divisible by 2**levels (use pad_to_multiple).
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


class Denoiser(nn.Module):
    """Raw buffers in, linear HDR color out. Predicts a residual in log space."""

    def __init__(self, base: int = 32, levels: int = 3):
        super().__init__()
        self.base = base
        self.levels = levels
        self.unet = UNet(9, 3, base, levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        color = torch.log1p(torch.clamp(x[:, 0:3], min=0.0))
        normal = x[:, 3:6] * 2.0 - 1.0
        albedo = x[:, 6:9]
        y = color + self.unet(torch.cat([color, normal, albedo], dim=1))
        return torch.expm1(torch.clamp(y, min=0.0))

    def config(self) -> dict:
        return {"base": self.base, "levels": self.levels}


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
    """Denoises (N,9,H,W) of any size by padding to a valid multiple."""
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
