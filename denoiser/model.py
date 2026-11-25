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

Two stronger ways of sharing information between the eyes build on fusion
(both need fusion=True, i.e. the disparity channel and paired batches; the
exported model keeps exactly the same interface):

  exchange=True  multi-scale feature exchange. After every encoder level and
                 the bottleneck, each eye's feature map is reprojected into the
                 other eye (disparity scaled to that level) and merged through a
                 small residual block whose last convolution starts at zero.
                 The eyes then share what they have inferred, not only their
                 raw samples, at every scale of the U-Net.
  blend=True     learned stereo blending of the outputs. A per-pixel weight
                 alpha in [0, 0.5] moves each eye's prediction (log space)
                 toward the other eye's reprojected prediction:
                     y_l <- y_l + alpha_l * mask_l * (warp(y_r) - y_l)
                 alpha = 0.5 makes both eyes output the average. The network
                 predicts alpha from its last decoder features, so it can keep
                 it low where the eyes legitimately differ (highlights,
                 reflections). A scalar gain that starts at 0 multiplies alpha.

Both start out as the identity: a fusion (or plain) checkpoint loaded with
load_weights gives exactly the same output as before, and fine-tuning only
moves away from it where that helps.
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

    def forward(self, x: torch.Tensor, exchange=None, return_features: bool = False):
        """exchange: called on each encoder output and on the bottleneck as
        exchange(level, features), where level is the downsampling level (0 = full
        resolution). return_features: also return the last decoder features."""
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x)
            if exchange is not None:
                x = exchange(i, x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        if exchange is not None:
            x = exchange(self.levels, x)
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([x, skips.pop()], dim=1))
        out = self.head(x)
        return (out, x) if return_features else out


def other_eye_grid(disp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """disp: (2B,1,h,w) float32, ordered [lefts..., rights...], in pixels of this
    resolution. Returns the grid_sample grid with which each eye reads the other
    eye (the left eye at x - d_left, the right eye at x + d_right) from a batch
    whose halves are swapped (see swap_eyes), and the in-bounds mask. Built once
    per resolution and shared by every warp at it. Same convention as
    losses._horizontal_warp, but sizes are taken from tensors rather than Python
    ints so a traced module keeps working at other resolutions and batch sizes."""
    dl, dr = torch.chunk(disp, 2, dim=0)
    shift = torch.cat([-dl, dr], dim=0)
    ones = torch.ones_like(shift)
    xs = ones.cumsum(dim=3) - 1.0
    ys = ones.cumsum(dim=2) - 1.0
    w = ones.sum(dim=3, keepdim=True)
    h = ones.sum(dim=2, keepdim=True)
    src_x = xs + shift
    grid = torch.cat([(src_x + 0.5) / w * 2.0 - 1.0, (ys + 0.5) / h * 2.0 - 1.0], dim=1).permute(0, 2, 3, 1)
    in_bounds = ((src_x >= 0.0) & (src_x <= w - 1.0)).to(disp.dtype)
    return grid, in_bounds


def swap_eyes(t: torch.Tensor) -> torch.Tensor:
    """[lefts..., rights...] -> [rights..., lefts...]."""
    a, b = torch.chunk(t, 2, dim=0)
    return torch.cat([b, a], dim=0)


def sample_other_eye(t: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """The other eye's t reprojected into each eye with a grid from other_eye_grid
    (no masking). Samples in float32 (the grid's type) and returns t's type."""
    out = F.grid_sample(swap_eyes(t).float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out.to(t.dtype)


def cross_eye_features(color: torch.Tensor, disp: torch.Tensor, tol: float = 1.0,
                       grid: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """For a batch ordered [left eyes..., right eyes...]: the other eye's color
    reprojected into each eye, (2B,3,H,W), and a (2B,1,H,W) mask that is 1 where
    that pixel is really the same surface point (inside the other image and
    passing the left-right disparity check) and 0 elsewhere. The reprojected
    color is zeroed where the mask is 0. grid: other_eye_grid(disp), if already
    built.

    Runs in float32 whatever the input type: half precision cannot hold pixel
    coordinates of a 512-wide image to better than half a pixel."""
    dtype = color.dtype
    d = disp.float()
    grid_, in_bounds = other_eye_grid(d) if grid is None else grid
    # x_right = x_left - d: the left eye looks up the right image at x - d_left,
    # the right eye looks up the left image at x + d_right. Both eyes in one warp;
    # a pixel is valid if the other eye's disparity there agrees with its own.
    warped = sample_other_eye(torch.cat([color.float(), d], dim=1), grid_)
    mask = in_bounds * ((warped[:, 3:4] - d).abs() < tol).to(in_bounds.dtype)
    return (warped[:, 0:3] * mask).to(dtype), mask.to(dtype)


def warp_other_eye(t: torch.Tensor, dl: torch.Tensor, dr: torch.Tensor) -> torch.Tensor:
    """For t ordered [lefts..., rights...]: the other eye's t reprojected into each
    eye (no masking), with the same convention as cross_eye_features. The warp
    runs in float32 and the result has t's type."""
    grid, _ = other_eye_grid(torch.cat([dl, dr], dim=0).float())
    return sample_other_eye(t, grid)


def stereo_context(disp: torch.Tensor, mask: torch.Tensor, levels: int) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Per U-Net level 0..levels: (left disparity, right disparity, mask), float32,
    at that level's resolution and in that level's pixel units. disp and mask are
    the full-resolution (2B,1,H,W) disparity and cross_eye_features mask; a coarse
    pixel's mask is the fraction of its full-resolution pixels that are valid."""
    disp, mask = disp.float(), mask.float()
    out = []
    for level in range(levels + 1):
        k = 2 ** level
        d = disp if k == 1 else F.avg_pool2d(disp, k) / k
        m = mask if k == 1 else F.avg_pool2d(mask, k)
        dl, dr = torch.chunk(d, 2, dim=0)
        out.append((dl, dr, m))
    return out


def level_grids(disp: torch.Tensor, mask: torch.Tensor, grid0: torch.Tensor, levels) -> dict:
    """{level: (grid, mask)} for the given levels, float32: what stereo_context
    gives, with each level's warp grid built once. grid0 is level 0's grid (the one
    cross_eye_features used)."""
    disp, mask = disp.float(), mask.float()
    out = {}
    for level in levels:
        if level == 0:
            out[0] = (grid0, mask)
            continue
        k = 2 ** level
        out[level] = (other_eye_grid(F.avg_pool2d(disp, k) / k)[0], F.avg_pool2d(mask, k))
    return out


def _pad_conv_input(conv: nn.Conv2d, multiple: int) -> tuple[nn.Conv2d, int]:
    """A copy of conv taking extra all-zero input channels up to a multiple of
    `multiple` (zero weights there, so the output is unchanged), and how many were
    added. FP16 convolutions only get the fast tensor-core kernels when the channel
    counts are multiples of 8."""
    pad = (-conv.in_channels) % multiple
    if pad == 0:
        return conv, 0
    w = conv.weight
    new = nn.Conv2d(conv.in_channels + pad, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    bias=conv.bias is not None).to(device=w.device, dtype=w.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :conv.in_channels] = w
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new, pad


def _pad_channels(x: torch.Tensor, pad: int) -> torch.Tensor:
    return F.pad(x, (0, 0, 0, 0, 0, pad)) if pad else x


class ExchangeBlock(nn.Module):
    """f + g([f, mask * warp(f_other), mask]); g's last convolution starts at zero."""

    def __init__(self, c: int):
        super().__init__()
        self.mix = nn.Conv2d(2 * c + 1, c, 3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.out = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.pad_in = 0  # zero channels appended to mix's input (Denoiser.align_channels)

    def forward(self, f: torch.Tensor, grid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(f.dtype)
        other = sample_other_eye(f, grid) * m
        z = _pad_channels(torch.cat([f, other, m], dim=1), self.pad_in)
        return f + self.out(self.act(self.mix(z)))


class CrossEyeExchange(nn.Module):
    """One ExchangeBlock per exchanging U-Net level (encoder levels and the
    bottleneck; all of them by default). Keyed by level, so the parameter names
    are the same as with a block at every level."""

    def __init__(self, base: int, levels: int, at=None):
        super().__init__()
        at = range(levels + 1) if at is None else at
        self.at = tuple(sorted(int(level) for level in at))
        if not self.at or min(self.at) < 0 or max(self.at) > levels:
            raise ValueError(f"exchange levels must be in 0..{levels}, got {at}")
        self.blocks = nn.ModuleDict({str(level): ExchangeBlock(base * 2 ** level) for level in self.at})

    def forward(self, level: int, f: torch.Tensor, ctx: dict) -> torch.Tensor:
        if level not in self.at:
            return f
        grid, mask = ctx[level]
        return self.blocks[str(level)](f, grid, mask)


class StereoBlend(nn.Module):
    """y + alpha * mask * (warp(y_other) - y), alpha = 0.5 * gain * sigmoid(head(features)),
    gain clamped to [0, 1] and starting at 0 (so the block starts as the identity).
    Under a consistency loss the gain grows; the head decides where it applies."""

    def __init__(self, c: int):
        super().__init__()
        self.head = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.gain = nn.Parameter(torch.zeros(1))

    def alpha(self, features: torch.Tensor) -> torch.Tensor:
        # Clamped in the forward pass, straight-through in the backward pass: a plain
        # clamp has zero gradient at the initial gain of 0 and would never move.
        gain = self.gain.float()
        gain = gain + (gain.clamp(0.0, 1.0) - gain).detach()
        return 0.5 * gain * torch.sigmoid(self.head(features).float())

    def forward(self, y: torch.Tensor, features: torch.Tensor, grid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        yf = y.float()
        other = sample_other_eye(yf, grid)
        return (yf + self.alpha(features) * mask * (other - yf)).to(y.dtype)


FIRST_CONV = "unet.enc.0.body.0.weight"


class Denoiser(nn.Module):
    """Raw buffers in, linear HDR color out. Predicts a residual in log space.
    fusion=True adds the cross-eye input; exchange / blend add the multi-scale
    feature exchange and the learned output blending (see the module docstring)."""

    def __init__(self, base: int = 32, levels: int = 3, fusion: bool = False,
                 exchange: bool = False, blend: bool = False, exchange_levels=None):
        super().__init__()
        if (exchange or blend) and not fusion:
            raise ValueError("exchange and blend need fusion=True (the disparity channel and paired batches)")
        self.base = base
        self.levels = levels
        self.fusion = fusion
        self.exchange = exchange
        self.blend = blend
        if exchange_levels is not None and not exchange:
            raise ValueError("exchange_levels needs exchange=True")
        self.exchange_levels = None if exchange_levels is None else sorted(int(v) for v in exchange_levels)
        self.unet = UNet(13 if fusion else 9, 3, base, levels)
        self.in_pad = 0  # zero channels appended to the U-Net input (align_channels)
        if exchange:
            self.xchg = CrossEyeExchange(base, levels, self.exchange_levels)
        if blend:
            self.stereo_blend = StereoBlend(base)

    @property
    def in_channels(self) -> int:
        return 10 if self.fusion else 9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        color = torch.log1p(torch.clamp(x[:, 0:3], min=0.0))
        normal = x[:, 3:6] * 2.0 - 1.0
        albedo = x[:, 6:9]
        feats = [color, normal, albedo]
        if self.fusion:
            disp = x[:, 9:10].float()
            grid0 = other_eye_grid(disp)
            other, mask = cross_eye_features(color, disp, grid=grid0)
            feats.extend((other, mask))
        unet_in = _pad_channels(torch.cat(feats, dim=1), self.in_pad)
        if self.exchange or self.blend:
            used = set(self.xchg.at) if self.exchange else set()
            if self.blend:
                used.add(0)
            ctx = level_grids(disp, mask, grid0[0], sorted(used))
            exchange = (lambda level, f: self.xchg(level, f, ctx)) if self.exchange else None
            residual, last = self.unet(unet_in, exchange, return_features=True)
            y = color + residual
            if self.blend:
                grid, m0 = ctx[0]
                y = self.stereo_blend(y, last, grid, m0)
        else:
            y = color + self.unet(unet_in)
        return torch.expm1(torch.clamp(y, min=0.0))

    def align_channels(self, multiple: int = 8) -> "Denoiser":
        """For inference only (export, timing): zero-pads the inputs of the
        convolutions whose input channel count is not a multiple of `multiple` (the
        first convolution, 9 or 13 channels, and each exchange block's 2c+1) with
        zero-weight channels. The output is exactly the same; FP16 convolutions on
        NVIDIA GPUs only get tensor-core kernels with aligned channel counts. The
        state dict changes shape, so do not save or train the aligned model."""
        enc0 = self.unet.enc[0].body
        enc0[0], self.in_pad = _pad_conv_input(enc0[0], multiple)
        if self.exchange:
            for block in self.xchg.blocks.values():
                block.mix, block.pad_in = _pad_conv_input(block.mix, multiple)
        return self

    def config(self) -> dict:
        cfg = {"base": self.base, "levels": self.levels, "fusion": self.fusion}
        if self.exchange:  # (left out otherwise, so older code can still read the checkpoint)
            cfg["exchange"] = True
            if self.exchange_levels is not None:
                cfg["exchange_levels"] = list(self.exchange_levels)
        if self.blend:
            cfg["blend"] = True
        return cfg

    def new_parameter_names(self, state: dict) -> set[str]:
        """Parameters of this model that a checkpoint state does not have (or has
        with fewer input channels): what fine-tuning from it has to learn first."""
        names = set()
        for name, p in self.named_parameters():
            if name not in state or (name == FIRST_CONV and state[name].shape[1] < p.shape[1]):
                names.add(name)
        return names

    def load_weights(self, state: dict) -> None:
        """load_state_dict that also accepts a checkpoint of a simpler model: a
        non-fusion checkpoint for a fusion model (the first convolution gets zero
        weights for the new input channels), and a checkpoint without the
        exchange / blend modules (they keep their identity initialisation). The
        model starts out computing exactly what the checkpoint did."""
        own_state = self.state_dict()
        own = own_state[FIRST_CONV]
        src = state[FIRST_CONV]
        if src.shape[1] < own.shape[1]:
            padded = torch.zeros_like(own)
            padded[:, :src.shape[1]] = src
            state = {**state, FIRST_CONV: padded}
        extra = [k for k in own_state if k not in state]
        if any(not k.startswith(("xchg.", "stereo_blend.")) for k in extra):
            raise KeyError(f"checkpoint is missing parameters: {[k for k in extra if not k.startswith(('xchg.', 'stereo_blend.'))]}")
        state = {**state, **{k: own_state[k] for k in extra}}
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
