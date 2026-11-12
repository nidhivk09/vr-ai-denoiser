"""Cross-eye fusion: the reprojection inside the model must follow the same
stereo convention as the losses (x_right = x_left - d), and a fusion model
initialised from a plain checkpoint must start out identical to it.

    python -m unittest tests.test_fusion
"""
from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from denoiser import losses  # noqa: E402
from denoiser.model import Denoiser, cross_eye_features, pair_input  # noqa: E402

H, W = 16, 32


def scene(d: float = 4.0, batch: int = 1):
    """A fronto-parallel textured plane at constant disparity d: right(x) = left(x + d)."""
    torch.manual_seed(0)
    wide = torch.rand(batch, 3, H, W + 16)
    left = wide[..., 0:W]
    right = wide[..., int(d):int(d) + W]
    disp = torch.full((batch, 1, H, W), d)
    return left, right, disp, disp.clone()


def randomise(model: Denoiser) -> Denoiser:
    torch.manual_seed(1)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.1)
    return model


class CrossEyeFeaturesTest(unittest.TestCase):
    def test_other_eye_lands_on_the_same_surface_point(self):
        left, right, dl, dr = scene()
        other, mask = cross_eye_features(torch.cat([left, right]), torch.cat([dl, dr]))
        own = torch.cat([left, right])
        self.assertGreater(mask.mean().item(), 0.8)
        self.assertLess(((other - own).abs() * mask).max().item(), 1e-5)

    def test_matches_the_warp_and_mask_used_by_the_losses(self):
        torch.manual_seed(2)
        color = torch.rand(4, 3, H, W)
        disp = torch.rand(4, 1, H, W) * 5.0
        other, mask = cross_eye_features(color, disp)
        cl, cr = color.chunk(2)
        dl, dr = disp.chunk(2)
        r_in_l, _ = losses.warp_right_to_left(cr, dl)
        l_in_r, _ = losses.warp_left_to_right(cl, dr)
        lm, rm = losses.visibility_masks(dl, dr)
        self.assertLess((mask - torch.cat([lm, rm])).abs().max().item(), 1e-6)
        self.assertLess((other - torch.cat([r_in_l * lm, l_in_r * rm])).abs().max().item(), 1e-5)

    def test_pixels_outside_the_other_image_are_masked_and_zeroed(self):
        left, right, dl, dr = scene(d=4.0)
        other, mask = cross_eye_features(torch.cat([left, right]), torch.cat([dl, dr]))
        # Left eye looks up x - d: its first d columns fall outside the right image.
        self.assertEqual(mask[0, :, :, :4].sum().item(), 0.0)
        self.assertEqual(other[0, :, :, :4].abs().sum().item(), 0.0)
        # Right eye looks up x + d: its last d columns fall outside the left image.
        self.assertEqual(mask[1, :, :, -4:].sum().item(), 0.0)

    def test_half_precision_input_keeps_its_type(self):
        left, right, dl, dr = scene()
        other, mask = cross_eye_features(torch.cat([left, right]).half(), torch.cat([dl, dr]).half())
        self.assertEqual(other.dtype, torch.float16)
        self.assertEqual(mask.dtype, torch.float16)


class FusionModelTest(unittest.TestCase):
    def test_initialised_from_a_plain_model_it_gives_the_same_output(self):
        plain = randomise(Denoiser(8, 2)).eval()
        fusion = Denoiser(8, 2, fusion=True).eval()
        fusion.load_weights(plain.state_dict())
        left, right, dl, dr = scene()
        li, ri = (torch.cat([c, torch.rand(1, 6, H, W)], dim=1) for c in (left, right))
        with torch.no_grad():
            a = plain(pair_input(plain, li, ri, dl, dr))
            b = fusion(pair_input(fusion, li, ri, dl, dr))
        self.assertEqual(pair_input(plain, li, ri, dl, dr).shape[1], 9)
        self.assertEqual(pair_input(fusion, li, ri, dl, dr).shape[1], 10)
        self.assertLess((a - b).abs().max().item(), 1e-6)

    def test_output_depends_on_the_other_eye(self):
        model = randomise(Denoiser(8, 2, fusion=True)).eval()
        left, right, dl, dr = scene()
        li, ri = (torch.cat([c, torch.rand(1, 6, H, W)], dim=1) for c in (left, right))
        with torch.no_grad():
            a = model(pair_input(model, li, ri, dl, dr))[0]
            ri2 = ri.clone()
            ri2[:, 0:3] *= 2.0
            b = model(pair_input(model, li, ri2, dl, dr))[0]
        self.assertGreater((a - b).abs().max().item(), 1e-4)  # left output changed with the right input

    def test_config_round_trips(self):
        model = Denoiser(8, 2, fusion=True)
        clone = Denoiser(**model.config())
        clone.load_weights(model.state_dict())
        self.assertTrue(clone.fusion)
        self.assertEqual(clone.in_channels, 10)
        self.assertEqual(Denoiser(**{"base": 8, "levels": 2}).in_channels, 9)  # checkpoints from before fusion

    def test_traced_model_follows_other_resolutions_and_batch_sizes(self):
        model = randomise(Denoiser(8, 2, fusion=True)).eval()

        def sample(n, h, w):
            x = torch.rand(n, 10, h, w)
            x[:, 9] *= 6.0
            return x

        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            traced = torch.jit.trace(model, sample(2, 16, 32))
            x = sample(6, 24, 20)
            self.assertLess((traced(x) - model(x)).abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
