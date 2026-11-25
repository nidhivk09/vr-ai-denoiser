"""Multi-scale cross-eye feature exchange and learned stereo blending
(Denoiser(fusion=True, exchange=..., blend=...)).

The reprojection at every level must follow the stereo convention of the losses
(x_right = x_left - d), the new blocks must start as the identity so a plain or
fusion checkpoint is reproduced exactly, and they must still receive gradients.

    python -m unittest tests.test_exchange
"""
from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from denoiser import losses  # noqa: E402
from denoiser.model import (FIRST_CONV, Denoiser, cross_eye_features, pair_input,  # noqa: E402
                            level_grids, other_eye_grid, sample_other_eye, stereo_context, warp_other_eye)

H, W = 16, 32
VARIANTS = [dict(exchange=True), dict(blend=True), dict(exchange=True, blend=True)]


def scene(d: float = 4.0):
    """A fronto-parallel textured plane at constant disparity d: right(x) = left(x + d)."""
    torch.manual_seed(0)
    wide = torch.rand(1, 3, H, W + 16)
    left, right = wide[..., 0:W], wide[..., int(d):int(d) + W]
    disp = torch.full((1, 1, H, W), d)
    li, ri = (torch.cat([c, torch.rand(1, 6, H, W)], dim=1) for c in (left, right))
    return li, ri, disp, disp.clone()


def randomise(model: Denoiser, seed: int = 1) -> Denoiser:
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(std=0.1)
        if model.blend:
            model.stereo_blend.gain.fill_(0.8)
    return model


class WarpTest(unittest.TestCase):
    def test_feature_warp_matches_the_color_warp(self):
        torch.manual_seed(2)
        f = torch.rand(4, 5, H, W)
        disp = torch.rand(4, 1, H, W) * 5.0
        dl, dr = disp.chunk(2)
        fl, fr = f.chunk(2)
        r_in_l, _ = losses.warp_right_to_left(fr, dl)
        l_in_r, _ = losses.warp_left_to_right(fl, dr)
        self.assertLess((warp_other_eye(f, dl, dr) - torch.cat([r_in_l, l_in_r])).abs().max().item(), 1e-5)

    def test_context_scales_disparity_with_the_level(self):
        disp = torch.full((2, 1, H, W), 8.0)
        _, mask = cross_eye_features(torch.zeros(2, 3, H, W), disp)
        ctx = stereo_context(disp, mask, 2)
        self.assertEqual(len(ctx), 3)
        for level, (dl, dr, m) in enumerate(ctx):
            self.assertEqual(tuple(dl.shape), (1, 1, H >> level, W >> level))
            self.assertEqual(tuple(m.shape), (2, 1, H >> level, W >> level))
            self.assertAlmostEqual(dl.mean().item(), 8.0 / 2 ** level, places=5)
            self.assertGreaterEqual(m.min().item(), 0.0)
            self.assertLessEqual(m.max().item(), 1.0)

    def test_coarse_features_land_on_the_same_surface_point(self):
        """A feature map that is the downsampled scene, warped with the level's
        context, matches the other eye's downsampled map where the mask is 1."""
        d = 8.0
        torch.manual_seed(3)
        wide = torch.rand(1, 3, H, W + 16)
        left, right = wide[..., 0:W], wide[..., int(d):int(d) + W]
        disp = torch.full((2, 1, H, W), d)
        _, mask = cross_eye_features(torch.cat([left, right]), disp)
        dl, dr, m = stereo_context(disp, mask, 1)[1]
        f = torch.nn.functional.avg_pool2d(torch.cat([left, right]), 2)
        warped = warp_other_eye(f, dl, dr)
        full = (m == 1.0).float()
        self.assertGreater(full.mean().item(), 0.5)
        self.assertLess(((warped - f) * full).abs().max().item(), 1e-5)


class IdentityStartTest(unittest.TestCase):
    def check_same(self, src: Denoiser, **variant):
        model = Denoiser(8, 2, fusion=True, **variant).eval()
        model.load_weights(src.state_dict())
        li, ri, dl, dr = scene()
        x = pair_input(model, li, ri, dl, dr)
        with torch.no_grad():
            a = src(pair_input(src, li, ri, dl, dr))
            b = model(x)
        self.assertLess((a - b).abs().max().item(), 1e-6, f"{variant}")

    def test_from_a_plain_checkpoint(self):
        plain = randomise(Denoiser(8, 2)).eval()
        for v in VARIANTS:
            self.check_same(plain, **v)

    def test_from_a_fusion_checkpoint(self):
        fusion = randomise(Denoiser(8, 2, fusion=True)).eval()
        for v in VARIANTS:
            self.check_same(fusion, **v)

    def test_new_parameters_are_reported(self):
        fusion = Denoiser(8, 2, fusion=True)
        model = Denoiser(8, 2, fusion=True, exchange=True, blend=True)
        new = model.new_parameter_names(fusion.state_dict())
        self.assertTrue(new)
        self.assertTrue(all(n.startswith(("xchg.", "stereo_blend.")) for n in new))
        self.assertIn(FIRST_CONV, model.new_parameter_names(Denoiser(8, 2).state_dict()))

    def test_a_checkpoint_missing_other_parameters_is_refused(self):
        model = Denoiser(8, 2, fusion=True, exchange=True)
        state = dict(Denoiser(8, 2, fusion=True).state_dict())
        del state["unet.head.bias"]
        with self.assertRaises(KeyError):
            model.load_weights(state)


class GradientTest(unittest.TestCase):
    def test_new_blocks_get_gradients_from_the_first_step(self):
        model = Denoiser(8, 2, fusion=True, exchange=True, blend=True)
        model.load_weights(randomise(Denoiser(8, 2)).state_dict())
        li, ri, dl, dr = scene()
        target = torch.rand(2, 3, H, W)
        loss = (model(pair_input(model, li, ri, dl, dr)) - target).abs().mean()
        loss.backward()
        for name in ("xchg.blocks.0.out.weight", "xchg.blocks.2.out.weight", "stereo_blend.gain"):
            g = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(g, name)
            self.assertGreater(g.abs().max().item(), 0.0, name)


class BehaviourTest(unittest.TestCase):
    def test_exchange_alone_carries_information_across_eyes(self):
        """With the fusion input channels switched off, the left output still
        depends on the right input: through the feature exchange only."""
        model = randomise(Denoiser(8, 2, fusion=True, exchange=True)).eval()
        with torch.no_grad():
            model.unet.enc[0].body[0].weight[:, 9:] = 0.0
        li, ri, dl, dr = scene()
        ri2 = ri.clone()
        ri2[:, 0:3] *= 2.0
        with torch.no_grad():
            a = model(pair_input(model, li, ri, dl, dr))[0]
            b = model(pair_input(model, li, ri2, dl, dr))[0]
            plain = randomise(Denoiser(8, 2, fusion=True)).eval()
            plain.unet.enc[0].body[0].weight[:, 9:] = 0.0
            c = plain(pair_input(plain, li, ri, dl, dr))[0]
            e = plain(pair_input(plain, li, ri2, dl, dr))[0]
        self.assertGreater((a - b).abs().max().item(), 1e-4)
        self.assertLess((c - e).abs().max().item(), 1e-7)  # control: no path without the exchange

    def test_full_blend_makes_the_eyes_agree(self):
        """alpha = 0.5 everywhere: both eyes output the average of the pair, so on
        mutually visible pixels (integer disparity) the disagreement vanishes."""
        model = randomise(Denoiser(8, 2, fusion=True, blend=True)).eval()
        with torch.no_grad():
            model.stereo_blend.gain.fill_(1.0)
            model.stereo_blend.head.weight.zero_()
            model.stereo_blend.head.bias.fill_(30.0)
        unblended = Denoiser(8, 2, fusion=True).eval()
        unblended.load_weights({k: v for k, v in model.state_dict().items() if not k.startswith("stereo_blend.")})
        li, ri, dl, dr = scene()
        with torch.no_grad():
            lp, rp = model(pair_input(model, li, ri, dl, dr)).chunk(2)
            lu, ru = unblended(pair_input(unblended, li, ri, dl, dr)).chunk(2)
        before = losses.stereo_disagreement(lu, ru, dl, dr).item()
        after = losses.stereo_disagreement(lp, rp, dl, dr).item()
        self.assertGreater(before, 1e-3)
        self.assertLess(after, 1e-5)

    def test_blend_weight_is_bounded(self):
        model = randomise(Denoiser(8, 2, fusion=True, blend=True))
        with torch.no_grad():
            model.stereo_blend.gain.fill_(5.0)  # clamped to 1
        a = model.stereo_blend.alpha(torch.randn(2, 8, H, W) * 100)
        self.assertGreaterEqual(a.min().item(), 0.0)
        self.assertLessEqual(a.max().item(), 0.5)


class ConfigTest(unittest.TestCase):
    def test_config_round_trips(self):
        for v in VARIANTS:
            model = Denoiser(8, 2, fusion=True, **v)
            clone = Denoiser(**model.config())
            clone.load_state_dict(model.state_dict())
            self.assertEqual(clone.in_channels, 10)
            self.assertEqual((clone.exchange, clone.blend), (v.get("exchange", False), v.get("blend", False)))
        self.assertNotIn("exchange", Denoiser(8, 2, fusion=True).config())  # older checkpoints unchanged

    def test_needs_fusion(self):
        with self.assertRaises(ValueError):
            Denoiser(8, 2, exchange=True)
        with self.assertRaises(ValueError):
            Denoiser(8, 2, blend=True)

    def test_traced_model_follows_other_resolutions_and_batch_sizes(self):
        model = randomise(Denoiser(8, 2, fusion=True, exchange=True, blend=True)).eval()

        def sample(n, h, w):
            x = torch.rand(n, 10, h, w)
            x[:, 9] *= 6.0
            return x

        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            traced = torch.jit.trace(model, sample(2, 16, 32))
            x = sample(6, 24, 20)
            self.assertLess((traced(x) - model(x)).abs().max().item(), 1e-5)


class InferenceSpeedupTest(unittest.TestCase):
    """align_channels and exchange_levels (the frame-time work)."""

    @staticmethod
    def sample(n=2, h=H, w=W):
        torch.manual_seed(5)
        x = torch.rand(n, 10, h, w)
        x[:, 9] *= 6.0
        return x

    def test_aligned_model_gives_the_same_output(self):
        x = self.sample()
        for kw in [dict(fusion=False), dict(fusion=True), *({"fusion": True, **v} for v in VARIANTS)]:
            model = randomise(Denoiser(8, 2, **kw)).eval()
            xi = x if model.fusion else x[:, :9]
            with torch.no_grad():
                before = model(xi)
                aligned = model.align_channels()
                after = aligned(xi)
            self.assertLess((before - after).abs().max().item(), 1e-6, f"{kw}")
            for name, p in aligned.named_parameters():
                if name.endswith("weight") and p.dim() == 4 and p.shape[2] == 3:
                    self.assertEqual(p.shape[1] % 8, 0, name)

    def test_exchange_only_at_chosen_levels(self):
        model = Denoiser(8, 2, fusion=True, exchange=True, exchange_levels=[2, 1])
        blocks = {n.split(".")[2] for n, _ in model.named_parameters() if n.startswith("xchg.")}
        self.assertEqual(blocks, {"1", "2"})
        self.assertEqual(model.config()["exchange_levels"], [1, 2])
        again = Denoiser(**model.config())
        again.load_state_dict(model.state_dict())  # same parameter names as a full exchange model's
        full = Denoiser(8, 2, fusion=True, exchange=True).state_dict()
        self.assertTrue(set(model.state_dict()) <= set(full))
        with self.assertRaises(ValueError):
            Denoiser(8, 2, fusion=True, exchange=True, exchange_levels=[3])
        with self.assertRaises(ValueError):
            Denoiser(8, 2, fusion=True, exchange_levels=[1])

    def test_level_subset_starts_as_the_identity(self):
        fusion = randomise(Denoiser(8, 2, fusion=True)).eval()
        model = Denoiser(8, 2, fusion=True, exchange=True, exchange_levels=[2]).eval()
        model.load_weights(fusion.state_dict())
        x = self.sample()
        with torch.no_grad():
            self.assertLess((model(x) - fusion(x)).abs().max().item(), 1e-6)

    def test_level_grids_warp_like_the_context(self):
        """The once-per-level grids reproject exactly as warp_other_eye with stereo_context."""
        torch.manual_seed(6)
        disp = torch.rand(2, 1, H, W) * 6.0
        _, mask = cross_eye_features(torch.rand(2, 3, H, W), disp)
        g0, _ = other_eye_grid(disp)
        grids = level_grids(disp, mask, g0, [0, 1, 2])
        for level, (dl, dr, m) in enumerate(stereo_context(disp, mask, 2)):
            f = torch.rand(2, 4, H >> level, W >> level)
            grid, gm = grids[level]
            self.assertEqual((sample_other_eye(f, grid) - warp_other_eye(f, dl, dr)).abs().max().item(), 0.0)
            self.assertEqual((gm - m).abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
