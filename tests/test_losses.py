"""Unit tests for the stereo warp, the occlusion mask and the losses.

These cover denoiser/losses.py, which holds the paper's contribution: the
consistency term is the only thing that distinguishes the stereo model from the
baseline, and a sign error in the warp or an inverted mask would still train to
a plausible-looking loss curve. Every test here is therefore written against
the stereo convention stated in renderer/src/core/stereo.h:

    disparity d is positive, and x_right = x_left - d.

Run:  python -m unittest discover -s tests -v        (or: pytest tests)
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from denoiser import losses  # noqa: E402

H, W = 32, 64
TOL = 1e-5


def ramp(h: int = H, w: int = W, channels: int = 3, x0: float = 0.0) -> torch.Tensor:
    """A smooth, strictly positive (1,C,h,w) image sampled from an analytic f.

    Column x holds f(x + x0), so `ramp(x0=d)` is exactly `ramp()` shifted by d
    pixels - including outside the original column range, which lets a warp be
    checked for exactness rather than to interpolation accuracy.
    """
    ys = torch.arange(h, dtype=torch.float32).view(1, 1, h, 1)
    xs = torch.arange(w, dtype=torch.float32).view(1, 1, 1, w) + x0
    f = 0.5 + 0.25 * torch.sin(0.1 * xs) + 0.1 * torch.cos(0.2 * ys)
    scale = torch.arange(1, channels + 1, dtype=torch.float32).view(1, channels, 1, 1)
    return (f * scale).contiguous()


def const(value: float, h: int = H, w: int = W, channels: int = 1) -> torch.Tensor:
    return torch.full((1, channels, h, w), float(value))


class HorizontalWarpTest(unittest.TestCase):
    """warp_right_to_left samples the right eye at x - d; warp_left_to_right at x + d."""

    def test_zero_disparity_is_the_identity(self):
        img = ramp()
        warped, in_bounds = losses.warp_right_to_left(img, const(0.0))
        self.assertTrue(torch.allclose(warped, img, atol=TOL))
        self.assertTrue(torch.all(in_bounds == 1.0))

    def test_zero_disparity_is_the_identity_in_both_directions(self):
        img = ramp()
        warped, _ = losses.warp_left_to_right(img, const(0.0))
        self.assertTrue(torch.allclose(warped, img, atol=TOL))

    def test_integer_shift_matches_plain_indexing(self):
        d = 7
        right = ramp()
        warped, in_bounds = losses.warp_right_to_left(right, const(d))
        # left(x) <- right(x - d): the warped image is the right eye moved right.
        self.assertTrue(torch.allclose(warped[..., d:], right[..., : W - d], atol=TOL))
        self.assertTrue(torch.all(in_bounds[..., :d] == 0.0))
        self.assertTrue(torch.all(in_bounds[..., d:] == 1.0))

    def test_left_to_right_shifts_the_other_way(self):
        d = 7
        left = ramp()
        warped, in_bounds = losses.warp_left_to_right(left, const(d))
        # right(x) <- left(x + d): the warped image is the left eye moved left.
        self.assertTrue(torch.allclose(warped[..., : W - d], left[..., d:], atol=TOL))
        self.assertTrue(torch.all(in_bounds[..., : W - d] == 1.0))
        self.assertTrue(torch.all(in_bounds[..., W - d :] == 0.0))

    def test_half_pixel_shift_averages_two_columns(self):
        """Pins down the align_corners=False / +0.5 pixel-centre convention:
        an off-grid shift must land between the two neighbouring columns."""
        img = ramp()
        warped, _ = losses.warp_right_to_left(img, const(0.5))
        expected = 0.5 * (img[..., :-1] + img[..., 1:])
        self.assertTrue(torch.allclose(warped[..., 1:], expected, atol=1e-5))

    def test_out_of_range_samples_clamp_to_the_border(self):
        d = 9
        img = ramp()
        warped, _ = losses.warp_right_to_left(img, const(d))
        # Columns left of the first valid sample repeat column 0 (padding_mode="border").
        self.assertTrue(torch.allclose(warped[..., :d], img[..., 0:1].expand(-1, -1, -1, d), atol=TOL))

    def test_warp_is_row_independent(self):
        """A horizontal warp must not move anything vertically."""
        # Each row is resampled only from itself, so warping the image equals
        # warping each row on its own.
        img = ramp()
        disp = const(4.0)
        whole, _ = losses.warp_right_to_left(img, disp)
        for y in (0, H // 3, H - 1):
            row, _ = losses.warp_right_to_left(img[:, :, y : y + 1], disp[:, :, y : y + 1])
            self.assertTrue(torch.allclose(whole[:, :, y : y + 1], row, atol=TOL))
        # And shifting a single bright row leaves every other row dark.
        spike = torch.zeros(1, 1, H, W)
        spike[..., H // 2, :] = 1.0
        warped, _ = losses.warp_right_to_left(spike, const(3.0))
        self.assertTrue(torch.all(warped[..., : H // 2, :] == 0.0))
        self.assertTrue(torch.all(warped[..., H // 2 + 1 :, :] == 0.0))

    def test_per_pixel_disparity_is_honoured(self):
        """The shift is read per pixel, not taken from a single scalar."""
        img = ramp()
        disp = torch.zeros(1, 1, H, W)
        disp[..., W // 2 :] = 5.0
        warped, _ = losses.warp_right_to_left(img, disp)
        self.assertTrue(torch.allclose(warped[..., : W // 2], img[..., : W // 2], atol=TOL))
        self.assertTrue(torch.allclose(warped[..., W // 2 + 5 :], img[..., W // 2 : W - 5], atol=TOL))


class VisibilityMaskTest(unittest.TestCase):
    """The left-right disparity check that masks occluded and off-image pixels."""

    def test_constant_disparity_masks_only_the_entering_border(self):
        d = 6
        left_mask, right_mask = losses.visibility_masks(const(d), const(d))
        # Left-eye columns < d have no counterpart inside the right image.
        self.assertTrue(torch.all(left_mask[..., :d] == 0.0))
        self.assertTrue(torch.all(left_mask[..., d:] == 1.0))
        # Mirror image on the right: its last d columns leave the left image.
        self.assertTrue(torch.all(right_mask[..., : W - d] == 1.0))
        self.assertTrue(torch.all(right_mask[..., W - d :] == 0.0))

    def test_zero_disparity_is_fully_visible(self):
        left_mask, right_mask = losses.visibility_masks(const(0.0), const(0.0))
        self.assertTrue(torch.all(left_mask == 1.0))
        self.assertTrue(torch.all(right_mask == 1.0))

    def test_occluded_background_is_masked_out(self):
        """A foreground slab at disparity 10 hides part of the background from
        one eye. Those pixels have no valid counterpart, so including them in
        the consistency loss would train the network towards the wrong colour.

        Left image:  background | foreground at columns [20,40) | background
        Right image: the same slab, shifted left by 10 -> columns [10,30).
        """
        left_disp = torch.zeros(1, 1, H, W)
        left_disp[..., 20:40] = 10.0
        right_disp = torch.zeros(1, 1, H, W)
        right_disp[..., 10:30] = 10.0

        left_mask, right_mask = losses.visibility_masks(left_disp, right_disp)

        # Foreground: seen by both eyes, disparities agree.
        self.assertTrue(torch.all(left_mask[..., 21:39] == 1.0))
        # Background the left eye sees at [10,20) sits behind the slab in the
        # right eye -> masked out.
        self.assertTrue(torch.all(left_mask[..., 11:19] == 0.0))
        # Background well clear of the slab stays visible.
        self.assertTrue(torch.all(left_mask[..., 41:63] == 1.0))
        self.assertTrue(torch.all(left_mask[..., 1:9] == 1.0))

        # The right eye loses the symmetric band: background at [30,40) is
        # behind the slab in the left image.
        self.assertTrue(torch.all(right_mask[..., 31:39] == 0.0))
        self.assertTrue(torch.all(right_mask[..., 11:29] == 1.0))
        self.assertTrue(torch.all(right_mask[..., 41:63] == 1.0))

    def test_tolerance_controls_the_check(self):
        """tol is in pixels of disparity disagreement."""
        left_disp = const(0.0)
        right_disp = const(2.0)
        strict, _ = losses.visibility_masks(left_disp, right_disp, tol=1.0)
        loose, _ = losses.visibility_masks(left_disp, right_disp, tol=5.0)
        self.assertTrue(torch.all(strict == 0.0))
        self.assertTrue(torch.all(loose[..., 1:] == 1.0))

    def test_mask_shape_and_values(self):
        left_mask, right_mask = losses.visibility_masks(const(3.0), const(3.0))
        self.assertEqual(left_mask.shape, (1, 1, H, W))
        self.assertEqual(right_mask.shape, (1, 1, H, W))
        for m in (left_mask, right_mask):
            self.assertTrue(torch.all((m == 0.0) | (m == 1.0)))


class ConsistencyLossTest(unittest.TestCase):
    """stereo_consistency_loss over mutually visible pixels, in log1p space."""

    @staticmethod
    def consistent_pair(d: float = 8.0):
        """A left/right pair that satisfies x_right = x_left - d exactly."""
        return ramp(x0=0.0), ramp(x0=d), const(d), const(d)

    def test_zero_for_a_geometrically_consistent_pair(self):
        left, right, dl, dr = self.consistent_pair()
        self.assertLess(losses.stereo_consistency_loss(left, right, dl, dr).item(), TOL)

    def test_zero_in_the_one_sided_form_too(self):
        left, right, dl, dr = self.consistent_pair()
        loss = losses.stereo_consistency_loss(left, right, dl, dr, symmetric=False)
        self.assertLess(loss.item(), TOL)

    def test_positive_when_the_right_eye_disagrees(self):
        left, right, dl, dr = self.consistent_pair()
        # The same image in both eyes is *wrong* at a non-zero disparity.
        wrong = losses.stereo_consistency_loss(left, left.clone(), dl, dr)
        self.assertGreater(wrong.item(), 1e-2)
        self.assertGreater(wrong.item(), losses.stereo_consistency_loss(left, right, dl, dr).item())

    def test_grows_with_the_size_of_the_disagreement(self):
        left, right, dl, dr = self.consistent_pair()
        small = losses.stereo_consistency_loss(left, right * 1.05, dl, dr).item()
        large = losses.stereo_consistency_loss(left, right * 1.5, dl, dr).item()
        self.assertLess(small, large)

    def test_masked_mean_is_normalised_per_pixel_not_per_channel(self):
        """A constant log-space gap of delta must give exactly delta, whatever
        the channel count: the mask covers pixels, the mean covers values."""
        delta = 1.0
        left = const(math.expm1(delta), channels=3)  # log1p -> 1.0
        right = torch.zeros(1, 3, H, W)              # log1p -> 0.0
        zero = const(0.0)
        loss = losses.stereo_consistency_loss(left, right, zero, zero, symmetric=False)
        self.assertAlmostEqual(loss.item(), delta, places=5)
        # ... and the symmetric form averages two identical terms, so it agrees.
        sym = losses.stereo_consistency_loss(left, right, zero, zero, symmetric=True)
        self.assertAlmostEqual(sym.item(), delta, places=5)

    def test_fully_occluded_pair_gives_a_finite_loss(self):
        """Guards the clamp(min=1) in the mask normalisation: an empty mask must
        not divide by zero."""
        left, right, _, _ = self.consistent_pair()
        huge = const(10.0 * W)  # every pixel maps outside the other image
        loss = losses.stereo_consistency_loss(left, right, huge, huge)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_gradients_reach_both_eyes(self):
        """Consistency must pull on both predictions - if the warped branch were
        detached, the term would silently become a one-sided regulariser."""
        left = ramp(x0=0.0).requires_grad_(True)
        right = (ramp(x0=8.0) * 1.2).requires_grad_(True)
        losses.stereo_consistency_loss(left, right, const(8.0), const(8.0)).backward()
        self.assertIsNotNone(left.grad)
        self.assertIsNotNone(right.grad)
        self.assertGreater(left.grad.abs().sum().item(), 0.0)
        self.assertGreater(right.grad.abs().sum().item(), 0.0)

    def test_one_sided_form_still_grades_the_right_eye(self):
        left = ramp(x0=0.0).requires_grad_(True)
        right = (ramp(x0=8.0) * 1.2).requires_grad_(True)
        losses.stereo_consistency_loss(left, right, const(8.0), const(8.0), symmetric=False).backward()
        self.assertGreater(right.grad.abs().sum().item(), 0.0)

    def test_loss_is_a_scalar(self):
        left, right, dl, dr = self.consistent_pair()
        self.assertEqual(losses.stereo_consistency_loss(left, right, dl, dr).shape, ())


class DisagreementMetricTest(unittest.TestCase):
    """stereo_disagreement is the reported headline metric, on display-referred
    images (clamp to [0,1], gamma 2.2)."""

    def test_zero_for_a_consistent_pair(self):
        left, right, dl, dr = ConsistencyLossTest.consistent_pair()
        self.assertLess(losses.stereo_disagreement(left, right, dl, dr).item(), TOL)

    def test_one_value_per_batch_element(self):
        left, right, dl, dr = ConsistencyLossTest.consistent_pair()
        batch = lambda t: torch.cat([t, t], dim=0)  # noqa: E731
        out = losses.stereo_disagreement(batch(left), batch(right), batch(dl), batch(dr))
        self.assertEqual(out.shape, (2,))
        self.assertAlmostEqual(out[0].item(), out[1].item(), places=6)

    def test_larger_for_an_inconsistent_pair(self):
        left, right, dl, dr = ConsistencyLossTest.consistent_pair()
        good = losses.stereo_disagreement(left, right, dl, dr).item()
        bad = losses.stereo_disagreement(left, left.clone(), dl, dr).item()
        self.assertGreater(bad, good + 1e-2)

    def test_differences_above_white_are_clamped_away(self):
        """Display-referred means over-range radiance differences do not count -
        both eyes render as white."""
        zero = const(0.0)
        out = losses.stereo_disagreement(const(5.0, channels=3), const(9.0, channels=3), zero, zero)
        self.assertAlmostEqual(out.item(), 0.0, places=6)

    def test_gamma_encoding_is_applied(self):
        """A 0 vs 1 pair differs by 1 after gamma, not by some linear-space value."""
        zero = const(0.0)
        out = losses.stereo_disagreement(const(1.0, channels=3), const(0.0, channels=3), zero, zero)
        self.assertAlmostEqual(out.item(), 1.0, places=5)
        # Mid-grey: |0.5^(1/2.2) - 0| = 0.7297...
        out = losses.stereo_disagreement(const(0.5, channels=3), const(0.0, channels=3), zero, zero)
        self.assertAlmostEqual(out.item(), 0.5 ** (1 / 2.2), places=5)

    def test_no_gradient_is_tracked(self):
        left, right, dl, dr = ConsistencyLossTest.consistent_pair()
        out = losses.stereo_disagreement(left.requires_grad_(True), right, dl, dr)
        self.assertFalse(out.requires_grad)


class LogSpaceTest(unittest.TestCase):
    def test_negatives_are_clamped_before_the_log(self):
        x = torch.tensor([-1.0, -1e-6, 0.0, math.e - 1.0])
        out = losses.log_space(x)
        self.assertTrue(torch.all(torch.isfinite(out)))
        self.assertAlmostEqual(out[0].item(), 0.0, places=6)
        self.assertAlmostEqual(out[1].item(), 0.0, places=6)
        self.assertAlmostEqual(out[3].item(), 1.0, places=5)

    def test_hdr_l1_is_zero_for_identical_images(self):
        img = ramp()
        self.assertAlmostEqual(losses.hdr_l1(img, img).item(), 0.0, places=7)

    def test_hdr_l1_damps_bright_pixels(self):
        """The stated reason for log space: an absolute error on a bright pixel
        must weigh less than the same error on a dark one."""
        target_dark = const(0.1, channels=1)
        target_bright = const(100.0, channels=1)
        err = 0.5
        dark = losses.hdr_l1(target_dark + err, target_dark).item()
        bright = losses.hdr_l1(target_bright + err, target_bright).item()
        self.assertGreater(dark, 10.0 * bright)


if __name__ == "__main__":
    unittest.main()
