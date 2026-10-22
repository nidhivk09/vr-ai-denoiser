"""Unit tests for the stereo patch augmentation (denoiser/dataset.py::_flip_pair).

A horizontal flip mirrors the scene, which reverses the handedness of the rig:
the mirrored left eye *is* the new right eye. If the eye swap were dropped, the
patches would still look perfectly plausible while silently violating
x_right = x_left - d, and the consistency loss would be trained on garbage for
half the samples. These tests check the augmented patch against the loss that
consumes it, so the two conventions cannot drift apart.

Run:  python -m unittest discover -s tests -v        (or: pytest tests)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import layout  # noqa: E402
from denoiser import losses  # noqa: E402
from denoiser.dataset import _flip_pair  # noqa: E402

E = layout.EYE_CHANNELS
H, W = 16, 64
D = 8.0  # constant disparity, in pixels


def _f(h: int, w: int, x0: float) -> np.ndarray:
    """Smooth positive (3,h,w) image whose column x holds f(x + x0)."""
    ys = np.arange(h, dtype=np.float32)[None, :, None]
    xs = np.arange(w, dtype=np.float32)[None, None, :] + x0
    img = 0.5 + 0.25 * np.sin(0.1 * xs) + 0.1 * np.cos(0.2 * ys)
    return img * np.array([1.0, 2.0, 3.0], np.float32)[:, None, None]


def stereo_patch(d: float = D, h: int = H, w: int = W) -> np.ndarray:
    """A packed (2E,h,w) patch that satisfies x_right = x_left - d exactly.

    The right eye sees f(x + d) where the left eye sees f(x), which is the
    relation the renderer produces and the warp in losses.py undoes.
    """
    x = np.zeros((2 * E, h, w), np.float32)
    for eye, (off, x0) in enumerate(((0, 0.0), (E, d))):
        img = _f(h, w, x0)
        x[off + 0 : off + 3] = img          # noisy colour
        x[off + 9 : off + 12] = img * 0.9   # clean colour (same geometry)
        x[off + 3 : off + 6] = 0.5          # normal, encoded to [0,1]
        x[off + 6 : off + 9] = 0.3 + 0.4 * eye  # albedo: distinct per eye
        x[off + layout.DEPTH] = 1.0
        x[off + layout.DISPARITY] = d
    return x


def pair(x: np.ndarray, channels: slice = layout.COLOR) -> tuple[torch.Tensor, ...]:
    """Pulls (left, right, left_disp, right_disp) tensors out of a packed patch."""
    t = torch.from_numpy(np.ascontiguousarray(x))
    dl = layout.DISPARITY
    return (t[channels][None], t[E + channels.start : E + channels.stop][None],
            t[dl : dl + 1][None], t[E + dl : E + dl + 1][None])


def disagreement(x: np.ndarray, channels: slice = layout.COLOR) -> float:
    return losses.stereo_disagreement(*pair(x, channels)).item()


def consistency(x: np.ndarray, channels: slice = layout.COLOR) -> float:
    return losses.stereo_consistency_loss(*pair(x, channels)).item()


class StereoConventionTest(unittest.TestCase):
    """The augmented patch must still satisfy the renderer's stereo convention."""

    def test_the_unaugmented_patch_is_consistent(self):
        """Sanity check on the fixture itself, so a failure below means the
        augmentation broke, not the test data."""
        x = stereo_patch()
        self.assertLess(consistency(x), 1e-5)
        self.assertLess(consistency(x, layout.TARGET), 1e-5)
        self.assertLess(disagreement(x), 1e-5)

    def test_hflip_preserves_consistency(self):
        x = _flip_pair(stereo_patch(), hflip=True, vflip=False)
        self.assertLess(consistency(x), 1e-5)
        self.assertLess(consistency(x, layout.TARGET), 1e-5)

    def test_vflip_preserves_consistency(self):
        x = _flip_pair(stereo_patch(), hflip=False, vflip=True)
        self.assertLess(consistency(x), 1e-5)

    def test_both_flips_preserve_consistency(self):
        x = _flip_pair(stereo_patch(), hflip=True, vflip=True)
        self.assertLess(consistency(x), 1e-5)
        self.assertLess(consistency(x, layout.TARGET), 1e-5)

    def test_mirroring_without_the_eye_swap_breaks_consistency(self):
        """Confirms the tests above actually discriminate: this is what the
        augmentation would produce if the left<->right swap were removed."""
        broken = np.ascontiguousarray(stereo_patch()[:, :, ::-1])
        self.assertGreater(consistency(broken), 1e-2)

    def test_disparity_stays_positive_after_augmentation(self):
        for hflip in (False, True):
            for vflip in (False, True):
                x = _flip_pair(stereo_patch(), hflip=hflip, vflip=vflip)
                for off in (0, E):
                    d = x[off + layout.DISPARITY]
                    self.assertTrue(np.all(d > 0.0), f"hflip={hflip} vflip={vflip}")
                    self.assertTrue(np.allclose(d, D))


class EyeSwapTest(unittest.TestCase):
    """Which channel blocks move where."""

    @staticmethod
    def tagged_patch() -> np.ndarray:
        """Left eye filled with 1.0, right eye with 2.0, in every channel."""
        x = np.ones((2 * E, H, W), np.float32)
        x[E:] = 2.0
        return x

    def test_hflip_swaps_the_eyes(self):
        x = _flip_pair(self.tagged_patch(), hflip=True, vflip=False)
        self.assertTrue(np.all(x[0 : 0 + 3] == 2.0))   # new left came from the right eye
        self.assertTrue(np.all(x[E : E + 3] == 1.0))

    def test_vflip_keeps_the_eye_order(self):
        x = _flip_pair(self.tagged_patch(), hflip=False, vflip=True)
        self.assertTrue(np.all(x[0 : 0 + 3] == 1.0))
        self.assertTrue(np.all(x[E : E + 3] == 2.0))

    def test_no_flip_is_a_no_op(self):
        x = stereo_patch()
        self.assertTrue(np.array_equal(_flip_pair(x.copy(), hflip=False, vflip=False), x))

    def test_the_swap_moves_every_channel_of_the_eye(self):
        """Not just the colour: albedo, depth and disparity must travel with it,
        or the inputs would describe one eye and the disparity the other."""
        x = stereo_patch()
        flipped = _flip_pair(x.copy(), hflip=True, vflip=False)
        for ch in (*range(6, 9), layout.DEPTH, layout.DISPARITY):  # albedo, depth, disparity
            self.assertTrue(
                np.allclose(flipped[ch], x[E + ch][:, ::-1]), f"channel {ch} did not follow the swap"
            )
            self.assertTrue(np.allclose(flipped[E + ch], x[ch][:, ::-1]), f"channel {ch}")


class FlipGeometryTest(unittest.TestCase):
    """Mirroring of the pixel grid and of the encoded normals."""

    @staticmethod
    def normal_patch() -> np.ndarray:
        """Encoded normals set to distinguishable per-axis gradients."""
        x = np.zeros((2 * E, H, W), np.float32)
        xs = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]
        ys = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
        for off in (0, E):
            x[off + 3] = xs          # normal.x encoded in [0,1]
            x[off + 4] = ys          # normal.y
            x[off + 5] = 0.25        # normal.z: unaffected by either flip
        return x

    def test_hflip_mirrors_the_columns(self):
        x = stereo_patch()
        flipped = _flip_pair(x.copy(), hflip=True, vflip=False)
        # New left eye = old right eye, mirrored left-to-right.
        self.assertTrue(np.allclose(flipped[0:3], x[E : E + 3][:, :, ::-1]))

    def test_vflip_mirrors_the_rows(self):
        x = stereo_patch()
        flipped = _flip_pair(x.copy(), hflip=False, vflip=True)
        self.assertTrue(np.allclose(flipped[0:3], x[0:3][:, ::-1, :]))

    def test_hflip_negates_the_x_normal_only(self):
        x = self.normal_patch()
        flipped = _flip_pair(x.copy(), hflip=True, vflip=False)
        for off in (0, E):
            self.assertTrue(np.allclose(flipped[off + 3], 1.0 - x[off + 3][:, ::-1]))
            self.assertTrue(np.allclose(flipped[off + 4], x[off + 4][:, ::-1]))
            self.assertTrue(np.allclose(flipped[off + 5], 0.25))

    def test_vflip_negates_the_y_normal_only(self):
        x = self.normal_patch()
        flipped = _flip_pair(x.copy(), hflip=False, vflip=True)
        for off in (0, E):
            self.assertTrue(np.allclose(flipped[off + 4], 1.0 - x[off + 4][::-1, :]))
            self.assertTrue(np.allclose(flipped[off + 3], x[off + 3][::-1, :]))
            self.assertTrue(np.allclose(flipped[off + 5], 0.25))

    def test_normals_stay_inside_the_encoded_range(self):
        x = self.normal_patch()
        for hflip in (False, True):
            for vflip in (False, True):
                n = _flip_pair(x.copy(), hflip=hflip, vflip=vflip)[3:6]
                self.assertGreaterEqual(n.min(), 0.0)
                self.assertLessEqual(n.max(), 1.0)

    def test_double_flip_returns_to_the_original(self):
        """Mirroring twice is the identity, eye swap included."""
        x = stereo_patch()
        once = _flip_pair(x.copy(), hflip=True, vflip=True)
        twice = _flip_pair(once.copy(), hflip=True, vflip=True)
        self.assertTrue(np.allclose(twice, x, atol=1e-6))

    def test_output_is_contiguous(self):
        """torch.from_numpy in the dataset needs a positive-stride array."""
        for hflip in (False, True):
            for vflip in (False, True):
                out = _flip_pair(stereo_patch(), hflip=hflip, vflip=vflip)
                self.assertTrue(out.flags["C_CONTIGUOUS"])
                self.assertEqual(out.shape, (2 * E, H, W))
                self.assertEqual(out.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
