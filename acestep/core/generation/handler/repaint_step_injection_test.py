"""Unit tests for repaint step injection and boundary blending."""

import unittest

import torch

from acestep.core.generation.handler.repaint_step_injection import (
    apply_repaint_boundary_blend,
    apply_repaint_step_injection,
    build_soft_repaint_mask,
)


class TestApplyRepaintStepInjection(unittest.TestCase):
    """Tests for per-step source latent replacement in non-repaint regions."""

    def setUp(self):
        self.B, self.T, self.C = 2, 100, 64
        self.xt = torch.randn(self.B, self.T, self.C)
        self.clean_src = torch.randn(self.B, self.T, self.C)
        self.noise = torch.randn(self.B, self.T, self.C)
        self.mask = torch.ones(self.B, self.T, dtype=torch.bool)
        self.mask[0, :20] = False
        self.mask[0, 40:] = False
        self.mask[1, :10] = False
        self.mask[1, 60:] = False

    def test_non_repaint_regions_match_noised_source(self):
        t_next = 0.5
        result = apply_repaint_step_injection(
            self.xt, self.clean_src, self.mask, t_next, self.noise,
        )
        expected = t_next * self.noise + (1.0 - t_next) * self.clean_src
        torch.testing.assert_close(result[0, :20], expected[0, :20])
        torch.testing.assert_close(result[0, 40:], expected[0, 40:])

    def test_repaint_regions_unchanged(self):
        result = apply_repaint_step_injection(
            self.xt, self.clean_src, self.mask, 0.3, self.noise,
        )
        torch.testing.assert_close(result[0, 20:40], self.xt[0, 20:40])
        torch.testing.assert_close(result[1, 10:60], self.xt[1, 10:60])

    def test_all_true_mask_is_noop(self):
        full_mask = torch.ones(self.B, self.T, dtype=torch.bool)
        result = apply_repaint_step_injection(
            self.xt, self.clean_src, full_mask, 0.5, self.noise,
        )
        torch.testing.assert_close(result, self.xt)

    def test_t_zero_returns_clean_source_in_preserved(self):
        result = apply_repaint_step_injection(
            self.xt, self.clean_src, self.mask, 0.0, self.noise,
        )
        torch.testing.assert_close(result[0, :20], self.clean_src[0, :20])

    def test_t_one_returns_pure_noise_in_preserved(self):
        result = apply_repaint_step_injection(
            self.xt, self.clean_src, self.mask, 1.0, self.noise,
        )
        torch.testing.assert_close(result[0, :20], self.noise[0, :20])


class TestBuildSoftRepaintMask(unittest.TestCase):
    """Tests for soft crossfade mask construction."""

    def test_core_region_is_one(self):
        mask = torch.zeros(1, 100, dtype=torch.bool)
        mask[0, 20:40] = True
        soft = build_soft_repaint_mask(mask, crossfade_frames=5)
        self.assertTrue((soft[0, 20:40] == 1.0).all())

    def test_far_preserved_is_zero(self):
        mask = torch.zeros(1, 100, dtype=torch.bool)
        mask[0, 20:40] = True
        soft = build_soft_repaint_mask(mask, crossfade_frames=5)
        self.assertTrue((soft[0, :15] == 0.0).all())
        self.assertTrue((soft[0, 45:] == 0.0).all())

    def test_crossfade_zone_is_monotonic(self):
        mask = torch.zeros(1, 100, dtype=torch.bool)
        mask[0, 20:60] = True
        soft = build_soft_repaint_mask(mask, crossfade_frames=8)
        left_ramp = soft[0, 12:20]
        for i in range(len(left_ramp) - 1):
            self.assertLess(left_ramp[i].item(), left_ramp[i + 1].item())
        right_ramp = soft[0, 60:68]
        for i in range(len(right_ramp) - 1):
            self.assertGreater(right_ramp[i].item(), right_ramp[i + 1].item())

    def test_all_true_mask_returns_ones(self):
        mask = torch.ones(2, 50, dtype=torch.bool)
        soft = build_soft_repaint_mask(mask, crossfade_frames=10)
        self.assertTrue((soft == 1.0).all())

    def test_all_false_mask_returns_zeros(self):
        mask = torch.zeros(2, 50, dtype=torch.bool)
        soft = build_soft_repaint_mask(mask, crossfade_frames=10)
        self.assertTrue((soft == 0.0).all())

    def test_zero_crossfade_is_hard_mask(self):
        mask = torch.zeros(1, 100, dtype=torch.bool)
        mask[0, 30:70] = True
        soft = build_soft_repaint_mask(mask, crossfade_frames=0)
        torch.testing.assert_close(soft, mask.float())

    def test_crossfade_clamped_at_boundaries(self):
        mask = torch.zeros(1, 50, dtype=torch.bool)
        mask[0, 2:48] = True
        soft = build_soft_repaint_mask(mask, crossfade_frames=10)
        self.assertTrue((soft[0] >= 0.0).all())
        self.assertTrue((soft[0] <= 1.0).all())
        self.assertTrue((soft[0, :2] < 1.0).all())


class TestApplyRepaintBoundaryBlend(unittest.TestCase):
    """Tests for post-loop boundary blending."""

    def test_preserved_region_uses_source(self):
        B, T, C = 1, 100, 16
        x_gen = torch.ones(B, T, C)
        clean = torch.zeros(B, T, C)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[0, 30:60] = True
        result = apply_repaint_boundary_blend(x_gen, clean, mask, crossfade_frames=5)
        torch.testing.assert_close(result[0, :25], clean[0, :25])
        torch.testing.assert_close(result[0, 65:], clean[0, 65:])

    def test_core_repaint_uses_generated(self):
        B, T, C = 1, 100, 16
        x_gen = torch.ones(B, T, C) * 2.0
        clean = torch.zeros(B, T, C)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[0, 30:60] = True
        result = apply_repaint_boundary_blend(x_gen, clean, mask, crossfade_frames=5)
        torch.testing.assert_close(result[0, 30:60], x_gen[0, 30:60])

    def test_full_repaint_mask_returns_generated(self):
        B, T, C = 2, 50, 8
        x_gen = torch.randn(B, T, C)
        clean = torch.randn(B, T, C)
        mask = torch.ones(B, T, dtype=torch.bool)
        result = apply_repaint_boundary_blend(x_gen, clean, mask, crossfade_frames=10)
        torch.testing.assert_close(result, x_gen)

    def test_blending_zone_is_interpolated(self):
        B, T, C = 1, 100, 4
        x_gen = torch.ones(B, T, C)
        clean = torch.zeros(B, T, C)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[0, 20:80] = True
        result = apply_repaint_boundary_blend(x_gen, clean, mask, crossfade_frames=5)
        blend_zone = result[0, 15:20, 0]
        for i in range(len(blend_zone)):
            val = blend_zone[i].item()
            self.assertGreater(val, 0.0)
            self.assertLess(val, 1.0)


class TestFloatMaskStepInjection(unittest.TestCase):
    """Tests for step injection with float (soft) masks used by the extend task.

    The extend task produces a seam-ramped ``[B, T]`` float mask in ``[0, 1]``
    where 1.0 = generate, 0.0 = preserve, fractional = blend.  ``apply_repaint_
    step_injection`` must implement the blend ``xt_new = m * xt + (1 - m) *
    zt_src`` for these masks while preserving the hard-boolean behavior used
    by repaint/lego for back-compat.
    """

    def setUp(self):
        self.B, self.T, self.C = 1, 40, 8
        self.xt = torch.randn(self.B, self.T, self.C)
        self.clean_src = torch.randn(self.B, self.T, self.C)
        self.noise = torch.randn(self.B, self.T, self.C)

    def test_float_mask_all_ones_returns_xt(self):
        mask = torch.ones(self.B, self.T, dtype=torch.float32)
        result = apply_repaint_step_injection(self.xt, self.clean_src, mask, 0.5, self.noise)
        torch.testing.assert_close(result, self.xt)

    def test_float_mask_all_zeros_returns_noised_source(self):
        mask = torch.zeros(self.B, self.T, dtype=torch.float32)
        t_next = 0.3
        result = apply_repaint_step_injection(self.xt, self.clean_src, mask, t_next, self.noise)
        expected = t_next * self.noise + (1.0 - t_next) * self.clean_src
        torch.testing.assert_close(result, expected)

    def test_float_mask_half_blends_equally(self):
        mask = torch.full((self.B, self.T), 0.5, dtype=torch.float32)
        t_next = 0.4
        result = apply_repaint_step_injection(self.xt, self.clean_src, mask, t_next, self.noise)
        zt_src = t_next * self.noise + (1.0 - t_next) * self.clean_src
        expected = 0.5 * self.xt + 0.5 * zt_src
        torch.testing.assert_close(result, expected)

    def test_float_mask_ramp_is_monotonic(self):
        mask = torch.zeros(self.B, self.T, dtype=torch.float32)
        # Ramp up from 0 to 1 across 20 frames
        mask[0, 10:30] = torch.linspace(0.0, 1.0, 20)
        mask[0, 30:] = 1.0
        t_next = 0.5
        result = apply_repaint_step_injection(self.xt, self.clean_src, mask, t_next, self.noise)
        # Inside the ramp, successive frames must blend more and more toward xt.
        # Measure L2 distance from zt_src: it should be non-decreasing as the
        # mask rises from 0 to 1 (assuming xt and zt_src differ, which they do
        # here because they are drawn from independent Gaussians).
        zt_src = t_next * self.noise + (1.0 - t_next) * self.clean_src
        dist = (result[0, 10:30] - zt_src[0, 10:30]).norm(dim=-1)
        diffs = dist[1:] - dist[:-1]
        # Allow a small numerical tolerance but all jumps should be >= 0
        self.assertTrue((diffs >= -1e-5).all().item(),
                        f"Expected monotonic distance increase, got {diffs}")

    def test_float_mask_preserves_boolean_equivalence(self):
        bool_mask = torch.zeros(self.B, self.T, dtype=torch.bool)
        bool_mask[0, 10:30] = True
        float_mask = bool_mask.float()
        t_next = 0.25
        result_bool = apply_repaint_step_injection(
            self.xt, self.clean_src, bool_mask, t_next, self.noise,
        )
        result_float = apply_repaint_step_injection(
            self.xt, self.clean_src, float_mask, t_next, self.noise,
        )
        torch.testing.assert_close(result_bool, result_float)


class TestBuildSoftMaskAcceptsFloat(unittest.TestCase):
    """Ensure build_soft_repaint_mask passes pre-built float masks through."""

    def test_float_input_returned_unchanged(self):
        soft_in = torch.zeros(1, 50, dtype=torch.float32)
        soft_in[0, 20:40] = 1.0
        soft_in[0, 15:20] = torch.linspace(0.0, 1.0, 5)
        out = build_soft_repaint_mask(soft_in, crossfade_frames=5)
        torch.testing.assert_close(out, soft_in)
        # It must return a distinct tensor so upstream writes don't mutate in.
        self.assertIsNot(out, soft_in)


if __name__ == "__main__":
    unittest.main()
