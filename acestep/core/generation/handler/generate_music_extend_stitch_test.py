"""Tests for extend chunk output stitching helpers."""

import unittest

import torch

from acestep.core.generation.handler.generate_music import _build_extend_chunk_output


class BuildExtendChunkOutputTests(unittest.TestCase):
    """Validate crossfade stitching for chunk-based extend output."""

    def test_stitch_builds_expected_duration(self):
        """Output length should equal original_crop_time + generated_extension duration."""
        sample_rate = 10
        original = torch.arange(0, 50, dtype=torch.float32).view(1, -1)
        overlap_sec = 2.0
        extend_sec = 3.0
        overlap_samples = int(overlap_sec * sample_rate)
        extend_samples = int(extend_sec * sample_rate)
        generated_chunk = torch.ones(1, overlap_samples + extend_samples)

        output = _build_extend_chunk_output(
            original_source=original,
            generated_chunk=generated_chunk,
            sample_rate=sample_rate,
            original_crop_time=4.0,
            overlap_sec=overlap_sec,
        )

        self.assertEqual(int((4.0 + extend_sec) * sample_rate), output.shape[-1])

    def test_zero_overlap_appends_extension_directly(self):
        """Zero overlap should keep crop prefix and append extension frames."""
        sample_rate = 4
        original = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
        generated_chunk = torch.tensor([[9.0, 8.0, 7.0]])

        output = _build_extend_chunk_output(
            original_source=original,
            generated_chunk=generated_chunk,
            sample_rate=sample_rate,
            original_crop_time=1.0,
            overlap_sec=0.0,
        )

        expected = torch.tensor([[0.0, 1.0, 2.0, 3.0, 9.0, 8.0, 7.0]])
        self.assertTrue(torch.equal(expected, output))


if __name__ == "__main__":
    unittest.main()
