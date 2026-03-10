"""Unit tests for ConditioningMaskMixin chunk-mask and source-latent behavior."""

import unittest
from typing import List, Optional

import torch

from acestep.core.generation.handler.conditioning_masks import ConditioningMaskMixin


class _Host(ConditioningMaskMixin):
    """Minimal host implementing ConditioningMaskMixin dependencies."""

    def __init__(self):
        self.device = "cpu"
        self.sample_rate = 48000


def _make_host():
    return _Host()


def _build(
    host,
    batch_size: int = 1,
    max_latent_length: int = 100,
    instructions: Optional[List[str]] = None,
    audio_code_hints: Optional[List[Optional[str]]] = None,
    target_wavs: Optional[torch.Tensor] = None,
    target_latents: Optional[torch.Tensor] = None,
    repainting_start: Optional[List[float]] = None,
    repainting_end: Optional[List[float]] = None,
):
    """Call _build_chunk_masks_and_src_latents with sensible defaults."""
    if instructions is None:
        instructions = ["Fill the audio semantic mask based on the given conditions:"] * batch_size
    if audio_code_hints is None:
        audio_code_hints = [None] * batch_size
    if target_wavs is None:
        target_wavs = torch.ones(batch_size, 2, 48000)
    if target_latents is None:
        # Non-zero so we can detect if they were replaced with silence
        target_latents = torch.ones(batch_size, max_latent_length, 16)
    silence_latent_tiled = torch.zeros(max_latent_length, 16)
    return host._build_chunk_masks_and_src_latents(
        batch_size=batch_size,
        max_latent_length=max_latent_length,
        instructions=instructions,
        audio_code_hints=audio_code_hints,
        target_wavs=target_wavs,
        target_latents=target_latents,
        repainting_start=repainting_start,
        repainting_end=repainting_end,
        silence_latent_tiled=silence_latent_tiled,
    )


class ConditioningMaskLegoBehaviorTests(unittest.TestCase):
    """Verify lego mode keeps source audio latents intact (no silence replacement)."""

    def test_lego_no_repainting_preserves_source_latents(self):
        """With repainting_start/end=None (lego path), src_latents equal target_latents.

        After the fix, lego must NOT call prepare_padding_info with can_use_repainting=True,
        so repainting_start/end arrive as None here. The source audio must be passed
        unchanged to the DiT as musical context.
        """
        host = _make_host()
        target_latents = torch.ones(1, 100, 16) * 2.5
        chunk_masks, spans, is_covers, src_latents = _build(
            host,
            target_latents=target_latents,
            repainting_start=None,
            repainting_end=None,
        )
        self.assertTrue(
            torch.allclose(src_latents, target_latents),
            "lego src_latents must equal the source audio latents, not be silenced",
        )
        self.assertEqual(spans[0][0], "full", "lego span should be 'full'")

    def test_repaint_full_range_silences_repainting_region(self):
        """Repaint with full range should overwrite the repainting region with silence.

        This verifies the existing repaint behavior is preserved: src_latents for
        the masked region should be silence so the DiT regenerates that section.
        """
        host = _make_host()
        target_latents = torch.ones(1, 100, 16) * 2.5
        chunk_masks, spans, is_covers, src_latents = _build(
            host,
            target_latents=target_latents,
            repainting_start=[0.0],
            repainting_end=[4.0],  # 4 seconds at sample_rate=48000, stride=1920 → latents 0..100
        )
        # With full-range repainting the src_latents region should be silenced
        start_l, end_l = spans[0][1], spans[0][2]
        self.assertEqual(spans[0][0], "repainting")
        # The repainting region in src_latents should be zeros (silence)
        self.assertTrue(
            src_latents[0, start_l:end_l].abs().sum().item() < 1e-6,
            "repaint src_latents in masked region should be silence",
        )

    def test_repaint_partial_range_silences_only_masked_region(self):
        """Partial repaint leaves source audio outside the mask intact."""
        host = _make_host()
        target_latents = torch.ones(1, 100, 16) * 3.0
        # Repaint 1s-2s (roughly latents 25-50 at 48000/1920=25 latents/sec)
        chunk_masks, spans, is_covers, src_latents = _build(
            host,
            target_latents=target_latents,
            repainting_start=[1.0],
            repainting_end=[2.0],
        )
        self.assertEqual(spans[0][0], "repainting")
        start_l, end_l = spans[0][1], spans[0][2]
        # Repainting region should be silenced
        self.assertTrue(
            src_latents[0, start_l:end_l].abs().sum().item() < 1e-6,
            "masked region in repaint src_latents should be silence",
        )
        # Outside region should keep original values
        if start_l > 0:
            self.assertAlmostEqual(
                src_latents[0, 0, 0].item(),
                3.0,
                places=4,
                msg="src_latents outside repaint mask should preserve original audio",
            )

    def test_no_audio_produces_silence_src_latents(self):
        """Without source audio, src_latents should be silence (text2music behavior)."""
        host = _make_host()
        # target_wavs all zeros = no audio
        target_wavs = torch.zeros(1, 2, 48000)
        chunk_masks, spans, is_covers, src_latents = _build(
            host,
            target_wavs=target_wavs,
            repainting_start=None,
            repainting_end=None,
        )
        self.assertTrue(
            src_latents.abs().sum().item() < 1e-6,
            "src_latents should be silence when no source audio is present",
        )


if __name__ == "__main__":
    unittest.main()
