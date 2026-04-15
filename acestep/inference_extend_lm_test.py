"""Regression tests for extend-task LM skip behavior."""

import unittest
from unittest.mock import MagicMock

from acestep.inference import GenerationConfig, GenerationParams, generate_music


class ExtendLmInferenceTests(unittest.TestCase):
    """Ensure extend/outpaint bypasses LM and runs DiT-only generation path."""

    def _build_dit_handler(self) -> MagicMock:
        """Create a minimal DiT handler mock for inference.generate_music."""
        dit_handler = MagicMock()
        dit_handler.prepare_seeds.return_value = ([12345], None)
        dit_handler.generate_music.return_value = {
            "success": True,
            "audios": [],
            "status_message": "ok",
            "extra_outputs": {},
        }
        return dit_handler

    def test_extend_skips_lm_even_with_cot_flags_enabled(self):
        """Extend should bypass LM and never call generate_with_stop_condition."""
        dit_handler = self._build_dit_handler()
        llm_handler = MagicMock()
        llm_handler.llm_initialized = True
        llm_handler.generate_with_stop_condition.return_value = {
            "success": True,
            "metadata": {},
            "audio_codes": "<|audio_code_1|>",
            "extra_outputs": {"time_costs": {}},
        }

        params = GenerationParams(
            task_type="extend",
            thinking=False,
            use_cot_metas=True,
            use_cot_caption=True,
            use_cot_language=True,
            src_audio="dummy.wav",
            caption="extend this music",
            lyrics="[Instrumental]",
            duration=155.0,
            crop_time=1.0,
            extend_duration=28.0,
        )
        config = GenerationConfig(batch_size=1, allow_lm_batch=False)

        result = generate_music(dit_handler, llm_handler, params, config)

        self.assertTrue(result.success)
        llm_handler.generate_with_stop_condition.assert_not_called()

    def test_non_extend_can_still_use_lm_when_thinking_enabled(self):
        """Non-extend tasks should still call LM when thinking is enabled."""
        dit_handler = self._build_dit_handler()
        llm_handler = MagicMock()
        llm_handler.llm_initialized = True
        llm_handler.generate_with_stop_condition.return_value = {
            "success": True,
            "metadata": {},
            "audio_codes": "<|audio_code_1|>",
            "extra_outputs": {"time_costs": {}},
        }

        params = GenerationParams(
            task_type="text2music",
            thinking=True,
            use_cot_metas=False,
            use_cot_caption=False,
            use_cot_language=False,
            caption="new song",
            lyrics="[Instrumental]",
            duration=30.0,
        )
        config = GenerationConfig(batch_size=1, allow_lm_batch=False)

        result = generate_music(dit_handler, llm_handler, params, config)
        self.assertTrue(result.success)
        llm_handler.generate_with_stop_condition.assert_called_once()


if __name__ == "__main__":
    unittest.main()
