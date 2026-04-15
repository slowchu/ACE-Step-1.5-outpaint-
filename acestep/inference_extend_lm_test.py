"""Regression tests for extend-task LM code generation behavior."""

import unittest
from unittest.mock import MagicMock

from acestep.inference import GenerationConfig, GenerationParams, _should_force_lm_codes, generate_music


class ExtendLmInferenceTests(unittest.TestCase):
    """Ensure extend/outpaint runs LM Phase 2 audio-code generation."""

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

    def test_extend_forces_llm_dit_when_thinking_disabled(self):
        """Extend should still request ``infer_type='llm_dit'`` without thinking."""
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
            use_cot_metas=False,
            use_cot_caption=False,
            use_cot_language=False,
            src_audio="dummy.wav",
            caption="extend this music",
            lyrics="[Instrumental]",
            duration=28.0,
        )
        config = GenerationConfig(batch_size=1, allow_lm_batch=False)

        result = generate_music(dit_handler, llm_handler, params, config)

        self.assertTrue(result.success)
        llm_handler.generate_with_stop_condition.assert_called_once()
        _, call_kwargs = llm_handler.generate_with_stop_condition.call_args
        self.assertEqual("llm_dit", call_kwargs.get("infer_type"))

    def test_force_lm_codes_only_for_extend_without_user_codes(self):
        """Only extend tasks without user codes should force LM code generation."""
        self.assertTrue(_should_force_lm_codes("extend", True))
        self.assertFalse(_should_force_lm_codes("extend", False))
        self.assertFalse(_should_force_lm_codes("text2music", True))


if __name__ == "__main__":
    unittest.main()
