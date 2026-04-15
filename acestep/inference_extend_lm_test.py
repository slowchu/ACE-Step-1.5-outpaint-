"""Regression tests for extend-task LM skip behavior."""

import unittest
from unittest.mock import MagicMock

import torch

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

    def test_extend_source_analysis_fills_missing_metas_for_dit(self):
        """Extend should inject analyzed source metadata when user leaves metas empty."""

        class _CaptureDiTHandler:
            def __init__(self):
                self.kwargs = None
                self.lora_loaded = False
                self.use_lora = False
                self.lora_scale = 1.0

            def prepare_seeds(self, batch_size, _seed_for_generation, _use_random_seed):
                return [111] * batch_size, 111

            def generate_music(
                self,
                *,
                bpm=None,
                key_scale="",
                time_signature="",
                **kwargs,
            ):
                self.kwargs = {
                    "bpm": bpm,
                    "key_scale": key_scale,
                    "time_signature": time_signature,
                    **kwargs,
                }
                return {
                    "success": True,
                    "audios": [],
                    "status_message": "ok",
                    "extra_outputs": {},
                }

        dit_handler = _CaptureDiTHandler()
        llm_handler = MagicMock()
        llm_handler.llm_initialized = True

        params = GenerationParams(
            task_type="extend",
            src_audio="dummy.wav",
            thinking=False,
            bpm=None,
            keyscale="",
            timesignature="",
            caption="extend this music",
            lyrics="[Instrumental]",
            crop_time=2.0,
            extend_duration=10.0,
        )
        config = GenerationConfig(batch_size=1, allow_lm_batch=False)

        with unittest.mock.patch(
            "acestep.audio_analysis.analyze_source_audio",
            return_value={"bpm": 88, "keyscale": "B minor", "timesignature": 4},
        ):
            result = generate_music(dit_handler, llm_handler, params, config)

        self.assertTrue(result.success)
        self.assertEqual(88, dit_handler.kwargs.get("bpm"))
        self.assertEqual("B minor", dit_handler.kwargs.get("key_scale"))
        self.assertEqual("4", dit_handler.kwargs.get("time_signature"))

    def test_extend_uses_chunk_duration_and_tail_overlap_context(self):
        """Extend should pass overlap+extend duration and overlap-tail src audio."""

        class _CaptureDiTHandler:
            def __init__(self):
                self.kwargs = None
                self.lora_loaded = False
                self.use_lora = False
                self.lora_scale = 1.0

            def prepare_seeds(self, batch_size, _seed_for_generation, _use_random_seed):
                return [111] * batch_size, 111

            def generate_music(
                self,
                *,
                audio_duration=None,
                repainting_start=None,
                repainting_end=None,
                extend_overlap_seconds=None,
                src_audio=None,
                **kwargs,
            ):
                self.kwargs = kwargs
                self.kwargs.update(
                    {
                        "audio_duration": audio_duration,
                        "repainting_start": repainting_start,
                        "repainting_end": repainting_end,
                        "extend_overlap_seconds": extend_overlap_seconds,
                        "src_audio": src_audio,
                    }
                )
                return {
                    "success": True,
                    "audios": [],
                    "status_message": "ok",
                    "extra_outputs": {},
                }

        dit_handler = _CaptureDiTHandler()
        llm_handler = MagicMock()
        llm_handler.llm_initialized = True
        src_audio_tensor = torch.zeros(2, int(12.0 * 48000))
        params = GenerationParams(
            task_type="extend",
            src_audio=src_audio_tensor,
            thinking=False,
            caption="extend this music",
            lyrics="[Instrumental]",
            crop_time=10.0,
            extend_duration=15.0,
            extend_overlap_seconds=6.0,
        )
        config = GenerationConfig(batch_size=1, allow_lm_batch=False)

        result = generate_music(dit_handler, llm_handler, params, config)

        self.assertTrue(result.success)
        self.assertEqual(21.0, dit_handler.kwargs.get("audio_duration"))
        self.assertEqual(6.0, dit_handler.kwargs.get("repainting_start"))
        self.assertEqual(21.0, dit_handler.kwargs.get("repainting_end"))
        self.assertEqual(6.0, dit_handler.kwargs.get("extend_overlap_seconds"))
        sliced_src_audio = dit_handler.kwargs.get("src_audio")
        self.assertEqual(int(6.0 * 48000), sliced_src_audio.shape[-1])


if __name__ == "__main__":
    unittest.main()
