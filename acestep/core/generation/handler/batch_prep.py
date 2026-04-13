"""Batch preparation helpers for handler decomposition."""

from typing import Dict, List, Optional, Union

import torch
from loguru import logger

from acestep.constants import DEFAULT_DIT_INSTRUCTION


class BatchPrepMixin:
    """Mixin containing batch and input normalization helpers.

    Depends on host members:
    - Attributes: ``device``, ``dtype``.
    - Methods: ``tiled_encode``, ``extract_caption_from_sft_format``,
      ``_build_metadata_dict``.
    """

    def _normalize_audio_code_hints(
        self, audio_code_hints: Optional[Union[str, List[str]]], batch_size: int
    ) -> List[Optional[str]]:
        """Normalize ``audio_code_hints`` into a batch-length list."""
        if audio_code_hints is None:
            normalized: List[Optional[str]] = [None] * batch_size
        elif isinstance(audio_code_hints, str):
            normalized = [audio_code_hints] * batch_size
        elif len(audio_code_hints) == 1 and batch_size > 1:
            normalized = audio_code_hints * batch_size
        elif len(audio_code_hints) != batch_size:
            normalized = list(audio_code_hints[:batch_size])
            while len(normalized) < batch_size:
                normalized.append(None)
        else:
            normalized = list(audio_code_hints)
        return [hint if isinstance(hint, str) and hint.strip() else None for hint in normalized]

    def _normalize_instructions(
        self,
        instructions: Optional[Union[str, List[str]]],
        batch_size: int,
        default: Optional[str] = None,
    ) -> List[str]:
        """Normalize instructions into a batch-length list."""
        if instructions is None:
            default_instruction = default or DEFAULT_DIT_INSTRUCTION
            return [default_instruction] * batch_size
        if isinstance(instructions, str):
            return [instructions] * batch_size
        if len(instructions) == 1:
            return instructions * batch_size
        if len(instructions) != batch_size:
            normalized = list(instructions[:batch_size])
            default_instruction = default or DEFAULT_DIT_INSTRUCTION
            while len(normalized) < batch_size:
                normalized.append(default_instruction)
            return normalized
        return list(instructions)

    def _create_fallback_vocal_languages(self, batch_size: int) -> List[str]:
        """Create default vocal-language values for missing inputs."""
        return ["en"] * batch_size

    def _encode_audio_to_latents(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to latents using tiled VAE encode path."""
        input_was_2d = audio.dim() == 2
        if input_was_2d:
            audio = audio.unsqueeze(0)

        with torch.inference_mode():
            latents = self.tiled_encode(audio, offload_latent_to_cpu=True)

        latents = latents.to(self.device).to(self.dtype)
        latents = latents.transpose(1, 2)
        if input_was_2d:
            latents = latents.squeeze(0)
        return latents

    def prepare_batch_data(
        self,
        actual_batch_size,
        processed_src_audio,
        audio_duration,
        captions,
        lyrics,
        vocal_language,
        instruction,
        bpm,
        key_scale,
        time_signature,
    ):
        """Prepare repeated batch-level caption/instruction/metadata values."""
        pure_caption = self.extract_caption_from_sft_format(captions)
        captions_batch = [pure_caption] * actual_batch_size
        instructions_batch = [instruction] * actual_batch_size
        lyrics_batch = [lyrics] * actual_batch_size
        vocal_languages_batch = [vocal_language] * actual_batch_size

        # Metadata duration MUST reflect the TOTAL output length, not the
        # src-audio length.  For the extend task the handler sets
        # ``audio_duration = crop_time + extend_duration`` — if we use the
        # src audio length here instead, the DiT text prompt encodes the
        # wrong duration (e.g. "13 seconds" when the latent is 28.5s), which
        # corrupts the model's internal timing/position expectations and
        # produces gibberish beyond the kept region.
        calculated_duration = None
        if audio_duration is not None and float(audio_duration) > 0:
            calculated_duration = float(audio_duration)
        elif processed_src_audio is not None:
            calculated_duration = processed_src_audio.shape[-1] / 48000.0

        src_len = (
            processed_src_audio.shape[-1] / 48000.0
            if processed_src_audio is not None else None
        )
        logger.info(
            "[extend-trace][batch_prep] audio_duration(in)={} src_len={} "
            "-> metadata duration={} (encoded into text prompt)",
            audio_duration, src_len, calculated_duration,
        )

        metadata_dict: Dict[str, Union[str, int]] = self._build_metadata_dict(
            bpm, key_scale, time_signature, calculated_duration
        )
        metas_batch = [metadata_dict.copy() for _ in range(actual_batch_size)]
        return captions_batch, instructions_batch, lyrics_batch, vocal_languages_batch, metas_batch
