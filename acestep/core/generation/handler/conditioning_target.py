"""Target-latent preparation helpers for handler batch conditioning."""

from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger


class ConditioningTargetMixin:
    """Mixin containing target-audio to latent preparation helpers.

    Depends on host members:
    - Attributes: ``device``, ``dtype``, ``sample_rate``, ``silence_latent``.
    - Methods: ``_ensure_silence_latent_on_device ``, ``_load_model_context``,
      ``is_silence``, ``_encode_audio_to_latents``, ``_decode_audio_codes_to_latents``.
    """

    def _get_silence_latent_slice(self, length: int) -> torch.Tensor:
        """Return a silence-latent slice of exactly ``length`` frames.

        When the pre-computed ``silence_latent`` tensor is shorter than
        ``length``, it is tiled (repeated) along the time axis to cover
        the needed span.  This prevents a silent shape mismatch that
        previously occurred when ``audio_duration`` was null and the
        generated code count exceeded the stored silence latent size.
        """
        available = self.silence_latent.shape[1]
        if length <= available:
            return self.silence_latent[0, :length, :]
        # Tile to cover the needed length
        repeats = (length + available - 1) // available  # ceil division
        tiled = self.silence_latent[0].repeat(repeats, 1)  # (repeats*available, C)
        return tiled[:length, :]

    def _prepare_target_latents_and_wavs(
        self,
        batch_size: int,
        target_wavs: torch.Tensor,
        audio_code_hints: List[Optional[str]],
        extend_specs: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Encode target audio/codes to latents and pad batch tensors.

        For items with a non-None ``extend_specs`` entry, only the cropped
        source region is VAE-encoded.  The extension region is filled with
        random-noise latents (``torch.randn_like``) rather than VAE-encoded
        silence so the model is not biased toward fading out.
        """
        self._ensure_silence_latent_on_device()

        if extend_specs is None:
            extend_specs = [None] * batch_size

        with torch.inference_mode():
            target_latents_list = []
            latent_lengths = []
            target_wavs_list = [target_wavs[i].clone() for i in range(batch_size)]
            if target_wavs.device != self.device:
                target_wavs = target_wavs.to(self.device)

            with self._load_model_context("vae"):
                _cached_wav_ref: Optional[torch.Tensor] = None
                _cached_latent: Optional[torch.Tensor] = None

                for i in range(batch_size):
                    spec = extend_specs[i] if i < len(extend_specs) else None
                    code_hint = audio_code_hints[i]

                    if code_hint and spec is not None:
                        logger.info(
                            "[generate_music] Decoding audio codes for extend item {} (hybrid target)...",
                            i,
                        )
                        decoded_latents = self._decode_audio_codes_to_latents(code_hint)
                        if decoded_latents is not None:
                            lm_latents = decoded_latents.squeeze(0)

                            crop_t = float(spec.get("crop_time", 0.0))
                            ext_d = float(spec.get("extend_duration", 0.0))
                            crop_samples = int(max(0.0, crop_t) * 48000)
                            full_wav = target_wavs_list[i]
                            crop_samples = min(crop_samples, full_wav.shape[-1])
                            cropped_wav = full_wav[..., :crop_samples].to(self.device).unsqueeze(0)

                            crop_latent_len = max(0, crop_samples // 1920)
                            ext_latent_len = max(1, int(round(ext_d * 25.0)))

                            if crop_latent_len > 0 and not self.is_silence(cropped_wav):
                                logger.info(
                                    f"[generate_music] Encoding cropped source ({crop_t:.2f}s) for extend item {i}..."
                                )
                                src_latent = self._encode_audio_to_latents(cropped_wav.squeeze(0))
                                if src_latent.shape[0] > crop_latent_len:
                                    src_latent = src_latent[:crop_latent_len]
                                elif src_latent.shape[0] < crop_latent_len:
                                    pad = self._get_silence_latent_slice(crop_latent_len - src_latent.shape[0])
                                    src_latent = torch.cat([src_latent, pad], dim=0)
                            elif crop_latent_len > 0:
                                src_latent = self._get_silence_latent_slice(crop_latent_len)
                            else:
                                src_latent = torch.zeros(
                                    0,
                                    self.silence_latent.shape[-1],
                                    device=self.device,
                                    dtype=self.silence_latent.dtype,
                                )

                            if lm_latents.shape[0] > crop_latent_len:
                                ext_latent = lm_latents[crop_latent_len:]
                            else:
                                ext_latent = torch.randn(
                                    ext_latent_len,
                                    self.silence_latent.shape[-1],
                                    device=self.device,
                                    dtype=src_latent.dtype,
                                )
                            target_latent = torch.cat([src_latent, ext_latent], dim=0)
                            logger.info(
                                "[extend-trace][conditioning_target] HYBRID item {}: "
                                "src_latent={} lm_ext={} target={}",
                                i,
                                tuple(src_latent.shape),
                                tuple(ext_latent.shape),
                                tuple(target_latent.shape),
                            )
                            target_latents_list.append(target_latent)
                            latent_lengths.append(target_latent.shape[0])
                            continue
                        logger.warning(
                            "[extend-trace][conditioning_target] HYBRID item {} failed to decode LM codes; "
                            "falling back to extend source+noise path",
                            i,
                        )

                    if code_hint:
                        logger.info(f"[generate_music] Decoding audio codes for item {i}...")
                        decoded_latents = self._decode_audio_codes_to_latents(code_hint)
                        if decoded_latents is not None:
                            decoded_latents = decoded_latents.squeeze(0)
                            target_latents_list.append(decoded_latents)
                            latent_lengths.append(decoded_latents.shape[0])
                            frames_from_codes = max(1, int(decoded_latents.shape[0] * 1920))
                            target_wavs_list[i] = torch.zeros(2, frames_from_codes)
                            continue

                    if spec is not None:
                        crop_t = float(spec.get("crop_time", 0.0))
                        ext_d = float(spec.get("extend_duration", 0.0))
                        crop_samples = int(max(0.0, crop_t) * 48000)
                        full_wav = target_wavs_list[i]
                        crop_samples = min(crop_samples, full_wav.shape[-1])
                        cropped_wav = full_wav[..., :crop_samples].to(self.device).unsqueeze(0)

                        crop_latent_len = max(0, crop_samples // 1920)
                        ext_latent_len = max(1, int(round(ext_d * 25.0)))
                        logger.info(
                            "[extend-trace][conditioning_target] item {}: spec={} "
                            "full_wav.shape={} crop_samples={} crop_latent_len={} "
                            "ext_latent_len={}",
                            i, spec, tuple(full_wav.shape), crop_samples,
                            crop_latent_len, ext_latent_len,
                        )

                        if crop_latent_len > 0 and not self.is_silence(cropped_wav):
                            logger.info(
                                f"[generate_music] Encoding cropped source ({crop_t:.2f}s) for extend item {i}..."
                            )
                            src_latent = self._encode_audio_to_latents(cropped_wav.squeeze(0))
                            if src_latent.shape[0] > crop_latent_len:
                                src_latent = src_latent[:crop_latent_len]
                            elif src_latent.shape[0] < crop_latent_len:
                                pad = self._get_silence_latent_slice(crop_latent_len - src_latent.shape[0])
                                src_latent = torch.cat([src_latent, pad], dim=0)
                        elif crop_latent_len > 0:
                            src_latent = self._get_silence_latent_slice(crop_latent_len)
                        else:
                            src_latent = torch.zeros(
                                0,
                                self.silence_latent.shape[-1],
                                device=self.device,
                                dtype=self.silence_latent.dtype,
                            )

                        # Use random-noise latents for the extension region so
                        # the diffusion process is seeded with noise (not a
                        # silence bias) over the generated span.
                        noise_latent = torch.randn(
                            ext_latent_len,
                            self.silence_latent.shape[-1],
                            device=self.device,
                            dtype=src_latent.dtype,
                        )
                        target_latent = torch.cat([src_latent, noise_latent], dim=0)
                        logger.info(
                            "[extend-trace][conditioning_target] item {}: "
                            "src_latent.shape={} noise_latent.shape={} "
                            "target_latent.shape={}",
                            i, tuple(src_latent.shape),
                            tuple(noise_latent.shape),
                            tuple(target_latent.shape),
                        )
                        target_latents_list.append(target_latent)
                        latent_lengths.append(target_latent.shape[0])
                        continue

                    current_wav = target_wavs_list[i].to(self.device).unsqueeze(0)
                    if self.is_silence(current_wav):
                        expected_latent_length = current_wav.shape[-1] // 1920
                        target_latent = self._get_silence_latent_slice(expected_latent_length)
                    else:
                        if (
                            _cached_wav_ref is not None
                            and _cached_latent is not None
                            and _cached_wav_ref.shape == current_wav.shape
                            and torch.equal(_cached_wav_ref, current_wav)
                        ):
                            logger.info(
                                f"[generate_music] Reusing cached VAE latents for item {i} (same audio as previous item)"
                            )
                            target_latent = _cached_latent.clone()
                        else:
                            logger.info(f"[generate_music] Encoding target audio to latents for item {i}...")
                            target_latent = self._encode_audio_to_latents(current_wav.squeeze(0))
                            _cached_wav_ref = current_wav
                            _cached_latent = target_latent
                    target_latents_list.append(target_latent)
                    latent_lengths.append(target_latent.shape[0])

            max_target_frames = max(wav.shape[-1] for wav in target_wavs_list)
            padded_target_wavs = []
            for wav in target_wavs_list:
                if wav.shape[-1] < max_target_frames:
                    pad_frames = max_target_frames - wav.shape[-1]
                    wav = torch.nn.functional.pad(wav, (0, pad_frames), "constant", 0)
                padded_target_wavs.append(wav)
            target_wavs = torch.stack(padded_target_wavs)

            max_latent_length = max(latent.shape[0] for latent in target_latents_list)
            max_latent_length = max(128, max_latent_length)
            silence_latent_tiled = self._get_silence_latent_slice(max_latent_length)

            padded_latents = []
            for latent in target_latents_list:
                latent_length = latent.shape[0]
                if latent_length < max_latent_length:
                    pad_length = max_latent_length - latent_length
                    latent = torch.cat([latent, self._get_silence_latent_slice(pad_length)], dim=0)
                padded_latents.append(latent)

            target_latents = torch.stack(padded_latents)
            latent_masks = torch.stack(
                [
                    torch.cat(
                        [
                            torch.ones(l, dtype=torch.long, device=self.device),
                            torch.zeros(max_latent_length - l, dtype=torch.long, device=self.device),
                        ]
                    )
                    for l in latent_lengths
                ]
            )
            return target_wavs, target_latents, latent_masks, max_latent_length, silence_latent_tiled
