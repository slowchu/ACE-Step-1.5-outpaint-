"""Chunk-mask and source-latent helpers for batch conditioning."""

from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

# Phrase unique to lego task instructions — used to detect lego items from instruction text.
# Matches both TASK_INSTRUCTIONS["lego"] and TASK_INSTRUCTIONS["lego_default"].
_LEGO_INSTRUCTION_MARKER = "based on the audio context"


class ConditioningMaskMixin:
    """Mixin containing repaint mask/span and source-latent builders.

    Depends on host members:
    - Attributes: ``device``, ``sample_rate``.
    """

    def _build_chunk_masks_and_src_latents(
        self,
        batch_size: int,
        max_latent_length: int,
        instructions: List[str],
        audio_code_hints: List[Optional[str]],
        target_wavs: torch.Tensor,
        target_latents: torch.Tensor,
        repainting_start: Optional[List[float]],
        repainting_end: Optional[List[float]],
        silence_latent_tiled: torch.Tensor,
        chunk_mask_modes: Optional[List[str]] = None,
        extend_specs: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> Tuple[
        torch.Tensor,
        List[Tuple[str, int, int]],
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """Create chunk masks/spans, source latents, and repaint injection mask.

        Returns:
            Tuple of (chunk_masks, spans, is_covers, src_latents, repaint_mask).
            ``repaint_mask`` is either a boolean ``[B, T]`` tensor (True =
            generate, False = preserve) for repaint / lego, or, when any
            extend item is present, a float ``[B, T]`` tensor in ``[0, 1]``
            with a linear ramp in the preserve region just before the
            boundary so the step-injection blend transitions smoothly between
            the preserved source latents and the generated extension.  1.0 =
            generate (model free), 0.0 = preserve source, fractional = blend.

            The model-level step-injection in all six variants has been
            updated to handle float masks (``m * xt + (1 - m) * zt_src``), so
            the ramp is honored end-to-end.  A short waveform-level splice is
            still applied after VAE decode to eliminate reconstruction drift
            in the kept region.
        """
        if extend_specs is None:
            extend_specs = [None] * batch_size

        chunk_masks = []
        spans = []
        is_covers = []
        repainting_ranges: Dict[int, Tuple[int, int]] = {}
        extend_ranges: Dict[int, Tuple[int, int, int]] = {}

        for i in range(batch_size):
            has_code_hint = audio_code_hints[i] is not None
            if repainting_start is not None and repainting_end is not None:
                start_sec = repainting_start[i] if repainting_start[i] is not None else 0.0
                end_sec = repainting_end[i]
                if end_sec is not None and end_sec > start_sec:
                    left_padding_sec = max(0, -start_sec)
                    adjusted_start_sec = start_sec + left_padding_sec
                    adjusted_end_sec = end_sec + left_padding_sec
                    start_latent = int(adjusted_start_sec * self.sample_rate // 1920)
                    end_latent = int(adjusted_end_sec * self.sample_rate // 1920)
                    start_latent = max(0, min(start_latent, max_latent_length - 1))
                    end_latent = max(start_latent + 1, min(end_latent, max_latent_length))

                    mask = torch.zeros(max_latent_length, dtype=torch.bool, device=self.device)
                    mask[start_latent:end_latent] = True
                    chunk_masks.append(mask)
                    spans.append(("repainting", start_latent, end_latent))
                    repainting_ranges[i] = (start_latent, end_latent)
                    spec_i = extend_specs[i] if i < len(extend_specs) else None
                    if spec_i is not None:
                        seam_sec = float(spec_i.get("seam_overlap_sec", 0.5))
                        seam_frames = max(0, int(round(seam_sec * 25.0)))
                        extend_ranges[i] = (start_latent, end_latent, seam_frames)
                        logger.info(
                            "[conditioning_masks] extend item {}: start_sec={:.3f} end_sec={:.3f} "
                            "start_latent={} end_latent={} seam_frames={} max_latent_length={}",
                            i, start_sec, end_sec, start_latent, end_latent,
                            seam_frames, max_latent_length,
                        )
                    else:
                        logger.debug(
                            "[conditioning_masks] repaint item {}: start_sec={:.3f} end_sec={:.3f} "
                            "start_latent={} end_latent={}",
                            i, start_sec, end_sec, start_latent, end_latent,
                        )
                    is_covers.append(False)
                    continue

            chunk_masks.append(torch.ones(max_latent_length, dtype=torch.bool, device=self.device))
            spans.append(("full", 0, max_latent_length))
            instruction_i = instructions[i] if instructions and i < len(instructions) else ""
            instruction_lower = instruction_i.lower()
            is_cover = (
                "generate audio semantic tokens" in instruction_lower
                and "based on the given conditions" in instruction_lower
            ) or has_code_hint
            is_covers.append(is_cover)

        chunk_masks_tensor = torch.stack(chunk_masks)

        # Diagnostic: dump chunk_masks per extend item BEFORE any auto-mode
        # overwrite.  The DiT's ``prepare_condition`` consumes these per-frame
        # flags (context vs generation target), so if they disagree with
        # ``repaint_mask`` the model silently plays both sides against the
        # middle and the extension region degenerates into gibberish.
        for _i in extend_ranges.keys():
            _m = chunk_masks_tensor[_i]
            _true_count = int(_m.sum().item()) if _m.dtype == torch.bool else int((_m > 0).sum().item())
            _mode = (
                chunk_mask_modes[_i]
                if chunk_mask_modes and _i < len(chunk_mask_modes)
                else None
            )
            _s, _e, _seam = extend_ranges[_i]
            logger.info(
                "[conditioning_masks] chunk_masks[{}] (extend) BEFORE auto: "
                "dtype={} shape={} true/nonzero_count={} "
                "kept=[0:{}]={} ext=[{}:{}]={} mode={!r}",
                _i, _m.dtype, tuple(_m.shape), _true_count,
                _s, _m[:_s].float().mean().item() if _s > 0 else "n/a",
                _s, _e, _m[_s:_e].float().mean().item(),
                _mode,
            )

        if chunk_mask_modes:
            for i, mode in enumerate(chunk_mask_modes):
                if mode == "auto":
                    if i in extend_ranges:
                        # Root cause of extend gibberish: the auto-mode 2.0
                        # overwrite erases the explicit kept=False /
                        # extension=True split that prepare_condition needs to
                        # know which frames are real source-audio context and
                        # which are the noise-seeded generation target.  Extend
                        # items must keep the per-frame split that was built
                        # above — skip the overwrite for them.
                        logger.info(
                            "[conditioning_masks] chunk_masks[{}] (extend) "
                            "keeping explicit kept/gen split; skipping "
                            "auto-mode 2.0 overwrite",
                            i,
                        )
                        continue
                    chunk_masks_tensor[i] = 2.0

        # Diagnostic: dump chunk_masks per extend item AFTER auto-mode overwrite.
        for _i in extend_ranges.keys():
            _m = chunk_masks_tensor[_i]
            _s, _e, _seam = extend_ranges[_i]
            logger.info(
                "[conditioning_masks] chunk_masks[{}] (extend) AFTER auto: "
                "dtype={} kept_mean={} ext_mean={} (prepare_condition input)",
                _i, _m.dtype,
                _m[:_s].float().mean().item() if _s > 0 else "n/a",
                _m[_s:_e].float().mean().item(),
            )

        is_covers_tensor = torch.BoolTensor(is_covers).to(self.device)

        src_latents_list = []
        for i in range(batch_size):
            has_code_hint = audio_code_hints[i] is not None
            has_target_audio = has_code_hint or (target_wavs is not None and target_wavs[i].abs().sum() > 1e-6)
            if has_target_audio:
                if i in repainting_ranges:
                    src_latent = target_latents[i].clone()
                    start_latent, end_latent = repainting_ranges[i]
                    instruction_i = instructions[i] if instructions and i < len(instructions) else ""
                    is_lego = _LEGO_INSTRUCTION_MARKER in instruction_i.lower()
                    if i in extend_ranges:
                        # Seed the extension region with random noise — never
                        # VAE-encoded silence — so the model's context
                        # conditioning is not biased toward fade-out silence
                        # even though step-injection uses mask=True (no inject)
                        # for this region.
                        seg = torch.randn(
                            end_latent - start_latent,
                            src_latent.shape[-1],
                            device=src_latent.device,
                            dtype=src_latent.dtype,
                        )
                        src_latent[start_latent:end_latent] = seg
                        logger.info(
                            "[conditioning_masks] extend src_latent item {}: "
                            "kept=[0:{}] noise-filled=[{}:{}] total={}",
                            i, start_latent, start_latent, end_latent,
                            src_latent.shape[0],
                        )
                    elif not is_lego:
                        src_latent[start_latent:end_latent] = silence_latent_tiled[start_latent:end_latent]
                    src_latents_list.append(src_latent)
                else:
                    src_latents_list.append(target_latents[i].clone())
            else:
                src_latents_list.append(silence_latent_tiled.clone())
        src_latents = torch.stack(src_latents_list)

        repaint_mask: Optional[torch.Tensor] = None
        if repainting_ranges:
            if extend_ranges:
                # Float mask with a linear seam ramp on the kept-side edge.
                # Outside the repaint span: 0 (preserve source).
                # Inside  the repaint span: 1 (model generates freely).
                # Ramp (preserve → repaint) spans ``seam_frames`` just before
                # ``start_latent`` and linearly rises 0 → 1 so the step
                # injection blend transitions smoothly between the preserved
                # source latents and the generated extension.
                repaint_mask = torch.zeros(
                    batch_size, max_latent_length, dtype=torch.float32, device=self.device,
                )
                for i, (start_latent, end_latent) in repainting_ranges.items():
                    repaint_mask[i, start_latent:end_latent] = 1.0
                    if i in extend_ranges:
                        _s, _e, seam = extend_ranges[i]
                        if seam > 0:
                            ramp_start = max(0, _s - seam)
                            ramp_len = _s - ramp_start
                            if ramp_len > 0:
                                ramp = torch.linspace(
                                    0.0, 1.0, steps=ramp_len + 2, device=self.device,
                                )[1:-1]
                                repaint_mask[i, ramp_start:_s] = ramp
                logger.info(
                    "[conditioning_masks] repaint_mask built (extend, float): "
                    "shape={} dtype={} "
                    "sum_per_item={} (1.0=generate, 0.0=preserve, fractional=blend)",
                    tuple(repaint_mask.shape), repaint_mask.dtype,
                    [float(s) for s in repaint_mask.sum(dim=-1).tolist()],
                )
            else:
                repaint_mask = torch.ones(
                    batch_size, max_latent_length, dtype=torch.bool, device=self.device,
                )
                for i, (start_latent, end_latent) in repainting_ranges.items():
                    repaint_mask[i] = False
                    repaint_mask[i, start_latent:end_latent] = True
                logger.info(
                    "[conditioning_masks] repaint_mask built (bool): shape={} "
                    "dtype={} true_per_item={} (True=generate, False=preserve)",
                    tuple(repaint_mask.shape), repaint_mask.dtype,
                    repaint_mask.sum(dim=-1).tolist(),
                )

        return chunk_masks_tensor, spans, is_covers_tensor, src_latents, repaint_mask
