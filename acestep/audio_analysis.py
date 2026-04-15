"""Audio metadata analysis helpers for source-conditioned generation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _empty_analysis_result() -> Dict[str, Any]:
    """Return the default analysis payload used on failure."""
    return {"bpm": None, "keyscale": None, "timesignature": None}


def _estimate_timesignature_from_beats(beat_frames: np.ndarray) -> int:
    """Estimate a coarse meter (3 or 4), defaulting to 4."""
    if beat_frames is None or len(beat_frames) < 8:
        return 4

    intervals = np.diff(beat_frames)
    if len(intervals) < 4:
        return 4

    intervals = np.maximum(intervals.astype(np.float64), 1.0)
    interval_cv = float(np.std(intervals) / (np.mean(intervals) + 1e-8))
    if interval_cv > 0.35:
        return 4

    idx = np.arange(len(intervals), dtype=np.float64)
    phase3 = np.abs(np.sin((2.0 * np.pi / 3.0) * idx))
    phase4 = np.abs(np.sin((2.0 * np.pi / 4.0) * idx))
    score3 = float(np.mean(phase3))
    score4 = float(np.mean(phase4))
    return 3 if score3 < score4 * 0.92 else 4


def _infer_keyscale_from_chroma(chroma: np.ndarray) -> str | None:
    """Infer key name from chroma using Krumhansl-Schmuckler templates."""
    if chroma is None or chroma.size == 0:
        return None

    chroma_sum = np.sum(chroma, axis=1).astype(np.float64)
    if chroma_sum.shape[0] != 12 or not np.isfinite(chroma_sum).all():
        return None
    if np.allclose(chroma_sum, 0):
        return None

    major_profile = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        dtype=np.float64,
    )
    minor_profile = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
        dtype=np.float64,
    )

    chroma_vec = chroma_sum - np.mean(chroma_sum)
    chroma_norm = float(np.linalg.norm(chroma_vec))
    if chroma_norm <= 1e-8:
        return None

    best_score = -np.inf
    best_root = 0
    best_mode = "major"

    for shift in range(12):
        major_rot = np.roll(major_profile, shift)
        minor_rot = np.roll(minor_profile, shift)

        major_vec = major_rot - np.mean(major_rot)
        minor_vec = minor_rot - np.mean(minor_rot)

        major_score = float(np.dot(chroma_vec, major_vec) / (chroma_norm * (np.linalg.norm(major_vec) + 1e-8)))
        minor_score = float(np.dot(chroma_vec, minor_vec) / (chroma_norm * (np.linalg.norm(minor_vec) + 1e-8)))

        if major_score > best_score:
            best_score = major_score
            best_root = shift
            best_mode = "major"
        if minor_score > best_score:
            best_score = minor_score
            best_root = shift
            best_mode = "minor"

    key_names = ["C", "D♭", "D", "E♭", "E", "F", "F#", "G", "A♭", "A", "B♭", "B"]
    return f"{key_names[best_root]} {best_mode}"


def analyze_source_audio(audio_path_or_tensor: Any, sample_rate: int = 48000) -> Dict[str, Any]:
    """Extract BPM, key, and time signature from source audio.

    Args:
        audio_path_or_tensor: File path string or waveform tensor.
        sample_rate: Expected sample rate for loading/resampling.

    Returns:
        Dict with keys: ``bpm``, ``keyscale``, ``timesignature``.
    """
    try:
        import librosa

        if isinstance(audio_path_or_tensor, str):
            audio_np, sr = librosa.load(audio_path_or_tensor, sr=sample_rate, mono=True)
        else:
            import torch

            if isinstance(audio_path_or_tensor, torch.Tensor):
                tensor = audio_path_or_tensor
                if tensor.dim() == 1:
                    audio_np = tensor.detach().cpu().numpy()
                else:
                    audio_np = tensor.mean(dim=0).detach().cpu().numpy()
                sr = sample_rate
            else:
                return _empty_analysis_result()

        if audio_np is None or len(audio_np) == 0:
            return _empty_analysis_result()

        tempo, beat_frames = librosa.beat.beat_track(y=audio_np, sr=sr)
        bpm = int(round(float(tempo))) if tempo is not None and float(tempo) > 0 else None

        chroma = librosa.feature.chroma_cqt(y=audio_np, sr=sr)
        keyscale = _infer_keyscale_from_chroma(chroma)

        timesignature = _estimate_timesignature_from_beats(np.asarray(beat_frames))

        return {
            "bpm": bpm,
            "keyscale": keyscale,
            "timesignature": timesignature,
        }
    except Exception:
        return _empty_analysis_result()
