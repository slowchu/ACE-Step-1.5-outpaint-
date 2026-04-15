"""Unit tests for source-audio metadata analysis helpers."""

import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from acestep.audio_analysis import analyze_source_audio


class AudioAnalysisTests(unittest.TestCase):
    """Validate ``analyze_source_audio`` output shape and failure handling."""

    def test_returns_empty_payload_on_failure(self):
        """Failures should be swallowed and return null metadata."""
        with patch.dict("sys.modules", {"librosa": None}):
            result = analyze_source_audio("missing.wav")
        self.assertEqual({"bpm": None, "keyscale": None, "timesignature": None}, result)

    def test_tensor_input_returns_structured_metadata(self):
        """Tensor input should be converted and analyzed into expected keys."""
        fake_librosa = types.SimpleNamespace()
        fake_librosa.beat = types.SimpleNamespace(
            beat_track=lambda y, sr: (87.6, np.array([0, 10, 20, 30, 40, 50, 60, 70]))
        )
        chroma = np.zeros((12, 32), dtype=np.float32)
        chroma[11, :] = 1.0  # Strong B pitch class
        fake_librosa.feature = types.SimpleNamespace(
            chroma_cqt=lambda y, sr: chroma
        )
        fake_librosa.load = lambda path, sr, mono: (np.zeros(48000, dtype=np.float32), sr)

        wav = torch.ones(2, 48000)
        with patch.dict("sys.modules", {"librosa": fake_librosa}):
            result = analyze_source_audio(wav, sample_rate=48000)

        self.assertIsInstance(result.get("bpm"), int)
        self.assertIsInstance(result.get("timesignature"), int)
        self.assertIsInstance(result.get("keyscale"), str)
        self.assertTrue(
            result.get("keyscale").lower().endswith(" major")
            or result.get("keyscale").lower().endswith(" minor")
        )


if __name__ == "__main__":
    unittest.main()
