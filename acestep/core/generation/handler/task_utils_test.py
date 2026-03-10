"""Unit tests for TaskUtilsMixin helper methods."""

import unittest

from acestep.core.generation.handler.task_utils import TaskUtilsMixin


class _Host(TaskUtilsMixin):
    """Minimal host implementing TaskUtilsMixin dependencies."""


class DetermineTaskTypeTests(unittest.TestCase):
    """Validate task-mode boolean computation in determine_task_type."""

    def setUp(self):
        self.host = _Host()

    def test_repaint_task_enables_repainting(self):
        """Repaint task should set can_use_repainting=True."""
        is_repaint, is_lego, is_cover, can_repaint = self.host.determine_task_type("repaint", None)
        self.assertTrue(is_repaint)
        self.assertFalse(is_lego)
        self.assertFalse(is_cover)
        self.assertTrue(can_repaint)

    def test_lego_task_does_not_enable_repainting(self):
        """Lego task must NOT set can_use_repainting=True.

        Lego mode uses the source audio as musical context; passing it through
        the repainting path would overwrite source latents with silence and
        cause the DiT to receive no audio context.
        """
        is_repaint, is_lego, is_cover, can_repaint = self.host.determine_task_type("lego", None)
        self.assertFalse(is_repaint)
        self.assertTrue(is_lego)
        self.assertFalse(is_cover)
        self.assertFalse(can_repaint, "lego must not use the repainting path")

    def test_cover_task_is_not_repaint(self):
        """Cover task should not set can_use_repainting=True."""
        is_repaint, is_lego, is_cover, can_repaint = self.host.determine_task_type("cover", None)
        self.assertFalse(is_repaint)
        self.assertFalse(is_lego)
        self.assertTrue(is_cover)
        self.assertFalse(can_repaint)

    def test_text2music_task_is_not_repaint(self):
        """Text-to-music task should not set can_use_repainting=True."""
        is_repaint, is_lego, is_cover, can_repaint = self.host.determine_task_type("text2music", None)
        self.assertFalse(is_repaint)
        self.assertFalse(is_lego)
        self.assertFalse(is_cover)
        self.assertFalse(can_repaint)

    def test_audio_codes_upgrade_task_to_cover(self):
        """Providing audio codes should set is_cover_task=True regardless of task_type."""
        is_repaint, is_lego, is_cover, can_repaint = self.host.determine_task_type(
            "text2music", "<|audio_code_1|>"
        )
        self.assertFalse(is_repaint)
        self.assertFalse(is_lego)
        self.assertTrue(is_cover)
        self.assertFalse(can_repaint)

    def test_lego_with_audio_codes_still_not_repainting(self):
        """Lego with audio codes should remain can_use_repainting=False."""
        is_repaint, is_lego, is_cover, can_repaint = self.host.determine_task_type(
            "lego", "<|audio_code_1|>"
        )
        self.assertTrue(is_lego)
        self.assertTrue(is_cover)
        self.assertFalse(can_repaint, "audio codes must not re-enable repainting for lego")


if __name__ == "__main__":
    unittest.main()
