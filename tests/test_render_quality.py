import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "env"))

from phase4_scene import render_quality_from_environment


class RenderQualityContractTests(unittest.TestCase):
    def test_default_is_tested_native_dlaa(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(render_quality_from_environment(), ("DLAA", 2))

    def test_supported_modes_are_canonicalized(self):
        for requested, expected in (("off", "Off"), ("fxaa", "FXAA"),
                                    ("taa", "TAA"), ("dlaa", "DLAA"),
                                    ("dlss", "DLSS")):
            with self.subTest(requested=requested), patch.dict(
                    os.environ, {"P4_ANTIALIASING_MODE": requested}, clear=True):
                self.assertEqual(render_quality_from_environment(), (expected, 2))

    def test_invalid_mode_is_rejected(self):
        with patch.dict(os.environ, {"P4_ANTIALIASING_MODE": "magic"}, clear=True):
            with self.assertRaises(ValueError):
                render_quality_from_environment()

    def test_recorder_and_panel_lock_dataset_capture_to_dlaa(self):
        recorder = (ROOT / "src/entry/record.py").read_text(encoding="utf-8")
        panel = (ROOT / "src/ui/control_panel.py").read_text(encoding="utf-8")
        self.assertIn("os.environ['P4_ANTIALIASING_MODE'] = 'DLAA'", recorder)
        self.assertIn("env['P4_ANTIALIASING_MODE']='DLAA'", panel)


if __name__ == "__main__":
    unittest.main()
