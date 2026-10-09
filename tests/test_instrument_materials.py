import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.instrument_materials import INSTRUMENT_MATERIAL_OVERRIDES


class InstrumentMaterialContractTests(unittest.TestCase):
    def test_every_instrument_has_a_recorder_owned_material(self):
        self.assertEqual(
            set(INSTRUMENT_MATERIAL_OVERRIDES),
            {"scalpel", "scissor", "love_retractor", "kelly", "scalpel_type2"},
        )

    def test_every_finish_is_readable_brushed_metal(self):
        for name, material in INSTRUMENT_MATERIAL_OVERRIDES.items():
            with self.subTest(name=name):
                self.assertEqual(len(material.diffuse_color), 3)
                self.assertTrue(all(0.0 <= channel <= 1.0 for channel in material.diffuse_color))
                self.assertGreater(material.metallic, 0.5)
                self.assertGreaterEqual(material.roughness, 0.30)
                self.assertLessEqual(material.roughness, 0.50)

    def test_class_finishes_are_neutral_not_label_colours(self):
        for name, material in INSTRUMENT_MATERIAL_OVERRIDES.items():
            with self.subTest(name=name):
                self.assertLessEqual(max(material.diffuse_color) - min(material.diffuse_color), 0.05)

    def test_preview_material_mode_contract(self):
        text = (ROOT / 'env' / 'phase3_shared_env_cfg.py').read_text(encoding='utf-8')
        self.assertIn('P4_INSTRUMENT_MATERIALS', text)
        self.assertIn('(\"recorder\", \"usd\")', text)


if __name__ == "__main__":
    unittest.main()
