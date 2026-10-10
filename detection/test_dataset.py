"""Run with Isaac Python: -m unittest detection.test_dataset."""
import json
import random
import tempfile
import unittest
from pathlib import Path
from detection.dataset import camera_views, export_index, write_json


class DatasetTests(unittest.TestCase):
    def test_camera_determinism_and_views(self):
        layout = dict(grid_x=[.12,.52],grid_y=[-.69,.31],tray_xy=[.15,-.89])
        a = camera_views(random.Random(42),layout,8)
        self.assertEqual(a,camera_views(random.Random(42),layout,8))
        self.assertEqual(len(a),9)
        self.assertEqual(a[-1]['name'],'top')
        self.assertEqual(len({tuple(v['eye']) for v in a}),9)

    def test_missing_scene_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'train/scenes/scene_000001'; p.mkdir(parents=True)
            write_json(p/'scene.json',dict(scene_id=1,views=[]))
            with self.assertRaises(ValueError): export_index(tmp)

    def test_missing_view_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'train/scenes/scene_000000'; p.mkdir(parents=True)
            write_json(p/'scene.json',dict(scene_id=0,views=[dict(name='top',boxes=[])]))
            with self.assertRaises(ValueError): export_index(tmp)


if __name__ == '__main__': unittest.main()
