import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from instrument_bridge.live_feed import publish_frame
from instrument_bridge.studio_worker import associate_tracks, evaluate_live_frame


class LiveFeedTest(unittest.TestCase):
    def test_same_camera_track_association(self):
        previous = [dict(class_name='kelly', score=.9, bbox_xyxy=[0, 0, 10, 10], track_id=7)]
        current = [dict(class_name='kelly', score=.8, bbox_xyxy=[1, 1, 11, 11]),
                   dict(class_name='scalpel', score=.7, bbox_xyxy=[1, 1, 11, 11])]
        next_id = associate_tracks(previous, current, 8)
        self.assertEqual(current[0]['track_id'], 7)
        self.assertEqual(current[1]['track_id'], 8)
        self.assertEqual(next_id, 9)

    def test_atomic_manifest_points_to_complete_image(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            first = publish_frame(folder, 'front', np.full((8, 9, 3), 42, dtype=np.uint8), scene_id=5)
            info = json.loads((folder / 'front_latest.json').read_text())
            self.assertEqual(info['camera'], 'front')
            self.assertEqual(info['scene_id'], 5)
            self.assertEqual(info['image'], first.name)
            self.assertEqual(np.asarray(Image.open(first)).shape, (8, 9, 3))
            second = publish_frame(folder, 'front', np.zeros((8, 9, 3), dtype=np.uint8), scene_id=6)
            self.assertNotEqual(first, second)
            self.assertEqual(json.loads((folder / 'front_latest.json').read_text())['image'], second.name)
            self.assertTrue(first.exists())
            with self.assertRaises(ValueError):
                publish_frame(folder, '../bad', np.zeros((8, 9, 3), dtype=np.uint8))

    def test_live_prediction_has_overlay_and_no_fabricated_metric(self):
        class Detector:
            def predict(self, rgb):
                mask = np.zeros(rgb.shape[:2], dtype=bool)
                mask[2:5, 3:7] = True
                obj = dict(class_name='scalpel', score=.8, bbox_xyxy=[3, 2, 7, 5], center_xy=[5, 3.5])
                return dict(objects=[obj], counts={'scalpel': 1}), [mask]

        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / 'live'
            output = Path(root) / 'output'
            publish_frame(folder, 'front', np.full((16, 16, 3), 100, dtype=np.uint8))
            info = json.loads((folder / 'front_latest.json').read_text())
            row = evaluate_live_frame(Detector(), folder, info, output)
            self.assertEqual(row['predicted']['scalpel'], 1)
            self.assertIsNone(row['metrics'])
            self.assertTrue((output / row['preview']).is_file())
            self.assertTrue((output / row['rgb']).is_file())

    def test_live_ground_truth_scores_this_camera(self):
        class Detector:
            def predict(self, rgb):
                mask = np.zeros(rgb.shape[:2], dtype=bool)
                mask[2:5, 3:7] = True
                obj = dict(class_name='scalpel', score=.8, bbox_xyxy=[3, 2, 7, 5], center_xy=[5, 3.5])
                return dict(objects=[obj], counts={'scalpel': 1}), [mask]

        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / 'live'
            gt = np.zeros((16, 16), dtype=np.uint16)
            gt[2:5, 3:7] = 1
            publish_frame(folder, 'front', np.full((16, 16, 3), 100, dtype=np.uint8),
                          scene_id=8, instance_mask=gt,
                          instance_classes={'1': {'name': 'scalpel'}},
                          spawn_on_table={'scalpel': 1})
            info = json.loads((folder / 'front_latest.json').read_text())
            row = evaluate_live_frame(Detector(), folder, info, Path(root) / 'output')
            self.assertEqual(row['metrics']['overall']['F1'], 1.0)
            self.assertEqual(row['visible_ground_truth']['scalpel'], 1)
            self.assertTrue((Path(root) / 'output' / row['ground_truth_preview']).is_file())


if __name__ == '__main__':
    unittest.main()
