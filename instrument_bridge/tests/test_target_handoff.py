import tempfile
import unittest
from pathlib import Path

from instrument_bridge.target_handoff import build_request, matching_indices, pick_at_pixel, requested_class


class TargetHandoffTest(unittest.TestCase):
    def test_language_and_duplicate_disambiguation(self):
        self.assertEqual(requested_class('pick up a scalpel'), 'scalpel')
        self.assertEqual(requested_class('pick the scalpel type 2'), 'scalpel_type2')
        self.assertEqual(requested_class('grasp scissors'), 'scissor')
        with self.assertRaises(ValueError):
            requested_class('pick an instrument')
        with self.assertRaises(ValueError):
            requested_class('pick scalpel and kelly')
        objects = [dict(class_name='scalpel', score=.9, bbox_xyxy=[1, 1, 10, 10]),
                   dict(class_name='scalpel', score=.8, bbox_xyxy=[20, 1, 30, 10])]
        self.assertEqual(matching_indices(objects, 'scalpel'), [0, 1])
        self.assertEqual(pick_at_pixel(objects, 25, 5), 1)
        self.assertIsNone(pick_at_pixel(objects, 50, 50))

    def test_request_contains_detection_only(self):
        obj = dict(class_name='kelly', score=.91, bbox_xyxy=[1, 2, 3, 4],
                   center_xy=[2, 3], mask_rle=dict(size=[8, 8], counts='abc'))
        report = dict(model='B', threshold=.5, source='synthetic', path='/scene', frame=None,
                      spawn_on_table={'kelly': 9}, per_camera=[dict(name='top', objects=[obj])])
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / 'top.png'
            image.write_bytes(b'fake')
            request = build_request(report, 'top', 0, 'pick up kelly', 'click', image)
            self.assertEqual(request['target']['class_name'], 'kelly')
            self.assertEqual(request['coordinate_frame'], 'camera_pixels')
            self.assertIsNone(request['pose_3d'])
            self.assertNotIn('spawn_on_table', request)
            with self.assertRaises(ValueError):
                build_request(report, 'top', 0, 'pick up scalpel', 'click', image)


if __name__ == '__main__':
    unittest.main()
