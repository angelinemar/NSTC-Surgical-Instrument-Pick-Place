import unittest
from pathlib import Path
from unittest.mock import patch

from training.rfdetr_pipeline import CLASSES, validate_coco


def valid_coco():
    images = [{'id': 1, 'file_name': 'images/one.png', 'width': 448, 'height': 448}]
    annotations = []
    for category_id in range(1, 6):
        annotations.append({'id': category_id, 'image_id': 1, 'category_id': category_id,
                            'bbox': [category_id, category_id, 10, 10], 'area': 100, 'iscrowd': 0})
    return {'images': images, 'annotations': annotations,
            'categories': [{'id': i + 1, 'name': name} for i, name in enumerate(CLASSES)]}


class RFDETRDatasetValidationTests(unittest.TestCase):
    @patch.object(Path, 'is_file', return_value=True)
    def test_accepts_native_448_coco(self, _is_file):
        report = validate_coco(valid_coco(), Path('train'))
        self.assertEqual(report['images'], 1)
        self.assertEqual(report['annotations'], 5)

    @patch.object(Path, 'is_file', return_value=True)
    def test_rejects_non_448_source(self, _is_file):
        coco = valid_coco()
        coco['images'][0]['width'] = 224
        with self.assertRaisesRegex(ValueError, 'expected native 448x448'):
            validate_coco(coco, Path('train'))

    @patch.object(Path, 'is_file', return_value=True)
    def test_rejects_out_of_bounds_box(self, _is_file):
        coco = valid_coco()
        coco['annotations'][0]['bbox'] = [440, 440, 20, 20]
        with self.assertRaisesRegex(ValueError, 'out-of-bounds bbox'):
            validate_coco(coco, Path('train'))


if __name__ == '__main__':
    unittest.main()
