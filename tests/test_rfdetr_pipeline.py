import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from training.rfdetr_pipeline import CLASSES, validate_coco, validate_dataset


def valid_coco():
    images = [{'id': 1, 'file_name': 'images/one.png', 'width': 448, 'height': 448}]
    annotations = []
    for category_id in range(1, 6):
        annotations.append({'id': category_id, 'image_id': 1, 'category_id': category_id,
                            'bbox': [category_id, category_id, 10, 10], 'area': 100, 'iscrowd': 0})
    return {'images': images, 'annotations': annotations,
            'categories': [{'id': i + 1, 'name': name} for i, name in enumerate(CLASSES)]}


class RFDETRDatasetValidationTests(unittest.TestCase):
    def test_static_scene_split_rejects_shared_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'manifest.json').write_text(json.dumps(dict(export_complete=True,
                split_rule='static_scene_grouped_no_view_split',rfdetr_dataset_file='roboflow')))
            for scene,split in enumerate(('train','valid','test')):
                folder=root/split; (folder/'images').mkdir(parents=True)
                (folder/'images/one.png').touch()
                coco=valid_coco();coco['images'][0]['scene_id']=scene
                (folder/'_annotations.coco.json').write_text(json.dumps(coco))
            self.assertEqual(len(validate_dataset(root)['splits']),3)
            path=root/'valid/_annotations.coco.json'
            coco=json.loads(path.read_text());coco['images'][0]['scene_id']=0
            path.write_text(json.dumps(coco))
            with self.assertRaisesRegex(ValueError,'leak across splits'):
                validate_dataset(root)

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
