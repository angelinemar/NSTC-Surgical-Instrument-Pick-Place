import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.recorder.domain_randomization import sample_lighting
from training.export_detection import visible_boxes, export
from training.export_sensor_only import copy_policy_images
from training.perception import PerceptionModel, save_from_joint


class Contracts(unittest.TestCase):
    def test_visible_box_metrics_detect_false_positives(self):
        from training.detection_metrics import ap50, iou_xywh
        self.assertEqual(iou_xywh([0,0,10,10],[0,0,10,10]),1)
        good=dict(category_id=1,bbox_xywh=[0,0,10,10],confidence=.9)
        truth=dict(category_id=1,bbox=[0,0,10,10])
        self.assertEqual(ap50([dict(predictions=[good],truth=[truth])])['mean_ap50'],1.)
        bad=dict(category_id=1,bbox_xywh=[30,30,10,10],confidence=.99)
        self.assertEqual(ap50([dict(predictions=[bad,good],truth=[truth])])['mean_ap50'],.5)

    def test_light_seed_and_table_aim(self):
        layout = json.loads((ROOT/'env/scene_layout.json').read_text())
        first = sample_lighting(1, layout)
        self.assertEqual(first, sample_lighting(1, layout))
        self.assertNotEqual(first, sample_lighting(2, layout))
        for seed in range(100):
            for light in sample_lighting(seed, layout)['lights']:
                self.assertGreater(light['position'][2], 1.2)
                self.assertEqual(light['target'][2], 0.)
                self.assertTrue(.12 <= light['target'][0] <= .52)
                self.assertTrue(-.89 <= light['target'][1] <= .31)

    def test_native_camera_has_wider_projection_than_old_crop(self):
        # Native square recording keeps the no-crop path but uses a wider FOV
        # than the historical 224 crop so table edges are not clipped.
        old_fx = 16. / 20.955 * 448
        new_fx = 16. / (20.955 * .75) * 224
        self.assertLess(new_fx, old_fx)
        self.assertGreater(new_fx, 16. / 20.955 * 224)
        self.assertEqual(448/2 - (448-224)/2, 112)
        self.assertEqual(336/2 - (336-224)/2, 112)
        self.assertAlmostEqual(16./(20.955*.75)*448, new_fx*2)

    def test_collection_plan_keeps_sessions_and_targets_separate(self):
        from scripts.collect_training import plan
        jobs=plan(ROOT/'debug/test_runs/unused_test_plan','dp',1,448,
                  dict(train=[11,12],valid=[21],test=[31]))
        self.assertEqual(len(jobs),20)
        self.assertEqual(len({j['folder'] for j in jobs}),20)
        for job in jobs:
            self.assertEqual(job['config']['tray_mode'],'random')
            self.assertEqual(job['config']['mode'],'auto')
            self.assertIn('--camera-size',job['args'])

    def test_preview_keeps_image_edges(self):
        from types import SimpleNamespace
        from phase4_camera_preview import recorded_crop
        frame=np.zeros((448,448,3),np.uint8)
        frame[:20,:,0]=255
        env=SimpleNamespace(scene={'camera':SimpleNamespace(update=lambda *a,**kw:None)})
        preview=recorded_crop(env,{'read_camera_rgb_depth':lambda *a,**kw:(frame,None)},'camera')
        self.assertEqual(preview.shape,(224,224,3))
        self.assertEqual(int(preview[0,0,0]),255)

    def test_occlusion_fragments_remain_one_object(self):
        mask = np.zeros((20,30),np.uint16)
        mask[2:4,3:7]=3
        mask[12:14,19:23]=3
        boxes = visible_boxes(mask)
        self.assertEqual(len(boxes),1)
        self.assertEqual(boxes[0]['bbox'],[3,2,20,12])
        self.assertEqual(boxes[0]['area'],16)

    def test_small_absent_and_unknown_labels(self):
        mask=np.zeros((20,20),np.uint16)
        mask[1,1]=3
        self.assertEqual(visible_boxes(mask),[])
        mask[1,1]=11
        with self.assertRaises(ValueError): visible_boxes(mask)

    def test_split_alias_and_parent_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                export({'train':[folder], 'valid':[folder]},Path(folder)/'out')
            with self.assertRaises(ValueError):
                export({'train':[folder], 'valid':[Path(folder)/'nested']},Path(folder)/'out')

    def test_resize_keeps_semantic_ids_and_raw_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            with h5py.File(Path(folder)/'images.h5','w') as h:
                raw=np.zeros((1,448,448),np.uint16)
                raw[:,40:100,40:100]=7
                source=h.create_dataset('raw',data=raw)
                group=h.create_group('export')
                copy_policy_images(source,group,'mask',mask=True)
                self.assertEqual(group['mask'].shape,(1,224,224))
                self.assertEqual(set(np.unique(group['mask'][:])),{0,7})
                np.testing.assert_array_equal(source[:],raw)
                small=h.create_dataset('small',data=np.zeros((2,224,224,3),np.uint8))
                copy_policy_images(small,group,'rgb')
                self.assertEqual(group['rgb'].shape,(2,224,224,3))

    def test_independent_perception_matches_joint_head(self):
        from training.sensor_policy import SensorPolicy
        joint=SensorPolicy().eval()
        rgb=torch.rand(1,2,6,3,224,224)
        with torch.no_grad():
            _,logits=joint.encode(rgb,torch.zeros(1,2,16))
            batch=dict(rgb=rgb,proprio=torch.zeros(1,2,16),task_target=torch.tensor([[1.,0.,0.,0.,0.]]),actions=torch.zeros(1,16,8),semantic=torch.full((1,6,224,224),8,dtype=torch.long))
            self.assertTrue(torch.isfinite(joint.losses(batch)[0]))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'perception.pt'
            save_from_joint(joint,path,True)
            independent=PerceptionModel().eval()
            independent.load_state_dict(torch.load(path,weights_only=True)['model'])
            with torch.no_grad():
                actual=independent(rgb[:,-1].flatten(0,1))
            torch.testing.assert_close(actual,logits.flatten(0,1))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
