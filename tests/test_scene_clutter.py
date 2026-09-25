import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import h5py
from src.recorder.scene_clutter import limits, pool, disjoint, table_footprint_allowed, validate
from src.recorder.instance_labels import body_mask, append_segment, audit_instances, CAMERAS
from src.recorder.domain_randomization import sample_background
from training.export_detection import visible_boxes


class ClutterTests(unittest.TestCase):
    def test_entire_tray_is_reserved_even_inside_table_grid(self):
        layout = dict(grid_x=[-2,2], grid_y=[-2,2], tray_xy=[0,0],
                      tray_yaw_deg=0, tray_dimensions_local_xy=[.4,.8])
        for yaw in (0,45,90):
            layout['tray_yaw_deg'] = yaw
            self.assertFalse(table_footprint_allowed(np.array([[-.01,-.01,0],[.01,.01,0]]),layout))
            self.assertTrue(table_footprint_allowed(np.array([[1,1,0],[1.1,1.1,0]]),layout))
        layout['tray_yaw_deg'] = 0
        # Center outside tray, but body overlaps its edge: still forbidden.
        self.assertFalse(table_footprint_allowed(np.array([[.19,-.01,0],[.3,.01,0]]),layout))
        self.assertFalse(table_footprint_allowed(np.array([[1.9,1.9,0],[2.1,2.1,0]]),layout))

    def test_settle_validation_rejects_legacy_tray_clutter(self):
        with self.assertRaisesRegex(RuntimeError,'CLUTTER_TRAY_FORBIDDEN'):
            validate(None, {'_p4_clutter':{'instances':[{'instance':'clutter_00','region':'tray'}]}})

    def test_default_pool_repeats_only_non_targets(self):
        with patch.dict(os.environ, {'P4_DISTRACTOR_MIN':'12','P4_DISTRACTOR_MAX':'18','P4_RANDOMIZATION_SEED':'17'}), patch('phase4_session.load_session', return_value=None):
            for target in ('scalpel','scissor','love_retractor','kelly','scalpel_type2'):
                entries = pool(target)
                self.assertEqual(len(entries),14)
                self.assertEqual(len({key for key,name in entries}),14)
                self.assertNotIn(target,[name for key,name in entries])
                self.assertLess(len({name for key,name in entries}),len(entries))
                self.assertEqual(pool(target),entries)
            self.assertEqual(limits(),(12,18))

    def test_background_stays_green(self):
        for seed in range(100):
            r,g,b = sample_background(seed)['color']
            self.assertGreater(g,r)
            self.assertGreater(g,b)

    def test_unlabeled_robot_joint_resolves_from_rendered_prim_path(self):
        from src.recorder.scene_semantics import complete_environment_labels
        raw=np.array([[5,6,7,0]])
        labels={'5':'/World/envs/env_0/Robot/panda_link4/visuals/mesh',
                '6':'/World/envs/env_0/HospitalRoom/Furniture/Table_04/Cloth/mesh',
                '7':'/World/envs/env_0/HospitalRoom/Floor/mesh','0':'BACKGROUND'}
        np.testing.assert_array_equal(complete_environment_labels(np.zeros_like(raw),raw,labels),[[1,8,9,0]])

    def test_mesh_parts_merge_but_duplicate_bodies_do_not(self):
        raw = np.array([[1,2,0,3],[1,2,0,3]],np.int32)
        masks = body_mask(raw,{'1':'/A/mesh','2':'/A/handle','3':'/B/mesh'}, {'/A':1,'/B':2})
        semantic = np.where(masks>0,4,8).astype(np.uint16)
        boxes = visible_boxes(semantic,min_pixels=1,instances=masks,instance_classes={'1':{'semantic_id':4},'2':{'semantic_id':4}})
        self.assertEqual(len(boxes),2)
        self.assertEqual([b['bbox'] for b in boxes],[[0,0,2,2],[3,0,1,2]])

    def test_segment_instance_alignment(self):
        from types import SimpleNamespace
        stages = ['X_PICK','X_PICK','X_PLACE','X_PLACE']
        frames = [np.full((2,2),i,np.uint16) for i in range(4)]
        recorder = SimpleNamespace(stage_names=stages,actions=[0]*4,instance_masks={'front':frames},instance_classes={})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'test.h5'
            with h5py.File(path,'w') as h:
                h.create_group('observations')
                h.create_dataset('stage_names',data=np.array(stages[2:],dtype=h5py.string_dtype()))
            append_segment(path,recorder,'place')
            with h5py.File(path) as h:
                np.testing.assert_array_equal(h['observations/front_instance'][:],frames[2:])
                self.assertEqual(h.attrs['instance_contract'],'rigid_body_visible_masks_v1')
                self.assertEqual(json.loads(h.attrs['semantic_class_ids'])['table'],8)

    def test_overlap_rejected(self):
        self.assertFalse(disjoint(((0,0),(1,1)),((.9,.9),(2,2))))
        self.assertTrue(disjoint(((0,0),(1,1)),((1.1,0),(2,1))))

    def test_instance_audit_catches_wrong_class_pixels(self):
        with tempfile.TemporaryDirectory() as tmp, h5py.File(Path(tmp)/'audit.h5','w') as h:
            h.attrs['instance_contract']='rigid_body_visible_masks_v1'
            h.attrs['instance_classes']=json.dumps({'1':{'semantic_id':4}})
            obs=h.create_group('observations')
            for name in CAMERAS:
                obs.create_dataset(name+'_instance',data=np.ones((1,2,2),np.uint16))
                obs.create_dataset(name+'_semantic',data=np.full((1,2,2),4,np.uint16))
            audit_instances(h)
            obs['front_semantic'][0,0,0]=3
            with self.assertRaisesRegex(ValueError,'disagree'):
                audit_instances(h)


if __name__ == '__main__':
    unittest.main()
