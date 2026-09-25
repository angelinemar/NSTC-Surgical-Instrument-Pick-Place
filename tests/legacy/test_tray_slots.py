import unittest
import numpy as np
import torch
import types
import itertools
from phase4_tray_slots import ORDER, slot_spec, axes, select_occupancy, mul, SlotEvidence, EvidenceFailure
from phase4_grasp_validation import rigid_world_points, rotate

LAYOUT=dict(tray_xy=[.15,-.89],tray_yaw_deg=90.,tray_dimensions_local_xy=[.22,.55])


class TrayContractTests(unittest.TestCase):
    def test_rgb_qc_rejects_flat_middle_frame_and_solid_color(self):
        from phase4_feedback import rgb_has_spatial_detail,validate_recorded_rgb
        image=np.zeros((224,224,3),dtype=np.uint8); image[:112]=255
        solid=np.full_like(image,175)
        red=np.zeros_like(image); red[:,:,0]=255
        self.assertFalse(rgb_has_spatial_detail(solid))
        self.assertFalse(rgb_has_spatial_detail(red))
        frames=[image,image,image]
        rec=types.SimpleNamespace(stage_names=['a','b','c'],front_rgb=frames,grip_b_rgb=frames,
            extra_camera_rgb={n:frames for n in ('cam_top','cam_left','cam_right','cam_tray')})
        self.assertEqual(validate_recorded_rgb(rec),3)
        rec.front_rgb=[image,solid,image]
        with self.assertRaisesRegex(ValueError,'frame=1'):
            validate_recorded_rgb(rec)

    def test_metrics_respects_explicit_phase_not_displacement_substring(self):
        from phase4_metrics import RunMetrics
        metrics=object.__new__(RunMetrics)
        calls=[]
        metrics._finish_attempt=lambda category,reason:calls.append(category)
        metrics.observe_line('[PHASE FAIL] pick | TYPE2_LIFT_CLEAR | displacement=0.04m')
        metrics.observe_line('[PHASE FAIL] place | TYPE2_OPEN | error')
        metrics.observe_line('[SENSOR FAIL] front: blank RGB')
        self.assertEqual(calls,['pick_fail','place_fail','sensor_fail'])

    def test_slot_ids_match_existing_object_handlers(self):
        from object_handlers import HANDLERS
        self.assertEqual(set(HANDLERS),set(ORDER))
        for i,name in enumerate(ORDER):
            self.assertEqual(HANDLERS[name].object_type_id,i)

    def test_slots_disjoint_centered_fixed_ids(self):
        down,right=axes(LAYOUT)
        centers=np.array([slot_spec(n,LAYOUT)['center_xy'] for n in ORDER])
        np.testing.assert_allclose(centers.mean(axis=0),LAYOUT['tray_xy'])
        for i,n in enumerate(ORDER):
            s=slot_spec(n,LAYOUT)
            self.assertEqual(s['slot_id'],i)
            self.assertGreater(s['size_local_xy'][0],.17)
            self.assertGreater(s['size_local_xy'][1],.08)
        np.testing.assert_allclose(np.diff(centers,axis=0)@right[:2],[.104]*4)
        np.testing.assert_allclose((centers-LAYOUT['tray_xy'])@down[:2],0,atol=1e-10)

    def test_all_occupancy_counts_and_target_exclusion(self):
        for target in ORDER:
            rng=np.random.default_rng(42)
            counts=set()
            for _ in range(250):
                chosen=select_occupancy(target,rng,dict(tray_mode='random'))
                self.assertNotIn(target,chosen)
                self.assertEqual(len(chosen),len(set(chosen)))
                counts.add(len(chosen))
            self.assertEqual(counts,set(range(4)))

    def test_manual_rejects_target_duplicate_unknown(self):
        for names in (['scalpel'],['kelly','kelly'],['unknown']):
            with self.assertRaises(ValueError):
                select_occupancy('scalpel',np.random.default_rng(),dict(tray_mode='manual',tray_objects=names))

    def test_full_and_empty(self):
        for n in ORDER:
            self.assertEqual(len(select_occupancy(n,np.random.default_rng(),dict(tray_mode='full'))),4)
            self.assertEqual(select_occupancy(n,np.random.default_rng(),dict(tray_mode='empty')),[])

    def fixture(self,dx=0.,angle=0.,gap=.008):
        xy=slot_spec('scalpel',LAYOUT)['center_xy']
        yaw=np.deg2rad(90+angle)
        obj=types.SimpleNamespace(_p4_link_corners=np.array(list(itertools.product([-.075,.075],[-.010,.010],[-.001,.001]))),
             data=types.SimpleNamespace(root_link_pos_w=torch.tensor([[xy[0]+dx,xy[1],.003+gap+.001]],dtype=torch.float64),
                    root_link_quat_w=torch.tensor([[np.cos(yaw/2),0.,0.,np.sin(yaw/2)]],dtype=torch.float64)))
        env=types.SimpleNamespace(scene={'object':obj})
        ns=dict(PHASE3_TARGET_OBJECT='scalpel',phase3_scene_key_for_object=lambda n:'object',_p4_tray_floors={'scalpel':.003},_p4_tray_objects=[])
        ev=SlotEvidence(env,ns,LAYOUT)
        center=obj.data.root_link_pos_w[0].numpy(); points=rigid_world_points(obj)
        ee=center+np.array([0.,0.,.03]); q=np.array([1.,0.,0.,0.])
        ev.begin_lift(ee,q,center,points[:,2].min()); ev.lift_verified=True
        return ev,obj,ee,q,center,points

    def test_release_rejects_wrong_slot_yaw_height(self):
        for kwargs in (dict(dx=.025),dict(angle=15.),dict(gap=.035),dict(gap=.001)):
            ev,obj,ee,q,c,p=self.fixture(**kwargs)
            with self.assertRaises(EvidenceFailure):
                ev.begin_release(ee,q,c,p,p[:,2].min())

    def test_settled_aligned_release_is_accepted(self):
        ev,obj,ee,q,c,p=self.fixture()
        ev.begin_release(ee,q,c,p,p[:,2].min())
        obj.data.root_link_pos_w[0,2]-=.008
        p=rigid_world_points(obj); c=obj.data.root_link_pos_w[0].numpy()
        for _ in range(6): ev.check_retreat(ee+np.array([0,0,.1]),c,p,p[:,2].min())
        self.assertTrue(ev.place_verified)
        self.assertEqual(ev.summary()['slot']['slot_id'],0)


if __name__=='__main__': unittest.main()
