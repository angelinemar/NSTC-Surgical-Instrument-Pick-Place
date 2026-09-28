import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from phase4_grasp_validation import GraspEvidence,EvidenceFailure
from phase4_session import validate_positions,apply_manual_positions
from phase4_metrics import RunMetrics,require_saved_goal


class EvidenceTests(unittest.TestCase):
    def test_instanced_robot_mesh_vertices(self):
        from pxr import Usd,UsdGeom
        from phase4_grasp_validation import mesh_points_in_frame
        import itertools
        stage=Usd.Stage.CreateInMemory()
        proto=UsdGeom.Xform.Define(stage,'/Prototype').GetPrim()
        mesh=UsdGeom.Mesh.Define(stage,'/Prototype/Mesh')
        mesh.CreatePointsAttr(list(itertools.product((-1.,1.),repeat=3)))
        finger=UsdGeom.Xform.Define(stage,'/Finger').GetPrim()
        finger.GetReferences().AddInternalReference(proto.GetPath())
        finger.SetInstanceable(True)
        points=mesh_points_in_frame(finger)
        self.assertEqual(len(points),8)
        np.testing.assert_allclose(points.min(axis=0),[-1,-1,-1])

    def test_scalpel_vertices_not_stale_extent(self):
        from phase4_cell_spawn import instrument_envelopes
        e=instrument_envelopes()['scalpel']
        # Actual imported vertices are offset +6.5cm along link Y. Authored
        # mesh extent incorrectly put them near Y=0 and broke table clearance.
        self.assertGreater(e['local_min'][1],.060)
        self.assertLess(e['local_max'][1],.071)
        self.assertAlmostEqual(e['size_m'][2],.15005,places=4)
        self.assertLess(e['radius_m'],.09)

    def evidence(self):
        evidence=GraspEvidence([0.,0.],[.22,.55],90.,.008)
        evidence.begin_lift([0.,0.,.01],[1.,0.,0.,0.],[0.,0.,.005],0.)
        return evidence

    def test_empty_grasp_aborts_near_start(self):
        e=self.evidence()
        e.check_lift([0.,0.,.031],[1.,0.,0.,0.],[0.,0.,.005],0.,3)
        with self.assertRaisesRegex(EvidenceFailure,'empty_grasp_early'):
            e.check_lift([0.,0.,.035],[1.,0.,0.,0.],[0.,0.,.005],0.,4)
        self.assertFalse(e.lift_verified)
        with self.assertRaises(EvidenceFailure): e.finish_lift()

    def test_follow_then_slip(self):
        e=self.evidence()
        e.check_lift([0.,0.,.07],[1.,0.,0.,0.],[0.,0.,.065],.06,9)
        e.finish_lift()
        e.check_held([.08,0.,.10],[1.,0.,0.,0.],[0.,0.,.06],10,'MOVE_TO_TARGET')
        with self.assertRaisesRegex(EvidenceFailure,'object_slipped'):
            e.check_held([.09,0.,.10],[1.,0.,0.,0.],[0.,0.,.05],11,'MOVE_TO_TARGET')

    def test_release_and_settle_requires_whole_footprint(self):
        e=self.evidence(); e.lift_verified=True
        points=np.array([[-.07,-.01,.025],[.07,.01,.035]])
        e.begin_release([0.,0.,.035],[1.,0.,0.,0.],[0.,0.,.030],points,.025)
        for _ in range(6):
            e.check_retreat([0.,0.,.2],[0.,0.,.015],points-[0.,0.,.015],.010)
        self.assertTrue(e.place_verified)
        e.check_retreat([0.,0.,.2],[0.,0.,.015],points+[.5,0.,0.],.01)
        self.assertFalse(e.place_verified)

    def test_object_still_in_fingers_after_open(self):
        e=self.evidence(); e.lift_verified=True
        points=np.array([[-.05,-.01,.025],[.05,.01,.035]])
        e.begin_release([0.,0.,.035],[1.,0.,0.,0.],[0.,0.,.03],points,.025)
        with self.assertRaisesRegex(EvidenceFailure,'still_attached'):
            e.check_retreat([0.,0.,.1],[0.,0.,.095],points+[0.,0.,.065],.09)

    def test_manual_bounds_overlap_and_yaw(self):
        layout={'grid_x':[0.,1.],'grid_y':[0.,1.]}
        radii={'a':.1,'b':.1}
        positions={'a':dict(x=.2,y=.2,yaw_deg=365),'b':dict(x=.7,y=.7,yaw_deg=-90)}
        result=validate_positions(positions,layout,radii)
        self.assertEqual(result['a']['yaw_deg'],5)
        positions['a']['x']=.01
        with self.assertRaisesRegex(ValueError,'edge'): validate_positions(positions,layout,radii)
        positions['a']=positions['b']
        with self.assertRaisesRegex(ValueError,'overlaps'): validate_positions(positions,layout,radii)

    def test_success_requires_completed_save(self):
        with tempfile.TemporaryDirectory() as folder:
            metrics=RunMetrics('scalpel',['--out_dir',folder])
            metrics.observe_line('ATTEMPT 1 | SUCCESS 0/1')
            metrics.observe_line('[RESULT] success=True steps=200')
            self.assertEqual(len(metrics.attempts),0)
            with self.assertRaisesRegex(RuntimeError,'COLLECTION_INCOMPLETE'):
                require_saved_goal(metrics,['--episodes','1'])
            metrics.observe_line('[SAVE SPLIT] 1/1 -> pick+place')
            self.assertEqual(metrics.attempts[0]['status'],'success')


if __name__=='__main__': unittest.main()
