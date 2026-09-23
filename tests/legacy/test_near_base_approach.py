import unittest
import numpy as np
from phase4_approach import bounded_delta,shift_tcp_jacobian
import torch
from scipy.spatial.transform import Rotation


class TCPJacobianTests(unittest.TestCase):
    def test_exact_half_turn_other_arc_has_opposite_axis(self):
        from phase4_approach import long_arc_slerp
        start = torch.tensor([1.,0.,0.,0.])
        goal = torch.tensor([0.,0.,0.,1.])
        middle = long_arc_slerp(start, goal, .5)
        self.assertLess(float(middle[3]), 0.)
        self.assertAlmostEqual(float(torch.linalg.norm(middle)), 1., places=6)
        self.assertAlmostEqual(abs(float(torch.dot(long_arc_slerp(start,goal,1.),goal))), 1., places=6)

    def test_grasp_floor_uses_destination_wrist_orientation(self):
        from types import SimpleNamespace as NS
        from phase4_feedback import finger_floor_height
        rot=Rotation.from_euler('x',2,degrees=True)
        q=torch.tensor(np.r_[rot.as_quat()[3],rot.as_quat()[:3]],dtype=torch.float64)
        ee=torch.tensor([.2,0.,.1],dtype=torch.float64)
        offsets=np.array([[0.,-.04,-.01],[0.,.04,-.01]])
        robot=NS(_p4_finger_bounds=[(0,torch.zeros(1,3,dtype=torch.float64)),(1,torch.zeros(1,3,dtype=torch.float64))],
                 data=NS(body_pos_w=(torch.tensor(rot.apply(offsets))+ee)[None],body_quat_w=q.repeat(1,2,1)))
        env=NS(scene={'robot':robot,'ee_frame':NS(data=NS(target_pos_w=ee.reshape(1,1,3),target_quat_w=q.reshape(1,1,4)))})
        self.assertGreater(finger_floor_height(env,torch),.011)
        self.assertAlmostEqual(finger_floor_height(env,torch,torch.tensor([1.,0.,0.,0.],dtype=torch.float64)),.0102,places=7)
    def test_physx_com_to_tcp_shift_matches_origin_jacobian(self):
        rng=np.random.default_rng(27)
        for _ in range(20):
            rotation=Rotation.random(random_state=rng)
            com=np.array([.000075,-.001205,.023961]);tcp=np.array([0.,0.,.1034])
            origin=torch.tensor(rng.normal(size=(1,6,7)))
            at_com=shift_tcp_jacobian(origin,torch.tensor(rotation.apply(com)[None]))
            corrected=shift_tcp_jacobian(at_com,torch.tensor(rotation.apply(tcp-com)[None]))
            expected=shift_tcp_jacobian(origin,torch.tensor(rotation.apply(tcp)[None]))
            torch.testing.assert_close(corrected,expected)
    def test_downward_hand_uses_negative_z_lever(self):
        j=torch.zeros((1,6,1),dtype=torch.float64);j[0,4,0]=1
        corrected=shift_tcp_jacobian(j,torch.tensor([[0.,0.,-.1034]],dtype=torch.float64))
        self.assertAlmostEqual(float(corrected[0,0,0]),-.1034)
        self.assertEqual(float(j[0,0,0]),0.)  # borrowed PhysX tensor must not be mutated
    def test_matches_finite_difference_under_arbitrary_hand_rotation(self):
        rng=np.random.default_rng(7)
        for _ in range(20):
            rotation=Rotation.random(random_state=rng);offset=np.array([0.,0.,.1034])
            lever=rotation.apply(offset);j=rng.normal(size=(6,7))
            corrected=shift_tcp_jacobian(torch.tensor(j[None]),torch.tensor(lever[None]))[0].numpy()
            for k in range(7):
                epsilon=1e-7
                shifted=Rotation.from_rotvec(epsilon*j[3:,k]).apply(lever)+epsilon*j[:3,k]
                np.testing.assert_allclose(corrected[:3,k],(shifted-lever)/epsilon,atol=1e-6)
            np.testing.assert_allclose(corrected[3:],j[3:])


class BoundedIKTests(unittest.TestCase):
    def test_half_turn_lookahead_prefers_available_joint_room(self):
        from phase4_approach import rotation_arc_cost
        j=np.zeros((6,7));j[5,6]=1.;q=np.zeros(7);q[6]=1.56
        limits=np.tile([-2.897,2.897],(7,1))
        short=rotation_arc_cost(j,np.zeros(3),np.array([0.,0.,3.09]),q,limits)
        other=rotation_arc_cost(j,np.zeros(3),np.array([0.,0.,-(2*np.pi-3.09)]),q,limits)
        self.assertGreater(short,other);self.assertEqual(other,0.)
    def test_long_arc_same_end_orientation_opposite_direction(self):
        from phase4_approach import long_arc_slerp
        a=torch.tensor([1.,0.,0.,0.],dtype=torch.float64)
        angle=np.deg2rad(170)
        b=torch.tensor([np.cos(angle/2),0.,0.,np.sin(angle/2)],dtype=torch.float64)
        torch.testing.assert_close(long_arc_slerp(a,b,0),a)
        self.assertAlmostEqual(abs(float(torch.dot(long_arc_slerp(a,b,1),b))),1.)
        self.assertLess(float(long_arc_slerp(a,b,.5)[3]),0.)
    def test_only_failed_approach_changes_manual_wrist_branch(self):
        from phase4_feedback import record_wrist_outcome
        ns={'_p4_active_wrist_key':('scissor',.22,0.,0.)}
        for stage in ('SCISSOR_CLOSE','SCISSOR_LIFT_CLEAR','SCISSOR_MOVE_TO_TARGET','SCISSOR_LOWER_PLACE','SCISSOR_OPEN','SCISSOR_RETREAT'):
            record_wrist_outcome(ns,False,stage)
            self.assertFalse(ns.get('_p4_wrist_branches'))
        record_wrist_outcome(ns,False,'SCISSOR_OPEN_HOVER')
        self.assertTrue(ns['_p4_wrist_branches'][ns['_p4_active_wrist_key']])
    def test_commands_respect_joint_limits_and_step_cap(self):
        rng=np.random.default_rng(4)
        limits=np.tile([-2.,2.],(7,1))
        for _ in range(40):
            q=rng.uniform(-1.994,1.994,7);j=rng.normal(size=(6,7));error=rng.normal(size=6)*10
            target=bounded_delta(j,error,q,limits)
            self.assertTrue(np.all(target<=1.995));self.assertTrue(np.all(target>=-1.995))
            self.assertLessEqual(np.max(np.abs(target-q)),.2000001)
    def test_redundancy_used_when_primary_joint_at_limit(self):
        j=np.zeros((6,7));j[0,0]=1;j[0,6]=1
        q=np.zeros(7);q[0]=.995;limits=np.tile([-1.,1.],(7,1))
        target=bounded_delta(j,np.array([.04,0,0,0,0,0]),q,limits)
        self.assertLessEqual(target[0],.995);self.assertGreater(target[6],.03)
    def test_zero_error_holds_pose(self):
        q=np.zeros(7);j=np.eye(6,7);limits=np.tile([-2.,2.],(7,1))
        np.testing.assert_allclose(bounded_delta(j,np.zeros(6),q,limits),q,atol=1e-10)
    def test_nonfinite_input_is_rejected(self):
        with self.assertRaises((ValueError,RuntimeError)):
            bounded_delta(np.eye(6,7),np.full(6,np.nan),np.zeros(7),np.tile([-2.,2.],(7,1)))


if __name__=='__main__':unittest.main()
