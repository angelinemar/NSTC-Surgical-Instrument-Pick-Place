"""CPU tensor tests for all five backend bindings and failure transitions."""
import ast
import os
import types
import unittest
import tempfile
import sys
import h5py
from unittest.mock import patch
from pathlib import Path
import torch
from phase4_feedback import StageFailure, fingers_ready, finger_floor_height, align_control_tcp, install_feedback, completed_episode_count, long_transfer_steps, transfer_peak_speed
from phase4_feedback import FEEDBACK_VERSION

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'record.py').exists())


class Env:
    step_dt = .02
    def __init__(self, stuck=False, blocked=False):
        self.pos = torch.tensor([0., 0., .1])
        self.quat = torch.tensor([1., 0., 0., 0.])
        self.robot = types.SimpleNamespace(joint_names=['panda_finger_joint1','panda_finger_joint2'],
            data=types.SimpleNamespace(joint_pos=torch.tensor([[.04, .04]]), joint_vel=torch.zeros(1, 2)))
        self.scene = {'robot':self.robot}
        self.stuck, self.blocked, self.actions = stuck, blocked, []
    def step(self, action):
        pos, quat, grip = action
        self.actions.append(action)
        if not self.stuck:
            self.pos += .4*(pos-self.pos)
            self.quat = quat.clone()
        prev = self.robot.data.joint_pos.clone()
        if not self.blocked:
            # Intentionally unequal final contact positions.
            goal = torch.tensor([[.006, .010]]) if grip < 0 else torch.tensor([[.04, .04]])
            self.robot.data.joint_pos += .35*(goal-self.robot.data.joint_pos)
        self.robot.data.joint_vel = (self.robot.data.joint_pos-prev)/self.step_dt


def make_ns(path):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    names = ('step_dynamic','step_smooth_grip','step_hold_const_grip')
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(path),'exec'),ns)
    def pick_and_place_object(env, recorder, get_pos_fn, tray_slot_target_w=None):
        ns['step_dynamic'](env, recorder, 'SCISSOR_LOWER_GRASP', torch.tensor([1.,0.,0.]),
                           env.quat, 1., None, 1)
        raise AssertionError('Failed lower incorrectly continued')
    ns.update(torch=torch, MIN_INTERP_STEPS=12,PHASE3_TARGET_OBJECT='scalpel',phase3_scene_key_for_object=lambda n:'robot',
        MOTION_TIMING=types.SimpleNamespace(open_hover_profile=(7,50,2), lower_grasp_profile=(9,80,4),
            lift_clear_profile=(7,50,3),move_to_target_profile=(8,75,2),retreat_profile=(7,50,2)),
        args_cli=types.SimpleNamespace(debug_every=100), pick_and_place_object=pick_and_place_object,
        get_ee_pos_w=lambda e:e.pos, get_ee_quat_w=lambda e:e.quat,
        to_base_pos=lambda e,p:p,
        get_scalpel_pos_w=lambda e:torch.tensor([0.,0.,.1]),
        make_state=lambda *a:None, make_abs_action=lambda e,p,q,g:(p.clone(),q.clone(),g),
        slerp=lambda a,b,t:b)
    install_feedback(ns, ns['get_scalpel_pos_w'])
    return ns


class FeedbackTests(unittest.TestCase):
    @patch('phase4_tray_slots.yaw_alignment', return_value=[0.,0.,0.,1.])
    def test_detour_reassesses_wrist_arc_after_first_leg(self, alignment):
        env = Env()
        env.pos = torch.tensor([.22,.2,.16])
        env.robot.data.root_pos_w = torch.zeros(1,3)
        env.robot.data.root_quat_w = torch.tensor([[1.,0.,0.,0.]])
        controller = types.SimpleNamespace()
        env.action_manager = types.SimpleNamespace(get_term=lambda name:types.SimpleNamespace(_ik_controller=controller))
        ns = make_ns(next((ROOT/'backends').glob('*recorder.py')))
        ns['get_scalpel_pos_w'] = lambda e:e.pos.clone()
        rec = types.SimpleNamespace(add_step=lambda *a:None)
        fake_math = types.SimpleNamespace(quat_apply=lambda q,v:v)
        with patch.dict(sys.modules, {'isaaclab.utils.math':fake_math}), patch(
                'phase4_approach.select_long_arc', side_effect=[(False,178.),(True,182.)]) as arc:
            ns['step_dynamic'](env,rec,'TEST_MOVE_TO_TARGET',torch.tensor([.15,-.70,.18]),
                               env.quat,-1.,None,0,dist_thresh=.015)
        self.assertEqual(arc.call_count,2)
        torch.testing.assert_close(arc.call_args_list[1].args[1],torch.tensor([.45,-.15,.30]))
        self.assertTrue(ns['_p4_stage_progress']['TEST_MOVE_TO_TARGET']['waypoint_long_arc'])
        self.assertLess(float(torch.linalg.norm(env.pos-torch.tensor([.15,-.70,.18]))),.015)

    @patch('phase4_tray_slots.yaw_alignment', return_value=[1.,0.,0.,0.])
    def test_near_base_transfer_allows_progress_but_keeps_early_gates(self, alignment):
        class SpeedLimitedEnv(Env):
            max_translation_per_step = .005
            def step(self, action):
                before = self.pos.clone()
                super().step(action)
                delta = self.pos-before
                self.pos = before+delta*min(1., self.max_translation_per_step/max(float(torch.linalg.norm(delta)), 1e-12))

        class VerySlowEnv(SpeedLimitedEnv):
            max_translation_per_step = .0002

        path = next((ROOT/'backends').glob('*recorder.py'))
        target = torch.tensor([.707,0.,.1])
        for near, env, succeeds in ((True,SpeedLimitedEnv(),True),
                                    (False,SpeedLimitedEnv(),True),
                                    (False,VerySlowEnv(),False),
                                    (True,Env(stuck=True),False),
                                    (True,Env(),True)):
            with self.subTest(near=near, env=type(env).__name__, stuck=env.stuck):
                ns = make_ns(path)
                ns['_p4_near_base_approach'] = near
                ns['get_scalpel_pos_w'] = lambda e:e.pos.clone()
                rec = types.SimpleNamespace(add_step=lambda *a:None)
                def transfer():
                    ns['step_dynamic'](env,rec,'TEST_MOVE_TO_TARGET',target.clone(),
                                       env.quat,-1.,None,0,dist_thresh=.015)
                if succeeds:
                    transfer()
                    self.assertLess(float(torch.linalg.norm(env.pos-target)),.015)
                    self.assertLess(len(env.actions),160)
                    if isinstance(env,SpeedLimitedEnv):
                        self.assertGreater(len(env.actions),100)
                    else:
                        self.assertLess(len(env.actions),84)
                else:
                    with self.assertRaises(StageFailure):
                        transfer()
                    if isinstance(env,VerySlowEnv):
                        self.assertEqual(len(env.actions),204)
                    else:
                        self.assertLess(len(env.actions),100)

    def test_extension_uses_net_pose_progress_not_distance_to_goal(self):
        from phase4_feedback import motion_still_converging
        self.assertTrue(motion_still_converging([.1944,.1350],[47.,37.1],.025,15.))
        self.assertFalse(motion_still_converging([.135,.135],[37.1,37.1],.025,15.))
        self.assertFalse(motion_still_converging([.10,.14],[40.,39.],.025,15.))
        self.assertFalse(motion_still_converging([float('nan'),.14],[40.,39.],.025,15.))
        self.assertTrue(motion_still_converging([.001,.001],[.1,.1],.025,15.,stable=1))

    def test_long_transfer_speed_limit(self):
        for distance in (.61,.9,1.2):
            steps = long_transfer_steps(distance,.02)
            t = torch.arange(steps+1,dtype=torch.float64)/steps
            positions = distance*t*t*(3.-2.*t)
            self.assertLessEqual(float(torch.diff(positions).max())/.02,transfer_peak_speed(distance))
            self.assertAlmostEqual(float(positions[-1]),distance)
        with self.assertRaises(ValueError):
            long_transfer_steps(1.,0.)
        self.assertEqual(transfer_peak_speed(.75),1.)
        self.assertEqual(transfer_peak_speed(1.1),.6)
    def test_resume_counts_segments_and_rejects_partial_or_legacy(self):
        with tempfile.TemporaryDirectory(prefix='p4_resume_test_') as folder:
            paths = []
            for skill in ('pick','place'):
                path = Path(folder)/(skill+'_policy')/'scissor'/'episode_000000.h5'
                path.parent.mkdir(parents=True)
                with h5py.File(path,'w') as h5:
                    h5.attrs.update(p4_feedback_version=FEEDBACK_VERSION,success=True,
                        expert_grasp_evidence='{"geometry_source":"mesh_vertices_physx_link_v1","lift_verified":true,"place_verified":true,"tray_contract":"tray_slots_v1","slot":{"object":"scissor"},"initial_tray_objects":[]}',
                        control_tcp_matches_observed_ee=True,control_tcp_offset_m='[0.0, 0.0, 0.1034]',num_samples=1,
                        rgb_frame_qc='all_frames_spatial_v1')
                    keys = ['actions','observations/state','observations/robot_proprio','stage_names']
                    keys += ['observations/'+n+'_rgb' for n in ('front','wrist','cam_left','cam_right','cam_top','cam_tray')]
                    for key in keys:
                        h5.create_dataset(key,data=[0.])
                paths.append(path)
            self.assertEqual(completed_episode_count(folder,'scissor','both'),1)
            with self.assertRaisesRegex(RuntimeError,'pruned'):
                completed_episode_count(folder,'scissor','pick')
            with h5py.File(paths[0],'r+') as h5:
                h5.attrs['p4_feedback_version'] = 'legacy'
            with self.assertRaisesRegex(RuntimeError,'Incompatible'):
                completed_episode_count(folder,'scissor','both')
            with h5py.File(paths[0],'r+') as h5:
                h5.attrs['p4_feedback_version'] = FEEDBACK_VERSION
            paths[1].unlink()
            with self.assertRaisesRegex(RuntimeError,'pairs incomplete'):
                completed_episode_count(folder,'scissor','both')
    def test_tcp_contract(self):
        frame = types.SimpleNamespace(prim_path='/Robot/panda_hand',
                    offset=types.SimpleNamespace(pos=(0.,0.,.1034),rot=(1.,0.,0.,0.)))
        arm = types.SimpleNamespace(body_name='panda_hand',body_offset=types.SimpleNamespace(pos=(0.,0.,.107),rot=(1.,0.,0.,0.)))
        cfg = types.SimpleNamespace(actions=types.SimpleNamespace(arm_action=arm),
              scene=types.SimpleNamespace(ee_frame=types.SimpleNamespace(target_frames=[frame])))
        align_control_tcp(cfg)
        self.assertEqual(arm.body_offset.pos,frame.offset.pos)
        self.assertEqual(arm.body_offset.rot,frame.offset.rot)
        arm.body_name = 'wrong'
        with self.assertRaises(ValueError):
            align_control_tcp(cfg)
    def test_live_geometry_floor_translation_invariant(self):
        env = Env()
        env.robot._p4_finger_bounds = [(0,torch.tensor([[0.,0.,-.005],[0.,0.,.005]])),
                                     (1,torch.tensor([[0.,0.,-.005],[0.,0.,.005]]))]
        env.robot.data.body_pos_w = torch.tensor([[[.2,-.04,.045],[.2,.04,.045]]])
        env.robot.data.body_quat_w = torch.tensor([[[1.,0.,0.,0.],[1.,0.,0.,0.]]])
        ee = types.SimpleNamespace(data=types.SimpleNamespace(target_pos_w=torch.tensor([[[.2,0.,.05]]])))
        env.scene['ee_frame'] = ee
        self.assertAlmostEqual(finger_floor_height(env,torch), .0102, places=6)
        env.robot.data.body_pos_w[:,:,2] += .1
        ee.data.target_pos_w[:,:,2] += .1
        self.assertAlmostEqual(finger_floor_height(env,torch), .0102, places=6)

    def test_finger_gate(self):
        self.assertTrue(fingers_ready([[.006,.012]]*5,True,12,12))
        self.assertFalse(fingers_ready([[.04,.012]]*5,True,12,12))
        self.assertFalse(fingers_ready([[.01,.012]]*5,True,12,11))
        self.assertFalse(fingers_ready([[float('nan'),.01]]*5,True,12,20))
        self.assertFalse(fingers_ready([[.01+i*.002,.01] for i in range(5)],True,12,20))

    @patch('phase4_feedback.finger_floor_height', return_value=0.)
    @patch('phase4_tray_slots.yaw_alignment',return_value=[1.,0.,0.,0.])
    @patch('phase4_tray_slots.SlotEvidence',return_value=types.SimpleNamespace(tray_z=.003,lift_verified=False,place_verified=False))
    def test_all_backends(self, evidence, alignment, floor):
        paths = list((ROOT/'backends').glob('*recorder.py'))
        self.assertEqual(len(paths),5)
        for path in paths:
            with self.subTest(backend=path.name):
                ns = make_ns(path)
                rec = types.SimpleNamespace(add_step=lambda *a:None)
                env = Env()
                ns['step_smooth_grip'](env,rec,'TEST_CLOSE',env.pos,env.quat,1.,-1.,None,1,32)
                self.assertTrue(all(a[2] == -1 for a in env.actions))
                self.assertTrue(all(torch.equal(a[0],env.pos) for a in env.actions))
                self.assertGreaterEqual(len(env.actions),12)
                self.assertLessEqual(len(env.actions),45)
                ns['step_hold_const_grip'](env,rec,'TEST_OPEN',env.pos,env.quat,1.,None,1)
                self.assertTrue(bool((env.robot.data.joint_pos > .037).all()))
                with self.assertRaisesRegex(StageFailure,'finger_or_hold_timeout'):
                    e = Env(blocked=True)
                    ns['step_hold_const_grip'](e,rec,'TEST_CLOSE',e.pos,e.quat,-1.,None,1)
                target = torch.tensor([.05,0.,.02])
                ns['step_dynamic'](env,rec,'TEST_LOWER_GRASP',target,env.quat,1.,None,1)
                self.assertLess(float(torch.linalg.norm(env.pos-target)), .015)
                with self.assertRaises(StageFailure):
                    e = Env(stuck=True)
                    ns['step_dynamic'](e,rec,'TEST_MOVE_TO_TARGET',target,e.quat,-1.,None,1,max_steps=12)
                e = Env(stuck=True)
                result = ns['pick_and_place_object'](e,rec,ns['get_scalpel_pos_w'],torch.zeros(3))
                self.assertFalse(result[0])
                self.assertIn('pick | SCISSOR_LOWER_GRASP',ns['_p4_failure_message'])
                self.assertEqual(StageFailure('SCISSOR_LOWER_PLACE','timeout').category,'place')
                self.assertEqual(StageFailure('SCISSOR_OPEN_HOVER','timeout').category,'pick')

    def test_budget(self):
        ns = make_ns(next((ROOT/'backends').glob('*recorder.py')))
        previous = os.environ.get('P4_MAX_ATTEMPTS')
        try:
            os.environ['P4_MAX_ATTEMPTS'] = '2'
            ns['_p4_attempt_guard'](1,3,0)
            with self.assertRaisesRegex(RuntimeError,'ATTEMPT_BUDGET_EXHAUSTED'):
                ns['_p4_attempt_guard'](2,3,1)
        finally:
            if previous is None:
                os.environ.pop('P4_MAX_ATTEMPTS',None)
            else:
                os.environ['P4_MAX_ATTEMPTS'] = previous

    @patch('phase4_feedback.finger_floor_height',return_value=.0092)
    def test_love_reaches_thin_shaft_waypoint_without_lowering_floor(self, floor):
        ns=make_ns(next((ROOT/'backends').glob('*recorder.py')))
        ns['PHASE3_TARGET_OBJECT']='love_retractor'
        env=Env()
        rec=types.SimpleNamespace(add_step=lambda *a:None)
        target=torch.tensor([0.,0.,.0037])
        ns['step_dynamic'](env,rec,'LOVE_LOWER_GRASP',target,env.quat,1.,None,1)
        self.assertAlmostEqual(float(target[2]),.0092,places=6)
        self.assertLess(float(torch.linalg.norm(env.pos-target)),.00025)
        self.assertTrue(all(float(action[0][2])>=.00919 for action in env.actions))

    def test_quality_requires_feedback_proof_not_legacy_twenty_steps(self):
        ns = make_ns(next((ROOT/'backends').glob('*recorder.py')))
        counts = {'TEST_LOWER_GRASP':18,'TEST_LIFT_CLEAR':22,'TEST_MOVE_TO_TARGET':25,
                  'TEST_CLOSE':12,'TEST_OPEN':15}
        recorder = types.SimpleNamespace(stage_names=[n for n,c in counts.items() for _ in range(c)],
                    _p4_evidence=types.SimpleNamespace(lift_verified=True,place_verified=True),
                    _p4_gates={n:{'steps':c,'dt':.02} for n,c in counts.items()})
        import numpy as np
        frame=np.zeros((224,224,3),dtype=np.uint8); frame[:112]=255
        frames=[frame]*len(recorder.stage_names)
        recorder.front_rgb=frames; recorder.grip_b_rgb=frames
        recorder.extra_camera_rgb={n:frames for n in ('cam_top','cam_left','cam_right','cam_tray')}
        self.assertTrue(ns['quality_ok'](recorder,'TEST_'))
        del recorder._p4_gates['TEST_CLOSE']
        self.assertFalse(ns['quality_ok'](recorder,'TEST_'))


if __name__ == '__main__':
    unittest.main()
