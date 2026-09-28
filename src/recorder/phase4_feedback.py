"""Feedback gates for P4 expert demonstrations (not policy observations).

Object geometry, camera/data schema and phase IDs remain owned by backends.
No timeout is a successful transition. Finger stability is not proof of grasp;
the existing physical lift and final placement checks remain mandatory.
"""
import functools
import inspect
import json
import math
import os
from pathlib import Path

CONTROL_TCP = {}
FEEDBACK_VERSION = '20260916-tray-slots-v1'
POLICY_WINDOW_CONTRACT = 'lower_pre_to_lift_clear__lower_place_to_retreat_v1'


def rgb_has_spatial_detail(frame):
    """Flat gray/black/solid-color render buffers are not training images."""
    import numpy as np
    a=np.asarray(frame)
    return (a.ndim==3 and a.shape[-1]==3 and a.dtype==np.uint8
            and float(a.astype(np.float32).std(axis=(0,1)).max())>1.)


def validate_recorded_rgb(recorder, crop_size=224):
    streams={'front':getattr(recorder,'front_rgb',[]),
             'wrist':getattr(recorder,'grip_b_rgb',[])}
    extra=getattr(recorder,'extra_camera_rgb',{})
    streams.update({name:extra.get(name,[]) for name in ('cam_top','cam_left','cam_right','cam_tray')})
    total=len(recorder.stage_names)
    if total<1:
        raise ValueError('empty recording')
    for name,frames in streams.items():
        if len(frames)!=total:
            raise ValueError(f'{name}: RGB frames={len(frames)}, expected={total}')
        for index,frame in enumerate(frames):
            if crop_size:
                h,w=frame.shape[:2]
                if min(h,w)<crop_size:
                    raise ValueError(f'{name}: image smaller than recording crop')
                y,x=(h-crop_size)//2,(w-crop_size)//2
                frame=frame[y:y+crop_size,x:x+crop_size]
            if not rgb_has_spatial_detail(frame):
                raise ValueError(f'{name}: blank/flat RGB frame={index} stage={recorder.stage_names[index]}')
    return total


def completed_episode_count(out_dir, object_name, mode):
    """Resume only contiguous completed segments with the current TCP contract."""
    import h5py
    root = Path(out_dir)
    from phase4_storage import recover, require_commit
    recover(root)
    excluded = 'place' if mode == 'pick' else 'pick' if mode == 'place' else None
    if excluded and any(root.glob(excluded+'_policy/*/episode_*.h5')):
        raise RuntimeError('Existing opposite-skill data would be pruned; use a separate output directory')
    groups = {skill:sorted((root/(skill+'_policy')/object_name).glob('episode_*.h5'))
              for skill in ('pick','place')}
    selected = ('pick','place') if mode == 'both' else (mode,)
    if any(groups[s] for s in groups if s not in selected):
        raise RuntimeError('Resume record_mode differs from existing segments; use a new output directory')
    indices = []
    for skill in selected:
        files = groups[skill]
        ids = [int(p.stem.rsplit('_',1)[1]) for p in files]
        if ids != list(range(len(ids))):
            raise RuntimeError(f'Incomplete/noncontiguous {skill} episodes; refusing to overwrite or skip holes')
        for path in files:
            with h5py.File(path,'r') as h5:
                require_commit(path, h5)
                evidence=json.loads(h5.attrs.get('expert_grasp_evidence','{}'))
                if (h5.attrs.get('p4_feedback_version') != FEEDBACK_VERSION
                        or h5.attrs.get('policy_window_contract') != POLICY_WINDOW_CONTRACT
                        or evidence.get('geometry_source') != 'mesh_vertices_physx_link_v1'
                        or not evidence.get('lift_verified') or not evidence.get('place_verified')
                        or evidence.get('tray_contract') != 'tray_slots_v1'
                        or evidence.get('slot',{}).get('object') != object_name
                        or object_name in evidence.get('initial_tray_objects',[])
                        or not bool(h5.attrs.get('success',False))
                        or not h5.attrs.get('control_tcp_matches_observed_ee',False)
                        or h5.attrs.get('rgb_frame_qc') != 'all_frames_spatial_v1'
                        or json.loads(h5.attrs.get('control_tcp_offset_m','null')) != [0.,0.,.1034]):
                    raise RuntimeError(f'Incompatible/incomplete episode {path}; use a fresh tray-slots output directory')
                total = int(h5.attrs.get('num_samples',0))
                required = ['actions','observations/state','observations/robot_proprio','stage_names']
                required += ['observations/'+n+'_rgb' for n in ('front','wrist','cam_left','cam_right','cam_top','cam_tray')]
                if total < 1 or any(k not in h5 or len(h5[k]) != total for k in required):
                    raise RuntimeError(f'Incomplete H5 topics in {path}; refusing unsafe resume')
        indices.append(ids)
    if any(ids != indices[0] for ids in indices):
        raise RuntimeError('Pick/place episode pairs incomplete; refusing unsafe resume')
    return len(indices[0])


def align_control_tcp(env_cfg):
    arm = env_cfg.actions.arm_action
    frame = env_cfg.scene.ee_frame.target_frames[0]
    if frame.prim_path.rsplit('/',1)[-1] != arm.body_name:
        raise ValueError('Control body and observed EE body differ')
    previous = tuple(arm.body_offset.pos)
    arm.body_offset.pos = tuple(frame.offset.pos)
    arm.body_offset.rot = tuple(frame.offset.rot)
    if hasattr(arm,'class_type'):
        from phase4_approach import corrected_action_type
        arm.class_type=corrected_action_type()
    CONTROL_TCP.update(link=arm.body_name, pos=list(arm.body_offset.pos), rot=list(arm.body_offset.rot))
    print(f'[P4 TCP ALIGNED] {arm.body_name}: IK {previous} -> {arm.body_offset.pos}; observation/action same TCP', flush=True)


def finger_floor_height(env, torch, target_quat=None):
    """TCP height that keeps both finger meshes 0.2 mm above P4 table Z=0.

Use authored local finger bounds with live articulation body transforms, not
stale USD world transforms. P4 table_geometry aligns the support plane to Z=0.
"""
    robot = env.scene['robot']
    if not hasattr(robot, '_p4_finger_bounds'):
        from phase4_grasp_validation import mesh_points_in_frame
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        records = []
        for body in ('panda_leftfinger','panda_rightfinger'):
            index = robot.body_names.index(body)
            prim = stage.GetPrimAtPath('/World/envs/env_0/Robot/'+body)
            if not prim.IsValid():
                raise RuntimeError('Missing finger geometry: '+body)
            corners = torch.tensor(mesh_points_in_frame(prim),
                                   device=robot.data.body_pos_w.device, dtype=robot.data.body_pos_w.dtype)
            records.append((index,corners))
        robot._p4_finger_bounds = records
    ee = env.scene['ee_frame'].data.target_pos_w[0,0]
    def rotate(q,points):
        q=q/torch.linalg.norm(q)
        xyz=q[1:].expand_as(points)
        cross=2*torch.linalg.cross(xyz,points)
        return points+q[0]*cross+torch.linalg.cross(xyz,cross)
    inverse=None
    if target_quat is not None:
        inverse=env.scene['ee_frame'].data.target_quat_w[0,0].clone()
        inverse[1:]*=-1
    lowest = float('inf')
    for index,corners in robot._p4_finger_bounds:
        q = robot.data.body_quat_w[0,index]
        xyz = q[1:].expand_as(corners)
        cross = 2*torch.linalg.cross(xyz,corners)
        world = corners+q[0]*cross+torch.linalg.cross(xyz,cross)+robot.data.body_pos_w[0,index]
        if inverse is not None:
            # Predict the same finger geometry at the final grasp orientation.
            # A still-tilted pregrasp wrist must not freeze an over-high floor.
            world=rotate(target_quat,rotate(inverse,world-ee))+ee
        lowest = min(lowest,float(world[:,2].min()))
    floor = float(ee[2])-lowest+.0002
    if not -.005 < floor < .05:
        raise RuntimeError(f'Unexpected TCP/finger floor clearance: {floor}')
    return max(0.,floor)


class StageFailure(RuntimeError):
    def __init__(self, stage, reason):
        self.stage, self.reason = stage, reason
        self.category = 'place' if any(s in stage for s in
            ('MOVE_TO_TARGET', 'PLACE', '_OPEN', 'RETREAT')) and not stage.endswith('OPEN_HOVER') else 'pick'
        super().__init__(f'{self.category} | {stage} | {reason}')


def manual_wrist_key(ns):
    """Stable key for a requested custom pose, not noisy measured coordinates."""
    target=ns['PHASE3_TARGET_OBJECT']
    pose=ns.get('_p4_manual_positions',{}).get(target)
    return (target,float(pose['x']),float(pose['y']),float(pose['yaw_deg'])%360) if pose else None


def record_wrist_outcome(ns,success,stage=None):
    # A grasp that reached CLOSE/LIFT is not an approach-branch failure.
    # Do not replace its working branch because placement or image QC failed.
    if stage is not None and not stage.endswith(('OPEN_HOVER','LOWER_PRE','LOWER_GRASP','LOWER_EXTRA')):
        return
    key=ns.get('_p4_active_wrist_key')
    if key is not None and not success:
        cache=ns.setdefault('_p4_wrist_branches',{})
        cache[key]=not cache.get(key,False)


def fingers_ready(history, closing, minimum_steps, step):
    """Each finger must have stopped; unequal contact positions are allowed."""
    if step < minimum_steps or len(history) < 5:
        return False
    if not all(math.isfinite(v) for sample in history for v in sample):
        return False
    stable = all(max(h[j] for h in history)-min(h[j] for h in history) <= .001
                 for j in (0, 1))
    # A blocked fully-open finger is not a closed gripper. Thin objects can
    # bring fingers almost to zero, so don't reject them as empty here.
    position = all(v < .035 for v in history[-1]) if closing else all(v > .037 for v in history[-1])
    return stable and position


def transfer_peak_speed(distance):
    # Far-side thin-blade transport reproducibly slipped with a 1 m/s
    # command. Keep ordinary moves unchanged; lower inertial loading only
    # for >80 cm transfers. This is a command limit, not measured EE speed.
    return .6 if distance > .8 else 1.


def transfer_detour(start_base, goal_base):
    """Expert waypoint around the base corridor; no change to the final target."""
    import numpy as np
    start, goal = np.asarray(start_base, dtype=float), np.asarray(goal_base, dtype=float)
    if start.shape != (3,) or goal.shape != (3,) or not np.isfinite([start, goal]).all():
        raise ValueError('Invalid transfer endpoints')
    # Joint-branch problems also occur on the outer-slot paths that clear a
    # narrow base-radius test. Route every far-side-to-tray transfer consistently.
    if start[1] <= .10 or goal[1] >= -.30:
        return None
    return np.array([.45, -.15, max(.30, start[2], goal[2])])


def long_transfer_steps(distance, dt):
    """Cubic ease-in/out with a distance-specific command speed limit."""
    if not math.isfinite(distance+dt) or distance < 0 or dt <= 0:
        raise ValueError('Invalid transfer distance/dt')
    return max(1, math.ceil(1.5*distance/(transfer_peak_speed(distance)*dt)))


def motion_still_converging(distances, angles, position_tolerance, angle_tolerance, stable=0):
    """Allow bounded extra time for net pose improvement, even far from goal."""
    if stable > 0:
        return True
    if len(distances) < 2 or len(angles) < 2:
        return False
    gain = ((distances[0]-distances[-1])/position_tolerance
            + (angles[0]-angles[-1])/angle_tolerance)
    return math.isfinite(gain) and gain > .02


def install_feedback(ns, other_get):
    torch = ns['torch']
    timing = ns['MOTION_TIMING']
    original_smooth = ns['step_smooth_grip']
    original_hold = ns['step_hold_const_grip']
    ns['_p4_stage_targets'] = {}
    ns['_p4_stage_quats'] = {}
    from phase4_grasp_validation import GraspEvidence, EvidenceFailure, rotate

    def geometry(env):
        from phase4_grasp_validation import rigid_world_points
        name = ns['PHASE3_TARGET_OBJECT']
        obj = env.scene[ns['phase3_scene_key_for_object'](name)]
        points = rigid_world_points(obj)
        # Convex-hull vertex density is NOT an object's geometric midpoint.
        # Use the fixed midpoint of its link-frame extents (e.g. curved hooks
        # otherwise bias the mean toward the end with more mesh vertices).
        mid=(obj._p4_link_corners.min(axis=0)+obj._p4_link_corners.max(axis=0))/2
        center=rotate(obj.data.root_link_quat_w[0].detach().cpu().numpy(),mid)+obj.data.root_link_pos_w[0].detach().cpu().numpy()
        return center, float(points[:,2].min()), points

    def measured(env):
        center,bottom,points = geometry(env)
        return ns['get_ee_pos_w'](env).detach().cpu().numpy(), ns['get_ee_quat_w'](env).detach().cpu().numpy(),center,bottom,points

    def check_evidence(fn, name, *args):
        try:
            return fn(*args)
        except EvidenceFailure as error:
            raise StageFailure(name,str(error)) from error

    def mark_gate(recorder, name, count, dt):
        base = getattr(recorder, 'original', recorder)
        if not hasattr(base, '_p4_gates'):
            base._p4_gates = {}
        base._p4_gates[name] = {'steps':count, 'dt':float(dt)}

    def pose_error(env, target, quat):
        distance = float(torch.linalg.norm(ns['get_ee_pos_w'](env)-target))
        q = ns['get_ee_quat_w'](env)
        dot = float((q*quat).sum()) / max(float(torch.linalg.norm(q)*torch.linalg.norm(quat)), 1e-12)
        angle = math.degrees(2*math.acos(min(1., abs(dot))))
        if not math.isfinite(distance+angle):
            raise ValueError('Non-finite robot pose')
        return distance, angle

    def sample(env, recorder, name, i, pos, quat, grip, slot, phase):
        from phase4_session import check_interrupt
        if check_interrupt(ns):
            raise StageFailure(name,'manual_discard_requested')
        state = ns['make_state'](env, slot, phase)
        action = ns['make_abs_action'](env, pos, quat, grip)
        recorder.add_step(env, name, i, state, action)
        env.step(action)
        obj = ns['get_scalpel_pos_w'](env) if phase == 0 else other_get(env)
        return float(obj[2])

    def dynamic(env, recorder, name, target_w, quat_w, grip,
                active_target_w, phase_id, max_steps=80, dist_thresh=.015,
                settle_steps=6, force_wait=False):
        if name.endswith('OPEN_HOVER') and hasattr(env,'action_manager'):
            from phase4_approach import configure
            base=env.scene['robot'].data.root_pos_w[0]
            ns['_p4_near_base_approach']=float(torch.linalg.norm(target_w[:2]-base[:2]))<.32
            configure(env,True)
            ns['_p4_bounded_ik']=True
        if hasattr(env, 'action_manager'):
            from phase4_approach import set_precision_skill
            set_precision_skill(env, 'place' if name.endswith(('MOVE_TO_TARGET','LOWER_PLACE','RETREAT')) else 'pick')
        if name.endswith(('LOWER_GRASP', 'LOWER_EXTRA')):
            requested_z = float(target_w[2])
            floor = finger_floor_height(env, torch, quat_w)
            target_w[2] = max(requested_z, floor)
            print(f'[{name}] TABLE_CLEARANCE requested_tcp_z={requested_z:.4f} safe_tcp_z={float(target_w[2]):.4f} (finger vertices + 0.2mm)', flush=True)
        ns['_p4_stage_targets'][name] = target_w.clone()
        if name.endswith('OPEN_HOVER') and os.environ.get('P4_SYMMETRIC_GRASP', '1') == '1':
            # Parallel jaws admit a 180-degree yaw alternative at the same
            # grasp point. Select the equivalent yaw with less wrist rotation.
            half_turn = quat_w.new_tensor([0., 0., 0., 1.])
            alternate = ns['quat_mul'](half_turn.reshape(1,4), quat_w.reshape(1,4))[0]
            current = ns['get_ee_quat_w'](env)
            from phase4_tray_slots import yaw_alignment, mul
            from phase4_scene import LAYOUT
            obj=env.scene[ns['phase3_scene_key_for_object'](ns['PHASE3_TARGET_OBJECT'])]
            delta=check_evidence(yaw_alignment,name,obj,LAYOUT)
            # Both approaches pinch the same point with the same jaw axis.
            # Select the branch whose eventual aligned place orientation is
            # closest to the backend's proven base wrist orientation.
            reference=ns.get('_p4_transport_quat',current)
            destination=quat_w.new_tensor(mul(delta,quat_w.detach().cpu().numpy()))
            alternative_destination=quat_w.new_tensor(mul(delta,alternate.detach().cpu().numpy()))
            if abs(float((reference*alternative_destination).sum())) > abs(float((reference*destination).sum())) + .05:
                quat_w.copy_(alternate)
                print(f'[{name}] SYMMETRIC_GRASP yaw+180; same point/jaw axis, place-compatible wrist branch', flush=True)
            # Never flip a working branch just because the attempt is even.
            # Only a failed custom-pose motion tries the equivalent alternative;
            # a successful branch is retained for subsequent identical poses.
            key=manual_wrist_key(ns)
            ns['_p4_active_wrist_key']=key
            if key is not None and ns.get('_p4_wrist_branches',{}).get(key,False):
                quat_w.copy_(ns['quat_mul'](half_turn.reshape(1,4),quat_w.reshape(1,4))[0])
                print(f'[{name}] RETRY alternate equivalent wrist branch',flush=True)
        interp = ns['MIN_INTERP_STEPS']
        for suffix, profile in (
            ('OPEN_HOVER', timing.open_hover_profile),
            ('LOWER_GRASP', timing.lower_grasp_profile),
            ('LIFT_CLEAR', timing.lift_clear_profile),
            ('MOVE_TO_TARGET', timing.move_to_target_profile),
            ('RETREAT', timing.retreat_profile)):
            if name.endswith(suffix):
                interp, cap, settle = profile
                max_steps, settle_steps = min(max_steps, cap), min(settle_steps, settle)
                break
        if ns.get('_p4_near_base_approach') and name.endswith(('OPEN_HOVER','LOWER_PRE','LOWER_GRASP')):
            interp=max(interp,25 if name.endswith('OPEN_HOVER') else 15)
            max_steps=max(max_steps,120)
            print(f'[{name}] BOUNDED_APPROACH max={max_steps}; ends immediately after unchanged pose gate',flush=True)
        if ns.get('_p4_near_base_approach') and name.endswith('LIFT_CLEAR'):
            # Measured constrained near-base lift is still converging at the
            # old 50+12 cap. Keep the lift/grasp gates; allow completion.
            max_steps=max(max_steps,80)
        if name.endswith('SCALPEL_TYPE2_MOVE_TO_TARGET'):
            interp = max(interp, 16)
        lower = 'LOWER_' in name
        if name.endswith(('LOWER_GRASP', 'LOWER_EXTRA')):
            # CLOSE now holds still instead of completing the last descent.
            # Reach the contact waypoint first, rather than closing 1 cm high.
            # Love's thin shaft can settle with its center below 2 mm. A
            # 1 mm early stop leaves contact on the fingertip edge only.
            # Reach closer to the SAME safe tabletop waypoint; never lower
            # the target below the mesh-derived floor or add a hold stage.
            contact_tolerance = .00025 if ns.get('PHASE3_TARGET_OBJECT') == 'love_retractor' else .001
            dist_thresh = min(dist_thresh, contact_tolerance)
            settle_steps = 2
        angle_tol = 10. if lower else 15.
        if name.endswith('MOVE_TO_TARGET'):
            from phase4_tray_slots import yaw_alignment, mul
            from phase4_scene import LAYOUT
            obj=env.scene[ns['phase3_scene_key_for_object'](ns['PHASE3_TARGET_OBJECT'])]
            delta=check_evidence(yaw_alignment,name,obj,LAYOUT)
            current=ns['get_ee_quat_w'](env).detach().cpu().numpy()
            ns['_p4_transport_quat']=quat_w.new_tensor(mul(delta,current))
        if name.endswith(('MOVE_TO_TARGET','LOWER_PLACE','RETREAT')):
            quat_w = ns.get('_p4_transport_quat',quat_w).clone()
        if ns.get('_p4_evidence') and name.endswith(('MOVE_TO_TARGET','LOWER_PLACE')):
            # Position the actual instrument center over the chosen tray slot.
            ee,q,center,bottom,points = measured(env)
            target_w = target_w.clone()
            from phase4_tray_slots import RELEASE_GAP
            import numpy as np
            inverse=np.array(q); inverse[1:] *= -1
            local_points=rotate(inverse,points-ee)
            destination_q=quat_w.detach().cpu().numpy()
            dest_offset=rotate(destination_q,rotate(inverse,center-ee))
            target_w[:2] = active_target_w[:2]-target_w.new_tensor(dest_offset[:2])
            if name.endswith('LOWER_PLACE'):
                lowest=float(rotate(destination_q,local_points)[:,2].min())
                target_w[2] = ns['_p4_evidence'].tray_z+RELEASE_GAP-lowest
                # Reserve 4.5 mm for a <=2.5-mm pose residual, leaving >=2 mm
                # finger clearance; OPEN still independently requires 1.5 mm.
                target_w[2] = max(float(target_w[2]),ns['_p4_tray_rim_z']+finger_floor_height(env,torch)+.0045)
                # A millimeter-level IK residual is not failed placement.
                # Actual center/yaw/gap are independently gated before OPEN.
                # Reserve centering margin for the measured release/settle
                # displacement. A 2.8-mm release residual became 8.09 mm on
                # the tray; keep the final 8-mm QC gate, aim more accurately.
                dist_thresh = .0025
                angle_tol = 2.
        ns['_p4_stage_targets'][name] = target_w.clone()
        ns['_p4_stage_quats'][name] = quat_w.clone()
        start, q0 = ns['get_ee_pos_w'](env).clone(), ns['get_ee_quat_w'](env).clone()
        smooth_transfer = name.endswith('MOVE_TO_TARGET')
        long_arc=False
        via = None
        via_steps = 0
        if smooth_transfer:
            span = float(torch.linalg.norm(target_w-start))
            interp = max(interp, long_transfer_steps(span, float(env.step_dt)))
            _,turn = pose_error(env,start,quat_w)
            if hasattr(env,'action_manager'):
                from phase4_approach import select_long_arc,long_arc_slerp
                long_arc,turn=select_long_arc(env,start,target_w,q0,quat_w)
            interp = max(interp,math.ceil(1.5*turn/90./float(env.step_dt)))
            if interp > (180 if long_arc else 150):
                raise StageFailure(name, f'transfer_distance_outside_budget: {span:.3f}m')
            detour = transfer_detour(ns['to_base_pos'](env,start).detach().cpu().numpy(),
                                     ns['to_base_pos'](env,target_w).detach().cpu().numpy())
            if detour is not None:
                from isaaclab.utils.math import quat_apply
                robot = env.scene['robot']
                via = quat_apply(robot.data.root_quat_w[:1], start.new_tensor(detour)[None])[0] + robot.data.root_pos_w[0]
                via_steps = math.ceil(1.5 * float(torch.linalg.norm(via-start)) / .6 / float(env.step_dt))
                # Reserve either equivalent near-half-turn arc; the joint-space
                # risk must be reassessed after actually reaching the waypoint.
                turn_budget = max(turn, 360.-turn) if 150. <= turn <= 210. else turn
                destination_steps = max(math.ceil(1.5 * float(torch.linalg.norm(target_w-via)) / .6 / float(env.step_dt)),
                                        math.ceil(1.5 * turn_budget / 90. / float(env.step_dt)))
                interp = via_steps + destination_steps
                if interp > 300:
                    raise StageFailure(name, f'transfer_detour_outside_budget: {interp} steps')
                print(f'[{name}] BASE_CORRIDOR_DETOUR via_b={detour.tolist()} legs={via_steps}+{destination_steps}; retain grasp orientation on first leg',flush=True)
            max_steps = max(max_steps, interp+30)
            if ns.get('_p4_near_base_approach'):
                # Bounded IK can still be converging after the short wrist
                # arc's 84-step budget. Keep the same early pose/stall gates.
                max_steps = max(max_steps, 160)
                print(f'[{name}] BOUNDED_TRANSFER max={max_steps}; early pose/stall/slip gates retained', flush=True)
            print(f'[{name}] DISTANCE_AWARE_TRANSFER span={span:.3f}m interpolation={interp} steps, peak command speed<={transfer_peak_speed(span):.1f}m/s', flush=True)
        grace_steps = 120
        progress = dict(max_steps=max_steps, grace_steps=grace_steps, hard_max_steps=max_steps+grace_steps, interpolation_steps=interp,
                        position_tolerance_m=dist_thresh, angle_tolerance_deg=angle_tol,
                        position_error_tail_m=[], angle_error_tail_deg=[])
        ns.setdefault('_p4_stage_progress', {})[name] = progress
        stable, max_z, history = 0, -999., []
        angle_history = []
        evidence = ns.get('_p4_evidence')
        if evidence and name.endswith('LIFT_CLEAR'):
            ee,q,center,bottom,points = measured(env)
            evidence.begin_lift(ee,q,center,bottom)
        getter = ns['get_scalpel_pos_w'] if phase_id == 0 else other_get
        transfer = name.endswith(('MOVE_TO_TARGET', 'LOWER_PLACE'))
        def local_anchor():
            q = ns['get_ee_quat_w'](env)
            q = q / torch.linalg.norm(q)
            v = getter(env)-ns['get_ee_pos_w'](env)
            xyz = -q[1:]
            cross = 2*torch.linalg.cross(xyz, v)
            return v+q[0]*cross+torch.linalg.cross(xyz,cross)
        anchor = local_anchor().clone() if transfer else None
        slipped = 0
        for i in range(max_steps+grace_steps):
            if i >= max_steps:
                # The nominal time cap must not discard a clearly converging
                # far pose. Existing stall/slip gates and a hard cap still apply.
                if not motion_still_converging(history,angle_history,dist_thresh,angle_tol,stable):
                    break
                if i == max_steps:
                    print(f'[{name}] PROGRESS_EXTENSION nominal={max_steps} hard_cap={max_steps+grace_steps}; same pose gates', flush=True)
            t = min(1., (i+1)/max(interp, 1))
            blend = t*t*(3.-2.*t) if smooth_transfer else t
            if via is not None and i == via_steps:
                long_arc, _ = select_long_arc(env,via,target_w,q0,quat_w)
                progress['wrist_arc_reassessed_at_waypoint'] = True
                progress['waypoint_long_arc'] = long_arc
            command_pos = start+blend*(target_w-start)
            command_quat = long_arc_slerp(q0,quat_w,blend) if long_arc else ns['slerp'](q0, quat_w, blend)
            if via is not None:
                first_leg = i < via_steps
                leg_t = min(1., (i+1)/via_steps if first_leg else (i+1-via_steps)/(interp-via_steps))
                leg_blend = leg_t*leg_t*(3.-2.*leg_t)
                command_pos = start+leg_blend*(via-start) if first_leg else via+leg_blend*(target_w-via)
                command_quat = q0 if first_leg else (long_arc_slerp(q0,quat_w,leg_blend) if long_arc else ns['slerp'](q0,quat_w,leg_blend))
            max_z = max(max_z, sample(env, recorder, name, i,
                command_pos, command_quat, grip, active_target_w, phase_id))
            distance, angle = pose_error(env, target_w, quat_w)
            if evidence and (transfer or name.endswith(('LIFT_CLEAR','RETREAT'))):
                ee,q,center,bottom,points = measured(env)
                if name.endswith('LIFT_CLEAR'):
                    check_evidence(evidence.check_lift,name,ee,q,center,bottom,i)
                elif transfer:
                    check_evidence(evidence.check_held,name,ee,q,center,i,name)
                elif name.endswith('RETREAT'):
                    check_evidence(evidence.check_retreat,name,ee,center,points,bottom)
            elif transfer:
                shift = float(torch.linalg.norm(local_anchor()-anchor))
                slipped = slipped+1 if shift > .05 else 0
                if slipped >= 3:
                    raise StageFailure(name, f'object_not_following_gripper: relative_shift={shift:.4f}m')
            history.append(distance)
            history = history[-12:]
            angle_history.append(angle)
            angle_history = angle_history[-12:]
            progress.update(position_error_tail_m=list(history), angle_error_tail_deg=list(angle_history),
                            translation_gain_m=history[0]-history[-1],
                            rotation_gain_deg=angle_history[0]-angle_history[-1],
                            extension_eligible=motion_still_converging(history,angle_history,dist_thresh,angle_tol,stable))
            if i>=interp+20 and len(history)==12 and not name.endswith('RETREAT'):
                no_translation_gain=history[0]-history[-1]<.001
                no_rotation_gain=angle_history[0]-angle_history[-1]<.5
                if distance>max(2*dist_thresh,.015) and no_translation_gain and no_rotation_gain:
                    raise StageFailure(name,f'pose_stalled_or_diverging: error={distance:.4f}m angle={angle:.1f}deg; discard and reset')
            good = t == 1. and distance < dist_thresh and angle < angle_tol
            stable = stable+1 if good else 0
            if i == 0 or i % max(1, ns['args_cli'].debug_every) == 0:
                print(f'[{name}] i={i:04d} dist={distance:.4f} angle={angle:.1f} g={grip:.1f}', flush=True)
            if stable >= max(2, settle_steps):
                if evidence and name.endswith('LIFT_CLEAR'):
                    check_evidence(evidence.finish_lift,name)
                if evidence and name.endswith('RETREAT') and not evidence.place_verified:
                    continue
                mark_gate(recorder,name,i+1,env.step_dt)
                print(f'[{name}] POSE_REACHED steps={i+1}', flush=True)
                return max_z
            # Detect lack of progress only outside the valid position region.
            # This does NOT identify table contact; it aborts an invalid lower.
            if lower and i >= interp+12 and len(history) == 12 and distance >= dist_thresh and max(history)-min(history) < .0004:
                raise StageFailure(name, f'lower_stalled_outside_tolerance: error={distance:.4f}m angle={angle:.1f}deg')
        if evidence and name.endswith('RETREAT') and not evidence.place_verified:
            raise StageFailure(name,f'placement_unverified: {evidence.last_placement}')
        raise StageFailure(name, f'pose_timeout: error={distance:.4f}m angle={angle:.1f}deg; base_budget={max_steps}, progressing_grace<={grace_steps} steps')

    def gripper(env, recorder, name, target_w, quat_w, grip, slot, phase):
        closing = grip < 0
        evidence = ns.get('_p4_evidence')
        if not closing:
            quat_w = ns.get('_p4_transport_quat',quat_w).clone()
            if evidence:
                ee,q,center,bottom,points = measured(env)
                clearance=float(ee[2])-finger_floor_height(env,torch)+.0002-ns['_p4_tray_rim_z']
                if clearance < .0015:
                    raise StageFailure(name,f'finger_too_close_to_tray_lip: clearance={clearance:.4f}m')
                check_evidence(evidence.begin_release,name,ee,q,center,points,bottom)
        # Hold measured position: closing must not finish an unfinished descent.
        hold = ns['get_ee_pos_w'](env).clone()
        ns['_p4_stage_targets'][name] = hold.clone()
        ns['_p4_stage_quats'][name] = quat_w.clone()
        robot = env.scene['robot']
        ids = [i for i, joint in enumerate(robot.joint_names) if joint in ('panda_finger_joint1', 'panda_finger_joint2')]
        if len(ids) != 2:
            raise StageFailure(name, f'finger_joint_mapping_invalid: {robot.joint_names}')
        dt = float(env.step_dt)
        min_steps = math.ceil((.24 if closing else .16)/dt)
        max_steps = math.ceil((.9 if closing else .7)/dt)
        history, max_z = [], -999.
        for i in range(max_steps):
            max_z = max(max_z, sample(env, recorder, name, i, hold, quat_w,
                                     -1. if closing else 1., slot, phase))
            positions = [float(robot.data.joint_pos[0, j]) for j in ids]
            velocities = [float(robot.data.joint_vel[0, j]) for j in ids]
            history.append(positions)
            history = history[-5:]
            # Solver contact impulses can have high instantaneous velocity
            # despite a stationary finger. Gate on the measured position
            # window and its net velocity; log raw velocities for diagnosis.
            filtered_velocities = [(positions[j]-history[0][j])/max((len(history)-1)*dt,dt) for j in (0,1)]
            hold_error, hold_angle = pose_error(env, hold, quat_w)
            if (fingers_ready(history, closing, min_steps, i+1)
                    and all(abs(v) < .01 for v in filtered_velocities)
                    and hold_error < .008 and hold_angle < 10.):
                next_check = 'lift check still required' if closing else 'final placement check still required'
                print(f'[{name}] FINGERS_STABLE steps={i+1} positions={positions} velocities={velocities}; {next_check}', flush=True)
                mark_gate(recorder,name,i+1,dt)
                return max_z
        raise StageFailure(name, f'finger_or_hold_timeout: positions={positions}, raw_velocities={velocities}, filtered_velocities={filtered_velocities}, position_window={history}, hold_error={hold_error:.4f}m angle={hold_angle:.1f}deg, steps={max_steps}')

    @functools.wraps(original_smooth)
    def smooth(env, recorder, name, target_w, quat_w, grip_start, grip_end, tray_slot_w, phase_id, steps=40):
        if name.endswith(('_CLOSE', '_OPEN')):
            return gripper(env, recorder, name, target_w, quat_w, grip_end, tray_slot_w, phase_id)
        return original_smooth(env, recorder, name, target_w, quat_w, grip_start, grip_end, tray_slot_w, phase_id, steps)

    @functools.wraps(original_hold)
    def constant(env, recorder, name, target_w, quat_w, grip, tray_slot_w, phase_id, steps=80):
        if name.endswith(('_CLOSE', '_OPEN')):
            return gripper(env, recorder, name, target_w, quat_w, grip, tray_slot_w, phase_id)
        return original_hold(env, recorder, name, target_w, quat_w, grip, tray_slot_w, phase_id, steps)

    ns.update(step_dynamic=dynamic, step_smooth_grip=smooth, step_hold_const_grip=constant)
    original_pick = ns['pick_and_place_object']
    signature = inspect.signature(original_pick)

    @functools.wraps(original_pick)
    def pick(*args, **kwargs):
        ns.pop('_p4_failure_message', None)
        ns.pop('_p4_active_wrist_key',None)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        ns['_p4_control_dt_s'] = float(bound.arguments['env'].step_dt)
        bound.arguments['recorder']._p4_gates = {}
        from phase4_scene import LAYOUT
        from phase4_tray_slots import SlotEvidence, slot_spec
        evidence = SlotEvidence(bound.arguments['env'],ns,LAYOUT)
        slot=bound.arguments['tray_slot_target_w']
        slot[:2]=slot.new_tensor(slot_spec(ns['PHASE3_TARGET_OBJECT'],LAYOUT)['center_xy'])
        slot[2]=evidence.tray_z
        ns['_p4_evidence'] = evidence
        bound.arguments['recorder']._p4_evidence = evidence
        ns['_p4_transport_quat'] = bound.arguments.get('base_grip_quat',ns['get_ee_quat_w'](bound.arguments['env'])).clone()
        try:
            result = original_pick(*args, **kwargs)
            if result[0] and not (evidence.lift_verified and evidence.place_verified):
                raise StageFailure('FINAL_PLACE','missing_physical_success_evidence')
            if not result[0]:
                recorder = bound.arguments['recorder']
                names = getattr(recorder,'stage_names',[])
                last = names[-1] if names else 'GRASP_POSE'
                failure = StageFailure(last, 'physical_object_follow_check_failed' if names else 'grasp_pose_rejected')
                ns['_p4_failure_message'] = '[PHASE FAIL] ' + str(failure)
                print(ns['_p4_failure_message']+'; DISCARD before reset',flush=True)
                record_wrist_outcome(ns,False,last)
                failure_preview(recorder, last+'_physical_check_rejected')
                if hasattr(recorder,'_reset'): recorder._reset()
                reset_failed_scene(bound.arguments['env'])
            return result
        except StageFailure as exc:
            # Caller resets/discards the attempt; never save a partial failure.
            ns['_p4_failure_message'] = '[PHASE FAIL] ' + str(exc)
            print(ns['_p4_failure_message']+'; DISCARD before reset',flush=True)
            record_wrist_outcome(ns,False,exc.stage)
            failure_preview(bound.arguments['recorder'], exc.stage)
            pos = bound.arguments['get_pos_fn'](bound.arguments['env'])
            # Reset immediately, including while a manual session waits for
            # its next Prepare click. No failed samples survive into next try.
            bound.arguments['recorder']._reset() if hasattr(bound.arguments['recorder'],'_reset') else None
            reset_failed_scene(bound.arguments['env'])
            return False, float(pos[2]), float(pos[2]), pos

    def reset_failed_scene(env):
        if ns.get('_p4_spawn_templates'):
            from phase4_reset import reset_robot,restore_objects
            reset_robot(env)
            restore_objects(env,ns,ns['_p4_force_args'][1])
        elif hasattr(env,'reset'):
            env.reset()

    def failure_preview(recorder, stage):
        # Diagnostic PNGs only, never failed trajectories in policy H5 folders.
        if not hasattr(ns['args_cli'], 'out_dir'):
            return
        try:
            from PIL import Image
            output = Path(ns['args_cli'].out_dir)/'failure_previews'/f"attempt_{ns.get('_p4_attempt_number',0):04d}_{stage}"
            images = {'cam_front':getattr(recorder,'front_rgb',[]),
                      'grip_cam_b':getattr(recorder,'grip_b_rgb',[])}
            images.update(getattr(recorder,'extra_camera_rgb',{}))
            for name, frames in images.items():
                if len(frames):
                    output.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(frames[-1]).save(output/f'{name}.png')
            evidence=getattr(recorder,'_p4_evidence',None)
            if evidence:
                output.mkdir(parents=True,exist_ok=True)
                (output/'evidence.json').write_text(json.dumps(evidence.summary(),indent=2),encoding='utf-8')
            print('[P4 FAILURE PREVIEW]',output,'(last recorded frame, not training data)',flush=True)
        except Exception as exc:
            print('[P4 FAILURE PREVIEW ERROR]',repr(exc),flush=True)

    ns['pick_and_place_object'] = pick

    def quality_ok(recorder, pfx):
        from collections import Counter
        counts = Counter(recorder.stage_names)
        gates = getattr(recorder, '_p4_gates', {})
        required = {'LOWER_GRASP':5, 'LIFT_CLEAR':12, 'MOVE_TO_TARGET':12,
                    'CLOSE':1, 'OPEN':1}
        ok = True
        evidence = getattr(recorder,'_p4_evidence',None)
        if evidence is None or not (evidence.lift_verified and evidence.place_verified):
            print('[QUALITY FAIL] missing measured lift/place evidence',flush=True)
            return False
        try:
            validate_recorded_rgb(recorder,int(getattr(ns.get('args_cli'),'center_crop_size',224) or 0))
        except ValueError as error:
            print(f'[SENSOR FAIL] {error}; episode discarded before H5 save',flush=True)
            return False
        for suffix, minimum in required.items():
            name = pfx+suffix
            gate = gates.get(name)
            if gate and suffix in ('CLOSE','OPEN'):
                minimum = math.ceil((.24 if suffix == 'CLOSE' else .16)/gate['dt'])
            if counts[name] < minimum or not gate or gate['steps'] != counts[name]:
                print(f'[QUALITY] {name}: samples={counts[name]}, minimum={minimum}, measured_gate={gate} FAIL', flush=True)
                ok = False
        return ok
    ns['quality_ok'] = quality_ok

    if 'save_realcompat_segment' in ns:
        from phase4_storage import EpisodeTransaction
        transactions = {}
        original_save = ns['save_realcompat_segment']
        save_signature = inspect.signature(original_save)
        @functools.wraps(original_save)
        def save_segment(*args, **kwargs):
            bound = save_signature.bind(*args,**kwargs)
            bound.apply_defaults()
            evidence = getattr(bound.arguments['recorder'],'_p4_evidence',None)
            if not bound.arguments['success'] or evidence is None or not (evidence.lift_verified and evidence.place_verified):
                raise RuntimeError('SAVE_REJECTED: verified lift AND final placement required')
            checked=validate_recorded_rgb(bound.arguments['recorder'],int(getattr(ns.get('args_cli'),'center_crop_size',224) or 0))
            destination = Path(bound.arguments['out_dir'])/f"episode_{bound.arguments['ep_idx']:06d}.h5"
            if destination.exists():
                raise FileExistsError(f'Refusing to overwrite {destination}; resume or choose a fresh output directory')
            meta = dict(bound.arguments.get('meta') or {})
            meta.update(rgb_frame_qc='all_frames_spatial_v1',rgb_qc_episode_frames=checked)
            meta.update(policy_window_contract=POLICY_WINDOW_CONTRACT,
                        policy_window_start='LOWER_PRE' if bound.arguments['segment']=='pick' else 'LOWER_PLACE',
                        policy_window_end='LIFT_CLEAR' if bound.arguments['segment']=='pick' else 'RETREAT',
                        excluded_policy_stages='OPEN_HOVER,MOVE_TO_TARGET')
            meta.update(approach_controller='tcp_frame_correct_bounded_place_integral_v6',
                        precision_feedback_scope='place_skill_robot_tcp_only',
                        near_base_approach=bool(ns.get('_p4_near_base_approach')),
                        tcp_jacobian_contract='world_COM_to_base_observed_TCP_v1')
            meta.update(motion_timing_contract='far_side_waypoint_arc_v11',
                        observation_action_alignment='observation_before_env_step_action',
                        step_ids_scope='stage_local_index_use_dataset_row_for_sequence')
            if '_p4_control_dt_s' in ns:
                meta['control_dt_s'] = ns['_p4_control_dt_s']
            meta.update(p4_feedback_version=FEEDBACK_VERSION, expert_grasp_evidence=json.dumps(evidence.summary()), control_tcp_link=CONTROL_TCP['link'],
                        control_tcp_offset_m=json.dumps(CONTROL_TCP['pos']),
                        control_tcp_offset_quat_wxyz=json.dumps(CONTROL_TCP['rot']),
                        control_tcp_matches_observed_ee=True)
            from phase4_tray_slots import ORDER, VERSION
            other_objects = [n for n in ORDER if n != ns['PHASE3_TARGET_OBJECT']]
            meta.update(distractor_objects=','.join(other_objects),
                        distractor_note='Four base non-targets plus randomized duplicate instances; see scene_clutter',
                        distractor_policy_note='Extra duplicate non-targets are table-only; base tray objects use fixed slots; target type appears once')
            meta.update(tray_contract=VERSION,tray_slot_id=evidence.spec['slot_id'],
                        tray_slot_order=json.dumps(ORDER),
                        initial_tray_occupancy=json.dumps([int(n in ns.get('_p4_tray_objects',[])) for n in ORDER]),
                        tray_slot_target=json.dumps(evidence.spec),tray_support_z_m=evidence.tray_z)
            bound.arguments['meta'] = meta
            key = (str(Path(bound.arguments['out_dir']).resolve().parents[1]),
                   bound.arguments['object_name'], bound.arguments['ep_idx'])
            if key not in transactions:
                mode = getattr(ns.get('args_cli'), 'record_mode', 'both')
                transactions[key] = EpisodeTransaction(*key, mode)
            result = transactions[key].save(original_save, bound)
            if bound.arguments['segment'] == 'place':
                transactions.pop(key, None)
            return result
        ns['save_realcompat_segment'] = save_segment
    if 'get_existing_demo_count' in ns:
        ns['get_existing_demo_count'] = lambda out_dir: completed_episode_count(
            out_dir, ns['PHASE3_TARGET_OBJECT'], ns['phase3_get_record_mode']())

    def attempt_guard(attempt, target_success, saved_count):
        limit = int(os.environ.get('P4_MAX_ATTEMPTS', 0))
        if limit > 0 and attempt >= limit:
            raise RuntimeError(f'ATTEMPT_BUDGET_EXHAUSTED: {saved_count}/{target_success} successes in {attempt}/{limit} attempts')
        ns['_p4_attempt_number'] = attempt+1
        from phase4_session import before_attempt
        before_attempt(ns,attempt+1,saved_count,target_success)
    ns['_p4_attempt_guard'] = attempt_guard
    print('[P4 FEEDBACK] binary close/open + measured two-finger gate; pose timeout aborts; object-specific grasp geometry retained', flush=True)
