"""P4 stage telemetry, sample cleanup and shared feedback-gate installation.

Object-specific grasp construction and H5 IDs remain in the five backends.
Stage completion is not a substitute for physical lift/placement validation.
"""
import functools
import inspect
import json
import math
import time
from pathlib import Path


def grip_schedule(start, end, steps, remove_idle):
    steps = int(steps)
    if steps < 2:
        raise ValueError('Gripper ramp requires at least two steps')
    if abs(start-end) < 0.01:
        return [start]*steps
    idle = steps//4
    # Exact tail of the original ramp, including its first/start command.
    indices = range(idle, steps) if remove_idle else range(steps)
    return [start+min(1., max(0., (i-idle)/max(steps-idle-1,1)))*(end-start)
            for i in indices]


class CountRecorder:
    def __init__(self, original):
        self.original = original
        self.count = 0
        self.wall_s = 0.0

    def add_step(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            result = self.original.add_step(*args, **kwargs)
        finally:
            self.wall_s += time.perf_counter()-started
        self.count += 1
        return result

    def __getattr__(self, name):
        return getattr(self.original, name)


def install_fsm(ns):
    if ns.get('_p4_fsm_installed'):
        return
    from phase4_scene import LAYOUT
    cfg = LAYOUT.get('fsm', {})
    # Legacy add_step traverses the entire hospital stage repeatedly while
    # collecting ONE sample. No prims are created between these redundant
    # calls. Keep one full cleanup per sample, plus every call outside samples.
    if 'EpisodeRecorder' in ns and 'phase3_hide_all_debug_visuals' in ns:
        original_hide = ns['phase3_hide_all_debug_visuals']
        original_add = ns['EpisodeRecorder'].add_step
        cleanup_state = {'in_sample':False,'done':False}

        @functools.wraps(original_hide)
        def hide_once(stage, verbose=False):
            if cleanup_state['in_sample'] and cleanup_state['done'] and not verbose:
                return
            result = original_hide(stage,verbose=verbose)
            if cleanup_state['in_sample']:
                cleanup_state['done'] = True
            return result

        @functools.wraps(original_add)
        def add_sample(*args, **kwargs):
            cleanup_state.update(in_sample=True,done=False)
            try:
                return original_add(*args,**kwargs)
            finally:
                cleanup_state.update(in_sample=False,done=False)

        ns['phase3_hide_all_debug_visuals'] = hide_once
        ns['EpisodeRecorder'].add_step = add_sample
    original_smooth = ns['step_smooth_grip']
    # Each backend has its own secondary getter. Preserve that mapping.
    other_names = ('get_scissor_pos_w','get_love_retractor_pos_w',
                   'get_kelly_pos_w','get_scalpel_type2_pos_w')
    used = [name for name in other_names if name in original_smooth.__code__.co_names]
    if len(used) != 1:
        raise RuntimeError(f'Cannot identify backend active object getter: {used}')
    other_get = ns[used[0]]

    @functools.wraps(original_smooth)
    def smooth(env, recorder, name, target_w, quat_w, grip_start, grip_end,
               tray_slot_w, phase_id, steps=40):
        if name.endswith('_OPEN'):
            steps = min(steps, ns['MOTION_TIMING'].open_steps)
        commands = grip_schedule(grip_start, grip_end, steps, True)
        max_z = -999.
        for i, grip in enumerate(commands):
            state = ns['make_state'](env, tray_slot_w, phase_id)
            action = ns['make_abs_action'](env, target_w, quat_w, grip)
            recorder.add_step(env, name, i, state, action)
            env.step(action)
            obj = ns['get_scalpel_pos_w'](env) if phase_id == 0 else other_get(env)
            max_z = max(max_z, float(obj[2]))
            if i == 0 or i % ns['args_cli'].debug_every == 0:
                print(f'[{name}] i={i:03d} grip={grip:.3f} obj_z={float(obj[2]):.4f}', flush=True)
        return max_z

    if cfg.get('remove_smooth_grip_initial_idle', False):
        ns['step_smooth_grip'] = smooth

    from phase4_feedback import install_feedback
    install_feedback(ns, other_get)

    def instrument(original):
        signature = inspect.signature(original)

        @functools.wraps(original)
        def run(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            env = bound.arguments['env']
            counter = CountRecorder(bound.arguments['recorder'])
            bound.arguments['recorder'] = counter
            started = time.perf_counter()
            error = None
            print('[P4 ACTIVE STAGE]',bound.arguments['name'],flush=True)
            try:
                return original(*bound.args, **bound.kwargs)
            except BaseException as exc:
                error = type(exc).__name__ + ': ' + str(exc)
                raise
            finally:
                # Diagnostics must never hide the original exception/result.
                try:
                    requested_target = bound.arguments['target_w']
                    target = ns.get('_p4_stage_targets',{}).get(bound.arguments['name'],requested_target)
                    ee = ns['get_ee_pos_w'](env)
                    dist = float(ns['torch'].linalg.norm((ee-target)[:3]).item())
                    requested_dist = float(ns['torch'].linalg.norm((ee-requested_target)[:3]).item())
                    q = ns['get_ee_quat_w'](env)
                    qt = ns.get('_p4_stage_quats',{}).get(bound.arguments['name'],bound.arguments['quat_w'])
                    dot = float((q*qt).sum().item())
                    denom = float((ns['torch'].linalg.norm(q)*ns['torch'].linalg.norm(qt)).item())
                    angle = math.degrees(2*math.acos(min(1.,abs(dot)/max(denom,1e-12))))
                    dt = float(env.step_dt)
                    tolerance = float(bound.arguments.get('dist_thresh', .015))
                    if bound.arguments['name'].endswith(('LOWER_GRASP','LOWER_EXTRA')):
                        tolerance = min(tolerance,.002)
                    if bound.arguments['name'].endswith('LOWER_PLACE'):
                        tolerance = .0025
                    if bound.arguments['name'].endswith(('_CLOSE','_OPEN')):
                        tolerance = .008
                    progress = ns.get('_p4_stage_progress', {}).get(bound.arguments['name'])
                    if progress:
                        tolerance = progress['position_tolerance_m']
                    row = {'stage':bound.arguments['name'], 'steps':counter.count,
                           'attempt':ns.get('_p4_attempt_number'),
                           'control_dt_s':dt, 'simulation_s':counter.count*dt,
                           'wall_s':time.perf_counter()-started, 'ee_error_m':dist,
                           'legacy_waypoint_error_m':requested_dist,
                           'effective_target_w':target.detach().cpu().tolist(),
                           'effective_target_quaternion_wxyz':qt.detach().cpu().tolist(),
                           'observation_collection_wall_s':counter.wall_s,
                           'orientation_error_deg':angle, 'exception':error,
                           'required_position_tolerance_m':tolerance,
                           'position_within_requested_tolerance':dist < tolerance,
                           'outcome':'FAILED' if error else 'STAGE_COMPLETE',
                           'note':'stage completion is not episode success; physical lift/place checks remain required'}
                    if progress:
                        row['motion_progress'] = dict(progress)
                    row['actual_ee_position_w'] = ee.detach().cpu().tolist()
                    robot = env.scene['robot']
                    finger_ids = [j for j,n in enumerate(robot.joint_names) if n in ('panda_finger_joint1','panda_finger_joint2')]
                    row['finger_positions_m'] = [float(robot.data.joint_pos[0,j]) for j in finger_ids]
                    row['finger_velocities_m_s'] = [float(robot.data.joint_vel[0,j]) for j in finger_ids]
                    arm_ids=[j for j,n in enumerate(robot.joint_names) if n.startswith('panda_joint')]
                    row['arm_joint_positions_rad']=[float(robot.data.joint_pos[0,j]) for j in arm_ids]
                    if hasattr(robot.data,'soft_joint_pos_limits'):
                        row['arm_joint_limits_rad']=robot.data.soft_joint_pos_limits[0,arm_ids].detach().cpu().tolist()
                        margins=[min(q-lo,hi-q) for q,(lo,hi) in zip(row['arm_joint_positions_rad'],row['arm_joint_limits_rad'])]
                        nearest=min(range(len(margins)),key=margins.__getitem__)
                        row['nearest_joint_limit']={'joint':robot.joint_names[arm_ids[nearest]],'margin_rad':margins[nearest]}
                    for attr,key in (('joint_pos_target','arm_joint_targets_rad'),('joint_vel','arm_joint_velocities_rad_s'),('applied_torque','arm_applied_torques_nm')):
                        if hasattr(robot.data,attr):
                            row[key]=getattr(robot.data,attr)[0,arm_ids].detach().cpu().tolist()
                    row['near_base_bounded_ik']=bool(ns.get('_p4_near_base_approach'))
                    row['bounded_ik_active']=bool(ns.get('_p4_bounded_ik'))
                    output = Path(ns['args_cli'].out_dir)
                    output.mkdir(parents=True, exist_ok=True)
                    with (output/'stage_metrics.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(row)+'\n')
                    print('\n'+'-'*62,flush=True)
                    print(f"[P4 STAGE] {row['stage']}\n  Steps {counter.count} | sim {row['simulation_s']:.3f}s | wall {row['wall_s']:.2f}s\n  Observation collection {counter.wall_s:.2f}s\n  EE error {dist*1000:.1f}mm | angle {angle:.1f}deg\n  Status {row['outcome']}\n  Reason {error or 'stage gate passed; not episode success'}",flush=True)
                    print('-'*62,flush=True)
                    if error and row.get('nearest_joint_limit'):
                        limit=row['nearest_joint_limit']
                        print(f"[P4 MOTION DIAGNOSTIC] nearest limit={limit['joint']} margin={math.degrees(limit['margin_rad']):.2f}deg | near-base bounded IK={row['near_base_bounded_ik']} | joint targets/velocities/torques in stage_metrics.jsonl; not a contact diagnosis",flush=True)
                except Exception as diagnostic_error:
                    print('[P4 STAGE METRICS ERROR]',diagnostic_error,flush=True)
        return run

    if cfg.get('stage_telemetry', True):
        for name in ('step_dynamic','step_smooth_grip','step_hold_const_grip'):
            ns[name] = instrument(ns[name])
    ns['_p4_fsm_installed'] = True
    print('[P4 FSM] shared feedback gates active for every backend; no micro-lift/extra hold stage added',flush=True)
