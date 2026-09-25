"""File protocol between the recorder and its separate pre-run control panel."""
import json
import math
import os
import time
from pathlib import Path


def load_session():
    path = os.environ.get('P4_SESSION_CONFIG')
    return json.loads(Path(path).read_text(encoding='utf-8')) if path else None


def status(ns, state, **values):
    path = os.environ.get('P4_SESSION_CONFIG')
    if not path:
        return
    out = Path(path).with_suffix('.status.json')
    values.setdefault('saved',ns.get('_p4_saved_count',0))
    values.setdefault('requested',ns.get('_p4_requested_count',0))
    payload = dict(state=state,object=ns['PHASE3_TARGET_OBJECT'],attempt=ns.get('_p4_attempt_number',0),
                   tray_objects=ns.get('_p4_tray_objects',[]),coverage=ns.get('_p4_coverage'),**values)
    payload['instrument_geometry']=ns.get('_p4_panel_geometry')
    payload['scene_clutter']=ns.get('_p4_clutter',{})
    spawn=ns.get('_p4_force_args',(None,{},None))[1]
    payload['table_positions']={name:dict(x=p.get('center_x',p.get('x')),y=p.get('center_y',p.get('y')),yaw_deg=p['yaw_deg'])
                               for name,p in spawn.items() if name!='grid' and 'yaw_deg' in p}
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    os.replace(tmp,out)


def before_attempt(ns, attempt, saved, requested):
    cfg = load_session()
    if not cfg:
        return
    if cfg.get('stop'):
        raise RuntimeError('Session stopped before next attempt')
    if cfg['target'] != ns['PHASE3_TARGET_OBJECT']:
        raise RuntimeError('Session target changed during run; finish and relaunch to switch object handler')
    if cfg.get('collection')=='grid_cycles':
        from phase4_scene import LAYOUT
        from phase4_coverage import progress
        ns['_p4_coverage']=progress(saved,int(cfg['cycles']),LAYOUT['grid_rows'],LAYOUT['grid_cols'])
        if requested!=ns['_p4_coverage']['goal']: raise ValueError('Episode goal and coverage cycles disagree')
        if cfg['mode']!='auto': raise ValueError('Grid cycles require automatic bounded-cell spawning')
        c=ns['_p4_coverage']
        print(f"[P4 COVERAGE] round {c['cycle']}/{c['cycles']} | cell {c['cell_id']} | saved {saved}/{requested} | failed attempts never advance cell",flush=True)
    else: ns.pop('_p4_coverage',None)
    if cfg['mode'] == 'manual':
        previous = ns.get('_p4_prepare_id',0)
        while not cfg.get('auto_start',False) and int(cfg.get('prepare_id',0)) <= previous:
            status(ns,'waiting_prepare',saved=saved,requested=requested)
            if cfg.get('stop') or not ns['simulation_app'].is_running():
                raise RuntimeError('Session stopped before next attempt')
            ns['simulation_app'].update()
            time.sleep(.02)
            cfg = load_session()
        ns['_p4_prepare_id'] = int(cfg['prepare_id'])
        ns['_p4_manual_positions'] = validate_positions(cfg['positions'])
        validate_target_workspace(cfg['target'],ns['_p4_manual_positions'])
    else:
        ns.pop('_p4_manual_positions',None)
    ns['_p4_saved_count']=saved
    ns['_p4_requested_count']=requested
    ns['_p4_discard_id'] = int(cfg.get('discard_id',0))
    status(ns,'preparing_spawn',saved=saved,requested=requested)


def wait_for_start(env, ns):
    cfg = load_session()
    if not cfg or cfg.get('auto_start',cfg['mode'] != 'manual'):
        status(ns,'recording',saved=ns.get('_p4_saved_count',0),requested=ns.get('_p4_requested_count',0))
        return True
    previous = ns.get('_p4_start_id',0)
    status(ns,'ready_to_record',message='Inspect settled instruments, then click Start recording')
    print('[P4 MANUAL READY] settled scene; waiting for Start recording button',flush=True)
    while int(cfg.get('start_id',0)) <= previous:
        if cfg.get('auto_start',False):
            break
        if cfg.get('stop') or not ns['simulation_app'].is_running():
            raise RuntimeError('Session stopped before recording')
        env.sim.render()
        ns['simulation_app'].update()
        time.sleep(.02)
        cfg = load_session()
        if cfg['mode']=='manual' and int(cfg.get('prepare_id',0)) > ns.get('_p4_prepare_id',0):
            positions=validate_positions(cfg['positions'])
            validate_target_workspace(cfg['target'],positions)
            ns['_p4_prepare_id']=int(cfg['prepare_id'])
            ns['_p4_manual_positions']=positions
            _,spawn,pose_mode=ns['_p4_force_args']
            import numpy as np
            from phase4_tray_slots import select_occupancy
            ns['_p4_tray_objects']=select_occupancy(ns['PHASE3_TARGET_OBJECT'],np.random.default_rng(),cfg)
            spawn['grid']['initial_tray_objects']=json.dumps(ns['_p4_tray_objects'])
            apply_manual_positions(spawn,ns['PHASE3_TARGET_OBJECT'],positions)
            ns['force_episode_objects'](env,spawn,pose_mode)
            fresh=ns['_p4_capture_before'](env,ns['phase3_scene_key_for_object'])
            old=ns.get('_p4_before_reference')
            if old is not None:
                old.clear(); old.update(fresh)
            status(ns,'preparing_spawn',message='Applying updated positions and settling')
            return False
    ns['_p4_start_id'] = int(cfg['start_id'])
    ns['_p4_discard_id'] = int(cfg.get('discard_id',0))
    status(ns,'recording')
    return True


def check_interrupt(ns):
    if not os.environ.get('P4_SESSION_CONFIG'):
        return False
    cfg = load_session()
    if cfg.get('stop'):
        raise RuntimeError('Session stopped by operator; unfinished attempt not saved')
    return int(cfg.get('discard_id',0)) > ns.get('_p4_discard_id',0)


def validate_positions(positions, layout=None, radii=None):
    if layout is None:
        from phase4_scene import LAYOUT
        layout = LAYOUT
    if radii is None:
        from phase4_cell_spawn import instrument_envelopes
        radii = {n:v['radius_m'] for n,v in instrument_envelopes().items()}
    if set(positions) != set(radii):
        raise ValueError('Specify exactly all five instruments')
    clean = {}
    for name,radius in radii.items():
        x,y,yaw = [float(positions[name][key]) for key in ('x','y','yaw_deg')]
        if not all(math.isfinite(v) for v in (x,y,yaw)):
            raise ValueError(f'{name}: nonfinite pose')
        for value,bounds in ((x,layout['grid_x']),(y,layout['grid_y'])):
            if not bounds[0]+radius+.004 <= value <= bounds[1]-radius-.004:
                raise ValueError(f'{name}: instrument footprint crosses the spawn-area edge')
        for other,pose in clean.items():
            if math.hypot(x-pose['x'],y-pose['y']) < radius+radii[other]+.005:
                raise ValueError(f'{name} overlaps {other}; separate their footprint circles')
        clean[name] = dict(x=x,y=y,yaw_deg=yaw%360)
    return clean


def target_workspace_ok(x,y,layout):
    # Conservative expert down-grasp envelope, NOT a full IK certificate.
    # The user's failed pose was 0.254 m from the base; descent diverged.
    rx,ry,_=layout['robot_pos']
    return math.hypot(x-rx,y-ry)>=.32


def validate_target_workspace(target,positions,layout=None):
    if layout is None:
        from phase4_scene import LAYOUT
        layout=LAYOUT
    p=positions[target]
    if not target_workspace_ok(p['x'],p['y'],layout):
        raise ValueError(f'{target}: too close to robot base for the validated downward grasp. Move target outside the marked 0.32 m exclusion zone; positions are NOT changed automatically.')


def apply_manual_positions(spawn, target, positions):
    for name,p in positions.items():
        item = spawn[name]
        item['center_x' if name=='scalpel' else 'x'] = p['x']
        item['center_y' if name=='scalpel' else 'y'] = p['y']
        item['yaw_deg'] = p['yaw_deg']
        item.update(cell_id=-1,row=-1,col=-1)
    spawn['grid'].update(cell_id=-1,row=-1,col=-1,x=positions[target]['x'],y=positions[target]['y'],
                         target_yaw_deg=positions[target]['yaw_deg'],spawn_contract='manual_bounded_v1',
                         object_cell_assignments='{}',manual_positions=json.dumps(positions))
    return spawn
