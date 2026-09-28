"""Shared P4 initial placement: one instrument per cell, bounded random jitter.

Does not change instrument orientation construction, physics or grasp handlers.
The radius encloses the asset around the origin used by each spawn function,
so arbitrary yaw and the existing scalpel pose modes remain safe initially.
"""
import itertools
import json
import math
from functools import lru_cache


@lru_cache(maxsize=1)
def instrument_envelopes():
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    import numpy as np
    from phase4_grasp_validation import mesh_points_in_frame
    from phase3_shared_env_cfg import INSTRUMENTS, ASSET_DIR
    result = {}
    for name, spec in INSTRUMENTS.items():
        stage = Usd.Stage.Open(str(ASSET_DIR/spec.usd))
        bodies=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(bodies)!=1: raise RuntimeError(f'{name}: expected one rigid body')
        prim=bodies[0]
        scale=np.abs(np.asarray(Gf.Transform(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).GetScale()))*np.asarray(spec.scale)
        points=mesh_points_in_frame(prim)*scale
        lo=points.min(axis=0).tolist(); hi=points.max(axis=0).tolist()
        origin = (0.,0.0656,0.) if name == 'scalpel' else (0.,0.,0.)
        radius = float(np.linalg.norm(points-np.asarray(origin),axis=1).max())
        result[name] = {'size_m':[hi[i]-lo[i] for i in range(3)], 'radius_m':radius,
                        'local_min':lo,'local_max':hi}
    return result


def fit_spawn_to_cells(spawn, target, rng, target_cell=None):
    from phase4_scene import LAYOUT
    envelopes = instrument_envelopes()
    names = list(envelopes)
    nx, ny = LAYOUT['grid_cols'], LAYOUT['grid_rows']
    if nx*ny < len(names):
        raise ValueError('Need at least one distinct cell per instrument')
    x0,x1 = LAYOUT['grid_x']; y0,y1 = LAYOUT['grid_y']
    width,height = (x1-x0)/nx, (y1-y0)/ny
    margin = LAYOUT.get('cell_object_margin',0.01)
    if target_cell is None:
        cells = rng.choice(nx*ny, size=len(names), replace=False)
    else:
        if not 0<=target_cell<nx*ny: raise ValueError('Invalid coverage cell')
        others=iter(rng.choice([i for i in range(nx*ny) if i!=target_cell],size=len(names)-1,replace=False))
        cells=[target_cell if n==target else next(others) for n in names]
    assignments = {}
    for name, cid in zip(names,cells):
        cid = int(cid); row,col = divmod(cid,nx)
        radius = envelopes[name]['radius_m']
        jx,jy = width/2-radius-margin, height/2-radius-margin
        if min(jx,jy) < 0:
            raise ValueError(f'{name} needs cell side >= {2*(radius+margin):.4f}m')
        x = x0+(col+0.5)*width+float(rng.uniform(-jx,jx))
        y = y0+(row+0.5)*height+float(rng.uniform(-jy,jy))
        data = spawn[name]
        data['center_x' if name=='scalpel' else 'x'] = x
        data['center_y' if name=='scalpel' else 'y'] = y
        data.update(cell_id=cid, row=row, col=col)
        # Retain the existing randomized yaw; never force aligned orientations.
        assignments[name] = dict(cell_id=cid,row=row,col=col,x=x,y=y,radius_m=radius)
    chosen = assignments[target]
    spawn['grid'].update({k:chosen[k] for k in ('cell_id','row','col','x','y')})
    spawn['grid'].update(cell_width_m=width, cell_height_m=height,
                         object_cell_assignments=json.dumps(assignments,sort_keys=True),
                         spawn_contract='one_object_per_cell_radius_bounded_v1')
    return spawn


def install_cell_spawn(namespace):
    from phase4_scene import LAYOUT
    if not LAYOUT.get('fit_objects_to_cells') or namespace.get('_p4_cell_spawn_installed'):
        return
    original = namespace['phase3_ensure_all_5_objects_in_spawn']
    def complete_and_fit(spawn, target, rng):
        result = original(spawn,target,rng)
        manual = namespace.get('_p4_manual_positions')
        if manual:
            from phase4_session import apply_manual_positions
            result = apply_manual_positions(result,target,manual)
        else:
            from phase4_session import target_workspace_ok
            coverage=namespace.get('_p4_coverage')
            if coverage:
                result=fit_spawn_to_cells(result,target,rng,coverage['cell_id'])
                result['grid'].update(coverage_cycle=coverage['cycle'],coverage_cell_id=coverage['cell_id'],coverage_total_cycles=coverage['cycles'])
            else:
                for _ in range(100):
                    result = fit_spawn_to_cells(result,target,rng)
                    if target_workspace_ok(result['grid']['x'],result['grid']['y'],LAYOUT): break
                else: raise ValueError('No target cell in conservative downward-grasp workspace')
        namespace['_p4_cell_assignments'] = json.loads(result['grid']['object_cell_assignments'])
        from phase4_tray_slots import select_occupancy, VERSION, ORDER
        from phase4_session import load_session
        namespace['_p4_tray_objects'] = select_occupancy(target,rng,load_session())
        result['grid'].update(tray_contract=VERSION,tray_slot_order=json.dumps(ORDER),
                              initial_tray_objects=json.dumps(namespace['_p4_tray_objects']))
        print('[P4 CELL SPAWN]', result['grid']['object_cell_assignments'], flush=True)
        return result
    namespace['phase3_ensure_all_5_objects_in_spawn'] = complete_and_fit
    original_metadata = namespace['build_settle_metadata']
    def cell_metadata(spawn, before, report, drift):
        meta=original_metadata(spawn,before,report,drift)
        meta['domain_randomization'] = json.dumps(namespace.get('_p4_domain_randomization', {}), sort_keys=True)
        meta['scene_clutter'] = json.dumps(namespace.get('_p4_clutter', {}), sort_keys=True)
        meta['scene_label_audit'] = json.dumps(namespace.get('_p4_scene_label_audit', {}), sort_keys=True)
        import os
        meta['dataset_purpose'] = os.environ.get('P4_DATASET_PURPOSE','detection')
        meta['dataset_split'] = os.environ.get('P4_DATASET_SPLIT','unassigned')
        meta.update(cell_recenter_passes=int(report.get('cell_recenter_passes',0)),
                    cell_fit_scope='five_base_bodies; extra_instances_use_scene_clutter_footprints',
                    cell_fit_rule='settled_rotated_asset_bbox_in_assigned_cell',
                    cell_assignments_requested=spawn.get('grid',{}).get('object_cell_assignments','{}'))
        meta.update(spawn_control_mode='manual' if namespace.get('_p4_manual_positions') else 'auto',
                    manual_positions_requested=spawn.get('grid',{}).get('manual_positions','{}'))
        meta.update(tray_contract=spawn['grid']['tray_contract'],tray_slot_order=spawn['grid']['tray_slot_order'],
                    initial_tray_objects=spawn['grid']['initial_tray_objects'],
                    initial_tray_geometry=json.dumps(namespace.get('_p4_tray_initial',{})),
                    tray_floor_z=json.dumps(namespace.get('_p4_tray_floors',{})))
        if namespace.get('_p4_coverage'):
            meta.update(coverage_contract='successful_cell_cycles_v1',coverage_json=json.dumps(namespace['_p4_coverage']),
                        coverage_cycle=spawn['grid']['coverage_cycle'],coverage_cell_id=spawn['grid']['coverage_cell_id'])
        requested=json.loads(meta['spawn_requested_all'])
        for name,item in requested.items():
            if name in namespace.get('_p4_tray_requested',{}):
                requested[name]=dict(namespace['_p4_tray_requested'][name],table_candidate_pose=item)
            else:
                item['region']='table'
        meta['spawn_requested_all']=json.dumps(requested,sort_keys=True)
        candidates=json.loads(meta['cell_assignments_requested'])
        meta['grid_candidates_all']=json.dumps(candidates,sort_keys=True)
        meta['cell_assignments_requested']=json.dumps({n:v for n,v in candidates.items() if n not in namespace.get('_p4_tray_objects',[])},sort_keys=True)
        return meta
    namespace['build_settle_metadata'] = cell_metadata
    namespace['_p4_cell_spawn_installed'] = True


def settled_cell_failures(env, namespace, recenter=False):
    from phase4_scene import LAYOUT
    from phase4_grasp_validation import rigid_world_points
    assignments=namespace.get('_p4_cell_assignments',{})
    manual=namespace.get('_p4_manual_positions',{})
    if manual:
        assignments={name:dict(cell_id=-1,col=0,row=0,**pose) for name,pose in manual.items()}
    gx,gy=LAYOUT['grid_x'],LAYOUT['grid_y']
    w=(gx[1]-gx[0])/LAYOUT['grid_cols']; h=(gy[1]-gy[0])/LAYOUT['grid_rows']
    failures=[]
    for name,a in assignments.items():
        if name in namespace.get('_p4_tray_objects',[]):
            continue
        obj=env.scene[namespace['phase3_scene_key_for_object'](name)]
        points=rigid_world_points(obj)
        low=[min(p[i] for p in points) for i in range(2)]
        high=[max(p[i] for p in points) for i in range(2)]
        bounds=(gx[0]+a['col']*w,gy[0]+a['row']*h)
        size=(w,h)
        wanted=(bounds[0]+w/2,bounds[1]+h/2)
        if manual:
            bounds=(gx[0],gy[0]); size=(gx[1]-gx[0],gy[1]-gy[0]); wanted=(a['x'],a['y'])
        off_center=bool(manual) and any(abs((low[i]+high[i])/2-wanted[i])>.003 for i in range(2))
        if off_center or any(low[i]<bounds[i]+.002 or high[i]>bounds[i]+size[i]-.002 for i in range(2)):
            failures.append(f'{name}: settled footprint {low}..{high} outside cell {a["cell_id"]}')
            if recenter:
                import torch
                if high[0]-low[0]>size[0]-.02 or high[1]-low[1]>size[1]-.02:
                    raise RuntimeError(f'{name}: settled footprint cannot fit 20cm cell safely')
                pose=obj.data.root_state_w[:,:7].clone()
                # Change ONLY XY during spawn preparation, retain settled Z,
                # orientation and each object's original grasp conventions.
                pose[0,0] += wanted[0]-(low[0]+high[0])/2
                pose[0,1] += wanted[1]-(low[1]+high[1])/2
                obj.write_root_pose_to_sim(pose)
                obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_state_w[:,7:13]))
                print(f'[P4 CELL RECENTER] {name} -> cell {a["cell_id"]}, XY only, before recording',flush=True)
    return failures
