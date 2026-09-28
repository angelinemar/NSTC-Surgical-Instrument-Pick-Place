"""P4 class-designated tray slots. Geometry/QC are expert-only, not policy input.

Canonical top view: screen right = tray +Y, screen down = tray +X.
Slot identity is object_type_id, NEVER chronological placement order.
"""
import json
import math
import os
import numpy as np
from phase4_grasp_validation import GraspEvidence, EvidenceFailure, rotate, rigid_world_points

VERSION = 'tray_slots_v1'
ORDER = ('scalpel', 'scissor', 'love_retractor', 'kelly', 'scalpel_type2')
CENTER_TOL = .008
ANGLE_TOL = 8.
RELEASE_GAP = .008


def axes(layout):
    yaw = math.radians(layout['tray_yaw_deg'])
    return np.array([math.cos(yaw), math.sin(yaw), 0.]), np.array([-math.sin(yaw), math.cos(yaw), 0.])


def slot_spec(name, layout):
    index = ORDER.index(name)
    length, width = layout['tray_dimensions_local_xy']
    down, right = axes(layout)
    # 15 mm end margins plus five equally sized, disjoint lanes.
    pitch = (width-.030)/len(ORDER)
    center = np.array([*layout['tray_xy'], 0.]) + right*((index-2)*pitch)
    return dict(slot_id=index, object_type_id=index, object=name,
                center_xy=center[:2].tolist(), size_local_xy=[length-.020,pitch],
                long_axis_world=down.tolist(), frame='tray_local: right=+Y, down=+X')


def select_occupancy(target, rng, session=None):
    eligible = [n for n in ORDER if n != target]
    mode = (session or {}).get('tray_mode', os.environ.get('P4_TRAY_OCCUPANCY','random'))
    if mode == 'random':
        # Mixed clutter: target remains on the table, and at least one
        # non-target stays on the table as a visual/physical distractor.
        count = int(rng.integers(0,4)) if hasattr(rng,'integers') else int(rng.randint(0,4))
        chosen = [str(n) for n in rng.choice(eligible,size=count,replace=False)]
    elif mode == 'empty':
        chosen = []
    elif mode == 'full':
        chosen = eligible
    elif mode == 'manual':
        chosen = list((session or {}).get('tray_objects',[]))
    else:
        raise ValueError(f'Unknown tray occupancy mode: {mode}')
    if target in chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(eligible):
        raise ValueError('Tray must contain unique non-target types only; target slot must be empty')
    return [n for n in ORDER if n in chosen]


def mul(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    return np.r_[a[0]*b[0]-np.dot(a[1:],b[1:]), a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:],b[1:])]


def object_axis(obj):
    rigid_world_points(obj)  # initialize measured PhysX-link geometry
    basis = np.zeros(3)
    basis[int(np.argmax(np.ptp(obj._p4_link_corners,axis=0)))] = 1.
    return rotate(obj.data.root_link_quat_w[0].detach().cpu().numpy(),basis)


def yaw_alignment(obj, layout):
    current = object_axis(obj)
    wanted,_ = axes(layout)
    if np.linalg.norm(current[:2]) < .9:
        raise EvidenceFailure('instrument_not_flat_for_tray_alignment')
    delta = math.atan2(wanted[1],wanted[0])-math.atan2(current[1],current[0])
    delta = (delta+math.pi)%(2*math.pi)-math.pi
    return np.array([math.cos(delta/2),0.,0.,math.sin(delta/2)])


def floor_heights(ns):
    """Query actual PhysX tray support before populating it, not root-origin Z."""
    if '_p4_tray_floors' in ns:
        return ns['_p4_tray_floors']
    import omni.physx
    from phase4_scene import LAYOUT
    query = omni.physx.get_physx_scene_query_interface()
    floors = {}
    for name in ORDER:
        x,y = slot_spec(name,LAYOUT)['center_xy']
        hit = query.raycast_closest((float(x),float(y),.15),(0.,0.,-1.),.20)
        path = str(hit.get('collision',''))
        if not hit.get('hit') or not path.startswith('/World/RandomTray'):
            raise RuntimeError(f'Tray floor not found at {name} slot: {hit}')
        z = float(hit['position'][2])
        if not -.01 < z < .05:
            raise RuntimeError(f'Unexpected tray support height: {z}')
        floors[name] = z
    ns['_p4_tray_floors'] = floors
    # Expanded fingers at an outer slot can cross the lip even when the
    # held instrument is safely inside. Measure the real mesh, not root Z.
    import omni.usd
    from pxr import Usd, UsdGeom
    prim=omni.usd.get_context().get_stage().GetPrimAtPath('/World/RandomTray')
    heights=[]
    for child in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()):
        if child.IsA(UsdGeom.Mesh):
            pts=np.asarray(UsdGeom.Mesh(child).GetPointsAttr().Get(),dtype=float)
            transform=np.asarray(UsdGeom.Xformable(child).ComputeLocalToWorldTransform(0))
            world=np.column_stack((pts,np.ones(len(pts))))@transform
            heights.append(float(world[:,2].max()))
    if not heights:
        raise RuntimeError('Missing tray lip mesh')
    ns['_p4_tray_rim_z']=max(heights)
    print('[P4 TRAY SUPPORT] PhysX raycast floor Z:', floors, flush=True)
    print('[P4 TRAY LIP] mesh top Z:',ns['_p4_tray_rim_z'],flush=True)
    return floors


def prepare_tray(env, ns):
    """Move selected existing bodies to the tray BEFORE settle/recording."""
    from phase4_scene import LAYOUT
    import torch
    floors = floor_heights(ns)
    selected = ns.get('_p4_tray_objects',[])
    ns['_p4_tray_requested']={}
    for name in selected:
        obj = env.scene[ns['phase3_scene_key_for_object'](name)]
        rigid_world_points(obj)
        q = mul(yaw_alignment(obj,LAYOUT),obj.data.root_link_quat_w[0].detach().cpu().numpy())
        pts = rotate(q,obj._p4_link_corners)
        midpoint = rotate(q,(obj._p4_link_corners.min(axis=0)+obj._p4_link_corners.max(axis=0))/2)
        xy = np.asarray(slot_spec(name,LAYOUT)['center_xy'])-midpoint[:2]
        z = floors[name]+.0015-float(pts[:,2].min())
        pose = torch.tensor([[*xy,z,*q]],device=env.device,dtype=obj.data.root_state_w.dtype)
        ns['_p4_tray_requested'][name]=dict(region='tray',slot=slot_spec(name,LAYOUT),
                                           root_link_pose_xyz_wxyz=pose[0].detach().cpu().tolist())
        obj.write_root_link_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros((env.num_envs,6),device=env.device))
        obj.update(env.physics_dt)
    ns['_p4_tray_initial'] = {}
    print(f'[P4 TRAY OCCUPANCY] {len(selected)}/4 preloaded={selected}; empty target={ns["PHASE3_TARGET_OBJECT"]}',flush=True)


def slot_measurement(obj, name, layout):
    spec=slot_spec(name,layout)
    pts=rigid_world_points(obj)
    q=obj.data.root_link_quat_w[0].detach().cpu().numpy()
    center=rotate(q,(obj._p4_link_corners.min(axis=0)+obj._p4_link_corners.max(axis=0))/2)+obj.data.root_link_pos_w[0].detach().cpu().numpy()
    down,right=axes(layout)
    offsets=pts-np.r_[spec['center_xy'],0.]
    local=np.column_stack((offsets@down,offsets@right))
    inside=bool(np.all(np.abs(local)<=np.asarray(spec['size_local_xy'])/2-.003))
    error=float(np.linalg.norm(center[:2]-spec['center_xy']))
    axis=object_axis(obj)
    planar_norm=float(np.linalg.norm(axis[:2]))
    angle=180. if planar_norm<.9 else math.degrees(math.acos(float(np.clip(np.dot(axis[:2]/planar_norm,down[:2]),-1,1))))
    tilt=math.degrees(math.asin(float(np.clip(abs(axis[2]),0,1))))
    return dict(center=center.tolist(),center_error_m=error,axis_error_deg=angle,axis_tilt_deg=tilt,inside_slot=inside,bottom_m=float(pts[:,2].min()))


def check_preloaded(env, ns, capture=False):
    from phase4_scene import LAYOUT
    reports={n:slot_measurement(env.scene[ns['phase3_scene_key_for_object'](n)],n,LAYOUT) for n in ns.get('_p4_tray_objects',[])}
    for name,r in reports.items():
        if not r['inside_slot'] or r['center_error_m']>CENTER_TOL or r['axis_error_deg']>ANGLE_TOL:
            raise EvidenceFailure(f'preloaded_instrument_out_of_slot: {name}: {r}')
        if abs(r['bottom_m']-ns['_p4_tray_floors'][name])>.005:
            raise EvidenceFailure(f'preloaded_instrument_not_supported: {name}: {r}')
    if capture:
        ns['_p4_tray_initial']=reports
    return reports


class SlotEvidence(GraspEvidence):
    def __init__(self, env, ns, layout):
        self.env,self.ns,self.layout=env,ns,layout
        self.name=ns['PHASE3_TARGET_OBJECT']
        self.spec=slot_spec(self.name,layout)
        self.obj=env.scene[ns['phase3_scene_key_for_object'](self.name)]
        self.other_final={}
        self.release_measurement={}
        super().__init__(self.spec['center_xy'],self.spec['size_local_xy'],layout['tray_yaw_deg'],ns['_p4_tray_floors'][self.name])

    def begin_release(self,ee,quat,center,points,bottom):
        super().begin_release(ee,quat,center,points,bottom)
        r=slot_measurement(self.obj,self.name,self.layout)
        if r['center_error_m']>CENTER_TOL or r['axis_error_deg']>ANGLE_TOL or not r['inside_slot']:
            raise EvidenceFailure(f'slot_release_alignment_invalid: {r}')
        if not .003 <= bottom-self.tray_z <= .018:
            raise EvidenceFailure(f'slot_release_gap_invalid: {bottom-self.tray_z:.4f}m')
        check_preloaded(self.env,self.ns)
        self.release_measurement=dict(r,gap_above_support_m=float(bottom-self.tray_z))
        print(f'[P4 SLOT RELEASE READY] {self.name}: {self.release_measurement}',flush=True)

    def check_retreat(self,ee,center,points,bottom):
        super().check_retreat(ee,center,points,bottom)
        r=slot_measurement(self.obj,self.name,self.layout)
        self.place_verified=bool(self.place_verified and r['inside_slot'] and r['center_error_m']<=CENTER_TOL and r['axis_error_deg']<=ANGLE_TOL and abs(bottom-self.tray_z)<=.005)
        self.last_placement.update(r)
        if self.place_verified:
            self.other_final=check_preloaded(self.env,self.ns)
        return self.place_verified

    def summary(self):
        result=super().summary()
        result.update(version='20260916-tray-slots-v1',tray_contract=VERSION,slot=self.spec,slot_order=list(ORDER),
                      initial_tray_objects=list(self.ns.get('_p4_tray_objects',[])),
                      initial_tray_geometry=self.ns.get('_p4_tray_initial',{}),
                      final_other_tray_geometry=self.other_final,
                      release_measurement=self.release_measurement,
                      tray_support_z=self.tray_z,release_gap_m=RELEASE_GAP,
                      tray_lip_z=self.ns.get('_p4_tray_rim_z'),
                      center_tolerance_m=CENTER_TOL,axis_tolerance_deg=ANGLE_TOL,
                      direction_convention='positive longest PhysX-link axis points tray +X; not an anatomical tip label')
        return result
