"""Read USD vertices and existing successful H5; never mutate scene/assets."""
import sys,json,math,hashlib,itertools
from pathlib import Path
ROOT=Path(r'C:\IsaacLab\scripts\custom\i4h_project\p4')
sys.path.insert(0,str(ROOT))
import numpy as np,h5py
from pxr import Usd,UsdGeom,UsdPhysics,Gf
from scipy.spatial import ConvexHull
from phase4_scene import LAYOUT,table_geometry,tray_scale_for_layout
from phase3_shared_env_cfg import INSTRUMENTS,ASSET_DIR
from phase4_grasp_validation import mesh_points_in_frame,rotate
from phase4_tray_slots import mul,slot_spec

sources={
 'scalpel':'tray_slots_scalpel_retest_v2',
 'scissor':'tray_slots_full_validation/datasets/scissor',
 'love_retractor':'panel_retry_validation_v3',
 'kelly':'tray_slots_full_validation/datasets/kelly',
 'scalpel_type2':'tray_slots_type2_retest'}
instruments={}; catalog={}
for name,spec in INSTRUMENTS.items():
    path=ASSET_DIR/spec.usd; stage=Usd.Stage.Open(str(path))
    body=next(p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))
    scale=np.abs(np.asarray(Gf.Transform(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0)).GetScale()))*np.asarray(spec.scale)
    points=mesh_points_in_frame(body)*scale
    hpath=ROOT/'test_runs'/sources[name]/'pick_policy'/name/'episode_000000.h5'
    with h5py.File(hpath,'r') as h:
        settled=json.loads(h.attrs['spawn_settled_all'])[name]
        yaw=float(h.attrs['target_yaw_deg'])
        q=np.asarray(settled['quaternion_wxyz'])
        d=math.radians(-yaw)
        zero=rotate(mul([math.cos(d/2),0,0,math.sin(d/2)],q),points)
        zero-= (zero.min(0)+zero.max(0))/2
        hull=zero[ConvexHull(zero[:,:2]).vertices,:2]
        long=np.zeros(3); long[int(np.argmax(np.ptp(points,axis=0)))]=1
        axis=rotate(mul([math.cos(d/2),0,0,math.sin(d/2)],q),long)
        heading=math.degrees(math.atan2(axis[1],axis[0]))
        item=dict(asset=spec.usd,asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             hull_xy_m=hull.tolist(),axis_xy=axis[:2].tolist(),heading_offset_deg=heading,
             reference_yaw_deg=yaw,reference_h5=str(hpath.relative_to(ROOT)),
             note='Measured settled target pose, yaw removed; prediction only until fresh scene settles')
        catalog[name]=item
        instruments[name]=dict(**item,scale=list(spec.scale),size_link_xyz_m=np.ptp(points,axis=0).tolist(),
            length_width_thickness_m=sorted(np.ptp(points,axis=0).tolist(),reverse=True),
            footprint_yaw0_xyz_m=np.ptp(zero,axis=0).tolist(),target_mass_kg=.10,distractor_mass_kg=spec.mass)

geom=table_geometry()
stage=Usd.Stage.Open(str(ROOT/'assets'/LAYOUT['hospital_asset']))
prim=stage.GetPrimAtPath('/HospitalScene/Furniture/'+LAYOUT['selected_table'])
all_points=[]
for child in Usd.PrimRange(prim):
    if not child.IsA(UsdGeom.Boundable): continue
    if child.IsA(UsdGeom.Mesh): p=np.asarray(UsdGeom.Mesh(child).GetPointsAttr().Get(),dtype=float)
    elif child.IsA(UsdGeom.Cube):
        half=float(UsdGeom.Cube(child).GetSizeAttr().Get())/2
        p=np.array(list(itertools.product([-half,half],repeat=3)))
    else:
        ext=UsdGeom.Boundable(child).GetExtentAttr().Get()
        if not ext: continue
        p=np.array(list(itertools.product(*zip(ext[0],ext[1]))))
    matrix=np.asarray(UsdGeom.Xformable(child).ComputeLocalToWorldTransform(0))
    all_points.append((np.column_stack([p,np.ones(len(p))])@matrix)[:,:3])
world=rotate(geom['room_rot'],np.concatenate(all_points))+np.array(geom['room_pos'])
table=dict(**geom,full_bounds_world_min=world.min(0).tolist(),full_bounds_world_max=world.max(0).tolist(),
           full_world_size_m=np.ptp(world,axis=0).tolist())
tray_path=ASSET_DIR/'SurgicalTray.usd'; ts=Usd.Stage.Open(str(tray_path))
tb=UsdGeom.BBoxCache(0,['default','render']).ComputeWorldBound(ts.GetDefaultPrim()).ComputeAlignedRange()
tray_scale=tray_scale_for_layout(tray_path)
tray=dict(scale=tray_scale,asset_size_scaled_xyz_m=(np.asarray(tb.GetSize())*tray_scale).tolist(),
          slots=[slot_spec(n,LAYOUT) for n in INSTRUMENTS])

sample=ROOT/'test_runs/panel_retry_validation_v3/pick_policy/love_retractor/episode_000000.h5'
with h5py.File(sample,'r') as h:
    attrs={k:(v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else v) for k,v in h.attrs.items()}
    topics={}; calibrations={}
    def visit(name,obj):
        if isinstance(obj,h5py.Dataset):
            topics[name]=dict(shape=list(obj.shape),dtype=str(obj.dtype))
            if ('intrinsic' in name or 'extrinsic' in name or 'camera' in name or 'cam_' in name) and obj.size<10000:
                if obj.dtype.kind in 'fi': calibrations[name]=obj[0].tolist() if len(obj.shape)>=2 and obj.shape[0]==int(h.attrs['num_samples']) else obj[:].tolist()
    h.visititems(visit)
    proprio=h['observations/robot_proprio'][:]
    sample_actions=h['actions'][:]
    stage_names=[s.decode() if isinstance(s,bytes) else str(s) for s in h['stage_names'][:]]
with h5py.File(sample.parent.parent.parent/'place_policy/love_retractor/episode_000000.h5','r') as h:
    all_actions=np.concatenate([sample_actions,h['actions'][:]])
    all_ee=np.concatenate([proprio[:,9:12],h['observations/robot_proprio'][:,9:12]])
    all_names=stage_names+[s.decode() if isinstance(s,bytes) else str(s) for s in h['stage_names'][:]]
all_ee=rotate(LAYOUT['robot_rot_wxyz'],all_ee)+np.asarray(LAYOUT['robot_pos'])
stages=[]
for n in dict.fromkeys(all_names):
    ids=[i for i,s in enumerate(all_names) if s==n]; xyz=all_ee[ids]
    stages.append(dict(name=n,steps=len(ids),sim_seconds=len(ids)*.02,
                       start_xyz=xyz[0].tolist(),end_xyz=xyz[-1].tolist(),
                       chord_m=float(np.linalg.norm(xyz[-1]-xyz[0])),
                       path_m=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum())))
data=dict(layout=LAYOUT,table=table,tray=tray,instruments=instruments,
          cameras=json.loads((ROOT/'camera_layout.json').read_text())['cameras'],
          sample_attrs=attrs,topics=topics,calibrations=calibrations,stages=stages,
          trajectory_xyz=all_ee[::4].tolist(),reference_h5=str(sample.relative_to(ROOT)))
out=Path(__file__).parent
(out/'report_data.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
(out/'instrument_preview_geometry.json').write_text(json.dumps(dict(version=1,source='USD vertices + measured settled target poses',instruments=catalog),indent=2),encoding='utf-8')
print(json.dumps(dict(table=table,tray=tray,instruments={n:{k:v for k,v in d.items() if k in ('length_width_thickness_m','footprint_yaw0_xyz_m','heading_offset_deg','target_mass_kg','distractor_mass_kg')} for n,d in instruments.items()},calibration_keys=list(calibrations)),indent=2))
