"""Static physics-settled scenes, DLAA 448, ring + top PNG and COCO capture."""
import argparse
import json
import os
from pathlib import Path
import random
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'compat')]
from detection.dataset import CLASSES, camera_views, write_json, export_index


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenes',type=int,default=100)
    p.add_argument('--min-objects',type=int,default=8)
    p.add_argument('--max-objects',type=int,default=18)
    p.add_argument('--ring-cameras',type=int,default=8)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--robot-visible-probability',type=float,default=.25)
    p.add_argument('--tray',choices=('random','empty','full'),default='random')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-file',type=Path)
    a=p.parse_args()
    if not (1<=a.scenes and 5<=a.min_objects<=a.max_objects<=30 and 3<=a.ring_cameras<=16 and 0<=a.robot_visible_probability<=1):
        p.error('Use scenes >=1, 5<=min<=max<=30, 3..16 ring cameras, robot probability 0..1')
    a.output=a.output.resolve()
    return a


def main():
    args=arguments()
    contract=dict(contract='p4_static_detection_v1',resolution=448,antialiasing='DLAA',materials='recorder',
                  seed=args.seed,min_objects=args.min_objects,max_objects=args.max_objects,
                  ring_cameras=args.ring_cameras,robot_visible_probability=args.robot_visible_probability,tray=args.tray,
                  split='train',robot_pose='home_static',labels='visible_rigid_instance_boxes')
    config=args.output/'config.json'
    if args.resume:
        if not config.is_file() or json.loads(config.read_text())!=contract:
            raise ValueError('Resume configuration mismatch or missing config')
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        write_json(config,contract)
    (args.output/'train'/'scenes').mkdir(parents=True,exist_ok=True)
    os.environ.update(P4_CAMERA_SIZE='448',P4_ANTIALIASING_MODE='DLAA',P4_INSTRUMENT_MATERIALS='recorder',
                      P4_RANDOMIZATION='train',P4_RANDOMIZATION_SEED=str(args.seed),P4_TRAY_OCCUPANCY='empty')
    # Ignore DP panel environment settings in this separate process.
    for key in ('P4_SESSION_CONFIG','P4_SESSION_FILE'):
        os.environ.pop(key,None)
    backend=ROOT/'backends/phase3_grid_split_scalpel_recorder.py'
    sys.argv=[str(backend),'--episodes','0','--layout_tuner','--out_dir',str(args.output/'bootstrap')]
    if args.headless: sys.argv.append('--headless')
    loaded=runpy.run_path(str(backend),run_name='detection_scene_bootstrap')
    ns=loaded['main'].__globals__
    from phase3_shared_env_cfg import RecorderEnvRequest
    ns['RecorderEnvRequest']=lambda **kw: RecorderEnvRequest(target='scalpel',distractors=CLASSES[1:],
        environment=('hospital_room',),use_canonical_target_only=True)
    # A per-class pool permits duplicate instances of EVERY class.
    import src.recorder.scene_clutter as clutter
    pool=[(f'detect_{c}_{i:02d}',c) for c in CLASSES for i in range(args.max_objects)]
    clutter.pool=lambda target: pool
    def add_pool(scene,target,rigid_cfg):
        for j,(key,name) in enumerate(pool):
            cfg=rigid_cfg(name,prim_name=key); cfg.init_state.pos=(3.+j*.2,3.,-2.)
            setattr(scene,key,cfg)
    clutter.add_to_scene=add_pool
    def setup_camera(cfg,**kw):
        from isaaclab.sensors import CameraCfg
        import isaaclab.sim as sim
        for name in ('grip_cam_b','cam_top','cam_left','cam_right','cam_tray'):
            setattr(cfg.scene,name,None)
        cfg.scene.camera=CameraCfg(prim_path='{ENV_REGEX_NS}/detection_camera',width=448,height=448,
            data_types=['rgb','semantic_segmentation','instance_id_segmentation_fast'],
            colorize_semantic_segmentation=False,colorize_instance_id_segmentation=False,
            spawn=sim.PinholeCameraCfg(focal_length=16.,horizontal_aperture=20.955,clipping_range=(.01,10.)),
            update_period=0)
    ns['phase3_apply_final_cameras_to_env_cfg']=setup_camera
    def capture_and_exit(env,stage,path):
        collect(env,stage,ns,args)
        # Isaac 5.1 on Windows can access-violate during USD teardown. All
        # PNG writers and atomic indexes are closed before this success exit.
        # Never take this path on a capture/validation exception.
        if os.name == 'nt':
            print('[DETECTION EXIT] Files committed; bypassing unsafe Windows USD teardown.',flush=True)
            sys.stderr.flush()
            os._exit(0)
    ns['run_shared_layout_tuner']=capture_and_exit
    try:
        ns['main']()
    finally:
        if ns['simulation_app'].is_running(): ns['simulation_app'].close()


def collect(env,stage,ns,args):
    import numpy as np
    import torch
    from PIL import Image
    from pxr import UsdGeom
    from phase4_scene import LAYOUT
    from phase4_reset import table_pose
    from phase4_grasp_validation import rigid_world_points,rotate
    from phase4_tray_slots import slot_spec,axes
    from src.recorder.scene_clutter import disjoint,table_footprint_allowed
    from src.recorder.domain_randomization import apply_episode_lighting
    from src.recorder.instance_labels import body_mask
    from training.export_detection import visible_boxes
    env.reset()
    # Initialize flat orientation through existing mesh-specific spawn functions.
    rng0=np.random.default_rng(args.seed)
    spawn=ns['sample_episode_spawn_grid'](rng0,2,'scalpel')
    spawn=ns['phase3_ensure_all_5_objects_in_spawn'](spawn,'scalpel',rng0)
    ns['force_episode_objects'](env,spawn,ns['choose_scalpel_pose_mode'](2))
    for _ in range(60):
        env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(env.physics_dt)
    templates={}
    for name in CLASSES:
        obj=env.scene[ns['phase3_scene_key_for_object'](name)]; rigid_world_points(obj)
        templates[name]=dict(yaw=float(spawn[name]['yaw_deg']),quat=obj.data.root_link_quat_w[0].cpu().numpy().copy(),points=obj._p4_link_corners.copy())
    bodies=[(ns['phase3_scene_key_for_object'](c),c) for c in CLASSES]+[(f'detect_{c}_{i:02d}',c) for c in CLASSES for i in range(args.max_objects)]
    robot=UsdGeom.Imageable(stage.GetPrimAtPath('/World/envs/env_0/Robot'))
    if not robot.GetPrim().IsValid():
        robot=UsdGeom.Imageable(stage.GetPrimAtPath(env.scene['robot'].cfg.prim_path.replace('{ENV_REGEX_NS}','/World/envs/env_0')))
    def pose(key,value):
        obj=env.scene[key]
        obj.write_root_link_pose_to_sim(torch.tensor([value],device=env.device,dtype=torch.float32))
        obj.write_root_velocity_to_sim(torch.zeros((1,6),device=env.device)); obj.update(env.physics_dt)
    completed=export_index(args.output)['scenes']
    for index in range(completed,args.scenes):
        if args.stop_file and args.stop_file.exists(): break
        folder=args.output/'train'/'scenes'/f'scene_{index:06d}'
        pending=args.output/f'.pending_scene_{index:06d}'
        if pending.exists():
            # Preserve interrupted work as diagnostics; no destructive recovery.
            import time
            pending.rename(args.output/f'.interrupted_{index:06d}_{time.time_ns()}')
        pending.mkdir()
        seed=args.seed+index*1009; rng=random.Random(seed)
        for j,(key,_) in enumerate(bodies): pose(key,[3.+j*.2,3.,-2.,1.,0.,0.,0.])
        ns['_p4_attempt_number']=index+1
        apply_episode_lighting(env,ns)
        tray=list(CLASSES) if args.tray=='full' else [c for c in CLASSES if args.tray=='random' and rng.random()<.4]
        count=rng.randint(args.min_objects,args.max_objects)
        types=list(CLASSES)+rng.choices(CLASSES,k=count-5)
        rng.shuffle(types); used={c:0 for c in CLASSES}; rows=[]; occupied=[]
        # Place named base bodies into their orderly tray slots when selected.
        down,_=axes(LAYOUT)
        for c in tray:
            key=ns['phase3_scene_key_for_object'](c); t=templates[c]
            pts=rotate(t['quat'],t['points'])
            # Dominant planar extent gives the imported flat tool's long axis.
            centered=pts[:,:2]-pts[:,:2].mean(0); _,_,v=np.linalg.svd(centered)
            angle=np.degrees(np.arctan2(down[1],down[0])-np.arctan2(v[0,1],v[0,0]))+t['yaw']
            xy=slot_spec(c,LAYOUT)['center_xy']; value=table_pose(t,*xy,angle)
            from phase4_tray_slots import floor_heights
            value[2]+=floor_heights(ns)[c]
            pose(key,value.tolist()); rows.append(dict(key=key,object=c,region='tray'))
        for c in types:
            key=f'detect_{c}_{used[c]:02d}'; used[c]+=1; t=templates[c]
            for trial in range(5000):
                value=table_pose(t,rng.uniform(*LAYOUT['grid_x']),rng.uniform(*LAYOUT['grid_y']),rng.uniform(-180,180))
                pts=rotate(value[3:],t['points'])+value[:3]; bounds=(pts[:,:2].min(0),pts[:,:2].max(0))
                if table_footprint_allowed(pts,LAYOUT) and all(disjoint(bounds,b,.002) for b in occupied): break
            else: raise RuntimeError('Scatter cannot fit requested count. Reduce maximum objects and start a new collection.')
            occupied.append(bounds); pose(key,value.tolist()); rows.append(dict(key=key,object=c,region='table'))
        show=rng.random()<args.robot_visible_probability
        robot.MakeVisible() if show else robot.MakeInvisible()
        # Physics settling only: no policy action or pick/place execution.
        for _ in range(120):
            env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(env.physics_dt)
        roots={}; classes={}
        for j,row in enumerate(rows,1):
            obj=env.scene[row['key']]; points=rigid_world_points(obj)
            if float(obj.data.root_lin_vel_w[0].norm())>.015 or not np.isfinite(points).all():
                raise RuntimeError('Unsettled instrument: '+row['key'])
            if row['region']=='table' and (not table_footprint_allowed(points,LAYOUT,0) or abs(float(points[:,2].min()))>.01):
                raise RuntimeError('Unsupported/out-of-workspace instrument: '+row['key'])
            row['pose']=obj.data.root_link_state_w[0,:7].cpu().tolist()
            prim=obj.cfg.prim_path.replace('{ENV_REGEX_NS}','/World/envs/env_0').replace('env_.*','env_0')
            roots[prim]=j; classes[str(j)]=dict(name=row['object'],semantic_id=CLASSES.index(row['object'])+3,prim_path=prim)
        metadata=dict(scene_id=index,seed=seed,table_count=count,tray_objects=tray,robot_visible=show,
                      resolution=448,antialiasing='DLAA',materials='recorder',
                      objects=rows,instance_classes=classes,lighting=ns['_p4_domain_randomization'],views=[])
        camera=env.scene['camera']
        for view in camera_views(rng,LAYOUT,args.ring_cameras):
            camera.set_world_poses_from_view(torch.tensor([view['eye']],device=env.device),torch.tensor([view['target']],device=env.device))
            for _ in range(32): env.sim.render()
            camera.update(0.,force_recompute=True)
            rgb=camera.data.output['rgb'][0,:,:,:3].cpu().numpy().copy()
            semantic=ns['_phase3_extract_semantic_u16'](camera,448,448,cam_name='camera')
            raw=camera.data.output['instance_id_segmentation_fast'][0].cpu().numpy().squeeze(-1)
            info=camera.data.info; info=info[0] if isinstance(info,(list,tuple)) else info
            instance=body_mask(raw,info['instance_id_segmentation_fast']['idToLabels'],roots)
            if rgb.shape!=(448,448,3) or float(rgb.astype(float).std(axis=(0,1)).max())<=1:
                raise RuntimeError('Invalid/blank RGB '+view['name'])
            if np.any(((semantic>=3)&(semantic<=7)) & (instance==0)):
                raise RuntimeError('Unmapped instrument pixels '+view['name'])
            boxes=visible_boxes(semantic,instances=instance,instance_classes=classes)
            Image.fromarray(rgb).save(pending/f'{view["name"]}_rgb.png')
            Image.fromarray(semantic.astype(np.uint16)).save(pending/f'{view["name"]}_semantic.png')
            Image.fromarray(instance).save(pending/f'{view["name"]}_instance.png')
            view.update(boxes=boxes,intrinsics=camera.data.intrinsic_matrices[0].cpu().tolist())
            metadata['views'].append(view)
        if not any(v['boxes'] for v in metadata['views']): raise RuntimeError('No visible instruments in this scene')
        write_json(pending/'scene.json',metadata); pending.rename(folder)
        stats=export_index(args.output)
        write_json(args.output/'status.json',dict(state='recording',goal=args.scenes,**stats))
        print('[DETECTION SAVED] '+json.dumps(stats),flush=True)
    stats=export_index(args.output)
    write_json(args.output/'status.json',dict(state='complete' if stats['scenes']==args.scenes else 'stopped',goal=args.scenes,**stats))
    print('[DETECTION DONE] '+json.dumps(stats),flush=True)


if __name__=='__main__': main()
