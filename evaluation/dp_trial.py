"""One bounded hybrid trial. Simulator state is for FSM/scoring, never DP inputs."""
import argparse,inspect,json,os,runpy,sys,time,hashlib
from pathlib import Path
from evaluation.dp_contract import replace_policy_windows


def _workspace_root():
 override = os.environ.get('NSTC_WORKSPACE_ROOT')
 if override:
  return Path(override).expanduser().resolve()
 for parent in Path(__file__).resolve().parents:
  if (parent/'datasets').is_dir() and (parent/'versions/6.0').is_dir():
   return parent
 raise RuntimeError('Set NSTC_WORKSPACE_ROOT to the NSTC workspace')


def _repo_root():
 return Path(__file__).resolve().parents[1]

class TrialFinished(BaseException):pass

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--snapshot',type=Path,help='Held-out raw H5: reset directly to skill start, no FSM')
 p.add_argument('--snapshot-split',choices=['valid','test'],default='test')
 p.add_argument('--skill',choices=['both','pick','place'],default='both')
 p.add_argument('--pick',type=Path,required=True);p.add_argument('--place',type=Path,required=True)
 p.add_argument('--object',required=True);p.add_argument('--mode',choices=['grid','free'],required=True)
 p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--max-steps',type=int,default=300);p.add_argument('--headless',action='store_true')
 p.add_argument('--expert-replay',action='store_true',help='Diagnostic: execute saved expert actions from --snapshot')
 p.add_argument('--expert-approach',action='store_true',help='Diagnostic: replay expert actions up to first CLOSE, then hand control to DP')
 p.add_argument('--template-replay',type=Path,help='Diagnostic: transform a training demonstration around the saved target pose')
 p.add_argument('--camera-pose-json',type=Path,help='Scissor camera-estimated target pose JSON; only with --template-replay')
 p.add_argument('--live-camera-pose',action='store_true',help='Capture front RGB-D after reset and estimate Scissor pose with RF-DETR B')
 p.add_argument('--learned-camera-pose',action='store_true',help='Use RF-DETR B live target pose with a trained target-relative Pick policy')
 p.add_argument('--target-perturb-dx',type=float,default=0.0,help='Evaluation-only target X shift in robot base, meters')
 p.add_argument('--target-perturb-dy',type=float,default=0.0,help='Evaluation-only target Y shift in robot base, meters')
 p.add_argument('--target-perturb-yaw-deg',type=float,default=0.0,help='Evaluation-only target yaw shift, degrees')
 a=p.parse_args()
 if a.expert_replay and (not a.snapshot or a.skill not in ('pick','place')):p.error('Expert replay requires --snapshot and one skill')
 if a.expert_approach and (not a.snapshot or a.skill!='pick' or a.expert_replay):p.error('Expert approach requires Pick snapshot without --expert-replay')
 if a.template_replay and (not a.snapshot or a.skill!='pick' or a.expert_replay or a.expert_approach):p.error('Template replay requires Pick snapshot without other replay flags')
 if a.camera_pose_json and not a.template_replay:p.error('--camera-pose-json requires --template-replay')
 if a.live_camera_pose and (not a.template_replay or a.camera_pose_json or a.object!='scissor'):p.error('--live-camera-pose requires Scissor --template-replay without --camera-pose-json')
 if a.learned_camera_pose and (not a.snapshot or a.skill!='pick' or a.template_replay or a.expert_replay or a.expert_approach or a.camera_pose_json or a.live_camera_pose):p.error('--learned-camera-pose requires a Pick snapshot with no replay options')
 a.output.mkdir(parents=True,exist_ok=False)
 row=dict(initialization='snapshot_no_fsm' if a.snapshot else 'FSM',evaluated_skill=a.skill,object=a.object,mode=a.mode,seed=a.seed,pick_started=False,place_started=False,lift_success=False,place_success=False,protocol='learned_target_relative_dp_live_camera_v1' if a.learned_camera_pose else 'train_template_live_camera_pose_diagnostic_v1' if a.live_camera_pose else 'train_template_camera_pose_diagnostic_v1' if a.camera_pose_json else 'train_template_oracle_pose_diagnostic_v1' if a.template_replay else 'expert_replay_diagnostic_v1' if a.expert_replay else 'expert_approach_then_dp_diagnostic_v1' if a.expert_approach else 'snapshot_skill_only_v1' if a.snapshot else 'hybrid_skill_only_v1',max_steps=a.max_steps,action_horizon=4 if a.learned_camera_pose else 8,denoising_steps=100,expert_replay=a.expert_replay,expert_approach=a.expert_approach,template_replay=str(a.template_replay) if a.template_replay else None,model_inference_used=not (a.expert_replay or a.template_replay),oracle_object_pose_used=bool(a.template_replay and not (a.camera_pose_json or a.live_camera_pose)),camera_pose_estimate_used=bool(a.camera_pose_json or a.live_camera_pose or a.learned_camera_pose),detector_inference_used=bool(a.camera_pose_json or a.live_camera_pose or a.learned_camera_pose))
 for skill in ['pick','place']:
  ck=getattr(a,skill);row[skill+'_checkpoint']=str(ck);row[skill+'_sha256']=hashlib.sha256(ck.read_bytes()).hexdigest()
 def event(actor,stage,**extra):
  row['active_actor']=actor;row['active_stage']=stage
  record=dict(actor=actor,stage=stage,wall_time=time.time(),**extra)
  with (a.output/'events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  print('[CONTROL]',actor,stage,extra,flush=True)
  if label[0] is not None:label[0].text=f'{a.object} | {a.mode}\nCONTROL: {actor}\n{stage}'
 label=[None];ns=None;app=None;actor=[None];video=[]
 try:
  from src.entry.object_handlers import get_handler
  import phase3_shared_env_cfg as shared
  original=shared.apply_workspace_globals
  def setup(namespace,*args,**kw):
   nonlocal ns
   original(namespace,*args,**kw);ns=namespace
   import numpy as np,torch,omni.ui as ui
   from training.runtime import read_live_sensors
   from evaluation.dp_actor import Actor
   from phase4_grasp_validation import rigid_world_points,rotate
   from phase4_tray_slots import slot_measurement,slot_spec,CENTER_TOL,ANGLE_TOL,check_preloaded
   from phase4_scene import LAYOUT
   if not (a.expert_replay or a.template_replay):
    actor[0]=Actor(_workspace_root()/'.venv-dp/bin/python',a.pick,a.place,a.seed,a.output/'actor.log')
    row['input_contracts']={skill:meta['input_contract'] for skill,meta in actor[0].metadata.items()}
    if a.learned_camera_pose:
     expected=('target_relative_dp_scissor_v1' if a.object=='scissor'
               else f'target_relative_dp_{a.object}_pick_v1')
     if row['input_contracts']['pick'] not in (expected,'target_relative_dp_joint_pick_v1'):
      raise ValueError(f'Pick checkpoint must use {expected} or target_relative_dp_joint_pick_v1')
    if a.snapshot and a.skill=='place' and row['input_contracts']['place'].startswith('target_relative_dp_'):
     row['protocol']='trained_target_relative_place_known_slot_v1';row['action_horizon']=4
    for s,m in actor[0].metadata.items():
     if m['skill']!=s or m['image_size']!=448:raise ValueError('Checkpoint skill/resolution mismatch')
     if m['controller_contract'][0]!='tcp_frame_correct_bounded_place_integral_v6':raise ValueError('Controller mismatch')
   window=ui.Window('DP evaluation — controller ownership',width=480,height=160)
   with window.frame:label[0]=ui.Label('Preparing scene',word_wrap=True)
   ns['_dp_eval_window']=window
   def finish():raise TrialFinished()
   def measure(env):
    obj=env.scene[ns['phase3_scene_key_for_object'](a.object)]
    pts=rigid_world_points(obj);q=obj.data.root_link_quat_w[0].cpu().numpy()
    mid=(obj._p4_link_corners.min(0)+obj._p4_link_corners.max(0))/2
    center=rotate(q,mid)+obj.data.root_link_pos_w[0].cpu().numpy()
    return obj,center,float(pts[:,2].min()),pts
   def skill(env,which):
    owner='TEMPLATE' if a.template_replay else 'EXPERT' if a.expert_replay else 'DP'
    row[which+'_started']=True;event(owner,which.upper()+' transformed training template' if a.template_replay else which.upper()+' replayed expert actions' if a.expert_replay else which.upper()+' full learned window')
    from phase4_approach import configure,set_precision_skill
    configure(env,True);set_precision_skill(env,which)
    if abs(float(env.step_dt)-.02)>1e-6:raise ValueError('Control interval differs from training')
    if a.expert_approach:
     import h5py
     with h5py.File(a.snapshot) as f:prefix=f['actions'][:]
     switches=np.flatnonzero((prefix[1:,7]<0)&(prefix[:-1,7]>=0))+1
     if len(switches)!=1:raise ValueError('Expected exactly one expert CLOSE event')
     warmsteps=int(switches[0])
     row['oracle_expert_approach_steps']=warmsteps
     event('ORACLE_EXPERT_APPROACH','expert motion until just before CLOSE',steps=warmsteps)
     for expert_action in prefix[:warmsteps]:
      env.step(torch.as_tensor(expert_action,dtype=torch.float32,device=env.device)[None])
    obj,initial,bottom,pts=measure(env);evidence=ns['_p4_evidence']
    known_slot_b=None
    if which=='place' and row.get('input_contracts',{}).get('place','').startswith('target_relative_dp_'):
     # The task's assigned tray slot is a calibrated goal, not a measured
     # simulator object pose or a saved expert action.
     slot=slot_spec(a.object,LAYOUT)
     goal_w=torch.as_tensor([*slot['center_xy'],float(evidence.tray_z)],dtype=torch.float32,device=env.device)
     known_slot_b=ns['to_base_pos'](env,goal_w).detach().cpu().numpy().astype(np.float32)
     row['known_slot_goal_b']=known_slot_b.tolist()
     event('GOAL','calibrated assigned tray slot',slot_id=slot['slot_id'])
    robot=env.scene['robot']
    finger_ids=[robot.joint_names.index(name) for name in ('panda_finger_joint1','panda_finger_joint2')]
    stable=0;centers=[];anchor=None;peak=float(initial[2])
    expert_actions=None
    if a.expert_replay:
     import h5py
     with h5py.File(a.snapshot) as f:expert_actions=f['actions'][:]
    target_pose_b=None
    if a.learned_camera_pose:
     import subprocess
     cam=env.scene['camera']
     depth=cam.data.output['distance_to_image_plane'][0].detach().cpu().numpy().squeeze()
     sensor_file=a.output/'live_front_sensor.npz'
     np.savez_compressed(sensor_file,
      rgb=cam.data.output['rgb'][0].detach().cpu().numpy()[...,:3].astype(np.uint8),
      depth=depth.astype(np.float32),
      intrinsics=cam.data.intrinsic_matrices[0].detach().cpu().numpy().astype(np.float32),
      position_b=ns['to_base_pos'](env,cam.data.pos_w[0]).detach().cpu().numpy().astype(np.float32),
      quaternion_b_ros=ns['to_base_quat'](env,cam.data.quat_w_ros[0]).detach().cpu().numpy().astype(np.float32),
      ee_position_b=read_live_sensors(env,image_size=448)['robot_proprio'][9:12].astype(np.float32))
     pose_file=a.output/'live_camera_pose.json'
     root=_workspace_root()
     estimator=[str(root/'NSTC-Surgical-Instrument-Pick-Place/.venv-rfdetr/bin/python'),
      str(_repo_root()/'experiments/dp_target_relative/estimate_scissor_pose.py'),
      '--sensor-npz',str(sensor_file),'--output',str(pose_file),'--instrument',a.object]
     if a.object!='scissor':estimator+=['--threshold','0.1','--selection','nearest_tcp']
     event('RF-DETR B','estimate and lock '+a.object+' target from live front RGB-D')
     with (a.output/'pose_estimator.log').open('w') as log:
      subprocess.run(estimator,stdout=log,stderr=subprocess.STDOUT,check=True)
     estimate=json.loads(pose_file.read_text())
     if estimate.get('contract')!=f'camera_pose_{a.object}_v1':raise ValueError('Unexpected camera pose contract')
     target_pose_b=np.asarray(estimate['pose_b_wxyz'],dtype=np.float32)
     if target_pose_b.shape!=(7,) or not np.isfinite(target_pose_b).all():raise ValueError('Invalid camera target pose')
     row['camera_pose_file']=str(pose_file)
     row['camera_pose_sha256']=hashlib.sha256(pose_file.read_bytes()).hexdigest()
     row['camera_pose_source']=estimate['source']
    if a.template_replay:
     import h5py
     from scipy.spatial.transform import Rotation
     if a.live_camera_pose:
      import subprocess
      cam=env.scene['camera']
      depth=cam.data.output['distance_to_image_plane'][0].detach().cpu().numpy().squeeze()
      sensor_file=a.output/'live_front_sensor.npz'
      np.savez_compressed(sensor_file,
       rgb=cam.data.output['rgb'][0].detach().cpu().numpy()[...,:3].astype(np.uint8),
       depth=depth.astype(np.float32),
       intrinsics=cam.data.intrinsic_matrices[0].detach().cpu().numpy().astype(np.float32),
       position_b=ns['to_base_pos'](env,cam.data.pos_w[0]).detach().cpu().numpy().astype(np.float32),
       quaternion_b_ros=ns['to_base_quat'](env,cam.data.quat_w_ros[0]).detach().cpu().numpy().astype(np.float32))
      a.camera_pose_json=a.output/'live_camera_pose.json'
      root=_workspace_root()
      estimator=[str(root/'NSTC-Surgical-Instrument-Pick-Place/.venv-rfdetr/bin/python'),
       str(_repo_root()/'experiments/dp_target_relative/estimate_scissor_pose.py'),
       '--sensor-npz',str(sensor_file),'--output',str(a.camera_pose_json)]
      event('PERCEPTION','RF-DETR B + live front RGB-D pose estimate')
      with (a.output/'pose_estimator.log').open('w') as log:
       subprocess.run(estimator,stdout=log,stderr=subprocess.STDOUT,check=True)
     if a.template_replay==Path('AUTO'):
      if not a.camera_pose_json:raise ValueError('AUTO template requires a camera pose estimate')
      estimate=json.loads(a.camera_pose_json.read_text())
      target_for_selection=np.asarray(estimate['pose_b_wxyz'],dtype=float)
      train_dir=_workspace_root()/'datasets/DP datasets/NSTC/DP datasets/scissor/scissor/train/pick_policy/scissor'
      candidates=sorted(train_dir.glob('episode_*.h5'))
      if len(candidates)!=70:raise RuntimeError('Scissor training template inventory changed')
      def pose_distance(source_pose):
       qa=np.asarray(source_pose[3:],dtype=float);qb=target_for_selection[3:]
       qa/=np.linalg.norm(qa);qb/=np.linalg.norm(qb)
       angle=2*np.arccos(np.clip(abs(float(np.dot(qa,qb))),0,1))
       return float(np.linalg.norm(source_pose[:3]-target_for_selection[:3])+.025*angle)
      ranked=[]
      for path in candidates:
       with h5py.File(path) as f:
        if f.attrs['dataset_split']!='train':raise ValueError('Non-training template in bank')
        ranked.append((pose_distance(f['debug_gt/selected_object_pose_b'][0]),path))
      _,a.template_replay=min(ranked,key=lambda pair:pair[0])
      row['template_replay']=str(a.template_replay)
      event('RETRIEVAL','selected nearest training template from camera pose',template=str(a.template_replay))
     with h5py.File(a.template_replay) as f:
      if f.attrs['dataset_split']!='train' or f.attrs['target_object']!=a.object or f.attrs['policy_skill']!='pick':
       raise ValueError('Template must be training Pick of the same instrument')
      template_pose=f['debug_gt/selected_object_pose_b'][0]
      expert_actions=f['actions'][:].copy()
     if a.camera_pose_json:
      estimate=json.loads(a.camera_pose_json.read_text())
      if estimate.get('contract')!='camera_pose_scissor_v1' or a.object!='scissor':raise ValueError('Expected Scissor camera-pose contract')
      target_pose=np.asarray(estimate['pose_b_wxyz'],dtype=float)
      if target_pose.shape!=(7,) or not np.isfinite(target_pose).all():raise ValueError('Invalid camera pose estimate')
      row['camera_pose_file']=str(a.camera_pose_json)
      row['camera_pose_sha256']=hashlib.sha256(a.camera_pose_json.read_bytes()).hexdigest()
      row['camera_pose_source']=estimate['source']
     else:
      with h5py.File(a.snapshot) as f:target_pose=f['debug_gt/selected_object_pose_b'][0]
     def wxyz_to_xyzw(q):return np.asarray([q[1],q[2],q[3],q[0]])
     def xyzw_to_wxyz(q):return np.asarray([q[3],q[0],q[1],q[2]])
     delta=Rotation.from_quat(wxyz_to_xyzw(target_pose[3:]))*Rotation.from_quat(wxyz_to_xyzw(template_pose[3:])).inv()
     expert_actions[:,:3]=target_pose[:3]+delta.apply(expert_actions[:,:3]-template_pose[:3])
     for action in expert_actions:
      action[3:7]=xyzw_to_wxyz((delta*Rotation.from_quat(wxyz_to_xyzw(action[3:7]))).as_quat())
     row['template_source_sha256']=hashlib.sha256(a.template_replay.read_bytes()).hexdigest()
     row['template_target_translation_m']=float(np.linalg.norm(target_pose[:3]-template_pose[:3]))
     row['template_target_rotation_deg']=float(np.linalg.norm(delta.as_rotvec())*180/np.pi)
    # The actor sees only RGB/proprio/task. No expert waypoints are passed here.
    for step in range(min(a.max_steps,len(expert_actions)) if expert_actions is not None else a.max_steps):
     if not ns['simulation_app'].is_running():raise RuntimeError('Viewport closed')
     sensors=read_live_sensors(env,image_size=448) if not a.template_replay else None
     if a.learned_camera_pose:sensors['target_pose_b']=target_pose_b
     if known_slot_b is not None:sensors['target_slot_b']=known_slot_b
     action=expert_actions[step] if expert_actions is not None else actor[0].predict(which,a.object,sensors,reset=step==0)[0]
     if not np.isfinite(action).all():raise RuntimeError('Nonfinite model action')
     env.step(torch.as_tensor(action,dtype=torch.float32,device=env.device)[None])
     if step%5==0 or step==a.max_steps-1:
      front=env.scene['camera'].data.output['rgb'][0].detach().cpu().numpy()[...,:3].copy()
      top=env.scene['cam_top'].data.output['rgb'][0].detach().cpu().numpy()[...,:3].copy()
      video.append((step+1,front,top))
     obj,center,bottom,pts=measure(env);peak=max(peak,float(center[2]))
     ee=ns['get_ee_pos_w'](env).detach().cpu().numpy();q=ns['get_ee_quat_w'](env).detach().cpu().numpy()
     measured_fingers=robot.data.joint_pos[0,finger_ids].detach().cpu().numpy()
     current=evidence.local_anchor(ee,q,center)
     place_components=None
     if which=='pick':
      raised=center[2]-initial[2]>=.035 and bottom>=.025
      if raised and anchor is None:anchor=current.copy()
      held=raised and anchor is not None and np.linalg.norm(current-anchor)<=.025
      stable=stable+1 if held else 0
      success=stable>=12 and center[2]-initial[2]>=.10
     else:
      r=slot_measurement(obj,a.object,LAYOUT);centers.append(center.copy());centers=centers[-6:]
      settled=len(centers)==6 and np.ptp(centers,axis=0).max()<.002
      fingers=sensors['robot_proprio'][7:9]
      place_components=dict(inside_slot=bool(r['inside_slot']),center_error_m=float(r['center_error_m']),axis_error_deg=float(r['axis_error_deg']),bottom_error_m=float(abs(bottom-evidence.tray_z)),settled=bool(settled),fingers_open=bool(min(fingers)>.025),ee_clear=bool(ee[2]-center[2]>.06))
      success=bool(place_components['inside_slot'] and place_components['center_error_m']<=CENTER_TOL and place_components['axis_error_deg']<=ANGLE_TOL and place_components['bottom_error_m']<=.005 and place_components['settled'] and place_components['fingers_open'] and place_components['ee_clear'])
      stable=stable+1 if success else 0;success=stable>=12
     with (a.output/'actions.jsonl').open('a') as f:
      f.write(json.dumps(dict(actor=owner,skill=which,step=step,action=action.tolist(),object_center_for_scoring=center.tolist(),bottom_for_scoring=bottom,ee_position_w_for_scoring=ee.tolist(),fingers_after_action_m=measured_fingers.tolist(),tcp_object_center_distance_m=float(np.linalg.norm(ee-center)),place_components=place_components))+'\n')
     if step%25==0:event(owner,which,step=step)
     if success:
      if not video or video[-1][0]!=step+1:
       video.append((step+1,env.scene['camera'].data.output['rgb'][0].detach().cpu().numpy()[...,:3].copy(),env.scene['cam_top'].data.output['rgb'][0].detach().cpu().numpy()[...,:3].copy()))
      if which=='pick':
       row['lift_success']=True
       # Seed the original FSM's held-object monitor from achieved physical state.
       evidence.anchor=current.copy();evidence.lift_verified=True;evidence.peak_rise=peak-float(initial[2])
      else:
       check_preloaded(env,ns);row['place_success']=True
      event(owner,which+' SUCCESS',steps=step+1)
      return peak
    completed=min(a.max_steps,len(expert_actions)) if expert_actions is not None else a.max_steps
    row['failure']=which+'_template_replay_no_success' if a.template_replay else which+'_expert_replay_no_success' if a.expert_replay else which+'_timeout'
    event(owner,which+' FAILED',steps=completed);finish()
   ns['_dp_eval_skill']=skill;ns['_dp_eval_finish']=finish
   if a.snapshot:
    from evaluation.dp_snapshot import restore
    def snapshot_run(env):
     ns['_p4_env']=env
     row['reset']=restore(env,ns,a.snapshot,a.skill,event,expected_split=a.snapshot_split,
      target_perturb=(a.target_perturb_dx,a.target_perturb_dy,a.target_perturb_yaw_deg))
     skill(env,a.skill)
     finish()
    ns['_dp_run_snapshot']=snapshot_run
   # Replace the original function beneath the recorder's evidence wrapper.
   raw=inspect.unwrap(ns['pick_and_place_object'])
   source=inspect.getsource(raw)
   compiled=replace_policy_windows(source,a.skill)
   temp=dict(ns);exec(compiled,temp)
   raw.__code__=temp['pick_and_place_object'].__code__
   # Label unchanged expert approach/transport; these do not count as DP actions.
   original_step=ns['step_dynamic']
   def fsm(*args,**kw):
    name=args[2] if len(args)>2 else kw['name']
    event('FSM',name)
    return original_step(*args,**kw)
   ns['step_dynamic']=fsm
   for method in ['step_smooth_grip','step_hold_const_grip']:
    original_grip=ns[method]
    def tagged(*args,_fn=original_grip,**kw):
     event('FSM',args[2] if len(args)>2 else kw['name'])
     return _fn(*args,**kw)
    ns[method]=tagged
   guard=ns['_p4_attempt_guard']
   def once(attempt,*args):
    if attempt>=1:
     row['failure']=ns.get('_p4_failure_message','setup_or_FSM_failure');finish()
    return guard(attempt,*args)
   ns['_p4_attempt_guard']=once
   if a.mode=='free':
    old=ns['phase3_ensure_all_5_objects_in_spawn']
    def spawn(*args,**kw):
     result=old(*args,**kw)
     from phase4_cell_spawn import instrument_envelopes
     from phase4_session import target_workspace_ok
     rng=args[2];placed=[];envs=instrument_envelopes()
     for name in envs:
      radius=envs[name]['radius_m']+.01
      for _ in range(10000):
       x=rng.uniform(LAYOUT['grid_x'][0]+radius,LAYOUT['grid_x'][1]-radius)
       y=rng.uniform(LAYOUT['grid_y'][0]+radius,LAYOUT['grid_y'][1]-radius)
       if name==a.object and not target_workspace_ok(x,y,LAYOUT):continue
       if all(np.hypot(x-px,y-py)>radius+pr for px,py,pr in placed):break
      else:raise RuntimeError('Free placement sampling exhausted')
      placed.append((x,y,radius));d=result[name]
      d['center_x' if name=='scalpel' else 'x']=float(x);d['center_y' if name=='scalpel' else 'y']=float(y)
     chosen=result[a.object]
     result['grid'].update(x=chosen['center_x' if a.object=='scalpel' else 'x'],y=chosen['center_y' if a.object=='scalpel' else 'y'],cell_id=-1,spawn_contract='free_nonoverlap_workspace_v1')
     ns['_p4_cell_assignments']={}
     result['grid']['object_cell_assignments']='{}'
     row['requested_spawn']=result
     return result
    ns['phase3_ensure_all_5_objects_in_spawn']=spawn
  shared.apply_workspace_globals=setup
  root=Path(__file__).resolve().parents[1];backend=root/'backends'/get_handler(a.object).backend
  if a.snapshot:
   import h5py
   with h5py.File(a.snapshot) as f:
    scene_seed=json.loads(f.attrs['domain_randomization'])['session_seed']
   os.environ['P4_RANDOMIZATION_SEED']=str(scene_seed)
  else:scene_seed=a.seed
  os.environ.update(P4_CAMERA_SIZE='448',P4_DATASET_PURPOSE='dp',P4_RANDOMIZATION_SEED=str(scene_seed),P4_MAX_ATTEMPTS='1')
  sys.argv=[str(backend),'--enable_cameras','--episodes','1','--out_dir',str(a.output/'scratch')]
  if a.headless:sys.argv.append('--headless')
  loaded=runpy.run_path(str(backend),run_name='dp_evaluation_backend');app=loaded['simulation_app']
  namespace=loaded['main'].__globals__
  namespace['RANDOM_SEED']=scene_seed
  if a.snapshot:
   import ast
   tree=ast.parse(inspect.getsource(loaded['main']))
   fn=tree.body[0]
   boundary=next(i for i,node in enumerate(fn.body) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='base_grip_quat' for t in node.targets))
   fn.body=fn.body[:boundary]+ast.parse('_dp_run_snapshot(env)').body
   ast.fix_missing_locations(tree)
   exec(compile(tree,'<skill-only scene initialization>','exec'),namespace)
  namespace['main']()
 except TrialFinished:pass
 except BaseException as exc:
  row['failure']=type(exc).__name__+': '+str(exc)
  import traceback;traceback.print_exc()
 finally:
  if video:
   try:
    from evaluation.dp_video import save_gif
    contract=row.get('input_contracts',{}).get(a.skill)
    model_label=('TRAIN template + ORACLE object pose' if a.template_replay
                 else 'EXPERT replay diagnostic' if a.expert_replay
                 else 'RF-DETR B + trained target-relative Scissor DP' if contract=='target_relative_dp_scissor_v1'
                 else 'Trained target-relative '+a.object+' '+a.skill+' DP' if contract and contract.startswith('target_relative_dp_')
                 else 'V3 instrument-only gripper pilot' if contract=='dp_gripper_focused_v3'
                 else 'V2 separate gripper' if contract=='dp_gripper_transition_v2'
                 else 'V1 coupled')
    row['gif']=save_gif(video,a.output,a.skill,a.object,bool(row['lift_success'] if a.skill=='pick' else row['place_success']),len(video),model_label=model_label)
   except Exception as exc:row['gif_error']=str(exc)
  row['trial_complete']=True
  (a.output/'result.json').write_text(json.dumps(row,indent=2,default=str))
  print(json.dumps(row,default=str),flush=True)
  import threading
  # Bound actor and Kit shutdown after the result/GIF are already durable.
  timer=threading.Timer(45,lambda:os._exit(0));timer.daemon=True;timer.start()
  try:
   if actor[0]:actor[0].close()
   if app:
    if ns and ns.get('_p4_env') is not None:ns['_p4_env'].close()
    app.close()
  finally:timer.cancel()
if __name__=='__main__':main()
