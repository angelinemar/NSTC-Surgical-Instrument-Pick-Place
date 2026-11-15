"""Restore held-out skill initial conditions; no expert trajectories."""
import json
import numpy as np

def restore(env,ns,path,skill,event,expected_split='test',target_perturb=(0.0,0.0,0.0)):
 import h5py,torch
 from isaaclab.utils.math import combine_frame_transforms,quat_mul
 from phase4_grasp_validation import rigid_world_points
 from phase4_tray_slots import floor_heights,SlotEvidence,check_preloaded
 from phase4_scene import LAYOUT
 with h5py.File(path) as f:
  if expected_split not in ('valid','test') or f.attrs['dataset_split']!=expected_split or f.attrs['policy_skill']!=skill:
   raise ValueError(f'Expected {expected_split} matching skill')
  target=str(f.attrs['target_object'])
  if target!=ns['PHASE3_TARGET_OBJECT']:raise ValueError('Target mismatch')
  state=f['observations/robot_proprio'][0]
  target_b=f['debug_gt/selected_object_pose_b'][0]
  settled=json.loads(f.attrs['spawn_settled_all'])
  preload=json.loads(f.attrs['initial_tray_objects'])
 event('RESET','restore saved joint/object poses; no FSM actions')
 env.reset()
 robot=env.scene['robot']
 ids=[robot.joint_names.index(n) for n in ['panda_joint'+str(i) for i in range(1,8)]+['panda_finger_joint1','panda_finger_joint2']]
 q=robot.data.default_joint_pos.clone();q[0,ids]=q.new_tensor(state[:9])
 robot.write_joint_state_to_sim(q,torch.zeros_like(q))
 hold=q.clone()
 if skill=='place':hold[0,ids[7:9]]=0.0
 robot.set_joint_position_target(hold)
 robot.set_joint_velocity_target(torch.zeros_like(q));env.action_manager.reset()
 ns['_p4_tray_objects']=preload
 from src.recorder.scene_clutter import park
 park(env,ns)
 for name,row in settled.items():
  if name==target:continue
  key=name if name.startswith('clutter_') else ns['phase3_scene_key_for_object'](name)
  obj=env.scene[key]
  pose=q.new_tensor([[*row['position_w'],*row['quaternion_wxyz']]])
  obj.write_root_link_pose_to_sim(pose);obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_state_w[:,7:13]))
 obj=env.scene[ns['phase3_scene_key_for_object'](target)]
 b=q.new_tensor(target_b)
 dx,dy,yaw_deg=(float(v) for v in target_perturb)
 if abs(dx)>.025 or abs(dy)>.025 or abs(yaw_deg)>30:
  raise ValueError('Target perturbation exceeds audited ±2.5 cm / ±30° range')
 b[0]+=dx;b[1]+=dy
 yaw=np.deg2rad(yaw_deg)
 delta=b.new_tensor([np.cos(yaw/2),0,0,np.sin(yaw/2)])
 b[3:]=quat_mul(delta[None],b[3:][None])[0]
 pos,quat=combine_frame_transforms(robot.data.root_pos_w[:1],robot.data.root_quat_w[:1],b[:3][None],b[3:][None])
 pose=torch.cat([pos,quat],dim=-1)
 obj.write_root_pose_to_sim(pose);obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_state_w[:,7:13]))
 env.scene.write_data_to_sim();env.sim.forward();env.scene.update(env.physics_dt)
 initial=obj.data.root_pos_w[0].clone()
 # Reset validation holds measured joint positions, not an expert skill.
 for _ in range(20):
  robot.set_joint_position_target(hold);env.scene.write_data_to_sim()
  env.sim.step();env.scene.update(env.physics_dt)
 drift=float(torch.linalg.norm(obj.data.root_pos_w[0]-initial))
 from training.runtime import read_live_sensors
 observed=read_live_sensors(env,448)['robot_proprio']
 diagnostics=dict(target_drift_m=drift,recorded_proprio=state.tolist(),restored_proprio=observed.tolist(),joint_names=robot.joint_names,
  ee_position_error_m=float(np.linalg.norm(observed[9:12]-state[9:12])),
  arm_joint_max_error_rad=float(np.max(np.abs(observed[:7]-state[:7]))),
  gripper_hold='closed actuator target (0m), preserves grasp force' if skill=='place' else 'recorded opening')
 from pathlib import Path
 report=Path(ns['args_cli'].out_dir).parent/'reset_diagnostics.json'
 report.write_text(json.dumps(diagnostics,indent=2))
 if diagnostics['ee_position_error_m']>.01:raise RuntimeError('RESET_INVALID: robot TCP differs from recorded start')

 if skill=='place' and drift>.005:raise RuntimeError(f'RESET_INVALID: held object drifted {drift:.4f}m before policy')
 if skill=='pick' and drift>.01:raise RuntimeError(f'RESET_INVALID: target drifted {drift:.4f}m before policy')
 floor_heights(ns)
 check_preloaded(env,ns,capture=True)
 ns['_p4_evidence']=SlotEvidence(env,ns,LAYOUT)
 for _ in range(5):env.sim.render()
 event('RESET','valid initial state',target_drift_m=drift)
 return dict(source=str(path),target_drift_m=drift,target_perturb_dx_m=dx,target_perturb_dy_m=dy,target_perturb_yaw_deg=yaw_deg,
  velocities='zeroed; not exact physics checkpoint',contact_warmup_physics_steps=20)
