"""Canonical resets during preparation, never during recorded motion."""
import math
import numpy as np

def reset_robot(env):
    import torch
    robot=env.scene['robot']
    q=robot.data.default_joint_pos.clone(); qd=torch.zeros_like(robot.data.default_joint_vel)
    robot.reset(); robot.write_joint_state_to_sim(q,qd)
    robot.set_joint_position_target(q); robot.set_joint_velocity_target(qd)
    env.action_manager.reset(); env.scene.write_data_to_sim(); env.sim.forward()
    env.scene.update(env.physics_dt)
    env.scene['ee_frame'].update(env.physics_dt,force_recompute=True)
    print('[P4 RESET] home robot, zero velocities, cleared old action',flush=True)

def cache_templates(env,ns):
    if ns.get('_p4_spawn_templates'): return
    from phase4_tray_slots import ORDER
    from phase4_grasp_validation import rigid_world_points
    spawn=ns['_p4_force_args'][1]; templates={}
    for name in ORDER:
        obj=env.scene[ns['phase3_scene_key_for_object'](name)]; rigid_world_points(obj)
        templates[name]=dict(yaw=float(spawn[name]['yaw_deg']),
            quat=obj.data.root_link_quat_w[0].detach().cpu().numpy().copy(),points=obj._p4_link_corners.copy())
    ns['_p4_spawn_templates']=templates
    ns['_p4_canonical_scalpel_mode']=ns['_p4_force_args'][2] or 'BROAD_FLAT'

def table_pose(template,x,y,yaw):
    from phase4_tray_slots import mul
    from phase4_grasp_validation import rotate
    delta=math.radians(yaw-template['yaw'])
    q=mul([math.cos(delta/2),0,0,math.sin(delta/2)],template['quat'])
    pts=rotate(q,template['points'])
    mid=rotate(q,(template['points'].min(axis=0)+template['points'].max(axis=0))/2)
    return np.array([x-mid[0],y-mid[1],.0015-pts[:,2].min(),*q])

def restore_objects(env,ns,spawn):
    import torch
    for name,t in ns['_p4_spawn_templates'].items():
        p=spawn[name]; obj=env.scene[ns['phase3_scene_key_for_object'](name)]
        pose=table_pose(t,p['center_x' if name=='scalpel' else 'x'],p['center_y' if name=='scalpel' else 'y'],float(p['yaw_deg']))
        obj.write_root_link_pose_to_sim(torch.tensor([pose.tolist()],device=env.device,dtype=obj.data.root_state_w.dtype))
        obj.write_root_velocity_to_sim(torch.zeros_like(obj.data.root_state_w[:,7:13])); obj.update(env.physics_dt)
    from phase4_tray_slots import prepare_tray
    prepare_tray(env,ns)
    env.scene.write_data_to_sim(); env.sim.forward()
    print('[P4 RESET] canonical body poses, zero velocities, tray restored before physics settling',flush=True)
