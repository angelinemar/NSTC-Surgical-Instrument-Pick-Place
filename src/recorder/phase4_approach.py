"""Joint-limit-aware differential IK for P4 expert trajectories.

Same absolute TCP action and observed frame. No teleport, force-success or
relaxed Cartesian gate. All expert phases share the same bounded controller.
"""
import numpy as np


def shift_tcp_jacobian(body_jacobian,lever_in_base):
    """J_tcp_v = J_hand_v + omega x r, with BOTH terms in base frame."""
    import torch
    result=body_jacobian.clone()
    lever=lever_in_base.unsqueeze(-1).expand_as(result[...,3:,:])
    result[...,:3,:]-=torch.linalg.cross(lever,result[...,3:,:],dim=-2)
    return result


def corrected_action_type():
    """Fix TCP-offset Jacobian frame locally, without editing IsaacLab itself."""
    from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
    from isaaclab.utils import math as math_utils
    import torch
    class CorrectTCPDifferentialIKAction(DifferentialInverseKinematicsAction):
        def _compute_frame_jacobian(self):
            # PhysX Jacobian is world-frame and at the rigid-body COM. Both the
            # angular velocity AND lever arm must be expressed in base frame.
            j=self.jacobian_w.clone()
            inverse=math_utils.quat_inv(self._asset.data.root_quat_w)
            rotation=math_utils.matrix_from_quat(inverse)
            j[:,:3,:]=torch.bmm(rotation,j[:,:3,:])
            j[:,3:,:]=torch.bmm(rotation,j[:,3:,:])
            hand_b=math_utils.quat_mul(inverse,self._asset.data.body_quat_w[:,self._body_idx])
            com=self._asset.data.body_com_pos_b[:,self._body_idx]
            offset=self._offset_pos if self.cfg.body_offset is not None else torch.zeros_like(com)
            lever=math_utils.quat_apply(hand_b,offset-com)
            j=shift_tcp_jacobian(j,lever)
            return j
    CorrectTCPDifferentialIKAction.__qualname__='CorrectTCPDifferentialIKAction'
    globals()['CorrectTCPDifferentialIKAction']=CorrectTCPDifferentialIKAction
    return CorrectTCPDifferentialIKAction


def bounded_delta(jacobian,error,q,limits,step=.20):
    from scipy.optimize import lsq_linear
    q=np.asarray(q,dtype=float);limits=np.asarray(limits,dtype=float)
    lower=limits[:,0]+.005;upper=limits[:,1]-.005
    anchor=np.clip(q,lower+1e-7,upper-1e-7)
    weights=np.array([1.,1.,1.,.3,.3,.3])
    a=np.vstack((weights[:,None]*jacobian,.03*np.eye(len(q))))
    b=np.r_[weights*error,np.zeros(len(q))]
    lo=np.maximum(lower-anchor,-step);hi=np.minimum(upper-anchor,step)
    solution=lsq_linear(a,b,bounds=(lo,hi),method='bvls',tol=1e-8,max_iter=50)
    if not solution.success or not np.isfinite(solution.x).all():
        raise RuntimeError('Bounded IK failed to find a finite incremental solution')
    return np.clip(anchor+solution.x,lower,upper)


def precision_integral(previous, position_error, rotation_error, stationary):
    """Bounded servo-bias correction from measured TCP only, never object pose.

    Reset on moving commands/far errors to avoid windup during transport or contact.
    Actual physical pose gates are evaluated against the original requested pose.
    """
    previous = np.asarray(previous, dtype=float)
    error = np.asarray(position_error, dtype=float)
    if (not stationary or not np.isfinite(error).all()
            or np.linalg.norm(error) > .01 or np.linalg.norm(rotation_error) > .05):
        return np.zeros(3)
    updated = previous + .08 * error
    return updated * min(1., .004 / max(np.linalg.norm(updated), 1e-12))


def set_precision_skill(env, skill):
    """An explicit pick/place command, not an expert stage or object observation."""
    if skill not in ('pick', 'place'):
        raise ValueError('Unknown controller skill')
    controller = env.action_manager.get_term('arm_action')._ik_controller
    if getattr(controller, '_p4_precision_skill', None) != skill:
        controller._p4_previous_desired = None
        controller._p4_integral = np.zeros((env.num_envs if hasattr(env, 'num_envs') else 1, 3))
    controller._p4_precision_skill = skill


def rotation_arc_cost(jacobian,position_error,rotation_vector,q,limits):
    """Local joint-limit lookahead used only to choose near-half-turn arcs."""
    weights=np.array([1.,1.,1.,.3,.3,.3])
    a=np.vstack((weights[:,None]*jacobian,.03*np.eye(len(q))))
    delta=np.linalg.lstsq(a,np.r_[weights*np.r_[position_error,rotation_vector],np.zeros(len(q))],rcond=None)[0]
    predicted=q+delta
    span=np.maximum(limits[:,1]-limits[:,0],.01)
    violation=np.maximum(limits[:,0]+.005-predicted,0)+np.maximum(predicted-(limits[:,1]-.005),0)
    return float(np.sum((violation/span)**2))


def select_long_arc(env,start,goal,q0,q1):
    """Same final orientation; choose the other half-turn if less limit risk.

    This is local IK lookahead, not a collision-free global planner. Actual
    position/orientation, grasp and slip gates remain authoritative.
    """
    import torch
    from isaaclab.utils.math import quat_inv,quat_mul,quat_apply
    a=q0/torch.linalg.norm(q0);b=q1/torch.linalg.norm(q1)
    if float(torch.dot(a,b))<0:b=-b
    relative=quat_mul(b[None],quat_inv(a[None]))[0]
    sine=torch.linalg.norm(relative[1:]);angle=2*torch.atan2(sine,relative[0])
    degrees=float(angle)*180/np.pi
    if degrees<150:return False,degrees
    term=env.action_manager.get_term('arm_action');robot=env.scene['robot']
    inverse=quat_inv(robot.data.root_quat_w[:1])
    axis=quat_apply(inverse,(relative[1:]/sine)[None])[0].detach().cpu().numpy()
    position=quat_apply(inverse,(goal-start)[None])[0].detach().cpu().numpy()
    jac=term._compute_frame_jacobian()[0].detach().cpu().numpy()
    q=robot.data.joint_pos[0,term._joint_ids].detach().cpu().numpy()
    limits=robot.data.soft_joint_pos_limits[0,term._joint_ids].detach().cpu().numpy()
    short_cost=rotation_arc_cost(jac,position,axis*float(angle),q,limits)
    long_cost=rotation_arc_cost(jac,position,-axis*(2*np.pi-float(angle)),q,limits)
    use_long=long_cost+1e-5<short_cost
    print(f'[P4 WRIST ARC] short={degrees:.1f}deg risk={short_cost:.5f}; other={360-degrees:.1f}deg risk={long_cost:.5f}; selected={"other" if use_long else "short"}; same final quaternion',flush=True)
    return use_long,360-degrees if use_long else degrees


def long_arc_slerp(a,b,t):
    import torch
    a=a/torch.linalg.norm(a);b=b/torch.linalg.norm(b)
    if float(torch.dot(a,b))>=0:b=-b
    angle=torch.acos(torch.clamp(torch.dot(a,b),-1.,1.))
    if abs(float(torch.sin(angle)))<1e-6:
        raise ValueError('Long-arc interpolation requires a nondegenerate half turn')
    result=(torch.sin((1-t)*angle)*a+torch.sin(t*angle)*b)/torch.sin(angle)
    return result/torch.linalg.norm(result)


def configure(env,enabled):
    """Activate only per near-base episode; reset activation at every hover."""
    term=env.action_manager.get_term('arm_action')
    controller=term._ik_controller
    if not hasattr(controller,'_p4_original_compute'):
        import torch
        from isaaclab.utils.math import compute_pose_error
        controller._p4_original_compute=controller.compute
        def compute(ee_pos,ee_quat,jacobian,joint_pos):
            if not controller._p4_bounded_active:
                return controller._p4_original_compute(ee_pos,ee_quat,jacobian,joint_pos)
            dp,dr=compute_pose_error(ee_pos,ee_quat,controller.ee_pos_des,controller.ee_quat_des,rot_error_type='axis_angle')
            errors=torch.cat((dp,dr),dim=1).detach().cpu().numpy()
            desired=controller.ee_pos_des.detach().cpu().numpy()
            previous_desired=getattr(controller,'_p4_previous_desired',None)
            integral=getattr(controller,'_p4_integral',np.zeros_like(desired))
            for i in range(len(desired)):
                stationary=(getattr(controller, '_p4_precision_skill', 'pick') == 'place'
                            and previous_desired is not None
                            and np.linalg.norm(desired[i]-previous_desired[i]) < 1e-7)
                integral[i]=precision_integral(integral[i],errors[i,:3],errors[i,3:],stationary)
            controller._p4_previous_desired=desired.copy()
            controller._p4_integral=integral
            errors[:,:3]+=integral
            limits=term._asset.data.soft_joint_pos_limits[:,term._joint_ids].detach().cpu().numpy()
            q=joint_pos.detach().cpu().numpy();j=jacobian.detach().cpu().numpy()
            result=np.stack([bounded_delta(j[i],errors[i],q[i],limits[i]) for i in range(len(q))])
            return joint_pos.new_tensor(result)
        controller.compute=compute
    controller._p4_bounded_active=bool(enabled)
    controller._p4_previous_desired=None
    controller._p4_integral=np.zeros((env.num_envs if hasattr(env,'num_envs') else 1,3))
    controller._p4_precision_skill='pick'
    if enabled:
        print('[P4 APPROACH] bounded IK: joint-limit constrained commands for all phases, same TCP/actions and success gates',flush=True)
