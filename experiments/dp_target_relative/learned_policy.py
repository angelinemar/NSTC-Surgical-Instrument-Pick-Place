"""Scissor-only target-relative Diffusion Policy, trained from Pick episodes.

The target pose is a training label here. At inference it must be estimated from
RGB-D and RF-DETR; validation/test simulator object poses are forbidden inputs.
"""
import argparse
import csv
import json
import random
import time
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from robomimic.models.diffusion_policy_nets import ConditionalUnet1D
from scipy.spatial.transform import Rotation
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


try:
    from .workspace_root import workspace_root
except ImportError:
    from workspace_root import workspace_root
ROOT = workspace_root()
DATA = ROOT/'datasets/DP datasets/NSTC/DP datasets/scissor/scissor'
HORIZON = 16


def rotation(q):
    return Rotation.from_quat(np.asarray(q)[..., [1, 2, 3, 0]])


def wxyz(r):
    q = r.as_quat()
    q = q[..., [3, 0, 1, 2]]
    return np.where(q[..., :1] < 0, -q, q)


def relative_state_action(proprio, actions, target):
    """Convert base-frame EE state/action to a fixed initial target frame."""
    base_from_target = rotation(target[3:])
    target_from_base = base_from_target.inv()
    ee_xyz = target_from_base.apply(proprio[:, 9:12]-target[:3])
    ee_quat = wxyz(target_from_base*rotation(proprio[:, 12:16]))
    action_xyz = target_from_base.apply(actions[:, :3]-target[:3])
    action_quat = wxyz(target_from_base*rotation(actions[:, 3:7]))
    state = np.concatenate((proprio[:, :9], ee_xyz, ee_quat), axis=1).astype(np.float32)
    motion = np.concatenate((action_xyz, action_quat), axis=1).astype(np.float32)
    close = (actions[:, 7] < 0).astype(np.float32)
    return state, motion, close


def load_split(split):
    files = sorted((DATA/split/'pick_policy/scissor').glob('episode_*.h5'))
    expected = {'train':70, 'valid':20, 'test':10}[split]
    if len(files) != expected:
        raise RuntimeError(f'{split}: expected {expected} Scissor episodes, found {len(files)}')
    episodes = []
    for path in files:
        with h5py.File(path) as h:
            if h.attrs['dataset_split'] != split or h.attrs['target_object'] != 'scissor' or h.attrs['policy_skill'] != 'pick':
                raise ValueError(f'Wrong episode contract: {path}')
            proprio = h['observations/robot_proprio'][:]
            actions = h['actions'][:]
            target = h['debug_gt/selected_object_pose_b'][0]
        if len(proprio) != len(actions) or len(actions) < HORIZON:
            raise ValueError(f'Misaligned or short episode: {path}')
        switches = np.flatnonzero((actions[1:, 7] < 0) & (actions[:-1, 7] >= 0))+1
        if len(switches) != 1:
            raise ValueError(f'Expected one CLOSE event: {path}')
        state, motion, close = relative_state_action(proprio, actions, target)
        episodes.append(dict(path=str(path), state=state, motion=motion, close=close,
                             close_at=int(switches[0])))
    return episodes


def fit_stats(episodes):
    states = np.concatenate([e['state'] for e in episodes])
    motion = np.concatenate([e['motion'] for e in episodes])
    return dict(state_mean=states.mean(0).tolist(), state_std=np.maximum(states.std(0), .01).tolist(),
                action_center=((motion.max(0)+motion.min(0))/2).tolist(),
                action_scale=np.maximum((motion.max(0)-motion.min(0))/2, .01).tolist())


class Windows(Dataset):
    def __init__(self, episodes, stats):
        self.episodes = episodes
        self.indices = [(i, t) for i, e in enumerate(episodes)
                        for t in range(len(e['close'])-HORIZON+1)]
        self.stats = {k:np.asarray(v,dtype=np.float32) for k,v in stats.items()}
        self.weights = np.asarray([24. if t==episodes[i]['close_at'] else
                                   10. if -4 <= t-episodes[i]['close_at'] <= 4 else
                                   3. if -12 <= t-episodes[i]['close_at'] <= 12 else 1.
                                   for i,t in self.indices],dtype=np.float64)

    def __len__(self):return len(self.indices)

    def __getitem__(self, index):
        i,t = self.indices[index]
        e = self.episodes[i]
        state = e['state'][[max(0,t-1),t]]
        state = (state-self.stats['state_mean'])/self.stats['state_std']
        motion = (e['motion'][t:t+HORIZON]-self.stats['action_center'])/self.stats['action_scale']
        return dict(state=torch.from_numpy(state.astype(np.float32)),
                    motion=torch.from_numpy(motion.astype(np.float32)),
                    close=torch.from_numpy(e['close'][t:t+HORIZON].copy()),
                    offset=torch.tensor(t-e['close_at'],dtype=torch.int16))


class RelativeDiffusionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(32,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        self.denoiser = ConditionalUnet1D(input_dim=7,global_cond_dim=128,
            diffusion_step_embed_dim=64,down_dims=[64,128,256])
        self.gripper = nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,HORIZON))
        self.scheduler = DDPMScheduler(num_train_timesteps=100,beta_schedule='squaredcos_cap_v2',
                                       clip_sample=True,prediction_type='epsilon')

    def losses(self,batch):
        context = self.context(batch['state'].flatten(1))
        target = batch['motion']
        noise = torch.randn_like(target)
        steps = torch.randint(self.scheduler.config.num_train_timesteps,(len(target),),device=target.device)
        pred = self.denoiser(self.scheduler.add_noise(target,noise,steps),steps,global_cond=context)
        motion = F.mse_loss(pred,noise)
        logits = self.gripper(context)
        element = F.binary_cross_entropy_with_logits(logits,batch['close'],reduction='none')
        offset = batch['offset']
        event_weight = torch.where(offset==0,12.,torch.where((offset>=-8)&(offset<0),5.,1.))
        horizon_weight = logits.new_tensor([4.,2.,1.,1.,1.,1.,1.,1.,.5,.5,.5,.5,.5,.5,.5,.5])
        grip = (element*horizon_weight*event_weight[:,None]).sum()/(horizon_weight.sum()*event_weight.sum())
        return motion+grip,dict(motion=motion.detach(),gripper=grip.detach())

    @torch.no_grad()
    def sample(self,state,stats,steps=100):
        context=self.context(state.flatten(1))
        trajectory=torch.randn((len(state),HORIZON,7),device=state.device)
        self.scheduler.set_timesteps(steps)
        for step in self.scheduler.timesteps:
            eps=self.denoiser(trajectory,step.to(state.device),global_cond=context)
            trajectory=self.scheduler.step(eps,step,trajectory).prev_sample
        center=trajectory.new_tensor(stats['action_center'])
        scale=trajectory.new_tensor(stats['action_scale'])
        motion=trajectory*scale+center
        motion[...,3:7]=F.normalize(motion[...,3:7],dim=-1)
        close=self.gripper(context).sigmoid()>.5
        return motion,close


class Runtime:
    def __init__(self,checkpoint,device='cuda'):
        data=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if data.get('input_contract')!='target_relative_dp_scissor_v1' or data.get('smoke_only'):
            raise ValueError('Expected a full Scissor target-relative DP checkpoint')
        self.model=RelativeDiffusionPolicy().to(device)
        self.model.load_state_dict(data['model']);self.model.eval()
        self.stats=data['stats'];self.device=device;self.skill='pick';self.image_size=448
        self.controller_contract=data['controller_contract']
        self.history=deque(maxlen=2);self.pose=None

    def set_target(self,instrument):
        if instrument!='scissor':raise ValueError('This policy was trained only for Scissor')
        self.history.clear();self.pose=None

    def observe(self,sensors):
        if 'target_pose_b' not in sensors:raise ValueError('Missing camera-estimated target_pose_b')
        pose=np.asarray(sensors['target_pose_b'],dtype=np.float32)
        if pose.shape!=(7,) or not np.isfinite(pose).all():raise ValueError('Invalid target pose')
        if self.pose is None:self.pose=pose.copy()
        elif np.linalg.norm(self.pose-pose)>1e-4:raise ValueError('Target lock changed during Pick')
        proprio=np.asarray(sensors['robot_proprio'],dtype=np.float32)
        if proprio.shape!=(16,) or not np.isfinite(proprio).all():raise ValueError('Invalid robot state')
        state,_,_=relative_state_action(proprio[None],np.r_[proprio[9:12],proprio[12:16],1.][None],self.pose)
        self.history.append(state[0])

    @torch.no_grad()
    def predict(self,inference_steps=100):
        if not self.history or self.pose is None:raise ValueError('Observe before prediction')
        states=np.stack([self.history[0],self.history[-1]])
        norm=(states-np.asarray(self.stats['state_mean']))/np.asarray(self.stats['state_std'])
        state=torch.as_tensor(norm,dtype=torch.float32,device=self.device)[None]
        motion,close=self.model.sample(state,self.stats,inference_steps)
        motion=motion[0].cpu().numpy();close=close[0].cpu().numpy()
        base=rotation(self.pose[3:])
        xyz=base.apply(motion[:,:3])+self.pose[:3]
        q=wxyz(base*rotation(motion[:,3:7]))
        actions=np.concatenate((xyz,q,np.where(close,-1.,1.)[:,None]),axis=1).astype(np.float32)
        return actions,None


def train():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--patience',type=int,default=12)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.output.exists():p.error('Choose a new output directory')
    if not torch.cuda.is_available():p.error('CUDA required')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    train_episodes=load_split('train');valid_episodes=load_split('valid')
    stats=fit_stats(train_episodes)
    training=Windows(train_episodes,stats);validation=Windows(valid_episodes,stats)
    sampler=WeightedRandomSampler(training.weights,len(training),replacement=True,
                                  generator=torch.Generator().manual_seed(a.seed))
    tr=DataLoader(training,batch_size=a.batch_size,sampler=sampler,num_workers=0,pin_memory=True)
    va=DataLoader(validation,batch_size=a.batch_size,shuffle=False,num_workers=0,pin_memory=True)
    model=RelativeDiffusionPolicy().cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-5)
    a.output.mkdir(parents=True)
    config=dict(variant='target_relative_dp_scissor_v1',train_episodes=70,valid_episodes=20,test_episodes=10,
                train_windows=len(training),valid_windows=len(validation),epochs=a.epochs,batch_size=a.batch_size,
                patience=a.patience,lr=a.lr,seed=a.seed,smoke=a.smoke,
                target_pose_training_source='train-only simulator label',
                target_pose_inference_source='RF-DETR B + live RGB-D shape registration',
                action_representation='target-relative XYZ + quaternion; separate CLOSE head')
    (a.output/'training_config.json').write_text(json.dumps(config,indent=2))
    best=float('inf');bad=0;best_epoch=0
    def epoch(loader,training_mode,limit=None):
        model.train(training_mode);total=0.;parts={};count=0
        for j,batch in enumerate(loader):
            if limit is not None and j>=limit:break
            batch={k:v.cuda(non_blocking=True) for k,v in batch.items()}
            with torch.set_grad_enabled(training_mode),torch.autocast('cuda',dtype=torch.bfloat16):
                loss,detail=model.losses(batch)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            if training_mode:
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            n=len(batch['state']);count+=n;total+=float(loss)*n
            for k,v in detail.items():parts[k]=parts.get(k,0.)+float(v)*n
        return dict(loss=total/max(1,count),**{k:v/max(1,count) for k,v in parts.items()})
    for ep in range(1,(2 if a.smoke else a.epochs+1)):
        start=time.monotonic()
        train_metrics=epoch(tr,True,2 if a.smoke else None)
        valid_metrics=epoch(va,False,2 if a.smoke else None)
        score=valid_metrics['loss'];improved=score<best-1e-4
        if improved:best=score;best_epoch=ep;bad=0
        else:bad+=1
        checkpoint=dict(input_contract='target_relative_dp_scissor_v1',model=model.state_dict(),stats=stats,
                        config=config,epoch=ep,skill='pick',policy_image_size=448,
                        controller_contract=['tcp_frame_correct_bounded_place_integral_v6'],smoke_only=a.smoke)
        torch.save(checkpoint,a.output/'checkpoint_latest.pt')
        if improved:torch.save(checkpoint,a.output/'checkpoint_best.pt')
        row=dict(epoch=ep,train_loss=train_metrics['loss'],valid_loss=score,train_motion=train_metrics['motion'],
                 valid_motion=valid_metrics['motion'],train_gripper=train_metrics['gripper'],
                 valid_gripper=valid_metrics['gripper'],best_epoch=best_epoch,bad_epochs=bad,seconds=time.monotonic()-start)
        with (a.output/'metrics.csv').open('a',newline='') as f:
            w=csv.DictWriter(f,row.keys())
            if f.tell()==0:w.writeheader()
            w.writerow(row)
        print(json.dumps(row),flush=True)
        if bad>=a.patience:break
    result=dict(training_complete=True,model='target_relative_dp_scissor_v1',best_epoch=best_epoch,
                epochs_completed=ep,checkpoint=str(a.output/'checkpoint_best.pt'),
                smoke_only=a.smoke,closed_loop_verified=False)
    (a.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':train()
