"""Joint RGB segmentation and conditional diffusion, with no simulator inputs."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from robomimic.models.diffusion_policy_nets import ConditionalUnet1D

if __package__:
    from .export_sensor_only import CAMERAS, assert_sensor_inputs
else:
    from export_sensor_only import CAMERAS, assert_sensor_inputs


def fit_training_stats(path):
    with h5py.File(path, 'r') as f:
        ids = [v.decode() for v in f['mask/train'][:]]
        if not ids:
            raise ValueError('Training split is empty')
        actions = np.concatenate([f['data/' + key + '/actions'][:] for key in ids])
        proprio = np.concatenate([f['data/' + key + '/obs/robot_proprio'][:] for key in ids])
    return dict(action_center=((actions.max(0) + actions.min(0)) / 2).tolist(),
                action_scale=np.maximum((actions.max(0) - actions.min(0)) / 2, .01).tolist(),
                proprio_mean=proprio.mean(0).tolist(),
                proprio_std=np.maximum(proprio.std(0), .01).tolist())


class SensorDataset(Dataset):
    def __init__(self, root, skill, split, stats, horizon=16, obs_horizon=2, stride=1):
        self.root, self.skill = Path(root), skill
        manifest = json.loads((self.root / 'manifest.json').read_text())
        if not manifest.get('export_complete') or manifest.get('labels_are_inputs') is not False:
            raise ValueError('Incomplete or unsafe export')
        if not manifest.get('task_command_required'):
            raise ValueError('Re-export this dataset with explicit task commands before training the clutter-aware policy')
        self.path = self.root / (skill + '.hdf5')
        self.stats, self.horizon, self.obs_horizon = stats, horizon, obs_horizon
        self.indices = []
        with h5py.File(self.path, 'r') as f:
            train = set(f['mask/train'][:])
            valid = set(f['mask/valid'][:])
            if train & valid:
                raise ValueError('Episode leakage across train and validation')
            for key in f['mask/' + split][:]:
                key = key.decode()
                demo = f['data/' + key]
                assert_sensor_inputs(demo['obs'])
                self.indices.extend((key, i) for i in range(0, len(demo['actions']) - horizon + 1, stride))
        if not self.indices:
            raise ValueError('Split contains no full action chunks')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        key, t = self.indices[index]
        times = [max(0, t - self.obs_horizon + 1 + j) for j in range(self.obs_horizon)]
        with h5py.File(self.path, 'r') as f:
            demo = f['data/' + key]
            rgb = np.stack([np.stack([demo['obs/' + c + '_rgb'][i] for c in CAMERAS]) for i in times])
            proprio = np.stack([demo['obs/robot_proprio'][i] for i in times])
            actions = demo['actions'][t:t + self.horizon]
            task_target = demo['task_target'][:]
        with h5py.File(self.root / 'perception_labels.hdf5', 'r') as f:
            semantic = np.stack([f[self.skill + '/' + key + '/' + c][t] for c in CAMERAS])
        proprio = (proprio - np.asarray(self.stats['proprio_mean'])) / np.asarray(self.stats['proprio_std'])
        actions = (actions - np.asarray(self.stats['action_center'])) / np.asarray(self.stats['action_scale'])
        # Action/semantic supervision stays separate from sensors and task command.
        return dict(rgb=torch.from_numpy(rgb).permute(0, 1, 4, 2, 3).float() / 255,
                    proprio=torch.from_numpy(proprio.astype(np.float32)),
                    task_target=torch.from_numpy(task_target.astype(np.float32)),
                    actions=torch.from_numpy(actions.astype(np.float32)),
                    semantic=torch.from_numpy(semantic.astype(np.int64)))


class SensorPolicy(nn.Module):
    def __init__(self, obs_horizon=2, horizon=16, diffusion_steps=100):
        super().__init__()
        self.obs_horizon, self.horizon = obs_horizon, horizon
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.SiLU())
        self.spatial = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(1024, 64), nn.SiLU())
        self.segmentation = nn.Conv2d(64, 11, 1)
        self.denoiser = ConditionalUnet1D(input_dim=8,
            global_cond_dim=obs_horizon * (len(CAMERAS) * 64 + 16) + 5,
            diffusion_step_embed_dim=64, down_dims=[64, 128, 256])
        self.scheduler = DDPMScheduler(num_train_timesteps=diffusion_steps,
            beta_schedule='squaredcos_cap_v2', clip_sample=True, prediction_type='epsilon')

    def encode(self, rgb, proprio, task_target=None):
        b, t, views, channels, height, width = rgb.shape
        if (t, views, channels, proprio.shape[-1]) != (self.obs_horizon, len(CAMERAS), 3, 16):
            raise ValueError('Invalid sensor dimensions')
        features = self.encoder(rgb.reshape(b * t * views, channels, height, width))
        context = torch.cat((self.spatial(features).reshape(b, t, views * 64), proprio), dim=-1).flatten(1)
        # Optional only for inspecting the RGB semantic head; act/losses require
        # an explicit task. This vector identifies the request, never a pose.
        if task_target is None:
            task_target = context.new_zeros((b,5))
        elif task_target.shape != (b,5) or not torch.all((task_target==0)|(task_target==1)) or not torch.all(task_target.sum(-1)==1):
            raise ValueError('Task target must be a one-hot requested instrument')
        context = torch.cat((context,task_target),dim=-1)
        features = features.reshape(b, t, views, *features.shape[1:])[:, -1].flatten(0, 1)
        logits = F.interpolate(self.segmentation(features), (height, width), mode='bilinear', align_corners=False)
        return context, logits.reshape(b, views, 11, height, width)

    def losses(self, batch):
        context, logits = self.encode(batch['rgb'], batch['proprio'], batch['task_target'])
        noise = torch.randn_like(batch['actions'])
        time = torch.randint(self.scheduler.config.num_train_timesteps, (len(noise),), device=noise.device)
        noisy = self.scheduler.add_noise(batch['actions'], noise, time)
        prediction = self.denoiser(noisy, time, global_cond=context)
        diffusion_loss = F.mse_loss(prediction, noise)
        # Balance instrument pixels against large background; labels never condition DP.
        weight = logits.new_tensor([.1, .3, .5, 1., 1., 1., 1., 1., .2, .1, .1])
        perception_loss = F.cross_entropy(logits.flatten(0, 1), batch['semantic'].flatten(0, 1), weight=weight)
        return diffusion_loss + .2 * perception_loss, diffusion_loss, perception_loss

    @torch.no_grad()
    def act(self, rgb, proprio, inference_steps=20, generator=None, *, task_target=None):
        if task_target is None:
            raise ValueError('Choose the requested target before DP inference')
        context, logits = self.encode(rgb, proprio, task_target)
        trajectory = torch.randn((len(rgb), self.horizon, 8), device=rgb.device, generator=generator)
        self.scheduler.set_timesteps(inference_steps)
        for time in self.scheduler.timesteps:
            prediction = self.denoiser(trajectory, time.to(rgb.device), global_cond=context)
            trajectory = self.scheduler.step(prediction, time, trajectory, generator=generator).prev_sample
        return trajectory, logits.argmax(2)


def decode_actions(normalized, stats):
    values = normalized * normalized.new_tensor(stats['action_scale']) + normalized.new_tensor(stats['action_center'])
    if not torch.isfinite(values).all():
        raise ValueError('Nonfinite predicted action')
    norm = values[..., 3:7].norm(dim=-1, keepdim=True)
    if torch.any(norm < 1e-6):
        raise ValueError('Degenerate predicted quaternion; refuse execution')
    values[..., 3:7] /= norm
    values[..., 7] = torch.where(values[..., 7] < 0, -1., 1.)
    return values
