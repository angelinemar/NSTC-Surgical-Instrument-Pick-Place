"""Train one target-relative Pick or Place DP per instrument.

Pick uses the selected object pose from training demonstrations as its target
frame. Place uses the assigned tray-slot center and the robot-base axes. At
runtime, Pick needs a camera-estimated object pose; Place needs the known tray
slot. Neither validation nor test target labels enter training or normalization.
The learned event is CLOSE for Pick and OPEN for Place.
"""
from __future__ import annotations

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
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

if __package__:
    from .learned_policy import (RelativeDiffusionPolicy, Windows, fit_stats,
                                 relative_state_action, rotation, wxyz)
else:
    from learned_policy import (RelativeDiffusionPolicy, Windows, fit_stats,
                                relative_state_action, rotation, wxyz)

try:
    from .workspace_root import workspace_root
except ImportError:
    from workspace_root import workspace_root
ROOT = workspace_root()
CLASSES = ('scalpel', 'scissor', 'love_retractor', 'kelly', 'scalpel_type2')
HORIZON = 16


def episode_files(instrument: str, skill: str, split: str) -> list[Path]:
    folder = ROOT / 'datasets/DP datasets/NSTC/DP datasets' / instrument / instrument / split / f'{skill}_policy' / instrument
    paths = sorted(folder.glob('episode_*.h5'))
    expected = {'train': 70, 'valid': 20, 'test': 10}[split]
    if len(paths) != expected:
        raise RuntimeError(f'{folder}: expected {expected} episodes, found {len(paths)}')
    return paths


def load_episodes(instrument: str, skill: str, split: str) -> list[dict]:
    episodes = []
    for path in episode_files(instrument, skill, split):
        with h5py.File(path) as h:
            if (h.attrs['dataset_split'], h.attrs['target_object'], h.attrs['policy_skill']) != (split, instrument, skill):
                raise ValueError(f'Wrong episode contract: {path}')
            proprio = h['observations/robot_proprio'][:]
            actions = h['actions'][:]
            if skill == 'pick':
                target = h['debug_gt/selected_object_pose_b'][0]
            else:
                slot = h['debug_gt/target_slot_b'][0]
                target = np.r_[slot, 1., 0., 0., 0.]
        if len(proprio) != len(actions) or len(actions) < HORIZON:
            raise ValueError(f'Misaligned or short episode: {path}')
        # The shared policy head predicts the positive event state. For Place,
        # OPEN is positive; the gripper command is decoded accordingly.
        event = (actions[:, 7] < 0) if skill == 'pick' else (actions[:, 7] > 0)
        transitions = np.flatnonzero(event[1:] & ~event[:-1]) + 1
        if len(transitions) != 1:
            raise ValueError(f'Expected one {"CLOSE" if skill == "pick" else "OPEN"} transition: {path}')
        state, motion, _ = relative_state_action(proprio, actions, target)
        episodes.append(dict(path=str(path), state=state, motion=motion,
                             close=event.astype(np.float32), close_at=int(transitions[0])))
    return episodes


class Runtime:
    """Inference adapter; callers must supply an observed Pick pose or tray goal."""
    def __init__(self, checkpoint: str | Path, device: str = 'cuda'):
        data = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if data.get('smoke_only') or not str(data.get('input_contract', '')).startswith('target_relative_dp_'):
            raise ValueError('Expected a full target-relative family checkpoint')
        self.skill = data['skill']
        self.instrument = data['instrument']
        if data['input_contract'] != f'target_relative_dp_{self.instrument}_{self.skill}_v1':
            raise ValueError('Checkpoint contract does not match instrument and skill')
        self.model = RelativeDiffusionPolicy().to(device)
        self.model.load_state_dict(data['model'])
        self.model.eval()
        self.stats = data['stats']
        self.device = device
        self.image_size = 448  # actor sensor adapter; the policy consumes no RGB
        self.controller_contract = data['controller_contract']
        self.history = deque(maxlen=2)
        self.pose = None

    def set_target(self, instrument: str) -> None:
        if instrument != self.instrument:
            raise ValueError(f'This checkpoint is for {self.instrument}, not {instrument}')
        self.pose = None
        self.history.clear()

    def observe(self, sensors: dict) -> None:
        key = 'target_pose_b' if self.skill == 'pick' else 'target_slot_b'
        if key not in sensors:
            raise ValueError(f'Missing {key}; no simulator oracle pose may be substituted')
        pose = np.asarray(sensors[key], dtype=np.float32)
        if self.skill == 'place' and pose.shape == (3,):
            pose = np.r_[pose, 1., 0., 0., 0.].astype(np.float32)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError(f'Invalid {key}')
        if self.pose is None:
            self.pose = pose.copy()
        elif np.linalg.norm(self.pose - pose) > 1e-4:
            raise ValueError('Target lock changed during the skill')
        proprio = np.asarray(sensors['robot_proprio'], dtype=np.float32)
        if proprio.shape != (16,) or not np.isfinite(proprio).all():
            raise ValueError('Invalid robot state')
        dummy = np.r_[proprio[9:12], proprio[12:16], 1.][None]
        state, _, _ = relative_state_action(proprio[None], dummy, self.pose)
        self.history.append(state[0])

    @torch.no_grad()
    def predict(self, inference_steps: int = 100):
        if self.pose is None or not self.history:
            raise ValueError('Observe before prediction')
        states = np.stack([self.history[0], self.history[-1]])
        norm = (states - np.asarray(self.stats['state_mean'])) / np.asarray(self.stats['state_std'])
        inputs = torch.as_tensor(norm, dtype=torch.float32, device=self.device)[None]
        motion, event = self.model.sample(inputs, self.stats, inference_steps)
        motion = motion[0].cpu().numpy()
        event = event[0].cpu().numpy()
        base = rotation(self.pose[3:])
        xyz = base.apply(motion[:, :3]) + self.pose[:3]
        quat = wxyz(base * rotation(motion[:, 3:7]))
        grip = np.where(event, -1., 1.) if self.skill == 'pick' else np.where(event, 1., -1.)
        actions = np.concatenate((xyz, quat, grip[:, None]), axis=1).astype(np.float32)
        return actions, None


def train(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f'Choose a new run directory: {args.output}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; check GPU access before starting training')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    train_episodes = load_episodes(args.instrument, args.skill, 'train')
    valid_episodes = load_episodes(args.instrument, args.skill, 'valid')
    # Train split only supplies fitted statistics and sampling weights.
    stats = fit_stats(train_episodes)
    training = Windows(train_episodes, stats)
    validation = Windows(valid_episodes, stats)
    sampler = WeightedRandomSampler(training.weights, len(training), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    tr = DataLoader(training, batch_size=args.batch_size, sampler=sampler,
                    num_workers=0, pin_memory=True)
    va = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                    num_workers=0, pin_memory=True)
    model = RelativeDiffusionPolicy().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    args.output.mkdir(parents=True)
    variant = f'target_relative_dp_{args.instrument}_{args.skill}_v1'
    config = dict(variant=variant, instrument=args.instrument, skill=args.skill,
                  train_episodes=70, valid_episodes=20, test_episodes=10,
                  train_windows=len(training), valid_windows=len(validation),
                  epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
                  lr=args.lr, seed=args.seed, smoke=args.smoke, horizon=HORIZON,
                  target_pose_training_source=('train-only simulator object label' if args.skill == 'pick'
                                               else 'train-only assigned tray-slot center'),
                  target_pose_inference_source=('camera-estimated object pose; estimator must be validated per class'
                                                if args.skill == 'pick' else 'known/calibrated tray-slot center'),
                  action_representation='target-relative XYZ + quaternion; separate binary event head',
                  event_name='CLOSE' if args.skill == 'pick' else 'OPEN',
                  controller_contract='tcp_frame_correct_bounded_place_integral_v6',
                  closed_loop_verified=False)
    (args.output / 'training_config.json').write_text(json.dumps(config, indent=2))

    def run_epoch(loader: DataLoader, training_mode: bool, limit: int | None) -> dict:
        model.train(training_mode)
        totals = {'loss': 0., 'motion': 0., 'gripper': 0.}
        count = 0
        for j, batch in enumerate(loader):
            if limit is not None and j >= limit:
                break
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            with torch.set_grad_enabled(training_mode), torch.autocast('cuda', dtype=torch.bfloat16):
                loss, detail = model.losses(batch)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            if training_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            n = len(batch['state'])
            count += n
            for k, v in (('loss', loss), ('motion', detail['motion']), ('gripper', detail['gripper'])):
                totals[k] += float(v) * n
        return {k: v / max(1, count) for k, v in totals.items()}

    best = float('inf')
    best_epoch = 0
    bad = 0
    for ep in range(1, (2 if args.smoke else args.epochs + 1)):
        start = time.monotonic()
        train_metrics = run_epoch(tr, True, 2 if args.smoke else None)
        valid_metrics = run_epoch(va, False, 2 if args.smoke else None)
        improved = valid_metrics['loss'] < best - 1e-4
        if improved:
            best = valid_metrics['loss']
            best_epoch = ep
            bad = 0
        else:
            bad += 1
        checkpoint = dict(input_contract=variant, model=model.state_dict(), stats=stats,
                          config=config, epoch=ep, skill=args.skill,
                          instrument=args.instrument, controller_contract=[config['controller_contract']],
                          smoke_only=args.smoke)
        torch.save(checkpoint, args.output / 'checkpoint_latest.pt')
        if improved:
            torch.save(checkpoint, args.output / 'checkpoint_best.pt')
        row = dict(epoch=ep, train_loss=train_metrics['loss'], valid_loss=valid_metrics['loss'],
                   train_motion=train_metrics['motion'], valid_motion=valid_metrics['motion'],
                   train_gripper=train_metrics['gripper'], valid_gripper=valid_metrics['gripper'],
                   best_epoch=best_epoch, bad_epochs=bad, seconds=time.monotonic() - start)
        with (args.output / 'metrics.csv').open('a', newline='') as handle:
            writer = csv.DictWriter(handle, row.keys())
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row), flush=True)
        if bad >= args.patience:
            break
    result = dict(training_complete=True, model=variant, best_epoch=best_epoch,
                  epochs_completed=ep, checkpoint=str(args.output / 'checkpoint_best.pt'),
                  smoke_only=args.smoke, closed_loop_verified=False)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instrument', choices=CLASSES, required=True)
    parser.add_argument('--skill', choices=('pick', 'place'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke', action='store_true')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
