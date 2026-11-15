"""Five-class target-relative DP: exactly one Pick or one Place checkpoint.

The two skills remain separate because their initial conditions and gripper
events differ. A one-hot class identity conditions each policy. All five
classes contribute train episodes; validation and test remain episode-disjoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

if __package__:
    from .learned_policy import RelativeDiffusionPolicy, Windows, fit_stats, relative_state_action, rotation, wxyz
    from .train_family import CLASSES, ROOT, load_episodes
else:
    from learned_policy import RelativeDiffusionPolicy, Windows, fit_stats, relative_state_action, rotation, wxyz
    from train_family import CLASSES, ROOT, load_episodes


class JointPolicy(RelativeDiffusionPolicy):
    def __init__(self):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(42, 128), nn.SiLU(),
                                     nn.Linear(128, 128), nn.SiLU())


def pooled_episodes(skill: str, split: str) -> list[dict]:
    episodes = []
    for index, instrument in enumerate(CLASSES):
        one_hot = np.eye(len(CLASSES), dtype=np.float32)[index]
        for episode in load_episodes(instrument, skill, split):
            state = episode['state']
            episode['state'] = np.concatenate((state,
                        np.broadcast_to(one_hot, (len(state), len(CLASSES)))), axis=1)
            episode['instrument'] = instrument
            episodes.append(episode)
    return episodes


class Runtime:
    """One of the two joint checkpoints; class is locked for a skill rollout."""
    def __init__(self, checkpoint: str | Path, device: str = 'cuda'):
        data = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if data.get('smoke_only') or data.get('input_contract') not in (
                'target_relative_dp_joint_pick_v1', 'target_relative_dp_joint_place_v1'):
            raise ValueError('Expected a complete five-class joint DP checkpoint')
        self.skill = data['skill']
        self.model = JointPolicy().to(device)
        self.model.load_state_dict(data['model'])
        self.model.eval()
        self.device = device
        self.stats = data['stats']
        self.controller_contract = data['controller_contract']
        self.image_size = 448  # actor adapter; no raw RGB enters the policy
        self.history = deque(maxlen=2)
        self.instrument = None
        self.pose = None

    def set_target(self, instrument: str) -> None:
        if instrument not in CLASSES:
            raise ValueError(f'Unknown instrument: {instrument}')
        self.instrument = instrument
        self.pose = None
        self.history.clear()

    def observe(self, sensors: dict) -> None:
        if self.instrument is None:
            raise ValueError('Set target class before observation')
        key = 'target_pose_b' if self.skill == 'pick' else 'target_slot_b'
        if key not in sensors:
            raise ValueError(f'Missing {key}')
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
        one_hot = np.eye(len(CLASSES), dtype=np.float32)[CLASSES.index(self.instrument)]
        self.history.append(np.r_[state[0], one_hot])

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
        raise RuntimeError('CUDA required')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    train_episodes = pooled_episodes(args.skill, 'train')
    valid_episodes = pooled_episodes(args.skill, 'valid')
    stats = fit_stats(train_episodes)
    training = Windows(train_episodes, stats)
    validation = Windows(valid_episodes, stats)
    counts = Counter(train_episodes[i]['instrument'] for i, _ in training.indices)
    class_balanced_weights = np.asarray([
        training.weights[j] / counts[train_episodes[i]['instrument']]
        for j, (i, _) in enumerate(training.indices)], dtype=np.float64)
    sampler = WeightedRandomSampler(class_balanced_weights, len(training), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    tr = DataLoader(training, batch_size=args.batch_size, sampler=sampler,
                    num_workers=0, pin_memory=True)
    va = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                    num_workers=0, pin_memory=True)
    model = JointPolicy().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    args.output.mkdir(parents=True)
    contract = f'target_relative_dp_joint_{args.skill}_v1'
    config = dict(variant=contract, skill=args.skill, class_order=list(CLASSES),
                  train_episodes=350, valid_episodes=100, test_episodes=50,
                  train_windows=len(training), valid_windows=len(validation),
                  train_windows_by_class=dict(counts), class_sampling='uniform class x transition weight',
                  epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
                  lr=args.lr, seed=args.seed, smoke=args.smoke, horizon=16,
                  target_pose_training_source=('train-only simulator object pose' if args.skill == 'pick'
                                               else 'train-only assigned tray slot'),
                  target_pose_inference_source=('RF-DETR B + live calibrated RGB-D' if args.skill == 'pick'
                                                else 'known calibrated assigned tray slot'),
                  observation='two frames of 16D target-relative robot state + 5D class one-hot',
                  action='16 x target-relative 7D TCP pose + separate binary gripper event',
                  event='CLOSE' if args.skill == 'pick' else 'OPEN',
                  controller_contract='tcp_frame_correct_bounded_place_integral_v6',
                  closed_loop_verified=False)
    (args.output / 'training_config.json').write_text(json.dumps(config, indent=2))

    def epoch(loader: DataLoader, training_mode: bool, limit: int | None) -> dict:
        model.train(training_mode)
        totals = dict(loss=0., motion=0., gripper=0.)
        n_total = 0
        for j, batch in enumerate(loader):
            if limit is not None and j >= limit:
                break
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            with torch.set_grad_enabled(training_mode), torch.autocast('cuda', dtype=torch.bfloat16):
                loss, detail = model.losses(batch)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            if training_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            n = len(batch['state'])
            n_total += n
            for k, value in (('loss', loss), ('motion', detail['motion']),
                             ('gripper', detail['gripper'])):
                totals[k] += float(value) * n
        return {k: v / max(1, n_total) for k, v in totals.items()}

    best, best_epoch, bad = float('inf'), 0, 0
    for ep in range(1, (2 if args.smoke else args.epochs + 1)):
        start = time.monotonic()
        train_metrics = epoch(tr, True, 2 if args.smoke else None)
        valid_metrics = epoch(va, False, 2 if args.smoke else None)
        improved = valid_metrics['loss'] < best - 1e-4
        if improved:
            best, best_epoch, bad = valid_metrics['loss'], ep, 0
        else:
            bad += 1
        checkpoint = dict(input_contract=contract, model=model.state_dict(), stats=stats,
                          config=config, epoch=ep, skill=args.skill,
                          controller_contract=[config['controller_contract']],
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
    result = dict(training_complete=True, model=contract, best_epoch=best_epoch,
                  epochs_completed=ep, checkpoint=str(args.output / 'checkpoint_best.pt'),
                  smoke_only=args.smoke, closed_loop_verified=False)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
