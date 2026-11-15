"""Recorded-test event-head audit for trained target-relative DP checkpoints.

This is teacher-forced event prediction on held-out demonstrations. It is not a
closed-loop Pick or Place success rate and cannot replace robot rollouts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_family import CLASSES, ROOT, Runtime, load_episodes
from learned_policy import Windows


def audit(path: Path) -> dict:
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('smoke_only'):
        raise ValueError(f'Smoke checkpoint: {path}')
    config = data['config']
    instrument, skill = config['instrument'], config['skill']
    episodes = load_episodes(instrument, skill, 'test')
    windows = Windows(episodes, data['stats'])
    runtime = Runtime(path, device='cpu')
    counts = dict(TP=0, FP=0, FN=0, TN=0)
    with torch.no_grad():
        for start in range(0, len(windows), 256):
            rows = [windows[i] for i in range(start, min(start + 256, len(windows)))]
            state = torch.stack([r['state'] for r in rows]).flatten(1)
            expected = np.array([bool(r['close'][0]) for r in rows])
            predicted = (runtime.model.gripper(runtime.model.context(state)).sigmoid()[:, 0] > .5).numpy()
            counts['TP'] += int(np.count_nonzero(predicted & expected))
            counts['FP'] += int(np.count_nonzero(predicted & ~expected))
            counts['FN'] += int(np.count_nonzero(~predicted & expected))
            counts['TN'] += int(np.count_nonzero(~predicted & ~expected))
    tp, fp, fn, tn = (counts[k] for k in ('TP', 'FP', 'FN', 'TN'))
    n = tp + fp + fn + tn
    return dict(contract='target_relative_dp_offline_test_event_v1',
                checkpoint=str(path), instrument=instrument, skill=skill,
                event='CLOSE' if skill == 'pick' else 'OPEN', test_episodes=len(episodes),
                test_windows=n, **counts,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                accuracy=(tp + tn) / n if n else None,
                closed_loop_verified=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='Audit all completed target-relative family checkpoints')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    paths = []
    if args.all:
        runs = ROOT / 'datasets/DP_training/runs'
        paths = sorted(p for p in runs.glob('target_relative_dp_*_v1_20261103/checkpoint_best.pt')
                       if p.parent.name != 'target_relative_dp_scissor_v1_20261103')
        # The original Scissor checkpoint predates the family naming convention.
    elif args.checkpoint:
        paths = [args.checkpoint]
    else:
        parser.error('Specify --all or --checkpoint')
    reports = []
    for path in paths:
        report = audit(path)
        target = path.parent / 'offline_test_event_audit.json'
        target.write_text(json.dumps(report, indent=2) + '\n')
        reports.append(report)
        print(f"{report['instrument']:16} {report['skill']:5} {report['event']:5} "
              f"test F1={report['f1']:.3f} recall={report['recall']:.3f} "
              f"precision={report['precision']:.3f} windows={report['test_windows']}")
    if args.all:
        target = ROOT / 'datasets/DP_training/runs/target_relative_family_20261103_offline_summary.json'
        target.write_text(json.dumps(reports, indent=2) + '\n')
        print(target)


if __name__ == '__main__':
    main()
