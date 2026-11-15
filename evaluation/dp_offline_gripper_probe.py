"""Probe V2 gripper decisions on held-out expert observations at the true switch."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from dp_gripper_v2.runtime import TransitionRuntime
from training.export_sensor_only import CAMERAS

ROOT = Path(__file__).resolve().parents[3]
TEST = ROOT / 'datasets/DP_evaluation/gripper_v2_vs_v1/paired_test_1_per_class'
CHECKPOINT = ROOT / 'datasets/DP_training/runs/gripper_v2_pick_rgb448_b24_20261028_223111/checkpoint_best.pt'
CLASSES = ('kelly', 'love_retractor', 'scalpel', 'scalpel_type2', 'scissor')


def main():
    torch.set_num_threads(4)
    model = TransitionRuntime(CHECKPOINT, device='cpu')
    output = []
    for name in CLASSES:
        result = next(TEST.glob(f'*_pick_{name}_r0_v2_separate_gripper')) / 'result.json'
        source = Path(json.loads(result.read_text())['reset']['source'])
        with h5py.File(source) as data:
            close = np.flatnonzero(data['actions'][:, 7] < 0)
            if len(close) == 0:
                raise ValueError(f'No expert CLOSE in {source}')
            first = int(close[0])
            for offset in (-10, -1, 0, 1, 10, 20):
                t = first + offset
                if not 1 <= t < len(data['actions']):
                    continue
                model.set_target(name)
                for frame in (t - 1, t):
                    sensors = {'robot_proprio': data['observations/robot_proprio'][frame]}
                    sensors.update({camera + '_rgb': data['observations/' + camera + '_rgb'][frame] for camera in CAMERAS})
                    model.observe(sensors)
                rgb = torch.from_numpy(np.stack([row[0] for row in model.history])).permute(0, 1, 4, 2, 3).float()[None] / 255
                state = (np.stack([row[1] for row in model.history]) - model.stats['proprio_mean']) / model.stats['proprio_std']
                proprio = torch.as_tensor(state, dtype=torch.float32)[None]
                with torch.no_grad():
                    model.model.encode(rgb, proprio, model.task_target)
                    logits = model.model.gripper_logits(rgb, proprio, model.task_target)
                    probabilities = torch.sigmoid(logits[0, :8]).cpu().numpy()
                    probability = float(probabilities[0])
                output.append(dict(instrument=name, source=str(source), expert_first_close_step=first,
                                   frame=t, offset=offset, expert_command='CLOSE' if data['actions'][t, 7] < 0 else 'OPEN',
                                   model_close_probability=probability, model_command='CLOSE' if probability >= .5 else 'OPEN',
                                   first8_max_close_probability=float(probabilities.max()),
                                   first8_close_commands=int((probabilities >= .5).sum())))
        print(name, 'first CLOSE', first, 'p(CLOSE) at switch',
              next(round(row['model_close_probability'], 4) for row in output if row['instrument'] == name and row['offset'] == 0), flush=True)
    destination = ROOT / 'datasets/DP_evaluation/gripper_v2_vs_v1/offline_gripper_probe.json'
    destination.write_text(json.dumps(output, indent=2))
    print(destination)


if __name__ == '__main__':
    main()
