"""Show actual finger response to V2's short CLOSE versus the held-out expert."""
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'datasets/DP_evaluation/gripper_v2_vs_v1'
DIAGNOSTIC = BASE / 'diagnostic_v2_pick_kelly'


def main():
    rows = [json.loads(line) for line in (DIAGNOSTIC / 'actions.jsonl').read_text().splitlines()]
    command = np.asarray([row['action'][7] < 0 for row in rows], dtype=int)
    fingers = np.asarray([np.mean(row['fingers_after_action_m']) for row in rows]) * 1000
    model_first = int(np.flatnonzero(command)[0])
    source = json.loads((DIAGNOSTIC / 'result.json').read_text())['reset']['source']
    with h5py.File(source) as data:
        expert_command = (data['actions'][:, 7] < 0).astype(int)
        expert_fingers = np.mean(data['observations/robot_proprio'][:, 7:9], axis=1) * 1000
    expert_first = int(np.flatnonzero(expert_command)[0])
    horizon = min(42, len(expert_command) - expert_first, len(command) - model_first)
    times = np.arange(horizon)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, facecolor='white')
    axes[0].step(times, expert_command[expert_first:expert_first + horizon], where='post', color='#2468a4', label='Expert command', linewidth=2.5)
    axes[0].step(times, command[model_first:model_first + horizon], where='post', color='#d47720', label='V2 command', linewidth=2.5)
    axes[0].set_yticks((0, 1), labels=('OPEN', 'CLOSE'))
    axes[0].set_ylim(-.15, 1.25)
    axes[0].set_ylabel('Command')
    axes[0].legend(loc='lower right')
    axes[1].plot(times, expert_fingers[expert_first:expert_first + horizon], color='#2468a4', label='Expert measured fingers', linewidth=2.5)
    axes[1].plot(times, fingers[model_first:model_first + horizon], color='#d47720', label='V2 measured fingers', linewidth=2.5)
    axes[1].set_ylabel('Each finger opening (mm)')
    axes[1].set_xlabel('Steps after each sequence\'s first CLOSE (20 ms/step)')
    axes[1].set_ylim(0, 43)
    axes[1].legend(loc='lower right')
    for axis in axes:axis.grid(alpha=.2)
    fig.suptitle('Kelly Pick: V2 reopens before the fingers finish closing', fontsize=16, fontweight='bold')
    fig.text(.5, .015, 'Matched held-out H5 start; timelines aligned at first CLOSE for visualization, not at equal robot pose.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(BASE / 'kelly_finger_response.png', dpi=180, facecolor='white')
    print(BASE / 'kelly_finger_response.png')


if __name__ == '__main__':
    main()
