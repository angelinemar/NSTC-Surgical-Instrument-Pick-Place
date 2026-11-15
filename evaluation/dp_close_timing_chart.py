"""Plot V2 CLOSE probability against the real expert switch in held-out Pick H5s."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'datasets/DP_evaluation/gripper_v2_vs_v1'


def main():
    rows = json.loads((BASE / 'offline_gripper_probe.json').read_text())
    fig, ax = plt.subplots(figsize=(11, 5.8), facecolor='white')
    labels = {'kelly': 'Kelly', 'love_retractor': 'Love retractor', 'scalpel': 'Scalpel',
              'scalpel_type2': 'Scalpel type 2', 'scissor': 'Scissor'}
    for name, label in labels.items():
        points = sorted((row for row in rows if row['instrument'] == name), key=lambda row: row['offset'])
        ax.plot([point['offset'] for point in points],
                [point['model_close_probability'] for point in points],
                marker='o', linewidth=2, label=label)
    ax.axvline(0, color='#222', linestyle='--', linewidth=1.3, label='Expert first CLOSE')
    ax.axhline(.5, color='#777', linestyle=':', linewidth=1.2, label='Model decision threshold')
    ax.set(xlabel='Control steps relative to expert first CLOSE (20 ms/step)',
           ylabel='V2 gripper-head P(CLOSE) for next action', ylim=(-.03, 1.05), xlim=(-11, 21))
    ax.set_title('V2 CLOSE prediction lags the expert transition',
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=.2)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, .5), frameon=False)
    fig.text(.5, .02, 'Held-out expert RGB + robot state; Pick checkpoint at best epoch 4. Lines connect sampled frames only.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, .82, 1))
    fig.savefig(BASE / 'offline_close_probabilities.png', dpi=180, facecolor='white')
    print(BASE / 'offline_close_probabilities.png')


if __name__ == '__main__':
    main()
