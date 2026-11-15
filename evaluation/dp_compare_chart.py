"""Create a clear white comparison chart from completed paired DP rollouts."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT / 'datasets/DP_evaluation/gripper_v2_vs_v1/paired_test_1_per_class'


def build(output=DEFAULT):
    output = Path(output)
    data = json.loads((output / 'summary.json').read_text())
    rows = list(csv.DictReader((output / 'results.csv').open()))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white', sharey=True)
    labels = {'v1_coupled': 'V1 coupled motion + gripper', 'v2_separate_gripper': 'V2 separate gripper head'}
    colors = {'v1_coupled': '#2468a4', 'v2_separate_gripper': '#d47720'}
    for axis, skill in zip(axes, ('pick', 'place')):
        for index, model in enumerate(labels):
            subset = [row for row in rows if row['skill'] == skill and row['model'] == model and row['valid'] == 'True']
            wins = sum(row['success'] == 'True' for row in subset)
            rate = 100 * wins / len(subset) if subset else 0
            axis.barh(index, rate, color=colors[model], height=.55)
            axis.text(min(rate + 1.5, 76), index, f'{wins}/{len(subset)} valid · {rate:.1f}%' if subset else 'not yet evaluated',
                      va='center', fontsize=11, fontweight='bold')
        axis.set_yticks((0, 1), labels=[labels[m] for m in labels])
        axis.invert_yaxis()
        axis.set_xlim(0, 100)
        axis.set_title(skill.upper(), fontsize=15, fontweight='bold')
        axis.set_xlabel('Physical skill success (%)')
        axis.grid(axis='x', alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle('Held-out paired closed-loop comparison', fontsize=17, fontweight='bold')
    fig.text(.5, .02, f'Identical H5 starts and DDPM seeds per pair · 300-step timeout · {data["finished_rollouts"]}/{data["planned_rollouts"]} rollouts complete',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .9))
    fig.savefig(output / 'success_comparison.png', dpi=180, facecolor='white')
    plt.close(fig)
    classes = ('kelly', 'love_retractor', 'scalpel', 'scalpel_type2', 'scissor')
    columns = (('pick', 'v1_coupled'), ('pick', 'v2_separate_gripper'),
               ('place', 'v1_coupled'), ('place', 'v2_separate_gripper'))
    lookup = {(row['skill'], row['instrument'], row['model']): row for row in rows}
    fig, ax = plt.subplots(figsize=(11, 6), facecolor='white')
    for y, instrument in enumerate(classes):
        for x, (skill, model) in enumerate(columns):
            row = lookup.get((skill, instrument, model))
            success = row is not None and row['success'] == 'True'
            color = '#dcefe6' if success else '#f9e2df' if row else '#eeeeee'
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor='white', linewidth=5))
            label = f'SUCCESS\n{row["steps"]} steps' if success else 'FAIL\n300-step timeout' if row else 'PENDING'
            ax.text(x + .5, y + .5, label, ha='center', va='center', fontsize=10,
                    color='#125d3d' if success else '#8d2929' if row else '#666', fontweight='bold')
    ax.set_xlim(0, 4); ax.set_ylim(5, 0)
    ax.set_xticks((.5, 1.5, 2.5, 3.5), labels=('PICK\nV1 coupled', 'PICK\nV2 separate', 'PLACE\nV1 coupled', 'PLACE\nV2 separate'), fontsize=11)
    ax.xaxis.tick_top()
    ax.set_yticks([i + .5 for i in range(5)], labels=[s.replace('_', ' ').title() for s in classes], fontsize=11)
    ax.tick_params(length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.suptitle('Physical outcome on each held-out instrument start', fontsize=16, fontweight='bold', y=.98)
    fig.text(.5, .04, 'V1 and V2 share the same H5 start and DDPM seed within each instrument and skill', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .91))
    fig.savefig(output / 'per_instrument_outcomes.png', dpi=180, facecolor='white')
    plt.close(fig)
    print(output / 'success_comparison.png')


if __name__ == '__main__':
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
