"""Create audited comparison text and white-background figure after 400 trials."""
from __future__ import annotations

import json
from pathlib import Path
from build_joint_failure_report import build as build_failure_report

try:
    from .workspace_root import workspace_root
except ImportError:
    from workspace_root import workspace_root
ROOT = workspace_root()
BASE = ROOT / 'datasets/DP_evaluation/target_relative_individual_vs_joint_100_20261103'
CLASSES = ('scalpel', 'scissor', 'love_retractor', 'kelly', 'scalpel_type2')
COLORS = {'individual': '#222222', 'joint': '#a7a7a7'}


def main() -> None:
    summaries = {}
    for variant in ('individual', 'joint'):
        for skill in ('pick', 'place'):
            path = BASE / variant / skill / 'summary.json'
            if not path.exists():
                raise RuntimeError(f'Missing {path}')
            data = json.loads(path.read_text())
            if not data.get('complete') or data.get('completed') != 100:
                raise RuntimeError(f'Incomplete 100-trial group: {path}')
            if sum(data['per_class'][name]['completed'] for name in CLASSES) != 100:
                raise RuntimeError(f'Per-class denominator mismatch: {path}')
            if any(data['per_class'][name]['completed'] != 20 for name in CLASSES):
                raise RuntimeError(f'Expected 20 cases per class: {path}')
            if sum(row['success'] for row in data['trials']) != data['successes']:
                raise RuntimeError(f'Success count mismatch: {path}')
            summaries[(variant, skill)] = data

    lines = ['# Matched five-instrument local-skill DP comparison', '',
             'All values below are physical Isaac Sim outcomes from trained checkpoints.',
             'They are not teacher-forced F1, expert replay, or a full Pick-to-Place cycle.', '',
             '| Policy | Skill | Success / scheduled | Technical invalid |',
             '| --- | --- | ---: | ---: |']
    for variant in ('individual', 'joint'):
        for skill in ('pick', 'place'):
            data = summaries[(variant, skill)]
            lines.append(f'| {variant.title()} | {skill.title()} | '
                         f'{data["successes"]}/100 ({data["successes"]}%) | '
                         f'{data["technical_invalid"]} |')
    lines += ['', '| Instrument | Individual Pick | Joint Pick | Individual Place | Joint Place |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name in CLASSES:
        cells = [f'{summaries[(variant, skill)]["per_class"][name]["successes"]}/20'
                 for variant, skill in (('individual','pick'),('joint','pick'),
                                        ('individual','place'),('joint','place'))]
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Pick uses a live front RGB-D frame and RF-DETR B target-pose estimate.',
              'Place starts from a held-object test snapshot and uses a known calibrated tray slot.',
              'Each class has ten held-out base episodes with two planned variations/seeds.',
              'Two trials from one episode are correlated; this is not 100 independent scenes.',
              'Non-Scissor pose selection was tuned after inspecting two Type 2 scalpel test scenes.',
              'The Type 2 subgroup is developmental rather than blind evidence.',
              'See plan.json, summary.json, individual trial logs, actions, and GIFs for audit.']
    (BASE / 'RESULTS.md').write_text('\n'.join(lines) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':11,
                         'figure.facecolor':'white','axes.facecolor':'white'})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), constrained_layout=True)
    for ax, skill in zip(axes, ('pick', 'place')):
        names = [name.replace('_',' ').title() for name in CLASSES]
        y = np.arange(len(CLASSES))
        for offset, variant in ((-.19,'individual'),(.19,'joint')):
            values = [summaries[(variant, skill)]['per_class'][name]['successes'] for name in CLASSES]
            bars = ax.barh(y+offset, values, height=.35, color=COLORS[variant],
                           edgecolor='#222222', linewidth=.5,
                           hatch='///' if variant == 'joint' else None,
                           label=variant.title())
            for bar,value in zip(bars,values):
                ax.text(min(value+.2,19.2),bar.get_y()+bar.get_height()/2,
                        f'{value}/20',va='center',ha='left',fontsize=9)
        ax.set_yticks(y,names)
        ax.invert_yaxis()
        ax.set_xlim(0,22.5)
        ax.set_xticks([0,5,10,15,20])
        ax.set_xlabel('Physical successes / 20 scheduled attempts')
        ax.set_title(skill.title(),fontweight='bold')
        ax.grid(axis='x',alpha=.18)
        ax.set_axisbelow(True)
        ax.legend(loc='lower right',frameon=False)
    out = ROOT / 'competition_report/figures/joint_vs_individual_success.png'
    fig.savefig(out,dpi=190,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    build_failure_report(summaries)
    print(BASE / 'RESULTS.md')
    print(out)


if __name__ == '__main__':
    main()
