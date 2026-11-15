"""Evidence chart and readable summary for the four paired Kelly pilot rollouts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'datasets/DP_evaluation/gripper_v2_vs_v1'
CASES = {
    ('pick', 'V1 coupled'): OUTPUT / 'pilot_v1_pick_kelly',
    ('pick', 'V2 separate gripper'): OUTPUT / 'pilot_v2_pick_kelly',
    ('place', 'V1 coupled'): OUTPUT / 'pilot_v1_place_kelly',
    ('place', 'V2 separate gripper'): OUTPUT / 'pilot_v2_place_kelly',
}


def load(directory):
    result = json.loads((directory / 'result.json').read_text())
    actions = [json.loads(line) for line in (directory / 'actions.jsonl').read_text().splitlines()]
    close = np.asarray([row['action'][7] < 0 for row in actions], dtype=int)
    height = np.asarray([row['object_center_for_scoring'][2] for row in actions])
    transitions = np.flatnonzero(np.diff(close) != 0) + 1
    return result, close, height, transitions


def build():
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex='col', facecolor='white')
    colors = {'V1 coupled': '#2069a8', 'V2 separate gripper': '#d57516'}
    lines = ['# Matched Kelly pilot: V1 vs V2', '',
             'Both models started from the same held-out H5 scene and seed for each skill. '
             'This is one illustrative start per skill, not a class-wide success-rate estimate.', '',
             '| Skill | Model | Outcome | Model steps | First CLOSE | OPEN/CLOSE switches |',
             '|---|---|---|---:|---:|---:|']
    for column, skill in enumerate(('pick', 'place')):
        for model in colors:
            directory = CASES[(skill, model)]
            result, close, height, transitions = load(directory)
            steps = np.arange(1, len(close) + 1)
            axes[0, column].step(steps, close, where='post', color=colors[model], label=model, linewidth=1.8)
            axes[1, column].plot(steps, (height - height[0]) * 100, color=colors[model], label=model, linewidth=1.8)
            success = result['lift_success' if skill == 'pick' else 'place_success']
            first = int(np.flatnonzero(close)[0] + 1) if close.any() else 'never'
            lines.append(f'| {skill.title()} | {model} | {"SUCCESS" if success else "FAIL (300-step timeout)"} | {len(close)} | {first} | {len(transitions)} |')
        axes[0, column].set_title(skill.upper(), fontsize=13, fontweight='bold')
        axes[0, column].set_ylim(-.15, 1.2)
        axes[0, column].set_yticks((0, 1), labels=('OPEN', 'CLOSE'))
        axes[0, column].grid(alpha=.2)
        axes[1, column].axhline(0, color='#999', linewidth=.8)
        axes[1, column].grid(alpha=.2)
        axes[1, column].set_xlabel('Closed-loop control step (20 ms each)')
    axes[0, 0].set_ylabel('Commanded gripper state')
    axes[1, 0].set_ylabel('Object center height change (cm)')
    axes[0, 1].legend(loc='upper right')
    fig.suptitle('Same test starts, different trained DP models', fontsize=17, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(OUTPUT / 'pilot_case_study.png', dpi=180, facecolor='white')
    plt.close(fig)
    lines += ['', 'The V2 Pick checkpoint has held-out validation transition accuracy 53.5% at its best epoch; '
              'the V2 Place checkpoint has 76.8%. These offline values do not establish grasp or placement success.',
              '', 'The V2 Pick command briefly closed and reopened repeatedly in this rollout; the target did not lift. '
              'The V2 Place command also switched repeatedly; the placement success gate was not met. '
              'The command trace shows what the model requested, but does not alone prove every physical failure mechanism.',
              '', '## GIFs (actual front and top camera frames)', '']
    for skill in ('pick', 'place'):
        for model in colors:
            directory = CASES[(skill, model)]
            gif = next(directory.glob('rollout_*.gif'))
            lines.append(f'- {skill.title()} {model}: [{gif.stem}]({gif.relative_to(OUTPUT).as_posix()})')
    lines += ['', '## Continue with balanced paired test', '',
              'Run `./compare-dp-gripper-v2.sh --per-class 1 --repeats 1` from `/home/user/NSTC`. '
              'It evaluates one held-out test start for each of five instrument classes, separately for Pick and Place, '
              'under both models (20 rollouts). The frozen plan, per-trial GIFs, CSV and summary are written to '
              '`paired_test_1_per_class/`. Use `--per-class 10 --repeats 2 --output NEW_FOLDER` for '
              '100 starts per skill per model, which is a much longer run. The script resumes completed trials in the same folder.',
              '', 'The pilot was deliberately chosen to replay the old model’s known successful Kelly Pick start. '
              'It must not be combined with the balanced test rate.']
    (OUTPUT / 'PILOT_REPORT.md').write_text('\n'.join(lines) + '\n')
    print(OUTPUT / 'pilot_case_study.png')


if __name__ == '__main__':
    build()
