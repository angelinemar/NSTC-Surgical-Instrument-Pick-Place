"""Audit every completed Pick/Place rollout and plot observed failure gates.

The plots describe measurable endpoints. They do not assign physical root
causes to a timeout. Each failed trial receives one primary category in a
fixed order, while the JSON retains every unmet final Place gate.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from audit_joint_failure import PICK_LABELS, PLACE_LABELS, audit_trial

try:
    from .workspace_root import workspace_root
except ImportError:
    from workspace_root import workspace_root
ROOT = workspace_root()
BASE = ROOT / 'datasets/DP_evaluation/target_relative_individual_vs_joint_100_20261103'
COLORS = {'individual': '#222222', 'joint': '#a7a7a7'}


def build(summaries: dict, *, partial: bool = False) -> tuple[Path, list[Path]]:
    output = BASE / 'preliminary_failure_audit' if partial else BASE
    output.mkdir(parents=True, exist_ok=True)
    audited = []
    counts = {}
    for skill in ('pick', 'place'):
        for variant in ('individual', 'joint'):
            rows = summaries[(variant, skill)]['trials']
            group = [audit_trial(row, variant, skill) for row in rows]
            audited.extend(group)
            counts[(variant, skill)] = Counter(item['category'] for item in group)
            if len(group) != summaries[(variant, skill)]['completed']:
                raise RuntimeError(f'Audit row count mismatch: {variant} {skill}')
            if sum(item['category'] == 'success' for item in group) != summaries[(variant, skill)]['successes']:
                raise RuntimeError(f'Audit success count mismatch: {variant} {skill}')

    contract = 'dp_joint_failure_audit_v1'
    audit_file = output / 'failure_audit.json'
    audit_file.write_text(json.dumps(dict(contract=contract,
        status='preliminary' if partial else 'complete',
        caveat='Primary categories are observed final gates, not proven root causes.',
        method='One mutually exclusive primary endpoint per failed trial; Place also records all unmet final gates.',
        rows=audited), indent=2) + '\n')
    csv_file = output / 'failure_audit.csv'
    with csv_file.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            'variant', 'skill', 'trial_index', 'instrument', 'success',
            'technical_invalid', 'category', 'all_unmet_final_gates',
            'max_center_rise_mm', 'close_commanded', 'open_commanded',
            'final_center_error_mm', 'final_axis_error_deg', 'result_file', 'gif'))
        writer.writeheader()
        for item in audited:
            evidence = item['evidence']
            last = evidence.get('final_place_gates') or {}
            writer.writerow(dict(variant=item['variant'], skill=item['skill'],
                trial_index=item['trial_index'], instrument=item['instrument'],
                success=item['success'], technical_invalid=item['technical_invalid'],
                category=item['category'],
                all_unmet_final_gates='; '.join(item['all_unmet_final_gates']),
                max_center_rise_mm=evidence.get('max_center_rise_mm'),
                close_commanded=evidence.get('close_commanded'),
                open_commanded=evidence.get('open_commanded'),
                final_center_error_mm=(round(last['center_error_m'] * 1000, 2)
                                       if 'center_error_m' in last else None),
                final_axis_error_deg=last.get('axis_error_deg'),
                result_file=item['result_file'], gif=item['gif']))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'Liberation Serif', 'font.size': 11,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    figures = []
    report = ['# Observed failure endpoints', '',
              'Categories identify measured end conditions, not proven root causes.',
              'Each failed trial has exactly one plotted primary category; the JSON also retains all unmet final Place gates.',
              'Invalid resets remain in the scheduled denominator and appear separately.', '']
    for skill in ('pick', 'place'):
        labels = PICK_LABELS if skill == 'pick' else PLACE_LABELS
        relevant = [key for key in labels if any(counts[(variant, skill)][key]
                                                   for variant in ('individual', 'joint'))]
        relevant.sort(key=lambda key: sum(counts[(v, skill)][key]
                                          for v in ('individual', 'joint')), reverse=True)
        if not relevant:
            relevant = ['other']
        height = max(3.5, .65 * len(relevant) + 1.65)
        fig, ax = plt.subplots(figsize=(9.3, height), facecolor='white')
        y = np.arange(len(relevant))
        for offset, variant in ((-.18, 'individual'), (.18, 'joint')):
            values = [counts[(variant, skill)][key] for key in relevant]
            bars = ax.barh(y + offset, values, height=.32, color=COLORS[variant],
                           edgecolor='#222222', linewidth=.5,
                           hatch='///' if variant == 'joint' else None,
                           label=f'{variant.title()} policy')
            for bar, value in zip(bars, values):
                ax.text(value + .15, bar.get_y() + bar.get_height()/2,
                        str(value), va='center', ha='left', fontsize=10)
        ax.set_yticks(y, [labels[key] for key in relevant])
        ax.invert_yaxis()
        ax.set_xlim(0, max(4, max(counts[(v,skill)][k] for v in ('individual','joint')
                                 for k in relevant) + 3))
        ax.set_xlabel('Number of failed trials')
        ni = summaries[('individual',skill)]['completed']
        nj = summaries[('joint',skill)]['completed']
        ax.set_title(skill.title(),loc='left',fontweight='bold',fontsize=13)
        ax.grid(axis='x', alpha=.17)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc='lower right')
        fig.subplots_adjust(left=.34, right=.96, top=.82, bottom=.16)
        directory = output if partial else ROOT / 'competition_report/figures'
        path = directory / f'joint_{skill}_failure_modes.png'
        fig.savefig(path, dpi=185, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        figures.append(path)
        report += [f'## {skill.title()}', '',
                   '| Observed endpoint | Individual | Joint |',
                   '| --- | ---: | ---: |']
        for key in relevant:
            report.append(f'| {labels[key]} | {counts[("individual",skill)][key]} | '
                          f'{counts[("joint",skill)][key]} |')
        report += ['', f'Individual: {summaries[("individual",skill)]["successes"]}/{ni} '
                   f'successes; Joint: {summaries[("joint",skill)]["successes"]}/{nj} successes.', '']
    (output / 'FAILURE_RESULTS.md').write_text('\n'.join(report) + '\n')
    print(audit_file)
    print(csv_file)
    for path in figures:
        print(path)
    return audit_file, figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial', action='store_true',
                        help='Create clearly labeled preliminary artifacts in a separate folder')
    args = parser.parse_args()
    summaries = {}
    for variant in ('individual', 'joint'):
        for skill in ('pick', 'place'):
            path = BASE / variant / skill / 'summary.json'
            if not path.exists():
                raise RuntimeError(f'Missing {path}')
            summary = json.loads(path.read_text())
            if not args.allow_partial and not (summary.get('complete') and
                                               summary.get('completed') == 100):
                raise RuntimeError(f'100-trial group incomplete: {path}')
            summaries[(variant, skill)] = summary
    build(summaries, partial=args.allow_partial)


if __name__ == '__main__':
    main()
