"""Run the frozen 400-rollout comparison with up to four Isaac Sim workers.

Each trial writes its own directory. Only the parent writes the summaries, so
an interrupted run can be resumed without mixing model checkpoints or cases.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmark_joint_vs_individual import OUT, make_plan, row_for, run_trial, save_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.limit <= 100 or not 1 <= args.workers <= 4:
        parser.error('Require 1 <= limit <= 100 and 1 <= workers <= 4')
    args.output.mkdir(parents=True, exist_ok=True)
    plan = make_plan()
    plan_path = args.output / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError('Frozen plan/checkpoint hashes changed')
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    env = os.environ.copy()
    env['OMNI_KIT_ACCEPT_EULA'] = 'yes'
    from benchmark_joint_vs_individual import ROOT
    env['PYTHONPATH'] = ':'.join(map(str, [Path(__file__).resolve().parents[1], ROOT / 'versions/6.0',
                                          ROOT / 'versions/6.0/compat']))
    keys = [(variant, skill) for variant in ('individual', 'joint')
            for skill in ('pick', 'place')]
    completed = {key: {} for key in keys}
    pending = []
    for case in plan['attempts'][:args.limit]:
        for variant, skill in keys:
            run = args.output / variant / skill / f'trial_{case["index"]:03d}'
            if (run / 'result.json').exists():
                completed[(variant, skill)][case['index']] = row_for(run, case, variant,
                                                                     skill, plan)
            else:
                pending.append((case, variant, skill))
    for key in keys:
        folder = args.output / key[0] / key[1]
        folder.mkdir(parents=True, exist_ok=True)
        save_summary(folder, *key,
                     [completed[key][i] for i in sorted(completed[key])])
    print(f'Existing {sum(map(len, completed.values()))}; pending {len(pending)}; '
          f'workers {args.workers}', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_trial, args.output, plan, case, variant, skill, env):
                   (case, variant, skill) for case, variant, skill in pending}
        for future in as_completed(futures):
            case, variant, skill = futures[future]
            row = future.result()
            key = (variant, skill)
            completed[key][case['index']] = row
            summary = save_summary(args.output / variant / skill, variant, skill,
                                   [completed[key][i] for i in sorted(completed[key])])
            status = ('INVALID' if row['technical_invalid'] else
                      'SUCCESS' if row['success'] else 'FAIL')
            line = (f'{case["index"]:03d}/100 {case["instrument"]:16} '
                    f'{variant:10} {skill:5} {status:7} '
                    f'cumulative={summary["successes"]}/{summary["completed"]} '
                    f'invalid={summary["technical_invalid"]}')
            print(line, flush=True)
            with (args.output / 'progress.log').open('a') as handle:
                handle.write(line + '\n')
    print('RUN_COMPLETE' if args.limit == 100 else 'PARTIAL_LIMIT_REACHED',
          args.limit, flush=True)


if __name__ == '__main__':
    main()
