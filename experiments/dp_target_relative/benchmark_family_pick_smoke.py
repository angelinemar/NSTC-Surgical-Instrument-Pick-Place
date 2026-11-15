"""One held-out camera-guided Pick rollout per newly trained instrument.

This is a deployment smoke test; the Scissor 100-trial result remains separate.
"""
from __future__ import annotations

import json
import os
import subprocess

from train_family import CLASSES, ROOT, episode_files

RUNS = ROOT / 'datasets/DP_training/runs'
OUT = ROOT / 'datasets/DP_evaluation/target_relative_family_pick_smoke_20261103'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['OMNI_KIT_ACCEPT_EULA'] = 'yes'
    env['PYTHONPATH'] = ':'.join(map(str, [Path(__file__).resolve().parents[1], ROOT / 'versions/6.0',
                                          ROOT / 'versions/6.0/compat']))
    rows = []
    classes = [name for name in CLASSES if name != 'scissor']
    for i, instrument in enumerate(classes, 1):
        snapshot = episode_files(instrument, 'pick', 'test')[0]
        pick = RUNS / f'target_relative_dp_{instrument}_pick_v1_20261103/checkpoint_best.pt'
        place = RUNS / f'target_relative_dp_{instrument}_place_v1_20261103/checkpoint_best.pt'
        output = OUT / instrument
        result_path = output / 'result.json'
        if not result_path.exists():
            if output.exists():
                raise RuntimeError(f'Incomplete output directory: {output}')
            command = [str(ROOT / '.venv-isaacsim6/bin/python'), '-m', 'evaluation.dp_trial',
                       '--learned-camera-pose', '--snapshot', str(snapshot),
                       '--snapshot-split', 'test', '--skill', 'pick', '--object', instrument,
                       '--mode', 'grid', '--seed', str(20261113 + i), '--max-steps', '300',
                       '--headless', '--pick', str(pick), '--place', str(place),
                       '--output', str(output)]
            with (OUT / f'{instrument}.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=False)
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        valid = (bool(result.get('pick_started')) and (output / 'actions.jsonl').exists()
                 and 'RESET_INVALID' not in str(result.get('failure', '')))
        row = dict(instrument=instrument, snapshot=str(snapshot), valid=valid,
                   success=bool(result.get('lift_success')) if valid else False,
                   failure=result.get('failure'), gif=result.get('gif'),
                   model_inference_used=result.get('model_inference_used'),
                   detector_inference_used=result.get('detector_inference_used'),
                   oracle_object_pose_used=result.get('oracle_object_pose_used'))
        rows.append(row)
        summary = dict(protocol='target_relative_family_pick_smoke_v1',
                       planned=len(classes), completed=len(rows),
                       valid=sum(x['valid'] for x in rows),
                       successes=sum(x['success'] for x in rows), trials=rows,
                       interpretation='One held-out Pick test episode per new class; runtime smoke, not a reliable success rate')
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f"{instrument}: {'SUCCESS' if row['success'] else 'FAIL'}; "
              f"valid={valid}; cumulative {summary['successes']}/{summary['valid']}", flush=True)


if __name__ == '__main__':
    main()
