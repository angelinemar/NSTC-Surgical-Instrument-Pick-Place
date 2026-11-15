"""One held-out Isaac Sim Place rollout per class using trained checkpoints.

This verifies the runtime interface and gives a tiny physical pilot. It does
not establish a class success rate or a complete Pick-to-Place transfer.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from train_family import CLASSES, ROOT, episode_files

RUNS = ROOT / 'datasets/DP_training/runs'
OUT = ROOT / 'datasets/DP_evaluation/target_relative_family_place_smoke_20261103'


def pick_checkpoint(name: str) -> Path:
    directory = ('target_relative_dp_scissor_v1_20261103' if name == 'scissor'
                 else f'target_relative_dp_{name}_pick_v1_20261103')
    return RUNS / directory / 'checkpoint_best.pt'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['OMNI_KIT_ACCEPT_EULA'] = 'yes'
    env['PYTHONPATH'] = ':'.join(map(str, [Path(__file__).resolve().parents[1], ROOT / 'versions/6.0',
                                          ROOT / 'versions/6.0/compat']))
    rows = []
    for i, instrument in enumerate(CLASSES, 1):
        snapshot = episode_files(instrument, 'place', 'test')[0]
        pick = pick_checkpoint(instrument)
        place = RUNS / f'target_relative_dp_{instrument}_place_v1_20261103/checkpoint_best.pt'
        output = OUT / instrument
        result_path = output / 'result.json'
        if not result_path.exists():
            if output.exists():
                raise RuntimeError(f'Incomplete output directory: {output}')
            command = [str(ROOT / '.venv-isaacsim6/bin/python'), '-m', 'evaluation.dp_trial',
                       '--snapshot', str(snapshot), '--snapshot-split', 'test',
                       '--skill', 'place', '--object', instrument, '--mode', 'grid',
                       '--seed', str(20261103 + i), '--max-steps', '300', '--headless',
                       '--pick', str(pick), '--place', str(place), '--output', str(output)]
            with (OUT / f'{instrument}.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=False)
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        valid = (bool(result.get('place_started')) and (output / 'actions.jsonl').exists()
                 and 'RESET_INVALID' not in str(result.get('failure', '')))
        row = dict(instrument=instrument, snapshot=str(snapshot), valid=valid,
                   success=bool(result.get('place_success')) if valid else False,
                   failure=result.get('failure'), gif=result.get('gif'),
                   model_inference_used=result.get('model_inference_used'),
                   expert_replay=result.get('expert_replay'),
                   oracle_object_pose_used=result.get('oracle_object_pose_used'))
        rows.append(row)
        summary = dict(protocol='target_relative_family_place_smoke_v1',
                       planned=len(CLASSES), completed=len(rows),
                       valid=sum(x['valid'] for x in rows),
                       successes=sum(x['success'] for x in rows), trials=rows,
                       interpretation='One held-out Place test episode per class; runtime smoke, not a reliable success rate')
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f"{instrument}: {'SUCCESS' if row['success'] else 'FAIL'}; "
              f"valid={valid}; cumulative {summary['successes']}/{summary['valid']}", flush=True)


if __name__ == '__main__':
    main()
