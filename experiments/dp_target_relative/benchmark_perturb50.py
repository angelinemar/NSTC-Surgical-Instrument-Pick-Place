"""Fifty Scissor Pick perturbations: 10 held-out scenes x 5 new target poses.

All actions use live RGB-D / RF-DETR B / train-only trajectory retrieval.
The 50 attempts are correlated within 10 base scenes, not 50 independent scenes.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np


try:
    from .workspace_root import workspace_root
except ImportError:
    from workspace_root import workspace_root
ROOT = workspace_root()
DATA = ROOT/'datasets/DP datasets/NSTC/DP datasets/scissor/scissor/test/pick_policy/scissor'
PICK = ROOT/'datasets/DP_training/runs/gripper_v2_pick_rgb448_b24_20261028_223111/checkpoint_best.pt'
PLACE = ROOT/'datasets/DP_training/runs/gripper_v2_place_rgb448_b24_20261028_223111/checkpoint_best.pt'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261103)
    p.add_argument('--limit', type=int, default=50, help='Run first N planned attempts; resume later with same output')
    a = p.parse_args()
    files = sorted(DATA.glob('episode_*.h5'))
    if len(files) != 10:
        raise RuntimeError('Expected 10 original Scissor test scenes')
    if not 1 <= a.limit <= 50:
        p.error('--limit must be 1..50')
    a.output.mkdir(parents=True, exist_ok=True)
    plan_file = a.output/'plan.json'
    if plan_file.exists():
        plan = json.loads(plan_file.read_text())
        if plan['seed'] != a.seed:
            raise ValueError('Cannot resume with a different seed')
    else:
        rng = np.random.default_rng(a.seed)
        attempts = []
        for scene, snapshot in enumerate(files, 1):
            for variation in range(1, 6):
                # Fixed before results are seen. No original unperturbed scene repeats.
                while True:
                    dx, dy = rng.uniform(-.018, .018, size=2)
                    if np.hypot(dx, dy) >= .005:
                        break
                yaw = float(rng.uniform(-25, 25))
                attempts.append(dict(scene=scene, variation=variation, snapshot=str(snapshot.resolve()),
                                     dx_m=float(dx), dy_m=float(dy), yaw_deg=yaw))
        plan = dict(contract='scissor_test_target_perturb50_v1', seed=a.seed,
                    perturbation_range='dx,dy uniform ±1.8 cm; yaw uniform ±25 degrees; min XY shift 0.5 cm',
                    scene_count=10, variations_per_scene=5, attempts=attempts)
        plan_file.write_text(json.dumps(plan, indent=2))
    env = os.environ.copy()
    env['OMNI_KIT_ACCEPT_EULA'] = 'yes'
    env['PYTHONPATH'] = ':'.join(map(str, [Path(__file__).resolve().parents[1], ROOT/'versions/6.0', ROOT/'versions/6.0/compat'])) + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    rows = []
    summary_file = a.output/'summary.json'
    if summary_file.exists():
        old = json.loads(summary_file.read_text())
        rows = old['trials']
        if len(rows) > a.limit:
            raise ValueError('Existing completed attempts exceed --limit')
    for index in range(len(rows), a.limit):
        item = plan['attempts'][index]
        run_dir = a.output/f'{index+1:02d}_scene{item["scene"]:02d}_variation{item["variation"]}'
        cmd = [str(ROOT/'.venv-isaacsim6/bin/python'), '-m', 'evaluation.dp_trial',
               '--template-replay', 'AUTO', '--live-camera-pose',
               '--snapshot', item['snapshot'], '--snapshot-split', 'test',
               '--skill', 'pick', '--object', 'scissor', '--mode', 'grid',
               '--seed', str(a.seed+index), '--max-steps', '300', '--headless',
               '--target-perturb-dx', str(item['dx_m']), '--target-perturb-dy', str(item['dy_m']),
               '--target-perturb-yaw-deg', str(item['yaw_deg']),
               '--pick', str(PICK), '--place', str(PLACE), '--output', str(run_dir)]
        print(f'{index+1}/50 scene {item["scene"]} variation {item["variation"]}: '
              f'dx={item["dx_m"]*100:.1f}cm dy={item["dy_m"]*100:.1f}cm yaw={item["yaw_deg"]:.1f}°', flush=True)
        with (a.output/f'{index+1:02d}.log').open('w') as log:
            process = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        result_file = run_dir/'result.json'
        result = json.loads(result_file.read_text()) if result_file.exists() else {}
        failure = result.get('failure')
        valid = not (failure and ('RESET_INVALID' in failure or 'target drift' in failure)) and result.get('pick_started', False)
        row = dict(**item, result_dir=str(run_dir), returncode=process.returncode,
                   valid=bool(valid), success=bool(result.get('lift_success')) if valid else False,
                   failure=failure, template=result.get('template_replay'), gif=result.get('gif'))
        rows.append(row)
        summary = dict(contract='scissor_test_target_perturb50_v1', protocol='train_template_live_camera_pose_diagnostic_v1',
                       model_inference_used=False, detector_inference_used=True, oracle_object_pose_used=False,
                       source_scenes=10, planned_attempts=50, completed=len(rows),
                       valid_attempts=sum(r['valid'] for r in rows), successes=sum(r['success'] for r in rows),
                       success_pct_valid=100*sum(r['success'] for r in rows)/max(1,sum(r['valid'] for r in rows)), trials=rows)
        summary_file.write_text(json.dumps(summary, indent=2))
        print(f"  {'SUCCESS' if row['success'] else 'INVALID' if not row['valid'] else 'FAIL'} | "
              f"{summary['successes']}/{summary['valid_attempts']} valid attempts", flush=True)
    print(f"Finished {len(rows)}/50 planned attempts. Summary: {summary_file}", flush=True)


if __name__ == '__main__':
    main()
