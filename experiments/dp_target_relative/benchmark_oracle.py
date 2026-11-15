"""Scissor Pick template benchmark: oracle, recorded-camera, or live-camera pose.

This evaluates a retrieval controller, not a learned Diffusion Policy.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np


try:
    from .workspace_root import workspace_root
except ImportError:
    from workspace_root import workspace_root
ROOT = workspace_root()
DATA = ROOT / 'datasets/DP datasets/NSTC/DP datasets/scissor/scissor'
PICK = ROOT / 'datasets/DP_training/runs/gripper_v2_pick_rgb448_b24_20261028_223111/checkpoint_best.pt'
PLACE = ROOT / 'datasets/DP_training/runs/gripper_v2_place_rgb448_b24_20261028_223111/checkpoint_best.pt'


def pose(path):
    with h5py.File(path) as f:
        if f.attrs['target_object'] != 'scissor' or f.attrs['policy_skill'] != 'pick':
            raise ValueError(f'Wrong target or skill: {path}')
        return np.asarray(f['debug_gt/selected_object_pose_b'][0], dtype=float)


def distance(a, b):
    # Quaternion geodesic angle; both q and -q encode the same orientation.
    qa, qb = a[3:] / np.linalg.norm(a[3:]), b[3:] / np.linalg.norm(b[3:])
    angle = 2 * np.arccos(np.clip(abs(float(np.dot(qa, qb))), 0, 1))
    return float(np.linalg.norm(a[:3] - b[:3]) + .025 * angle)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=('valid', 'test'), default='valid')
    p.add_argument('--starts', type=int, default=5)
    p.add_argument('--start-index', type=int, default=0, help='Zero-based index into sorted held-out episodes')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gui', action='store_true')
    p.add_argument('--camera-pose', action='store_true', help='Estimate pose from RF-DETR B + recorded RGB-D; no validation/test object pose')
    p.add_argument('--live-camera', action='store_true', help='Estimate pose and select template from live RGB-D after simulator reset')
    a = p.parse_args()
    if a.camera_pose and a.live_camera:
        p.error('Choose either --camera-pose or --live-camera')
    train = sorted((DATA / 'train/pick_policy/scissor').glob('episode_*.h5'))
    held = sorted((DATA / a.split / 'pick_policy/scissor').glob('episode_*.h5'))
    if len(train) != 70 or len(held) != (20 if a.split == 'valid' else 10):
        raise RuntimeError('Unexpected episode inventory; audit split before benchmarking')
    if a.start_index < 0 or a.starts < 1 or a.start_index+a.starts > len(held):
        p.error(f'Choose a nonempty episode range within 0..{len(held)-1}')
    a.output.mkdir(parents=True, exist_ok=False)
    train_poses = [(path, pose(path)) for path in train]
    rows = []
    env = os.environ.copy()
    env['OMNI_KIT_ACCEPT_EULA'] = 'yes'
    env['PYTHONPATH'] = ':'.join(map(str, [Path(__file__).resolve().parents[1], ROOT / 'versions/6.0', ROOT / 'versions/6.0/compat'])) + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    for i, snapshot in enumerate(held[a.start_index:a.start_index+a.starts], 1):
        estimate_file = a.output / f'{i:02d}_camera_pose.json'
        if a.live_camera:
            target = None
            template = Path('AUTO')
            source_pose = None
        elif a.camera_pose:
            estimator = [str(ROOT/'NSTC-Surgical-Instrument-Pick-Place/.venv-rfdetr/bin/python'),
                         str(Path(__file__).resolve().parent/'estimate_scissor_pose.py'),
                         '--episode', str(snapshot), '--output', str(estimate_file)]
            with (a.output/f'{i:02d}_pose.log').open('w') as log:
                subprocess.run(estimator, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            target = np.asarray(json.loads(estimate_file.read_text())['pose_b_wxyz'], dtype=float)
        else:
            target = pose(snapshot)
        if not a.live_camera:
            template, source_pose = min(train_poses, key=lambda item: distance(item[1], target))
        run_dir = a.output / f'{i:02d}_{snapshot.stem}'
        command = [str(ROOT / '.venv-isaacsim6/bin/python'), '-m', 'evaluation.dp_trial',
                   '--template-replay', str(template), '--snapshot', str(snapshot),
                   '--snapshot-split', a.split, '--skill', 'pick', '--object', 'scissor',
                   '--mode', 'grid', '--seed', str(840000 + i), '--max-steps', '300',
                   '--pick', str(PICK), '--place', str(PLACE), '--output', str(run_dir)]
        if not a.gui:
            command.append('--headless')
        if a.camera_pose:
            command.extend(['--camera-pose-json', str(estimate_file)])
        if a.live_camera:
            command.append('--live-camera-pose')
        print(f'{i}/{a.starts}: {snapshot.name} <- ' + ('LIVE camera selection' if a.live_camera else f'TRAIN {template.name}'), flush=True)
        with (a.output / f'{i:02d}.log').open('w') as log:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        result_file = run_dir / 'result.json'
        result = json.loads(result_file.read_text()) if result_file.exists() else {}
        row = dict(snapshot=str(snapshot), template=str(result.get('template_replay', template)),
                   template_distance=distance(source_pose, target) if source_pose is not None else None,
                   success=bool(result.get('lift_success')), returncode=process.returncode,
                   failure=result.get('failure'), gif=result.get('gif'))
        rows.append(row)
        (a.output / 'summary.json').write_text(json.dumps(dict(protocol='train_template_live_camera_pose_diagnostic_v1' if a.live_camera else 'train_template_camera_pose_diagnostic_v1' if a.camera_pose else 'train_template_oracle_pose_upper_bound',
            model_inference_used=False, detector_inference_used=bool(a.camera_pose or a.live_camera), oracle_object_pose_used=not (a.camera_pose or a.live_camera), split=a.split,
            completed=len(rows), successes=sum(r['success'] for r in rows), trials=rows), indent=2))
        print(f"  {'SUCCESS' if row['success'] else 'FAIL'}; {sum(r['success'] for r in rows)}/{len(rows)} so far", flush=True)
    print(f"Done: {sum(r['success'] for r in rows)}/{len(rows)} {'camera-assisted' if (a.camera_pose or a.live_camera) else 'oracle-assisted'} training-template trials. This is not DP success.")


if __name__ == '__main__':
    main()
