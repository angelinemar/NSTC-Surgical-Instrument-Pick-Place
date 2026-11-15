"""Frozen, resumable 100-Pick + 100-Place benchmark for each DP approach.

Two approaches (class-specific and five-class joint) see the same held-out
episodes, perturbations, and random seeds. There are 20 scheduled trials per
class and skill: ten independent test episodes with two perturbations/seeds.
The 400 rollouts are run sequentially because Isaac Sim owns the GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from train_family import CLASSES, ROOT, episode_files

RUNS = ROOT / 'datasets/DP_training/runs'
OUT = ROOT / 'datasets/DP_evaluation/target_relative_individual_vs_joint_100_20261103'
DETECTOR = Path(__file__).resolve().parents[2] / 'models/object_detection/B/checkpoint_best_total.pth'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def checkpoints(variant: str, instrument: str) -> tuple[Path, Path]:
    if variant == 'joint':
        return tuple(RUNS / f'target_relative_dp_joint_{skill}_v1_20261103/checkpoint_best.pt'
                     for skill in ('pick', 'place'))
    pick_name = ('target_relative_dp_scissor_v1_20261103' if instrument == 'scissor'
                 else f'target_relative_dp_{instrument}_pick_v1_20261103')
    place_name = f'target_relative_dp_{instrument}_place_v1_20261103'
    return RUNS / pick_name / 'checkpoint_best.pt', RUNS / place_name / 'checkpoint_best.pt'


def make_plan() -> dict:
    rng = np.random.default_rng(20261103)
    episodes = {skill: {name: episode_files(name, skill, 'test') for name in CLASSES}
                for skill in ('pick', 'place')}
    attempts = []
    for repetition in range(2):
        for scene in range(10):
            for instrument in CLASSES:
                index = len(attempts) + 1
                while True:
                    dx, dy = rng.uniform(-.018, .018, size=2)
                    if np.hypot(dx, dy) >= .005:
                        break
                attempts.append(dict(index=index, instrument=instrument,
                    base_scene=scene + 1, repetition=repetition + 1,
                    pick_snapshot=str(episodes['pick'][instrument][scene].resolve()),
                    place_snapshot=str(episodes['place'][instrument][scene].resolve()),
                    pick_dx_m=float(dx), pick_dy_m=float(dy),
                    pick_yaw_deg=float(rng.uniform(-25, 25)),
                    seed=20261103 + index))
    if len(attempts) != 100:
        raise AssertionError('Expected exactly 100 cases per skill')
    files = {'detector_B': DETECTOR}
    for variant in ('individual', 'joint'):
        for name in CLASSES:
            pick, place = checkpoints(variant, name)
            files[f'{variant}_{name}_pick'] = pick
            files[f'{variant}_{name}_place'] = place
    hashes = {name: dict(path=str(path), sha256=digest(path)) for name, path in files.items()}
    return dict(contract='dp_individual_vs_joint_100_v1', seed=20261103,
                planned_per_variant_skill=100, classes=list(CLASSES),
                per_class_skill=20, independent_base_episodes_per_class_skill=10,
                pick_variations='Two predeclared XY/yaw shifts per base Pick scene; XY ±1.8 cm and yaw ±25°',
                place_variations='Two diffusion seeds per base Place scene; same held-object start',
                detector_note='Scissor uses historical threshold 0.5; other classes use 0.1 and nearest measured TCP candidate. Target class is given.',
                snapshots='held-out test split only', hashes=hashes, attempts=attempts)


def valid_reset(result: dict) -> bool:
    return bool(result.get('reset')) and 'RESET_INVALID' not in str(result.get('failure', ''))


def row_for(run: Path, case: dict, variant: str, skill: str, plan: dict) -> dict:
    result_path = run / 'result.json'
    if not result_path.exists():
        return dict(index=case['index'], instrument=case['instrument'],
                    base_scene=case['base_scene'], repetition=case['repetition'],
                    technical_invalid=True, success=False, failure='missing result.json',
                    model_action_steps=0, detector_pose_available=False,
                    gif=None, result_file=str(result_path))
    result = json.loads(result_path.read_text())
    if result.get('expert_replay') or result.get('expert_approach') or result.get('template_replay'):
        raise RuntimeError(f'Expert/template replay detected in {result_path}')
    if not result.get('model_inference_used') or result.get('oracle_object_pose_used'):
        raise RuntimeError(f'Model/oracle contract violated in {result_path}')
    if skill == 'pick' and (not result.get('detector_inference_used') or
                            result.get('protocol') != 'learned_target_relative_dp_live_camera_v1'):
        raise RuntimeError(f'Pick detector contract violated in {result_path}')
    if skill == 'place' and result.get('protocol') != 'trained_target_relative_place_known_slot_v1':
        raise RuntimeError(f'Place target contract violated in {result_path}')
    pick, place = checkpoints(variant, case['instrument'])
    if result.get('pick_sha256') != plan['hashes'][f'{variant}_{case["instrument"]}_pick']['sha256']:
        raise RuntimeError(f'Pick checkpoint changed in {result_path}')
    if result.get('place_sha256') != plan['hashes'][f'{variant}_{case["instrument"]}_place']['sha256']:
        raise RuntimeError(f'Place checkpoint changed in {result_path}')
    actions = run / 'actions.jsonl'
    action_steps = sum(1 for _ in actions.open()) if actions.exists() else 0
    reset_ok = valid_reset(result)
    outcome = bool(result.get('lift_success' if skill == 'pick' else 'place_success'))
    if outcome and not action_steps:
        raise RuntimeError(f'Success without model actions: {run}')
    pose_ok = (run / 'live_camera_pose.json').exists() if skill == 'pick' else False
    # A detector miss after a valid reset is a scheduled pipeline failure, not
    # a technical invalid that can be silently removed from the denominator.
    return dict(index=case['index'], instrument=case['instrument'],
                base_scene=case['base_scene'], repetition=case['repetition'],
                technical_invalid=not reset_ok, success=outcome if reset_ok else False,
                failure=result.get('failure'), model_action_steps=action_steps,
                detector_pose_available=pose_ok, gif=result.get('gif'),
                result_file=str(result_path))


def save_summary(folder: Path, variant: str, skill: str, rows: list[dict]) -> dict:
    per_class = {}
    for name in CLASSES:
        subset = [row for row in rows if row['instrument'] == name]
        per_class[name] = dict(completed=len(subset), successes=sum(x['success'] for x in subset),
                               technical_invalid=sum(x['technical_invalid'] for x in subset),
                               detector_pose_available=(sum(x['detector_pose_available'] for x in subset)
                                                        if skill == 'pick' else None))
    summary = dict(contract='dp_individual_vs_joint_100_v1', variant=variant,
                   skill=skill, planned=100, completed=len(rows),
                   successes=sum(row['success'] for row in rows),
                   technical_invalid=sum(row['technical_invalid'] for row in rows),
                   detector_pose_available=(sum(row['detector_pose_available'] for row in rows)
                                            if skill == 'pick' else None),
                   per_class=per_class, complete=len(rows) == 100,
                   shared_base_scenes_per_class=10, trials=rows)
    path = folder / 'summary.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(summary, indent=2) + '\n')
    temporary.replace(path)
    return summary


def no_action_gif(run: Path, skill: str) -> Path | None:
    sensor = run / 'live_front_sensor.npz'
    if skill != 'pick' or not sensor.exists():
        return None
    from PIL import Image, ImageDraw
    with np.load(sensor, allow_pickle=False) as bundle:
        rgb = bundle['rgb'].astype(np.uint8)
    frame = Image.fromarray(rgb, 'RGB')
    canvas = Image.new('RGB', (frame.width, frame.height + 54), 'white')
    canvas.paste(frame, (0, 54))
    ImageDraw.Draw(canvas).text((12, 16), 'NO ACTION: target detection or pose failed', fill='black')
    path = run / 'perception_failure_single_frame.gif'
    canvas.save(path, save_all=True, duration=2000, loop=0)
    return path


def run_trial(output: Path, plan: dict, case: dict, variant: str, skill: str,
              env: dict) -> dict:
    folder = output / variant / skill
    folder.mkdir(parents=True, exist_ok=True)
    run = folder / f'trial_{case["index"]:03d}'
    if not (run / 'result.json').exists():
        if run.exists():
            archive = folder / f'trial_{case["index"]:03d}_interrupted'
            serial = 1
            while archive.exists():
                archive = folder / f'trial_{case["index"]:03d}_interrupted_{serial}'
                serial += 1
            shutil.move(run, archive)
        pick, place = checkpoints(variant, case['instrument'])
        snapshot = case[f'{skill}_snapshot']
        command = [str(ROOT / '.venv-isaacsim6/bin/python'), '-m', 'evaluation.dp_trial',
                   '--snapshot', snapshot, '--snapshot-split', 'test',
                   '--skill', skill, '--object', case['instrument'], '--mode', 'grid',
                   '--seed', str(case['seed']), '--max-steps', '300', '--headless',
                   '--pick', str(pick), '--place', str(place), '--output', str(run)]
        if skill == 'pick':
            command += ['--learned-camera-pose',
                        '--target-perturb-dx', str(case['pick_dx_m']),
                        '--target-perturb-dy', str(case['pick_dy_m']),
                        '--target-perturb-yaw-deg', str(case['pick_yaw_deg'])]
        with (folder / f'trial_{case["index"]:03d}.log').open('w') as handle:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0 or not (run / 'result.json').exists():
            run.mkdir(parents=True, exist_ok=True)
            (run / 'launcher_exit.json').write_text(json.dumps(dict(returncode=completed.returncode)))
            log_path = folder / f'trial_{case["index"]:03d}.log'
            raise RuntimeError(f'Isaac trial failed: {variant} {skill} {case["index"]:03d}; '
                               f'inspect {log_path}')
    row = row_for(run, case, variant, skill, plan)
    if not row['model_action_steps'] and not row['gif']:
        gif = no_action_gif(run, skill)
        if gif:
            row['gif'] = str(gif)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--limit', type=int, default=100,
                        help='First N paired cases; resume by increasing N')
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        parser.error('Require 1 <= --limit <= 100')
    args.output.mkdir(parents=True, exist_ok=True)
    plan = make_plan()
    plan_path = args.output / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError('Frozen plan/checkpoint hash changed; refusing mixed evaluation')
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    env = os.environ.copy()
    env['OMNI_KIT_ACCEPT_EULA'] = 'yes'
    env['PYTHONPATH'] = ':'.join(map(str, [Path(__file__).resolve().parents[1], ROOT / 'versions/6.0',
                                          ROOT / 'versions/6.0/compat']))
    keys = [(variant, skill) for skill in ('pick', 'place')
            for variant in ('individual', 'joint')]
    rows = {key: [] for key in keys}
    for case in plan['attempts'][:args.limit]:
        for variant, skill in keys:
            row = run_trial(args.output, plan, case, variant, skill, env)
            rows[(variant, skill)].append(row)
            summary = save_summary(args.output / variant / skill, variant, skill,
                                   rows[(variant, skill)])
            status = ('INVALID' if row['technical_invalid'] else
                      'SUCCESS' if row['success'] else 'FAIL')
            line = (f"{case['index']:03d}/100 {case['instrument']:16} "
                    f"{variant:10} {skill:5} {status:7} "
                    f"cumulative={summary['successes']}/{summary['completed']} "
                    f"invalid={summary['technical_invalid']}")
            print(line, flush=True)
            with (args.output / 'progress.log').open('a') as handle:
                handle.write(line + '\n')
    print('RUN_COMPLETE' if args.limit == 100 else 'PARTIAL_LIMIT_REACHED',
          args.limit, flush=True)


if __name__ == '__main__':
    main()
