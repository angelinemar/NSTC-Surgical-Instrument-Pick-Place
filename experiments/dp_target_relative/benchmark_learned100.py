"""100 fixed, resumable Scissor Pick trials using RF-DETR B + trained DP actions.

Ten held-out test scenes each receive ten predeclared target XY/yaw changes.
No expert action or training template is sent to the robot during evaluation.
The 100 attempts share ten base scenes, so report both counts and dependence.
"""
import argparse
import hashlib
import json
import os
import signal
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

try:
    from .workspace_root import workspace_root
except ImportError:
    from workspace_root import workspace_root
ROOT = workspace_root()
DATA=ROOT/'datasets/DP datasets/NSTC/DP datasets/scissor/scissor/test/pick_policy/scissor'
PICK=ROOT/'datasets/DP_training/runs/target_relative_dp_scissor_v1_20261103/checkpoint_best.pt'
PLACE=ROOT/'datasets/DP_training/runs/gripper_v2_place_rgb448_b24_20261028_223111/checkpoint_best.pt'
DETECTOR=Path(__file__).resolve().parents[2]/'models/object_detection/B/checkpoint_best_total.pth'
OUTPUT=ROOT/'datasets/DP_evaluation/scissor_trained_dp_live_100_20261103'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def plan():
    files=sorted(DATA.glob('episode_*.h5'))
    if len(files)!=10:raise RuntimeError(f'Expected 10 distinct held-out Scissor scenes, got {len(files)}')
    rng=np.random.default_rng(20261103)
    attempts=[]
    for scene,snapshot in enumerate(files,1):
        for variation in range(1,11):
            while True:
                dx,dy=rng.uniform(-.018,.018,size=2)
                if np.hypot(dx,dy)>=.005:break
            yaw=float(rng.uniform(-25,25))
            attempts.append(dict(scene=scene,variation=variation,snapshot=str(snapshot.resolve()),
                                 dx_m=float(dx),dy_m=float(dy),yaw_deg=yaw))
    return dict(contract='trained_dp_rfdetr_b_scissor_pick_100_v1',seed=20261103,
                base_test_scenes=10,variations_per_scene=10,planned_attempts=100,
                perturbation='XY independently uniform ±1.8 cm, minimum XY shift 0.5 cm; yaw uniform ±25°',
                pick_checkpoint=str(PICK),pick_sha256=digest(PICK),
                detector_checkpoint=str(DETECTOR),detector_sha256=digest(DETECTOR),
                attempts=attempts)


def row_for(index,item,run):
    result_file=run/'result.json'
    if not result_file.exists():raise RuntimeError(f'Missing result for attempt {index}: {run}')
    result=json.loads(result_file.read_text())
    if result.get('protocol')!='learned_target_relative_dp_live_camera_v1':
        raise RuntimeError(f'Wrong protocol in {result_file}')
    if not result.get('model_inference_used') or not result.get('detector_inference_used') or result.get('oracle_object_pose_used'):
        raise RuntimeError(f'Checkpoint/detector contract not met in {result_file}')
    if result.get('expert_replay') or result.get('expert_approach') or result.get('template_replay'):
        raise RuntimeError(f'Expert/template replay detected in {result_file}')
    if result.get('pick_sha256')!=digest(PICK):raise RuntimeError(f'Pick checkpoint changed in {result_file}')
    actions=run/'actions.jsonl'
    action_steps=sum(1 for _ in actions.open()) if actions.exists() else 0
    if result.get('lift_success') and action_steps==0:
        raise RuntimeError(f'Success without model actions in {run}')
    failure=result.get('failure')
    valid=bool(result.get('pick_started')) and 'RESET_INVALID' not in str(failure)
    return dict(index=index,**item,valid=valid,success=bool(result.get('lift_success')) if valid else False,
                failure=failure,model_action_steps=action_steps,
                gif=result.get('gif'),result_file=str(result_file))


def save_summary(output,rows):
    successes=sum(x['success'] for x in rows)
    valid=sum(x['valid'] for x in rows)
    report=dict(contract='trained_dp_rfdetr_b_scissor_pick_100_v1',
                model_inference_used=True,detector_inference_used=True,
                expert_or_template_replay_used=False,oracle_object_pose_used=False,
                planned_attempts=100,completed_attempts=len(rows),valid_attempts=valid,
                technical_invalid=len(rows)-valid,successes=successes,
                failures=len(rows)-successes,
                success_pct_of_completed=100*successes/len(rows) if rows else None,
                success_pct_of_planned=100*successes/100,
                complete=len(rows)==100,shared_base_scenes=10,trials=rows)
    temp=output/'summary.json.tmp';temp.write_text(json.dumps(report,indent=2));temp.replace(output/'summary.json')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--start',type=int,default=1,help='First planned attempt to process')
    p.add_argument('--limit',type=int,default=100,help='Run the first N of the fixed 100 attempts; resume by increasing N')
    p.add_argument('--no-summary',action='store_true',help='Shard worker: write trial files without touching shared summary.json')
    p.add_argument('--trial-timeout-s',type=int,default=300,
                   help='Stop a hung Isaac Sim process group after this many seconds; keep a complete result if already saved')
    a=p.parse_args()
    if not 1<=a.start<=a.limit<=100:p.error('Require 1 <= --start <= --limit <= 100')
    if a.trial_timeout_s<60:p.error('--trial-timeout-s must be at least 60')
    a.output.mkdir(parents=True,exist_ok=True)
    fixed=plan()
    plan_file=a.output/'plan.json'
    if plan_file.exists() and json.loads(plan_file.read_text())!=fixed:
        raise RuntimeError('Saved plan or model hash changed; refusing to mix evaluations')
    plan_file.write_text(json.dumps(fixed,indent=2))
    env=os.environ.copy();env['OMNI_KIT_ACCEPT_EULA']='yes'
    env['PYTHONPATH']=':'.join(map(str,[Path(__file__).resolve().parents[1],ROOT/'versions/6.0',ROOT/'versions/6.0/compat']))
    rows=[]
    for index,item in enumerate(fixed['attempts'][a.start-1:a.limit],a.start):
        run=a.output/f'trial_{index:03d}'
        if not (run/'result.json').exists():
            if run.exists():
                archive=a.output/f'trial_{index:03d}_incomplete'
                serial=1
                while archive.exists():
                    archive=a.output/f'trial_{index:03d}_incomplete_{serial}'
                    serial+=1
                shutil.move(str(run),str(archive))
                print(f'Archived interrupted trial: {archive}',flush=True)
            cmd=[str(ROOT/'.venv-isaacsim6/bin/python'),'-m','evaluation.dp_trial',
                 '--learned-camera-pose','--snapshot',item['snapshot'],'--snapshot-split','test',
                 '--skill','pick','--object','scissor','--mode','grid','--seed',str(20261103+index),
                 '--max-steps','300','--headless','--pick',str(PICK),'--place',str(PLACE),
                 '--target-perturb-dx',str(item['dx_m']),
                 '--target-perturb-dy',str(item['dy_m']),
                 '--target-perturb-yaw-deg',str(item['yaw_deg']),
                 '--output',str(run)]
            with (a.output/f'trial_{index:03d}.log').open('w') as log:
                process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,
                                         stderr=subprocess.STDOUT,start_new_session=True)
                timed_out=False
                saved_at=None
                launched_at=time.monotonic()
                result_file=run/'result.json'
                while process.poll() is None:
                    if saved_at is None and result_file.exists():
                        try:
                            if json.loads(result_file.read_text()).get('trial_complete') is True:
                                saved_at=time.monotonic()
                        except (OSError,json.JSONDecodeError):
                            pass
                    if (saved_at is not None and time.monotonic()-saved_at>15) or \
                       time.monotonic()-launched_at>a.trial_timeout_s:
                        timed_out=True
                        try:os.killpg(process.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        process.wait()
                        break
                    time.sleep(1)
            result_file=run/'result.json'
            complete_result=(result_file.exists() and
                             json.loads(result_file.read_text()).get('trial_complete') is True)
            if not complete_result:
                raise RuntimeError(f'Trial {index} did not complete cleanly; inspect its log')
            if timed_out:
                print(f'Trial {index}: Isaac Sim shutdown timed out after result was saved; process group stopped',flush=True)
            elif process.returncode:
                print(f'Trial {index}: Isaac Sim exited {process.returncode} after saving a complete result',flush=True)
        row=row_for(index,item,run)
        rows.append(row)
        if not a.no_summary:
            if a.start!=1:raise RuntimeError('Only the main worker may update the ordered summary')
            summary=save_summary(a.output,rows)
        print(f"{index:03d}/100 {('SUCCESS' if row['success'] else 'FAIL' if row['valid'] else 'INVALID'):7s} "
              f"shard successes {sum(x['success'] for x in rows)}/{len(rows)}",flush=True)
    if a.no_summary:
        print(f'SHARD_COMPLETE {a.start}-{a.limit}',flush=True)
    elif a.limit==100:
        print('FINAL',json.dumps({k:v for k,v in summary.items() if k!='trials'}),flush=True)


if __name__=='__main__':main()
