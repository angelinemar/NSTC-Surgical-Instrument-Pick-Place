"""Evaluate RF-DETR B + trained Scissor DP on distinct held-out validation scenes.

The reset uses a saved validation scene. The policy receives only live camera
RGB-D-derived target pose and measured robot state, never expert actions or
validation object poses. Runs are sequential because Isaac Sim owns the GPU.
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
DATA = ROOT/'datasets/DP datasets/NSTC/DP datasets/scissor/scissor'
PICK = ROOT/'datasets/DP_training/runs/target_relative_dp_scissor_v1_20261103/checkpoint_best.pt'
PLACE = ROOT/'datasets/DP_training/runs/gripper_v2_place_rgb448_b24_20261028_223111/checkpoint_best.pt'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['valid','test'],default='valid')
    p.add_argument('--perturb',action='store_true',help='One preregistered XY/yaw perturbation per original scene')
    p.add_argument('--output',type=Path)
    p.add_argument('--limit',type=int,default=5)
    a=p.parse_args()
    files=sorted((DATA/a.split/'pick_policy/scissor').glob('episode_*.h5'))
    expected=20 if a.split=='valid' else 10
    if len(files)!=expected or not 1<=a.limit<=expected:
        p.error(f'Expected {expected} held-out {a.split} scenes and --limit 1..{expected}')
    if a.output is None:
        suffix='_perturbed' if a.perturb else ''
        a.output=ROOT/f'datasets/DP_evaluation/scissor_trained_dp_live_{a.split}{suffix}_20261103'
    a.output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261103)
    perturbations=[]
    for _ in files:
        dx,dy=rng.uniform(-.018,.018,size=2)
        yaw=float(rng.uniform(-25,25))
        perturbations.append((float(dx),float(dy),yaw))
    if a.perturb:
        plan=dict(seed=20261103,split=a.split,
                  contract='one_fixed_perturbation_per_held_out_scene_v1',
                  perturbations=[dict(snapshot=str(path),dx_m=dx,dy_m=dy,yaw_deg=yaw)
                                 for path,(dx,dy,yaw) in zip(files,perturbations)])
        plan_file=a.output/'plan.json'
        if plan_file.exists() and json.loads(plan_file.read_text())!=plan:
            raise ValueError('Perturbation plan differs from the existing output')
        plan_file.write_text(json.dumps(plan,indent=2))
    env=os.environ.copy()
    env['OMNI_KIT_ACCEPT_EULA']='yes'
    env['PYTHONPATH']=':'.join(map(str,[Path(__file__).resolve().parents[1],ROOT/'versions/6.0',ROOT/'versions/6.0/compat']))
    for i,snapshot in enumerate(files[:a.limit],1):
        run=a.output/f'trial_{i:02d}'
        if not (run/'result.json').exists():
            if run.exists():raise RuntimeError(f'Incomplete trial directory: {run}')
            cmd=[str(ROOT/'.venv-isaacsim6/bin/python'),'-m','evaluation.dp_trial',
                 '--learned-camera-pose','--snapshot',str(snapshot),'--snapshot-split',a.split,
                 '--skill','pick','--object','scissor','--mode','grid','--seed',str(20261102+i),
                 '--max-steps','300','--headless','--pick',str(PICK),'--place',str(PLACE),'--output',str(run)]
            if a.perturb:
                dx,dy,yaw=perturbations[i-1]
                cmd+=['--target-perturb-dx',str(dx),'--target-perturb-dy',str(dy),
                      '--target-perturb-yaw-deg',str(yaw)]
            with (a.output/f'trial_{i:02d}.log').open('w') as log:
                subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
        rows=[]
        for j in range(1,i+1):
            file=a.output/f'trial_{j:02d}/result.json'
            result=json.loads(file.read_text()) if file.exists() else {}
            valid=bool(result.get('pick_started')) and 'RESET_INVALID' not in str(result.get('failure',''))
            rows.append(dict(scene=j,snapshot=str(files[j-1]),valid=valid,
                             success=bool(result.get('lift_success')) if valid else False,
                             failure=result.get('failure'),gif=result.get('gif'),
                             model_inference_used=result.get('model_inference_used'),
                             detector_inference_used=result.get('detector_inference_used'),
                             oracle_object_pose_used=result.get('oracle_object_pose_used')))
        n=sum(r['valid'] for r in rows);s=sum(r['success'] for r in rows)
        summary=dict(protocol='learned_target_relative_dp_live_camera_v1',split=a.split,
                     perturbed=a.perturb,
                     model_inference_used=True,detector_inference_used=True,oracle_object_pose_used=False,
                     completed=i,valid=n,successes=s,success_pct_valid=100*s/n if n else None,
                     trials=rows)
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
        print(f'{a.split} {i}/{a.limit}: {"SUCCESS" if rows[-1]["success"] else "FAIL"}; cumulative {s}/{n}',flush=True)


if __name__=='__main__':main()
