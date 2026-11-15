"""Preflight and sequential GUI trials; never stops training."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from evaluation.dp_contract import summarize,replace_policy_windows
ROOT=Path(__file__).resolve().parents[3]
RUNS=ROOT/'datasets/DP_training/runs'
NAMES=['scalpel','scissor','love_retractor','kelly','scalpel_type2']

def checkpoints():
 result={}
 for s in ['pick','place']:
  folders=sorted(RUNS.glob(f'dp_{s}_all5_rgb448_b24_*'))
  folders=[p for p in folders if p.is_dir()]
  if not folders:raise RuntimeError(f'No {s} run')
  run=folders[-1];marker=run/'result.json'
  if not marker.exists():raise RuntimeError(f'{s.upper()} still lacks completed result.json: {run.name}; leave training running')
  data=json.loads(marker.read_text())
  if not data.get('training_complete') or data.get('smoke_only'):raise RuntimeError(f'{s} is not a completed production run')
  ck=Path(data['checkpoint'])
  if not ck.is_file():raise RuntimeError(f'Missing checkpoint: {ck}')
  result[s]=str(ck)
 return result

def main():
 p=argparse.ArgumentParser()
 p.add_argument('command',choices=['check','smoke','run'])
 p.add_argument('--trials-per-object',type=int,default=10)
 p.add_argument('--independent',action='store_true',help='100 Pick + 100 Place attempts with default trial count')
 p.add_argument('--seed',type=int,default=280001)
 p.add_argument('--output',type=Path)
 a=p.parse_args()
 if a.trials_per_object<1:p.error('Positive trial count required')
 for path in (ROOT/'versions/6.0/backends').glob('*recorder.py'):
  replace_policy_windows(path.read_text(encoding='utf-8-sig'))
 try:ck=checkpoints()
 except RuntimeError as e:print(str(e),flush=True);return 2
 print(json.dumps(ck,indent=2),flush=True)
 if a.command=='check':return 0
 import datetime
 output=a.output or ROOT/'datasets/DP_evaluation'/datetime.datetime.now().strftime('hybrid_%Y%m%d_%H%M%S')
 output.mkdir(parents=True,exist_ok=False)
 targets=NAMES if a.command=='run' else ['scalpel']
 n=a.trials_per_object if a.command=='run' else 1
 plan=[dict(skill=skill,object=obj,mode=mode,seed=a.seed+i*n+j) for skill in (['pick','place'] if a.independent else ['both']) for mode in ['grid','free'] for i,obj in enumerate(targets) for j in range(n)]
 (output/'plan.json').write_text(json.dumps(dict(checkpoints=ck,trials=plan,protocol='hybrid_skill_only_v1',smoke_only=a.command=='smoke',note='paired seeds; separate condition geometry; new generated scenes, not a certified blind benchmark'),indent=2))
 rows=[];initial_trials=len(plan)
 for i,trial in enumerate(plan):
  dest=output/f"{i:03d}_{trial['skill']}_{trial['mode']}_{trial['object']}"
  cmd=[str(ROOT/'.venv-isaacsim6/bin/python'),'-m','evaluation.dp_trial','--pick',ck['pick'],'--place',ck['place'],'--output',str(dest)]
  for key,value in trial.items():cmd+=['--'+key,str(value)]
  with (output/f'{i:03d}.log').open('w') as log:
   process=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
  if not (dest/'result.json').exists():
   raise RuntimeError(f'Trial process failed before result; inspect {output}/{i:03d}.log')
  row=json.loads((dest/'result.json').read_text());rows.append(row)
  if a.independent and not row.get(trial['skill']+'_started'):
   if len(plan)<initial_trials*3:
    replacement=dict(trial,seed=a.seed+1000000+len(plan))
    plan.append(replacement)
    with (output/'setup_replacements.jsonl').open('a') as f:f.write(json.dumps(dict(failed_trial=i,replacement=replacement))+'\n')
   else:raise RuntimeError('FSM setup budget exhausted; report actual model attempt counts, not 100')
  summary={mode:summarize([r for r in rows if r['mode']==mode]) for mode in ['grid','free']}
  if a.independent:
   summary['independent_models']={}
   for skill in ['pick','place']:
    selected=[r for r in rows if r.get('evaluated_skill')==skill]
    started=sum(bool(r.get(skill+'_started')) for r in selected)
    successes=sum(bool(r.get('lift_success' if skill=='pick' else 'place_success')) for r in selected)
    summary['independent_models'][skill]=dict(scenes_attempted=len(selected),model_attempts=started,successes=successes,success_percent=100*successes/started if started else None,FSM_setup_failures=len(selected)-started)
  summary['per_object']={mode:{obj:summarize([r for r in rows if r['mode']==mode and r['object']==obj]) for obj in targets} for mode in ['grid','free']}
  if a.independent:
   def independent_report(items,skill):
    entered=[r for r in items if r.get(skill+'_started')]
    successes=sum(bool(r.get('lift_success' if skill=='pick' else 'place_success')) for r in entered)
    return dict(model_attempts=len(entered),successes=successes,success_percent=100*successes/len(entered) if entered else None,FSM_setup_failures=len(items)-len(entered))
   summary=dict(protocol='independent_skill_v1',requested_per_model=2*len(targets)*n,completed=False,
    models={skill:independent_report([r for r in rows if r.get('evaluated_skill')==skill],skill) for skill in ['pick','place']},
    by_condition={mode:{skill:independent_report([r for r in rows if r.get('evaluated_skill')==skill and r['mode']==mode],skill) for skill in ['pick','place']} for mode in ['grid','free']},
    per_instrument={obj:{skill:independent_report([r for r in rows if r.get('evaluated_skill')==skill and r['object']==obj],skill) for skill in ['pick','place']} for obj in targets})
  (output/'summary.json').write_text(json.dumps(summary,indent=2))
  print(json.dumps(row),flush=True)
  # Infrastructure errors must not masquerade as valid model failure rates.
  failure=row.get('failure','')
  if failure and not (failure.endswith('_timeout') or failure.startswith('[PHASE FAIL]') or failure=='setup_or_FSM_failure'):
   raise RuntimeError('Infrastructure/evaluation error; stopped suite: '+failure)
 summary['completed']=True
 (output/'summary.json').write_text(json.dumps(summary,indent=2))
 print('Evaluation output:',output,flush=True)
 return 0
if __name__=='__main__':sys.exit(main())
