"""100 planned skill-only rollouts per model, from 50 test starts x two seeds."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from evaluation.dp_suite import checkpoints,ROOT

def write(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def report(rows,plan):
 result=dict(protocol='snapshot_skill_only_v2_full_ddpm',planned=len(plan),finished=len(rows),complete=len(rows)==len(plan),denoising_steps=100,action_prefix=8)
 def stats(items,skill):
  valid=[r for r in items if r.get(skill+'_started')]
  success=sum(bool(r.get('lift_success' if skill=='pick' else 'place_success')) for r in valid)
  return dict(planned=100,finished=len(items),valid_model_rollouts=len(valid),successes=success,failures=len(valid)-success,invalid_resets=len(items)-len(valid),success_rate_percent=100*success/len(valid) if valid else None)
 result['models']={s:stats([r for r in rows if r['evaluated_skill']==s],s) for s in ['pick','place']}
 result['per_class']={}
 for name in sorted({r['object'] for r in plan}):
  result['per_class'][name]={s:stats([r for r in rows if r['evaluated_skill']==s and r['object']==name],s) for s in ['pick','place']}
  for s in ['pick','place']:
   result['per_class'][name][s]['planned']=sum(r['skill']==s and r['object']==name for r in plan)
 return result

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--output',type=Path,default=ROOT/'datasets/DP_evaluation/model_only_100_each/results')
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 ck=checkpoints()
 inventory=ROOT/'datasets/Object datasets/trials/trial_03C_rfdetr_seg_syn_dp_balanced/dataset/source_episodes.json'
 sources=json.loads(inventory.read_text());plan=[]
 for repeat in range(2):
  for skill in ['pick','place']:
   rows=sorted([r for r in sources if r['split']=='test' and '/'+skill+'_policy/' in r['path']],key=lambda r:r['path'])
   if len(rows)!=50:raise ValueError('Expected 50 unique held-out starts per skill')
   for i,r in enumerate(rows):
    plan.append(dict(skill=skill,object=r['target'],source=r['path'],repeat=repeat,seed=700000+repeat*1000+i,mode='grid'))
 config=dict(protocol='snapshot_skill_only_v2_full_ddpm',checkpoints=ck,checkpoint_sha256={s:hashlib.sha256(Path(path).read_bytes()).hexdigest() for s,path in ck.items()},inventory_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),plan=plan,unique_starts_per_skill=50,repeats_per_start=2,denoising_steps=100,action_prefix=8)
 frozen=a.output/'plan.json'
 if frozen.exists() and json.loads(frozen.read_text())!=config:raise ValueError('Plan/checkpoints differ; use a new folder')
 write(frozen,config);completed=[]
 for i,trial in enumerate(plan):
  dest=a.output/f"{i:03d}_{trial['skill']}_{trial['object']}_r{trial['repeat']}"
  marker=dest/'result.json'
  if not marker.exists():
   if dest.exists():raise RuntimeError(f'Incomplete trial needs inspection: {dest}')
   cmd=[str(ROOT/'.venv-isaacsim6/bin/python'),'-m','evaluation.dp_trial','--pick',ck['pick'],'--place',ck['place'],'--skill',trial['skill'],'--object',trial['object'],'--mode','grid','--seed',str(trial['seed']),'--snapshot',trial['source'],'--output',str(dest)]
   print(f"[{i+1}/{len(plan)}] {trial['skill']} {trial['object']} repeat {trial['repeat']+1}",flush=True)
   with (a.output/f'{i:03d}.log').open('w') as log:
    process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:process.wait(timeout=900)
    except subprocess.TimeoutExpired:
     os.killpg(process.pid,signal.SIGKILL);process.wait()
     raise RuntimeError('Simulator timeout; no fabricated model result')
  if not marker.exists():raise RuntimeError(f'Missing result: inspect {a.output}/{i:03d}.log')
  row=json.loads(marker.read_text())
  if row.get('denoising_steps')!=100 or row.get('action_horizon')!=8:raise RuntimeError('Inference contract mismatch')
  failure=row.get('failure','')
  if failure and failure not in ['pick_timeout','place_timeout'] and 'RESET_INVALID:' not in failure:
   write(a.output/'blocked.json',dict(trial=i,error=failure))
   raise RuntimeError('Infrastructure/reset contract error: '+failure)
  if row.get('evaluated_skill')!=trial['skill']:raise RuntimeError('Wrong skill result')
  completed.append(row);summary=report(completed,plan);write(a.output/'summary.json',summary)
  print(json.dumps(summary['models']),flush=True)
 print('FINISHED',a.output,flush=True)
if __name__=='__main__':main()
