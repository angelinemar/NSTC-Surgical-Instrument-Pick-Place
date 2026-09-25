"""Collect reproducible independent sessions for DP or perception.

One process per instrument/session; existing footprint and physical success
gates remain mandatory. Nonzero recorder exits stop this orchestrator.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.recorder.capture_contract import storage_budget
OBJECTS = ('scalpel','scissor','love_retractor','kelly','scalpel_type2')


def plan(output, track, cycles, camera_size, sessions):
    jobs=[]
    for split, seeds in sessions.items():
        for seed in seeds:
            for name in OBJECTS:
                folder=Path(output)/split/f'{name}_seed{seed}'
                config=dict(target=name, mode='auto', auto_start=True, stop=False,
                            collection='grid_cycles',cycles=cycles,
                            tray_mode='full' if track in ('dp','both') else 'random')
                args=['scripts/p4.py','record','--object',name,'--episodes',str(10*cycles),
                      '--record_mode','both','--camera-size',str(camera_size),
                      '--randomization','train','--randomization-seed',str(seed),
                      '--dataset-purpose', 'detection' if track=='perception' else track,
                      '--dataset-split',split,
                      '--max-attempts',str(100*cycles),'--session-config',str(folder/'session.json'),
                      '--tray-occupancy',config['tray_mode'],'--out_dir',str(folder),'--headless']
                jobs.append(dict(split=split,seed=seed,object=name,folder=str(folder),config=config,args=args))
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--track',choices=('dp','perception','both'),required=True)
    p.add_argument('--python',type=Path,default=Path('C:/IsaacLab/_isaac_sim/python.bat'))
    p.add_argument('--cycles',type=int,default=1)
    p.add_argument('--camera-size',type=int,choices=(224,448),default=448)
    p.add_argument('--train-seeds',type=int,nargs='+',default=[1701,1702])
    p.add_argument('--valid-seeds',type=int,nargs='+',default=[2801])
    p.add_argument('--test-seeds',type=int,nargs='+',default=[3901])
    p.add_argument('--run',action='store_true',help='Without this flag prints the complete plan only')
    p.add_argument('--reserve-gb',type=float,default=30.,help='Free disk reserve, minimum 10 GB')
    args=p.parse_args()
    sessions={s:getattr(args,s+'_seeds') for s in ('train','valid','test')}
    seeds=[v for values in sessions.values() for v in values]
    if args.cycles<1 or len(seeds)!=len(set(seeds)):
        p.error('Positive cycles and distinct session seeds across splits required')
    if args.reserve_gb < 10:
        p.error('reserve-gb must be at least 10')
    jobs=plan(args.output.resolve(),args.track,args.cycles,args.camera_size,sessions)
    drive=args.output.resolve()
    while not drive.exists(): drive=drive.parent
    pair_bytes=int((.9 if args.camera_size==448 else .25)*1e9)
    reserve=int(args.reserve_gb*1e9)
    storage=storage_budget(args.output,len(jobs)*10*args.cycles,args.camera_size,args.reserve_gb)
    storage['note']='Conservative planning estimate, compressed sizes vary; checked before every job'
    if not args.run:
        print(json.dumps(dict(track=args.track, storage=storage,jobs=jobs),indent=2))
        return
    if not storage['capacity_pass']:
        raise RuntimeError('Insufficient disk for collection plan: '+json.dumps(storage))
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'collection_plan.json').write_text(json.dumps(jobs,indent=2))
    for job in jobs:
        if shutil.disk_usage(drive).free < reserve + 10*args.cycles*pair_bytes:
            raise RuntimeError('Disk reserve reached; completed jobs preserved, cohort remains incomplete')
        folder=Path(job['folder']); folder.mkdir(parents=True)
        (folder/'session.json').write_text(json.dumps(job['config'],indent=2))
        log=ROOT/'debug/logs'/args.output.name/job['split']/f"{job['object']}_{job['seed']}.log"
        log.parent.mkdir(parents=True,exist_ok=True)
        print('COLLECT',job['split'],job['object'],job['seed'],flush=True)
        with log.open('w') as stream:
            subprocess.run([str(args.python),*job['args']],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    (args.output/'collection_complete.json').write_text(json.dumps(dict(jobs=len(jobs),training_ready=False,
        next='Full pixel/commit audit, session-preserving export, model metrics and physical rollout required'),indent=2))


if __name__=='__main__': main()
