"""One terminal entry for Trial 02 preparation, recording, training and evaluation."""
import argparse,json,os,subprocess,sys,time
from datetime import datetime
from pathlib import Path
from detection.paths import object_datasets_root
from detection.prepare_seg_protocol import prepare
from detection.prepare_segmentation import prepare as export

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=object_datasets_root()/'trials/trial_02_rfdetr_seg_hemisphere768'


def capture(root,counts,headless,smoke=False):
    raw=root/('smoke_capture' if smoke else 'captures');logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
    for split,goal in counts.items():
        destination=raw/split;status=destination/'status.json'
        if (destination/'config.json').exists():
            previous_config=json.loads((destination/'config.json').read_text())
            expected_protocol=json.loads((root/'protocol'/(split+'.json')).read_text())
            if previous_config.get('segmentation_protocol')!=expected_protocol:
                raise ValueError('Capture protocol changed; use a new trial root')
        if status.exists():
            state=json.loads(status.read_text())
            if state['state']=='complete' and state['scenes']==goal:
                print(f'{split}: {goal} scenes already complete',flush=True);continue
        cmd=[str(ROOT/'scripts/launchers/run_linux6.sh'),'python','detection/record.py',
            '--seg-protocol',str(root/'protocol'/(split+'.json')),'--output',str(destination),'--scenes',str(goal),
            '--min-objects','10' if smoke else '5','--max-objects','15' if smoke else '25']
        if headless:cmd+=['--headless']
        if (destination/'config.json').exists():cmd+=['--resume']
        log=logs/(('smoke_' if smoke else '')+split+'.log')
        print(f'{split}: starting Isaac Sim; full log: {log}',flush=True)
        with log.open('a') as f:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            previous=None
            try:
                while proc.poll() is None:
                    if status.exists():
                        s=json.loads(status.read_text());progress=(s.get('scenes'),s.get('images'))
                        if progress!=previous:print(f'{split}: {progress[0]}/{goal} scenes, {progress[1]} images',flush=True);previous=progress
                    time.sleep(1)
            except KeyboardInterrupt:
                proc.terminate();proc.wait();raise
        if proc.returncode:raise RuntimeError(f'Capture failed ({proc.returncode}); inspect {log}. Rerun this command to resume committed scenes.')
    dataset=root/('smoke_dataset' if smoke else 'dataset')
    if dataset.exists():
        m=json.loads((dataset/'manifest.json').read_text())
        if any(m['splits'][s]['scenes']!=counts[s] for s in counts):raise ValueError('Existing exported dataset has different counts; use a separate trial root')
    else:print(json.dumps(export(raw,dataset),indent=2))
    print('Dataset ready:',dataset,flush=True)
    return dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial-root',type=Path,default=DEFAULT)
    sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--dp-source',type=Path,default=object_datasets_root().parent/'DP datasets/NSTC')
    rec=sub.add_parser('record');rec.add_argument('--train-scenes',type=int,default=1000);rec.add_argument('--valid-scenes',type=int,default=150);rec.add_argument('--test-scenes',type=int,default=150);rec.add_argument('--headless',action='store_true')
    smoke=sub.add_parser('smoke');smoke.add_argument('--headless',action='store_true')
    sub.add_parser('train');sub.add_parser('evaluate');sub.add_parser('evaluate-dp')
    a,extra=p.parse_known_args();root=a.trial_root.resolve();root.mkdir(parents=True,exist_ok=True)
    if a.command in ('prepare','record','smoke') and extra:p.error('Unrecognized arguments: '+' '.join(extra))
    if a.command=='prepare':print(json.dumps(prepare(a.dp_source,root/'protocol',ROOT),indent=2));return
    if a.command in ('record','smoke'):
        counts={s:2 if a.command=='smoke' else getattr(a,s+'_scenes') for s in ('train','valid','test')}
        if min(counts.values())<1:p.error('Every split needs at least one scene')
        dataset=capture(root,counts,a.headless,a.command=='smoke')
        if a.command=='record':return
        output=root/'smoke_training'
        if (output/'result.json').exists():print('Smoke training already complete:',output);return
        if output.exists() and any(output.iterdir()):output=root/('smoke_training_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        cmd=[sys.executable,'-m','training.rfdetr_seg_pipeline','--dataset',str(dataset),'--output',str(output),'--epochs','1','--batch-size','1','--grad-accum-steps','2','--num-workers','2']
    elif a.command=='train':
        output=root/'training'/datetime.now().strftime('%Y%m%d_%H%M%S')
        cmd=[sys.executable,'-m','training.rfdetr_seg_pipeline','--dataset',str(root/'dataset'),'--output',str(output),'--wandb',*extra]
    else:
        output=root/'evaluations'/(( 'dp_' if a.command=='evaluate-dp' else 'synthetic_')+datetime.now().strftime('%Y%m%d_%H%M%S'))
        source=['--dp-root',str(object_datasets_root().parent/'DP datasets/NSTC')] if a.command=='evaluate-dp' else ['--dataset',str(root/'dataset')]
        cmd=[sys.executable,'-m','training.evaluate_segmentation',*source,'--output',str(output),*extra]
    os.chdir(ROOT);os.execv(sys.executable,cmd)

if __name__=='__main__':main()
