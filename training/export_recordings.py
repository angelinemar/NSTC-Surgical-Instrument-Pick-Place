"""Export a GUI collection into independent detection and/or joint DP artifacts.

Each session must declare train/valid/test. No frame-level random splitting.
Run with Isaac Python; choose a NEW output outside the raw recording collection.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def inventory(source, purpose):
    splits={s:[] for s in ('train','valid','test')}
    pairs=[]; seeds={}
    for path in sorted(source.rglob('capture_contract.json')):
        c=json.loads(path.read_text())
        required={'detection'} if purpose=='detection' else {'dp_joint'} if purpose=='dp' else {'detection','dp_joint'}
        if not required.issubset(c['consumers']):
            continue
        split=c['split']; seed=c['session_seed']
        if split not in splits:
            raise ValueError(f'Assign a whole-session split before export: {path}')
        if seeds.setdefault(seed,split)!=split:
            raise ValueError('Session seed leakage across splits')
        folder=path.parent
        metrics=json.loads((folder/'run_metrics.json').read_text())
        if metrics.get('runtime_error'):
            raise ValueError(f'Recorder reported runtime error: {folder}')
        session=json.loads((folder/'session.json').read_text())
        status=json.loads((folder/'session.status.json').read_text())
        if status.get('state')!='complete':
            raise ValueError(f'Incomplete recording: {folder}')
        if session.get('collection')=='grid_cycles':
            if not json.loads((folder/'coverage_report.json').read_text()).get('complete'):
                raise ValueError('Grid coverage incomplete')
        skills=['pick','place'] if c['saved_skill']=='both' else [c['saved_skill']]
        files={s:sorted(folder.glob(s+'_policy/*/episode_*.h5')) for s in skills}
        names={s:{p.name for p in paths} for s,paths in files.items()}
        if not files[skills[0]] or any(v!=names[skills[0]] for v in names.values()):
            raise ValueError('Missing saved segments')
        if len(files[skills[0]])!=status.get('requested'):
            raise ValueError('Saved segment count differs from requested goal')
        splits[split].append(folder)
        for i,path0 in enumerate(files[skills[0]]):
            case=dict(id=f'{folder.parent.name}_{folder.name}_{path0.stem}',
                      object=session['target'],session_seed=seed,saved_skills=skills)
            pairs.append((case,split,[files[s][i] for s in skills]))
    if any(not runs for runs in splits.values()):
        raise ValueError('Need independently recorded train, valid AND test sessions for selected purpose')
    if purpose in ('dp','both'):
        for skill in ('pick','place'):
            covered={split for case,split,_ in pairs if skill in case['saved_skills']}
            if covered and covered!=set(splits):
                raise ValueError(f'{skill} needs independent train, valid AND test sessions')
    return splits,pairs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--purpose',choices=('detection','dp','both'),default='both')
    p.add_argument('--stride',type=int,default=20)
    a=p.parse_args(); a.source=a.source.resolve(); a.output=a.output.resolve()
    if a.output.is_relative_to(a.source):
        p.error('Output must be outside the raw source collection')
    splits,pairs=inventory(a.source,a.purpose)
    a.output.mkdir(parents=True,exist_ok=False)
    if a.purpose in ('detection','both'):
        from training.export_detection import export
        export(splits,a.output/'detection',a.stride)
    if a.purpose in ('dp','both'):
        from training.export_sensor_only import export
        export(a.source,a.output/'dp',set(),panel_pairs=pairs)
    (a.output/'export_complete.json').write_text(json.dumps(dict(purpose=a.purpose,
        production_ready=False,raw_duplicated=False),indent=2))


if __name__=='__main__': main()
