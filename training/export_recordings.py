"""Export a GUI collection into independent detection and/or joint DP artifacts.

Sessions declare either a fixed split or committed per-episode automatic splits.
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
    pairs=[]; seeds={}; episode_seeds={}; transactions={}
    for path in sorted(source.rglob('capture_contract.json')):
        c=json.loads(path.read_text())
        required={'detection'} if purpose=='detection' else {'dp_joint'} if purpose=='dp' else {'detection','dp_joint'}
        if not required.issubset(c['consumers']):
            continue
        split=c['split']; seed=c['session_seed']
        automatic = split == 'auto' and c.get('split_contract') == 'episode_grid_balanced_v1'
        if split not in splits and not automatic:
            raise ValueError(f'Assign a whole-session split before export: {path}')
        if not automatic and seeds.setdefault(seed,split)!=split:
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
        from src.recorder.episode_split import episode_files
        files={s:episode_files(folder,s) for s in skills}
        names={s:{p.name for p in paths} for s,paths in files.items()}
        if not files[skills[0]] or any(v!=names[skills[0]] for v in names.values()):
            raise ValueError('Missing saved segments')
        if len(files[skills[0]])!=status.get('requested'):
            raise ValueError('Saved segment count differs from requested goal')
        if not automatic:
            splits[split].append(folder)
        for i,path0 in enumerate(files[skills[0]]):
            assigned = path0.parents[2].name if automatic else split
            if assigned not in splits:
                raise ValueError('Missing automatic episode split folder')
            if automatic:
                import h5py
                from src.recorder.phase4_storage import require_commit
                for skill in skills:
                    segment = files[skill][i]
                    with h5py.File(segment,'r') as h:
                        require_commit(segment,h)
                        if (segment.parents[2].name != assigned or
                                h.attrs.get('dataset_split') != assigned or
                                h.attrs.get('split_contract') != 'episode_grid_balanced_v1'):
                            raise ValueError('Automatic pair split metadata disagrees')
                        tx=str(h.attrs['pair_transaction_id'])
                        if transactions.setdefault(tx,assigned)!=assigned:
                            raise ValueError('Episode transaction crosses splits')
                        randomization=json.loads(h.attrs.get('domain_randomization','{}'))
                        if 'seed' not in randomization:
                            raise ValueError('Auto split export requires episode randomization metadata')
                        key=(session['target'],seed,randomization['seed'])
                        if episode_seeds.setdefault(key,assigned)!=assigned:
                            raise ValueError('Repeated episode randomization crosses splits')
                if path0.parents[2] not in splits[assigned]:
                    splits[assigned].append(path0.parents[2])
            case=dict(id=f'{folder.parent.name}_{folder.name}_{path0.stem}',
                      object=session['target'],session_seed=seed,saved_skills=skills)
            pairs.append((case,assigned,[files[s][i] for s in skills]))
    if any(not runs for runs in splits.values()):
        raise ValueError('Need saved train, valid AND test episodes for selected purpose; small collections may not have all three yet')
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
