"""Windows Kit cleanup-crash workaround, gated by durable committed recordings.

Only call after recorder save/physical/coverage gates succeeded. Do not use this
to recover an arbitrary crashed process. Native cleanup remains available for
debugging; process isolation lets Windows reclaim the simulator's GPU resources.
"""
import json
import os
from pathlib import Path
import sys


def verify_and_exit(root, object_name, skill, requested, exit_process=None):
    import h5py
    from phase4_storage import require_commit
    root=Path(root)
    skills=('pick','place') if skill=='both' else (skill,)
    if any(s not in ('pick','place') for s in skills) or requested < 1:
        raise ValueError('Invalid completion goal')
    names=None; checked=[]
    for selected in skills:
        paths=sorted((root/(selected+'_policy')/object_name).glob('episode_*.h5'))
        current={p.name for p in paths}
        if len(paths)<requested or (names is not None and names!=current):
            raise RuntimeError('Incomplete committed segment set')
        names=current
        for path in paths:
            with h5py.File(path,'r') as h:
                if h.attrs.get('storage_contract')!='journaled_episode_v2':
                    raise ValueError('Verified exit requires checksummed journal v2')
                if not h.attrs.get('success') or h.attrs.get('policy_skill')!=selected:
                    raise ValueError('Invalid segment success/skill')
                require_commit(path,h)
                checked.append(str(path.relative_to(root)))
    payload=dict(recording_complete=True, requested=requested, checked_segments=checked,
                 shutdown_mode='verified_process_exit', native_cleanup_skipped=True,
                 production_ready=False)
    temp=root/'run_completion.tmp'
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(payload,stream,indent=2); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp,root/'run_completion.json')
    print('[P4 COMPLETE] committed segments verified; exiting isolated recorder (Windows Kit cleanup workaround)',flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    (exit_process or os._exit)(0)
