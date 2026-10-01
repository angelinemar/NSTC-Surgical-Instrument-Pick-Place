"""Deterministic successful-cell schedule; randomness stays inside each cell."""
def progress(saved,cycles,rows,cols):
    if min(cycles,rows,cols)<1 or saved<0: raise ValueError('Invalid coverage counts')
    cells=rows*cols; goal=cycles*cells
    if saved>goal: raise ValueError('Saved count exceeds coverage goal')
    counts=[saved//cells+int(i<saved%cells) for i in range(cells)]
    return dict(contract='successful_cell_cycles_v1',cell_id=None if saved==goal else saved%cells,
                cycle=min(cycles,saved//cells+1),cycles=cycles,rows=rows,cols=cols,
                saved=saved,goal=goal,counts=counts,complete=saved==goal)


def audit_directory(root,target,mode,cycles,rows,cols):
    """Independent on-disk coverage proof; never infer coverage from exit code."""
    import h5py
    from pathlib import Path
    selected=('pick','place') if mode=='both' else (mode,)
    from src.recorder.episode_split import episode_files
    paths={s:{p.name:p for p in episode_files(root,s,target)} for s in selected}
    names=sorted(set().union(*(set(p) for p in paths.values())))
    counts=[0]*(rows*cols); errors=[]; valid=[]
    for name in names:
        try:
            index=int(Path(name).stem.rsplit('_',1)[1]); cell=index%(rows*cols); cycle=index//(rows*cols)+1
            if cycle>cycles: raise ValueError('episode exceeds requested cycles')
            for skill in selected:
                if name not in paths[skill]: raise ValueError(f'missing {skill} segment')
                with h5py.File(paths[skill][name],'r') as h5:
                    from phase4_storage import require_commit
                    require_commit(paths[skill][name], h5)
                    if not bool(h5.attrs.get('success',False)): raise ValueError('not a successful segment')
                    if h5.attrs.get('coverage_contract')!='successful_cell_cycles_v1': raise ValueError('missing coverage contract')
                    if int(h5.attrs.get('coverage_cell_id',-1))!=cell or int(h5.attrs.get('coverage_cycle',-1))!=cycle:
                        raise ValueError('cell/round metadata differs from scheduled episode')
            counts[cell]+=1; valid.append(index)
        except (OSError,ValueError,KeyError) as error:
            errors.append(f'{name}: {error}')
    if valid!=list(range(len(valid))): errors.append('valid saved episodes have holes')
    goal=cycles*rows*cols
    return dict(contract='successful_cell_cycles_v1',source='saved_h5_metadata',target=target,cycles=cycles,
                rows=rows,cols=cols,goal=goal,saved=len(valid),counts=counts,errors=errors,
                complete=not errors and len(valid)==goal and counts==[cycles]*len(counts))
