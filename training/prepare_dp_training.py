"""Resumable episode-pair export; preserve the original DP splits and all five tasks."""
import argparse,collections,hashlib,json,shutil
from pathlib import Path
import h5py,numpy as np
from training.export_sensor_only import export

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--image-size',type=int,choices=[224,448],default=224);a=p.parse_args()
 entries=json.loads(a.inventory.read_text());pairs={};layouts={}
 for row in entries:
  pair=row['pair'];pairs.setdefault(pair,[]).append(row)
  old=layouts.setdefault(row['layout_sha256'],row['split'])
  if old!=row['split']:raise ValueError('Layout crosses splits')
 groups=[]
 for pair,rows in sorted(pairs.items()):
  if len(rows)!=2 or len({r['split'] for r in rows})!=1 or len({r['target'] for r in rows})!=1:raise ValueError('Invalid episode pair')
  groups.append((pair,rows))
 if a.smoke:
  groups=[next(g for g in groups if g[1][0]['split']==s) for s in ['train','valid','test']]
 a.output.mkdir(parents=True,exist_ok=True);frozen=dict(inventory_sha256=hashlib.sha256(a.inventory.read_bytes()).hexdigest(),smoke_only=a.smoke,policy_image_size=a.image_size,pairs=[p for p,_ in groups])
 index=a.output/'selection.json'
 if index.exists() and json.loads(index.read_text())!=frozen:raise ValueError('Output selection differs')
 index.write_text(json.dumps(frozen,indent=2));shards=a.output/'episodes';shards.mkdir(exist_ok=True)
 for n,(pair,rows) in enumerate(groups):
  target=shards/pair;done=target/'manifest.json'
  if done.exists() and (target/'source_stat.json').exists():
   m=json.loads(done.read_text())
   if m.get('policy_image_size')!=a.image_size:raise ValueError('Shard resolution mismatch')
   if not m.get('export_complete'):raise ValueError('Incomplete shard marker')
   # Refuse silent reuse if raw files changed since export.
   snapshots=json.loads((target/'source_stat.json').read_text())
   if any(Path(x['path']).stat().st_size!=x['size'] or Path(x['path']).stat().st_mtime_ns!=x['mtime_ns'] for x in snapshots):raise ValueError('Raw source changed; use a fresh export')
  else:
   if target.exists():shutil.rmtree(target) # Only an unfinished pair shard inside this dedicated export.
   paths=[Path(r['path']) for r in rows]
   with h5py.File(paths[0]) as h:seed=json.loads(h.attrs['domain_randomization'])['session_seed']
   case=dict(id=pair,object=rows[0]['target'],session_seed=seed,saved_skills=['pick','place'])
   export(paths[0].parents[5],target,set(),panel_pairs=[(case,rows[0]['split'],paths)],image_size=a.image_size)
   (target/'source_stat.json').write_text(json.dumps([dict(path=str(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths]))
  print(f'Prepared {n+1}/{len(groups)} episode pairs',flush=True)
 if (a.output/'manifest.json').exists():
  manifest=json.loads((a.output/'manifest.json').read_text())
  if manifest.get('selection')==frozen and manifest.get('export_complete'):
   print('DATASET ALREADY READY',a.output,flush=True);return
 # Assemble tiny index H5 files via external links; image arrays are not duplicated.
 all_episodes=[];counts={};base=None;contracts=None
 with h5py.File(a.output/'perception_labels.hdf5','w') as labels:
  for skill in ['pick','place']:
   splits={s:[] for s in ['train','valid','test']};labels.create_group(skill)
   with h5py.File(a.output/(skill+'.hdf5'),'w') as f:
    data=f.create_group('data');f.attrs['sensor_only_contract']='p4_sensor_task_v2'
    for n,(pair,rows) in enumerate(groups):
     folder=shards/pair;m=json.loads((folder/'manifest.json').read_text());base=m if base is None else base
     if contracts is None:contracts=m['controller_contract']
     if contracts!=m['controller_contract']:raise ValueError('Mixed controller contracts')
     key=f'demo_{n:06d}';data[key]=h5py.ExternalLink(f'episodes/{pair}/{skill}.hdf5','/data/demo_000000');labels[skill][key]=h5py.ExternalLink(f'episodes/{pair}/perception_labels.hdf5',f'/{skill}/demo_000000');splits[rows[0]['split']].append(key)
     e=next(e for e in m['episodes'] if e['skill']==skill);all_episodes.append(dict(e,id=key))
    masks=f.create_group('mask')
    for split,ids in splits.items():masks.create_dataset(split,data=np.asarray(ids,dtype='S'))
    data.attrs['total']=sum(len(data[key]['actions']) for key in data)
   counts[skill]={s:len(v) for s,v in splits.items()}
 hashes=[e['sha256'] for e in all_episodes]
 if len(hashes)!=len(set(hashes)):raise ValueError('Duplicate raw source content across episode pairs')
 base.update(episodes=all_episodes,split_counts=counts,smoke_only=a.smoke,export_complete=True,selection=frozen,storage='external H5 links to resumable episode-pair shards')
 tmp=a.output/'manifest.json.tmp';tmp.write_text(json.dumps(base,indent=2));tmp.replace(a.output/'manifest.json');print('DATASET READY',a.output,counts,flush=True)
if __name__=='__main__':main()
