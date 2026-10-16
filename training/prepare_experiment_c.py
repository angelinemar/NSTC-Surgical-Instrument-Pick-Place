"""Balanced synthetic/DP segmentation dataset with audited episode separation."""
import argparse,collections,hashlib,json,random
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from detection.dataset import CATEGORIES,write_json
from training.export_detection import CLASSES,CAMERAS
from training.evaluate_dp_cameras import instance_boxes
from training.evaluate_segmentation import rle
from phase4_storage import require_commit


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def rounded(value):
    if isinstance(value,float):return round(value,5)
    if isinstance(value,list):return [rounded(v) for v in value]
    if isinstance(value,dict):return {k:rounded(v) for k,v in value.items()}
    return value

def guard(seen,key,split,description):
    if key in seen and seen[key]!=split:raise ValueError(f'Cross-split leakage: {description}: {seen[key]} / {split}')
    seen[key]=split

def prepare(synthetic,dp_root,output):
    if output.exists():raise ValueError('Output exists; preserve it and select a new output for a fresh preparation')
    source=json.loads((synthetic/'manifest.json').read_text())
    assert source['export_complete'] and source['split_rule']=='configuration_disjoint_hemisphere_lighting_v1'
    groups={s:{c:[] for c in CLASSES} for s in ('train','valid','test')}
    pairs={};layouts={};inventory=[]
    for path in sorted(dp_root.rglob('episode_*.h5')):
        split=next((s for s in groups if s in path.parts),None)
        if not split:continue
        with h5py.File(path) as h:
            require_commit(path,h)
            assert str(h.attrs['dataset_split'])==split
            pair=str(h.attrs['pair_transaction_id']);layout=digest(rounded(json.loads(h.attrs['spawn_requested_all'])))
            guard(pairs,pair,split,'DP pair');guard(layouts,layout,split,'DP requested layout rounded to 10 micrometres')
            target=path.parent.name
            assert target in CLASSES and len(h['actions'])>0
            groups[split][target].append((path,len(h['actions'])))
            inventory.append(dict(path=str(path),split=split,pair=pair,layout_sha256=layout,frames=len(h['actions']),target=target))
    output.mkdir(parents=True);write_json(output/'source_episodes.json',inventory)
    # Hash every selected decoded RGB. Duplicate checks include SYN vs DP, all splits.
    rgb_seen={};selection=[];datasets={};reports={};syn_seeds={};syn_layouts={}
    for split in groups:
        folder=output/split;folder.mkdir();(folder/'synthetic').symlink_to((synthetic/split).resolve(),target_is_directory=True);(folder/'dp').mkdir()
        src=json.loads((synthetic/split/'_annotations.coco.json').read_text())
        assert len(src['images'])%30==0,'Need equal per-class/per-camera quotas'
        # Re-audit source scenes, not just a copied manifest assertion.
        scene_paths=sorted((synthetic/split/'scenes').glob('scene_*/scene.json'))
        assert len(scene_paths)==source['splits'][split]['scenes']
        for path in scene_paths:
            scene=json.loads(path.read_text());guard(syn_seeds,scene['seed'],split,'synthetic seed')
            signature=digest(sorted([(x['object'],rounded(x['pose'])) for x in scene['objects']]))
            guard(syn_layouts,signature,split,'synthetic layout')
        coco=dict(images=[],annotations=[],categories=CATEGORIES);byid={}
        for box in src['annotations']:byid.setdefault(box['image_id'],[]).append(box)
        for item in src['images']:
            path=synthetic/split/item['file_name'];rgb=np.asarray(Image.open(path).convert('RGB'));key=hashlib.sha256(rgb.tobytes()).hexdigest()
            guard(rgb_seen,key,split,'decoded RGB')
            image_id=len(coco['images'])+1
            coco['images'].append(dict(item,id=image_id,file_name='synthetic/'+item['file_name'],domain='synthetic'))
            for box in byid.get(item['id'],[]):coco['annotations'].append(dict(box,id=len(coco['annotations'])+1,image_id=image_id))
        target_count=len(src['images']);quota=target_count//30;class_counts=collections.Counter();camera_counts=collections.Counter();rejected=0
        # A low-discrepancy temporal order visits midpoint, quarters, eighths, ... .
        def frame_index(round_index,n):
            x=round_index+1;frac=0.;scale=.5
            while x:frac+=(x&1)*scale;x>>=1;scale*=.5
            return min(n-1,int(frac*n))
        for class_index,target in enumerate(CLASSES):
            episodes=list(groups[split][target]);random.Random(42+class_index).shuffle(episodes)
            if not episodes:raise ValueError('Missing DP target class '+target)
            for camera in CAMERAS:
                accepted=0;used=set()
                for attempt in range(len(episodes)*2048):
                    if accepted==quota:break
                    path,n=episodes[attempt%len(episodes)];t=frame_index(attempt//len(episodes),n)
                    if (path,t) in used:continue
                    used.add((path,t))
                    with h5py.File(path) as h:
                        rgb=h['observations/'+camera+'_rgb'][t];key=hashlib.sha256(rgb.tobytes()).hexdigest()
                        if key in rgb_seen:
                            # Reject duplicates rather than moving a test sample into training.
                            rejected+=1;continue
                        prefix='grip_b' if camera=='wrist' else camera
                        ins=h['observations/'+prefix+'_instance'][t];sem=h['observations/'+prefix+'_semantic'][t]
                        boxes=instance_boxes(sem,ins,json.loads(h.attrs['instance_classes']))
                        if not boxes:rejected+=1;continue
                        for b in boxes:
                            b['segmentation']=rle(ins==b['instance_id']);assert int(np.sum(ins==b['instance_id']))==b['area']
                        pair=str(h.attrs['pair_transaction_id'])
                    rgb_seen[key]=split;image_id=len(coco['images'])+1;name=f'dp/{image_id:07d}.png';Image.fromarray(rgb).save(folder/name)
                    meta=dict(id=image_id,file_name=name,width=rgb.shape[1],height=rgb.shape[0],domain='dp',camera=camera,source=str(path),frame=t,pair=pair,target=target)
                    coco['images'].append(meta)
                    for b in boxes:
                        coco['annotations'].append(dict(b,id=len(coco['annotations'])+1,image_id=image_id));class_counts[CLASSES[b['category_id']-1]]+=1
                    selection.append(dict(split=split,**meta,rgb_sha256=key));accepted+=1;camera_counts[camera]+=1
                if accepted!=quota:raise ValueError(f'Not enough unique frames for {split}/{target}/{camera}: {accepted}/{quota}')
                print(f'{split}: DP {target}/{camera} {accepted}/{quota}',flush=True)
        assert len(coco['images'])==target_count*2
        write_json(folder/'_annotations.coco.json',coco);datasets[split]=coco
        reports[split]=dict(images=len(coco['images']),synthetic_images=target_count,dp_images=target_count,annotations=len(coco['annotations']),dp_camera_counts=dict(camera_counts),dp_visible_class_counts=dict(class_counts),duplicate_or_empty_candidates_rejected=rejected)
    # Build separate domain views for evaluation with exactly the same image IDs/labels.
    for domain in ('synthetic','dp'):
        for split in ('valid','test'):
            view=output/'evaluation'/domain/split;view.mkdir(parents=True)
            for name in ('synthetic','dp'):(view/name).symlink_to((output/split/name).resolve(),target_is_directory=True)
            src=datasets[split];images=[i for i in src['images'] if i['domain']==domain];ids={i['id'] for i in images}
            write_json(view/'_annotations.coco.json',dict(images=images,annotations=[b for b in src['annotations'] if b['image_id'] in ids],categories=CATEGORIES))
    write_json(output/'selection.json',selection)
    write_json(output/'manifest.json',dict(contract='nstc_seg_dataset_v1',export_complete=True,split_rule='balanced_syn_dp_episode_disjoint_v1',splits=reports,
        source_synthetic=str(synthetic.resolve()),source_manifest_sha256=hashlib.sha256((synthetic/'manifest.json').read_bytes()).hexdigest(),seed=42,
        annotation_sha256={s:hashlib.sha256((output/s/'_annotations.coco.json').read_bytes()).hexdigest() for s in groups},
        audits=['committed DP checksum','original DP split','pick/place pair disjoint','rounded DP requested layout disjoint','synthetic seed/layout disjoint','selected decoded RGB hash disjoint across domains and splits'],
        balance='50/50 SYN/DP images; equal DP target-episode class and camera counts. Visible object annotation counts are not forced equal.',
        limitations=['DP camera identities are shared across splits; held out episodes/layouts, not camera identities.','Exact-duplicate and metadata audits cannot prove absence of visually similar scenes.','DP labels have known missing/partial instruments; masks preserved unchanged.','DP validation/test were inspected in previous experiments; these are not a fresh blinded final benchmark.']))
    print(json.dumps(reports,indent=2));print('Experiment C dataset ready:',output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--synthetic',type=Path,required=True);p.add_argument('--dp-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.synthetic,a.dp_root,a.output)
