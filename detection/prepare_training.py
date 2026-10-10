"""Audit every static frame and create scene-disjoint detector splits, retaining masks."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import numpy as np
from PIL import Image
from detection.dataset import CLASSES,write_json
from training.export_detection import visible_boxes
from training.rfdetr_pipeline import validate_coco


def scene_splits(count,seed=20261011):
    if count<10:raise ValueError('Need at least ten independent scenes')
    ids=list(range(count));random.Random(seed).shuffle(ids)
    train=int(count*.8);valid=int(count*.1)
    return {i:('train' if n<train else 'valid' if n<train+valid else 'test') for n,i in enumerate(ids)}


def prepare(source,output,seed=20261011):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('Use a new output folder')
    config=json.loads((source/'config.json').read_text());status=json.loads((source/'status.json').read_text())
    if config.get('contract')!='p4_static_detection_v1' or status['state']!='complete':raise ValueError('Source collection is not complete')
    coco=json.loads((source/'train/_annotations.coco.json').read_text())
    validate_coco(coco,source/'train')
    images_by_path={image['file_name']:image for image in coco['images']}
    if len(images_by_path)!=len(coco['images']):raise ValueError('Duplicate image paths')
    boxes_by_image=defaultdict(list)
    for ann in coco['annotations']:
        boxes_by_image[ann['image_id']].append({k:v for k,v in ann.items() if k not in ('id','image_id')})
    assignment=scene_splits(status['scenes'],seed)
    groups={s:dict(images=[],annotations=[],categories=coco['categories']) for s in ('train','valid','test')}
    audit=dict(contract='static_label_audit_v1',source=str(source),scenes=0,images=0,png_files=0,annotations=0,
               semantic_pixel_counts=Counter(),instances_by_class=Counter(),duplicate_rgb_images=0)
    seen={};index=set();scene_records=[]
    for scene_id in range(status['scenes']):
        folder=source/'train/scenes'/f'scene_{scene_id:06d}'
        meta=json.loads((folder/'scene.json').read_text())
        assert meta['scene_id']==scene_id and meta['resolution']==448 and meta['antialiasing']=='DLAA'
        assert meta['render_settings']['aa_op']==4
        for item in meta['instance_classes'].values():
            assert item['name']==CLASSES[int(item['semantic_id'])-3]
        expected={f'ring_{i:02d}' for i in range(config['ring_cameras'])}|{'top'}
        assert len(meta['views'])==len(expected) and {v['name'] for v in meta['views']}==expected
        split=assignment[scene_id];group=groups[split]
        for view in meta['views']:
            name=view['name'];arrays={}
            for kind in ('rgb','semantic','instance'):
                with Image.open(folder/f'{name}_{kind}.png') as image:
                    image.load();assert image.size==(448,448)
                    if kind=='rgb':assert image.mode=='RGB'
                    arrays[kind]=np.array(image)
            rgb,sem,inst=arrays['rgb'],arrays['semantic'],arrays['instance']
            assert rgb.dtype==np.uint8 and rgb.shape==(448,448,3) and rgb.std(axis=(0,1)).max()>1
            assert sem.shape==inst.shape==(448,448)
            assert np.issubdtype(sem.dtype,np.integer) and np.issubdtype(inst.dtype,np.integer)
            assert np.isin(sem,np.arange(11)).all()
            assert not np.any(((sem>=3)&(sem<=7))&(inst==0)),f'Unmapped tools in scene {scene_id}'
            boxes=visible_boxes(sem,instances=inst,instance_classes=meta['instance_classes'])
            assert boxes==view['boxes'],f'Metadata boxes mismatch: {scene_id}/{name}'
            relative=f'scenes/{folder.name}/{name}_rgb.png';item=images_by_path[relative];index.add(item['id'])
            assert boxes==boxes_by_image[item['id']],f'COCO mismatch: {scene_id}/{name}'
            digest=hashlib.sha256(rgb.tobytes()).hexdigest()
            if digest in seen:
                audit['duplicate_rgb_images']+=1
                assert seen[digest]==split,'Identical RGB leaks across splits'
            seen[digest]=split
            ids,counts=np.unique(sem,return_counts=True)
            audit['semantic_pixel_counts'].update({int(i):int(n) for i,n in zip(ids,counts)})
            audit['instances_by_class'].update(CLASSES[b['category_id']-1] for b in boxes)
            iid=len(group['images'])+1
            group['images'].append(dict(item,id=iid,scene_id=scene_id,rgb_sha256=digest,
                mask_file=relative.replace('_rgb.png','_semantic.png'),
                instance_mask_file=relative.replace('_rgb.png','_instance.png'),instance_classes=meta['instance_classes']))
            for box in boxes:group['annotations'].append(dict(box,id=len(group['annotations'])+1,image_id=iid))
            audit['images']+=1;audit['png_files']+=3;audit['annotations']+=len(boxes)
        audit['scenes']+=1
        scene_records.append(dict(scene_id=scene_id,split=split,seed=meta['seed'],views=len(meta['views'])))
        if (scene_id+1)%25==0:print(f'[AUDIT] {scene_id+1}/{status["scenes"]} scenes, {audit["images"]} aligned RGB/mask pairs',flush=True)
    assert len(index)==len(coco['images'])==status['images']==audit['images']
    assert len(coco['annotations'])==status['annotations']==audit['annotations']
    assert len(list((source/'train/scenes').glob('scene_*')))==status['scenes']
    # Materialize only after the full input has passed. Never modify raw capture.
    output.mkdir(parents=True)
    for record in scene_records:
        relative=Path('scenes')/f'scene_{record["scene_id"]:06d}'
        shutil.copytree(source/'train'/relative,output/record['split']/relative)
    splits={}
    for split,group in groups.items():
        write_json(output/split/'_annotations.coco.json',group)
        splits[split]=validate_coco(group,output/split)
        splits[split]['scenes']=sum(r['split']==split for r in scene_records)
    audit['semantic_pixel_counts']=dict(sorted(audit['semantic_pixel_counts'].items()))
    audit['instances_by_class']=dict(audit['instances_by_class'])
    write_json(output/'label_audit.json',audit)
    write_json(output/'manifest.json',dict(export_complete=True,production_ready=False,
        split_rule='static_scene_grouped_no_view_split',split_seed=seed,rfdetr_dataset_file='roboflow',
        source=str(source),source_config=config,splits=splits,scene_assignments=scene_records,
        note='Synthetic scene-held-out evaluation, not real-world or pick/place generalization. Masks retained; detector training uses RGB and boxes.'))
    print(json.dumps(dict(audit=audit,splits=splits),indent=2),flush=True)
    return audit

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261011)
    args=parser.parse_args();prepare(args.source,args.output,args.seed)
