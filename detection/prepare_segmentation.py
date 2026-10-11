"""Audit/export independently captured configuration-disjoint segmentation splits."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from detection.dataset import CLASSES,CATEGORIES,write_json
from detection.segmentation_protocol import LIGHT_MASKS,AZIMUTH_BANDS


def prepare(captures,output):
    output=Path(output)
    if output.exists():raise ValueError('Choose a new dataset output directory')
    reports={};records={};hashes={};layouts={};seeds={};configs={};reservation=None
    for split in ('train','valid','test'):
        source=Path(captures)/split;config=json.loads((source/'config.json').read_text());status=json.loads((source/'status.json').read_text())
        if config['contract']!='p4_static_segmentation_v2' or config['split']!=split or status['state']!='complete':raise ValueError('Incomplete or incorrectly assigned split '+split)
        protocol=config['segmentation_protocol'];configs[split]=config
        if reservation is None:reservation=protocol['dp_reservation_sha256']
        if reservation!=protocol['dp_reservation_sha256']:raise ValueError('DP reservation differs across splits')
        coco=dict(images=[],annotations=[],categories=CATEGORIES);class_counts={c:0 for c in CLASSES};qualities=[]
        scene_paths=sorted((source/'train/scenes').glob('scene_*/scene.json'))
        assert len(scene_paths)==status['scenes'],'Scene count does not match committed status'
        for expected,path in enumerate(scene_paths):
            scene=json.loads(path.read_text());assert scene['scene_id']==expected
            seed=scene['seed'];assert seeds.setdefault(seed,split)==split,'Scene seed leaks across splits'
            layout=[(r['object'],[round(x,5) for x in r['pose']]) for r in scene['objects']]
            fingerprint=hashlib.sha256(json.dumps(sorted(layout),sort_keys=True).encode()).hexdigest()
            assert layouts.setdefault(fingerprint,split)==split,'Instrument arrangement leaks across splits'
            assert scene['lighting']['enabled_mask'] in LIGHT_MASKS[split]
            for view in scene['views']:
                lo,hi=AZIMUTH_BANDS[split];assert lo-1e-6<=view['azimuth_deg']%60<=hi+1e-6
                assert view['dp_clearance_m']>=protocol['dp_exclusion_radius_m']+.02
                prefix=path.parent/view['name'];rgb=Path(str(prefix)+'_rgb.png');inst=Path(str(prefix)+'_instance.png');sem=Path(str(prefix)+'_semantic.png')
                with Image.open(rgb) as im:
                    im.load();assert im.size==(config['resolution'],)*2 and im.mode=='RGB'
                    digest=hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
                assert hashes.setdefault(digest,split)==split,'RGB leakage'
                instance=np.asarray(Image.open(inst));semantic=np.asarray(Image.open(sem))
                image_id=len(coco['images'])+1
                coco['images'].append(dict(id=image_id,file_name=str(rgb.relative_to(source/'train')),
                    width=config['resolution'],height=config['resolution'],scene_id=expected,camera=view['name'],
                    distance_m=view['distance_m'],distance_band=view['distance_band'],quality=view['quality']))
                for box in view['boxes']:
                    mask=instance==box['instance_id'];assert np.all(semantic[mask]==box['category_id']+2)
                    raw=box['segmentation'];rle=coco_mask.frPyObjects(raw,*raw['size'])
                    assert np.array_equal(coco_mask.decode(rle).astype(bool),mask)
                    assert int(coco_mask.area(rle))==box['area']
                    assert np.array_equal(coco_mask.toBbox(rle),box['bbox'])
                    rle['counts']=rle['counts'].decode('ascii')
                    annotation=dict(box,id=len(coco['annotations'])+1,image_id=image_id,segmentation=rle)
                    coco['annotations'].append(annotation);class_counts[CLASSES[box['category_id']-1]]+=1
                qualities.append(dict(scene=expected,camera=view['name'],distance_band=view['distance_band'],**view['quality']))
        assert len(coco['images'])==status['images']
        assert all(class_counts.values()),f'Missing class in {split}'
        reports[split]=dict(scenes=status['scenes'],images=len(coco['images']),annotations=len(coco['annotations']),class_counts=class_counts,quality=qualities)
        records[split]=coco
    output.mkdir(parents=True)
    for split,coco in records.items():
        folder=output/split;folder.mkdir();(folder/'scenes').symlink_to((Path(captures)/split/'train/scenes').resolve(),target_is_directory=True)
        write_json(folder/'_annotations.coco.json',coco)
    manifest=dict(contract='nstc_seg_dataset_v1',export_complete=True,split_rule='configuration_disjoint_hemisphere_lighting_v1',
        source_configs=configs,splits=reports,classes=list(CLASSES),dp_reservation_sha256=reservation,
        leakage_audits=['scene_seed','instrument_pose_hash','RGB_hash','azimuth_partition','lighting_mask_partition','DP_camera_clearance'],
        masks='exact visible per-body COCO RLE; holes and disconnected components preserved',production_ready=False)
    write_json(output/'manifest.json',manifest)
    return {s:{k:v for k,v in r.items() if k!='quality'} for s,r in reports.items()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--captures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.captures,a.output),indent=2))
