"""Build experiment B: unchanged synthetic train, original DP valid/test masks."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from detection.dataset import CATEGORIES,write_json
from training.evaluate_segmentation import samples


def prepare(synthetic,dp_root,output,max_files=0,frames_per_camera=0):
    source=json.loads((synthetic/'manifest.json').read_text())
    assert source['contract']=='nstc_seg_dataset_v1' and source['export_complete']
    config=dict(synthetic=str(synthetic.resolve()),dp_root=str(dp_root.resolve()),
        synthetic_manifest_sha256=hashlib.sha256((synthetic/'manifest.json').read_bytes()).hexdigest(),
        max_files=max_files,frames_per_camera=frames_per_camera)
    output.mkdir(parents=True,exist_ok=True)
    cp=output/'preparation.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Preparation settings differ; use another output')
    write_json(cp,config)
    train=output/'train'
    if not train.exists():train.symlink_to((synthetic/'train').resolve(),target_is_directory=True)
    assert train.resolve()==(synthetic/'train').resolve()
    reports={}
    for split in ('valid','test'):
        target=output/split;done=target/'_annotations.coco.json'
        if done.exists():
            coco=json.loads(done.read_text())
        else:
            target.mkdir(exist_ok=True);(target/'images').mkdir(exist_ok=True)
            coco=dict(images=[],annotations=[],categories=CATEGORIES)
            args=SimpleNamespace(dataset=None,dp_root=dp_root,split=split,max_files=max_files,frames_per_camera=frames_per_camera)
            for index,(rgb,boxes,meta) in enumerate(samples(args),1):
                name=f'images/{index:08d}.png'
                temp=(target/name).with_suffix('.tmp')
                Image.fromarray(rgb).save(temp,format='PNG');temp.replace(target/name)
                coco['images'].append(dict(id=index,file_name=name,width=rgb.shape[1],height=rgb.shape[0],**meta))
                for b in boxes:coco['annotations'].append(dict(b,id=len(coco['annotations'])+1,image_id=index))
                if index%1000==0:print(f'{split}: {index} images exported',flush=True)
            assert coco['images'] and coco['annotations']
            write_json(done,coco)
        reports[split]=dict(images=len(coco['images']),annotations=len(coco['annotations']))
    write_json(output/'manifest.json',dict(contract='nstc_seg_dataset_v1',export_complete=True,
        split_rule='synthetic_train_original_dp_valid_test_v1',source_configs=config,splits=dict(train=source['splits']['train'],**reports),
        label_warning='DP labels include known missing/partial instruments; use unchanged and interpret accordingly.',
        scope='sampled_smoke' if max_files or frames_per_camera else 'all_frames'))
    print('Experiment B dataset ready:',output,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic',type=Path,required=True);p.add_argument('--dp-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-files',type=int,default=0);p.add_argument('--frames-per-camera',type=int,default=0)
    a=p.parse_args();prepare(a.synthetic,a.dp_root,a.output,a.max_files,a.frames_per_camera)

if __name__=='__main__':main()
