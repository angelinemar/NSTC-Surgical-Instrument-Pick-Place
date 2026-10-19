"""BBox/mask COCO AP and fixed-threshold P/R/F1 on synthetic or original DP data."""
import argparse,contextlib,hashlib,io,json,pickle
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from pycocotools import mask as mask_api
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from training.evaluate_dp_cameras import summarize,instance_boxes
from training.export_detection import CAMERAS,CLASSES
from phase4_storage import require_commit
from detection.dataset import write_json


def rle(mask):
    value=mask_api.encode(np.asfortranarray(mask.astype(np.uint8)));value['counts']=value['counts'].decode('ascii');return value


def samples(a):
    if a.dataset:
        root=a.dataset/a.split;data=json.loads((root/'_annotations.coco.json').read_text());by_image={}
        for box in data['annotations']:by_image.setdefault(box['image_id'],[]).append(box)
        for item in data['images']:
            yield np.asarray(Image.open(root/item['file_name']).convert('RGB')),by_image.get(item['id'],[]),dict(source=str(root/item['file_name']),camera=item.get('distance_band',item.get('camera','unknown')),**{k:item[k] for k in ('quality','distance_m') if k in item})
    else:
        files=[p for p in sorted(a.dp_root.rglob('episode_*.h5')) if a.split in p.parts and '.pending' not in p.parts]
        if a.max_files:files=files[:a.max_files]
        for path in files:
            with h5py.File(path) as h:
                require_commit(path,h);mapping=json.loads(h.attrs['instance_classes']);n=len(h['actions'])
                indices=sorted(set(np.linspace(0,n-1,a.frames_per_camera,dtype=int))) if a.frames_per_camera else range(n)
                for camera in CAMERAS:
                    prefix='grip_b' if camera=='wrist' else camera
                    for t in indices:
                        instance=h['observations/'+prefix+'_instance'][t];semantic=h['observations/'+prefix+'_semantic'][t]
                        boxes=instance_boxes(semantic,instance,mapping)
                        for b in boxes:b['segmentation']=rle(instance==b['instance_id'])
                        yield h['observations/'+camera+'_rgb'][t],boxes,dict(source=str(path),camera=camera,frame=int(t))


def matches(images,annotations,predictions,kind):
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset=dict(info={},images=images,annotations=annotations,categories=[dict(id=i+1,name=n) for i,n in enumerate(CLASSES)]);gt.createIndex()
        if predictions:dt=gt.loadRes(predictions)
        else:dt=COCO();dt.dataset=dict(gt.dataset,annotations=[]);dt.createIndex()
        e=COCOeval(gt,dt,kind);e.params.areaRng=[[0,1e10]];e.params.areaRngLbl=['all'];e.params.maxDets=[100];e.evaluate()
    groups={};image_by_id={i['id']:i for i in images}
    for group in sorted(set(i['group'] for i in images)):
        groups[group]={}
        for cid,name in enumerate(CLASSES,1):
            rows=[x for x in e.evalImgs if x is not None and x['category_id']==cid and image_by_id[x['image_id']]['group']==group]
            groups[group][name]=dict(ngt=sum(len(x['gtIds']) for x in rows),
                scores=np.concatenate([np.asarray(x['dtScores'],np.float32) for x in rows]) if rows else np.zeros(0,np.float32),
                matches=np.concatenate([x['dtMatches']>0 for x in rows],axis=1) if rows else np.zeros((10,0),bool))
    return groups


def preview(path,rgb,boxes,predictions,threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,title in zip(axes,('RGB','Ground-truth instances','Predicted instances')):ax.imshow(rgb);ax.set_title(title);ax.axis('off')
    for ax,rows in [(axes[1],boxes),(axes[2],[p for p in predictions if p['score']>=threshold])]:
        overlay=np.zeros((*rgb.shape[:2],4),float)
        for i,row in enumerate(rows):
            value=row['segmentation'];encoded=mask_api.frPyObjects(value,*value['size']) if isinstance(value['counts'],list) else value
            mask=mask_api.decode(encoded).astype(bool);color=plt.get_cmap('tab20')(i%20);overlay[mask]=(*color[:3],.6)
            x,y,w,h=row['bbox'];ax.text(x,y,CLASSES[row['category_id']-1],fontsize=6,color='white',bbox=dict(facecolor='black',alpha=.5,pad=.5))
        ax.imshow(overlay)
    fig.savefig(path,dpi=120);plt.close(fig)


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--dataset',type=Path);g.add_argument('--dp-root',type=Path)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=('valid','test'),default='test');p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--resolution',type=int);p.add_argument('--max-files',type=int,default=0);p.add_argument('--frames-per-camera',type=int,default=0)
    a=p.parse_args()
    from training.segmentation_checkpoint import checkpoint_resolution
    a.resolution=checkpoint_resolution(a.checkpoint,a.resolution)
    a.output.mkdir(parents=True,exist_ok=True);cache=a.output/'cache';cache.mkdir(exist_ok=True)
    config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(evaluation_contract='segmentation_native_resolution_v2',checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),score_floor=.001,iou_match=.5,
        metrics_units='percent',scope='sampled_smoke' if a.max_files or a.frames_per_camera else 'all_frames',
        label_warning='DP source has documented missing/partial labels; scores use supplied masks unchanged.' if a.dp_root or (a.dataset and a.dataset.name=='dp') else None)
    cp=a.output/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Output configuration mismatch')
    write_json(cp,config)
    from rfdetr import RFDETRSegSmall
    import torch
    torch.set_num_threads(4);model=None;images=[];annotations=[];predictions=[];count=0;chunk=0;examples=[]
    case_path=a.output/'cases.json'
    cases=json.loads(case_path.read_text())['cases'] if case_path.exists() else []
    best=max((c['mask_F1'] for c in cases),default=-1.);worst=min((c['mask_F1'] for c in cases),default=2.)
    def flush():
        nonlocal images,annotations,predictions,chunk
        target=cache/f'{chunk:06d}.pkl'
        if not target.exists():
            record={kind:matches(images,annotations,predictions,kind) for kind in ('bbox','segm')}
            temp=target.with_suffix('.tmp')
            with temp.open('wb') as f:pickle.dump(dict(images=len(images),records=record),f,protocol=5)
            # Persist case descriptions before committing their metric chunk.
            write_json(case_path,dict(note='Class-correct mask IoU >= .50; smoke results are pipeline checks.',cases=cases))
            temp.replace(target)
        images=[];annotations=[];predictions=[];chunk+=1
        write_json(a.output/'status.json',dict(state='evaluating',images=count,chunks=chunk))
        print('Evaluated images:',count,flush=True)
    for rgb,boxes,meta in samples(a):
        image_id=len(images)+1;count+=1;images.append(dict(id=image_id,width=rgb.shape[1],height=rgb.shape[0],group=meta['camera']))
        if (cache/f'{chunk:06d}.pkl').exists():
            if len(images)==32:flush()
            continue
        if model is None:
            model=RFDETRSegSmall.from_checkpoint(str(a.checkpoint),resolution=a.resolution);model.inference(compile=False,dtype=torch.float16)
        det=model.predict(rgb,threshold=.001,include_source_image=False)
        assert det.mask is not None,'Checkpoint did not produce instance masks'
        for b in boxes:annotations.append(dict(b,id=len(annotations)+1,image_id=image_id))
        rows=[]
        for xyxy,score,name,mask in zip(det.xyxy,det.confidence,det.data['class_name'],det.mask):
            if name=='__background__':continue
            assert name in CLASSES
            x,y,x2,y2=map(float,xyxy);row=dict(image_id=image_id,category_id=CLASSES.index(name)+1,
                bbox=[x,y,x2-x,y2-y],score=float(score),segmentation=rle(mask));rows.append(row);predictions.append(row)
        # Deterministic examples from every camera/distance group, with all masks.
        if meta['camera'] not in examples:
            preview(a.output/(meta['camera']+'_example.png'),rgb,boxes,rows,a.threshold);examples.append(meta['camera'])
        selected=sorted([p for p in rows if p['score']>=a.threshold],key=lambda x:-x['score'])
        def compressed(value):
            return mask_api.frPyObjects(value,*value['size']) if isinstance(value['counts'],list) else value
        used=set();tp=0
        if boxes and selected:
            ious=mask_api.iou([compressed(p['segmentation']) for p in selected],
                [compressed(b['segmentation']) for b in boxes],[0]*len(boxes))
            for i,pred in enumerate(selected):
                candidates=[j for j,b in enumerate(boxes) if j not in used and b['category_id']==pred['category_id'] and ious[i,j]>=.5]
                if candidates:used.add(max(candidates,key=lambda j:ious[i,j]));tp+=1
        fp=len(selected)-tp;fn=len(boxes)-tp;f1=2*tp/max(2*tp+fp+fn,1)
        case=dict(meta,TP=tp,FP=fp,FN=fn,mask_F1=f1,image_index=count)
        cases[:]=[c for c in cases if c['image_index']!=count]
        cases.append(case)
        if f1>best:best=f1;preview(a.output/'best_available.png',rgb,boxes,rows,a.threshold)
        if f1<worst:worst=f1;preview(a.output/'worst_available.png',rgb,boxes,rows,a.threshold)
        if len(images)==32:flush()
    if images:flush()
    groups={kind:{} for kind in ('bbox','segm')}
    for path in sorted(cache.glob('*.pkl')):
        with path.open('rb') as f:d=pickle.load(f)
        for kind,record in d['records'].items():
            for camera,values in record.items():groups[kind].setdefault(camera,[]).append(values)
    result=dict(config=config,images=count,metrics={})
    for kind,cameras in groups.items():
        result['metrics'][kind]=dict(overall=summarize([r for rows in cameras.values() for r in rows],a.threshold),
            per_camera_or_distance={c:summarize(rows,a.threshold) for c,rows in cameras.items()})
    write_json(a.output/'metrics.json',result);write_json(a.output/'status.json',dict(state='complete',images=count))
    write_json(a.output/'cases.json',dict(note='Class-correct mask IoU >= .50. Best available is relative to this evaluation, not a guarantee of success.',cases=cases))
    lines=['# Instance segmentation evaluation', '', f'Images: {count}. Threshold: {a.threshold}. All values below are percentages.', '', config['label_warning'] or 'Synthetic visible-instance ground truth.', '', '| Metric type / group | AP50:95 | AP50 | Precision | Recall | F1 |', '|---|---:|---:|---:|---:|---:|']
    for kind,values in result['metrics'].items():
        for name,row in [('overall',values['overall']),*values['per_camera_or_distance'].items()]:
            lines.append('| '+kind+' / '+name+' | '+' | '.join(f"{row[k]:.2f}" for k in ('mAP_50_95','mAP_50','precision','recall','F1'))+' |')
    lines += ['', 'Per-class metrics: metrics.json. Per-image failures: cases.json. Visuals: *_example.png, best_available.png, worst_available.png.', 'AP is not a single classification accuracy; true negatives are not defined for object detection.']
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v['overall'] for k,v in result['metrics'].items()},indent=2))

if __name__=='__main__':main()
