"""Evaluate all committed validation/test H5 frames without training or copying RGB.

Cache per-file COCO matches for restartable evaluation. Report COCO-style AP
(all areas, maxDets=100 per category) and micro P/R/F1 at IoU .50. Confidence
selection uses validation only; the frozen threshold is then used for test.
"""
import argparse
import contextlib
import hashlib
import io
import json
import pickle
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'compat')]
from training.export_detection import CAMERAS, CLASSES, visible_boxes
from phase4_storage import require_commit
from detection.dataset import write_json


def instance_boxes(mask, instances, mapping, min_pixels=8):
    """Same visible-box contract as export_detection, using one label scan."""
    from scipy.ndimage import find_objects
    if mask.ndim!=2 or mask.shape!=instances.shape or mask.min()<0 or mask.max()>10:
        raise ValueError('Invalid semantic/instance map')
    maximum=int(instances.max())
    lookup=np.full(maximum+1,-1,dtype=np.int16);lookup[0]=0
    for key,item in mapping.items():
        if int(key)<=maximum:lookup[int(key)]=int(item['semantic_id'])
    nonzero=instances!=0
    if not np.all(lookup[instances[nonzero]]==mask[nonzero]):
        raise ValueError('Instance class disagrees with semantic pixels')
    if np.any(((mask>=3)&(mask<=7)) & ~nonzero):
        raise ValueError('Instrument semantic pixels have no instance')
    counts=np.bincount(instances.ravel(),minlength=maximum+1)
    boxes=[]
    for value,slices in enumerate(find_objects(instances),1):
        if slices is None or counts[value]<min_pixels:continue
        y,x=slices;category=int(lookup[value])-2
        if category not in range(1,6):raise ValueError('Non-instrument instance')
        boxes.append(dict(category_id=category,instance_id=value,
            bbox=[x.start,y.start,x.stop-x.start,y.stop-y.start],area=int(counts[value]),iscrowd=0))
    return boxes


def summarize(records, threshold):
    classes = {}
    total_tp = total_fp = total_gt = 0
    for name in CLASSES:
        parts = [r[name] for r in records if name in r]
        ngt = sum(p['ngt'] for p in parts)
        scores = np.concatenate([p['scores'] for p in parts]) if parts else np.array([])
        matches = np.concatenate([p['matches'] for p in parts], axis=1) if parts else np.zeros((10,0),bool)
        order = np.argsort(-scores, kind='stable'); scores=scores[order]; matches=matches[:,order]
        tp = np.cumsum(matches,axis=1); fp = np.cumsum(~matches,axis=1)
        aps=[]; recalls=[]
        for t in range(10):
            recall=tp[t]/max(ngt,1); precision=tp[t]/np.maximum(tp[t]+fp[t],1)
            precision=np.maximum.accumulate(precision[::-1])[::-1]
            indices=np.searchsorted(recall,np.linspace(0,1,101),side='left')
            sampled=np.zeros(101); ok=indices<len(precision);sampled[ok]=precision[indices[ok]]
            aps.append(float(sampled.mean()) if ngt else None)
            recalls.append(float(recall[-1]) if len(recall) else 0.)
        selected=scores>=threshold
        ntp=int(matches[0,selected].sum());nfp=int(selected.sum())-ntp
        p=ntp/max(ntp+nfp,1);r=ntp/max(ngt,1)
        classes[name]=dict(AP_50_95=100*np.mean(aps) if ngt else None,
            AP_50=100*aps[0] if ngt else None, AP_75=100*aps[5] if ngt else None,
            AR_100=100*np.mean(recalls) if ngt else None,
            precision=100*p,recall=100*r,F1=100*2*p*r/max(p+r,1e-12),TP=ntp,FP=nfp,FN=ngt-ntp,ground_truth=ngt)
        total_tp+=ntp;total_fp+=nfp;total_gt+=ngt
    p=total_tp/max(total_tp+total_fp,1);r=total_tp/max(total_gt,1)
    result=dict(threshold=float(threshold),precision=100*p,recall=100*r,F1=100*2*p*r/max(p+r,1e-12),
        TP=total_tp,FP=total_fp,FN=total_gt-total_tp,ground_truth=total_gt,per_class=classes)
    for source,dest in [('AP_50_95','mAP_50_95'),('AP_50','mAP_50'),('AP_75','mAP_75'),('AR_100','mAR_100')]:
        values=[v[source] for v in classes.values() if v[source] is not None]
        result[dest]=float(np.mean(values)) if values else None
    return result


def coco_records(images, annotations, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset=dict(info={},images=images,annotations=annotations,
            categories=[dict(id=i+1,name=n) for i,n in enumerate(CLASSES)]);gt.createIndex()
        if predictions: dt=gt.loadRes(predictions)
        else:
            dt=COCO();dt.dataset=dict(gt.dataset,annotations=[]);dt.createIndex()
        evaluator=COCOeval(gt,dt,'bbox');evaluator.params.areaRng=[[0,1e10]];evaluator.params.areaRngLbl=['all']
        evaluator.params.maxDets=[100];evaluator.evaluate()
    camera_by_id={v['id']:v['camera'] for v in images}
    records={c:{} for c in CAMERAS}
    for camera in CAMERAS:
        for cid,name in enumerate(CLASSES,1):
            rows=[r for r in evaluator.evalImgs if r is not None and r['category_id']==cid and camera_by_id[r['image_id']]==camera]
            records[camera][name]=dict(ngt=sum(int((~np.asarray(r['gtIgnore'],bool)).sum()) for r in rows),
                scores=np.concatenate([np.asarray(r['dtScores'],np.float32) for r in rows]) if rows else np.zeros(0,np.float32),
                matches=np.concatenate([r['dtMatches']>0 for r in rows],axis=1) if rows else np.zeros((10,0),bool))
            assert not any(np.asarray(r['dtIgnore']).any() for r in rows)
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True);cache=a.output/'cache';cache.mkdir(exist_ok=True)
    sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    config=dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=sha,source=str(a.source.resolve()),
        cameras=list(CAMERAS),stride=1,min_visible_pixels=8,prediction_score_floor=.001,max_detections_per_category=100,inference_dtype='float16',
        metrics_units='percent',iou_thresholds=[round(.5+.05*i,2) for i in range(10)],
        threshold_selection='maximum pooled validation F1 on grid 0.05..0.95; frozen for test',
        note='Detection metrics, not classification accuracy; all frames are correlated within episodes. No training.')
    cp=a.output/'evaluation_config.json'
    if cp.exists(): assert json.loads(cp.read_text())==config, 'Output belongs to a different evaluation'
    else:write_json(cp,config)
    files=[]; transactions={};seeds={};expected={s:0 for s in ('valid','test')}
    for split in ('valid','test'):
        for path in sorted(a.source.rglob('episode_*.h5')):
            if split not in path.relative_to(a.source).parts:continue
            if not any(x in ('pick_policy','place_policy') for x in path.parts):continue
            with h5py.File(path) as h:
                assert h.attrs['dataset_split']==split
                tx=str(h.attrs['pair_transaction_id']);assert transactions.setdefault(tx,split)==split,'Episode leakage'
                dr=json.loads(h.attrs['domain_randomization']);seed=(str(h.attrs['target_object']),dr.get('session_seed'),dr.get('seed'))
                assert seeds.setdefault(seed,split)==split,'Randomization leakage'
                n=len(h['actions']);expected[split]+=n*len(CAMERAS)
            files.append((split,path,n))
    write_json(a.output/'inventory.json',dict(files=len(files),images=expected))
    from rfdetr import RFDETRSmall
    import torch
    torch.set_num_threads(8)
    model=None; start=time.time();done={s:0 for s in expected}
    pool=ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn'))
    pending=[];completed=0
    def finish(item):
        nonlocal completed
        future,target,saved=item
        saved['records']=future.result()
        temp=target.with_suffix('.tmp')
        with temp.open('wb') as stream:pickle.dump(saved,stream,protocol=5)
        temp.replace(target);done[saved['split']]+=saved['images'];completed+=1
        status=dict(state='evaluating',files_done=completed,files_total=len(files),images_done=done,
            images_expected=expected,elapsed_seconds=time.time()-start)
        write_json(a.output/'status.json',status);print(json.dumps(status),flush=True)
    for index,(split,path,n) in enumerate(files):
        key=hashlib.sha256(str(path.relative_to(a.source)).encode()).hexdigest()[:20]
        target=cache/(key+'.pkl')
        stat=path.stat();signature=(stat.st_size,stat.st_mtime_ns)
        if target.exists():
            with target.open('rb') as stream: saved=pickle.load(stream)
            assert saved['source_signature']==signature,'Source modified since cache'
            done[split]+=n*len(CAMERAS);completed+=1;continue
        if model is None:
            model=RFDETRSmall.from_checkpoint(str(a.checkpoint));print('Class names:',model.class_names,flush=True)
            assert set(CLASSES).issubset(set(model.class_names))
            model.inference(compile=False,dtype=torch.float16)
        images=[];annotations=[];predictions=[]; previews=[]
        with h5py.File(path) as h:
            require_commit(path,h)
            mapping=json.loads(h.attrs['instance_classes']);sem_classes=json.loads(h.attrs['semantic_class_ids'])
            assert all(sem_classes[c]==i+3 for i,c in enumerate(CLASSES))
            for camera in CAMERAS:
                prefix='grip_b' if camera=='wrist' else camera
                rgb=h['observations/'+camera+'_rgb'];sem=h['observations/'+prefix+'_semantic'];inst=h['observations/'+prefix+'_instance']
                assert rgb.shape==(n,448,448,3) and sem.shape==inst.shape==(n,448,448)
                for first in range(0,n,a.batch_size):
                    frames=rgb[first:first+a.batch_size];masks=sem[first:first+a.batch_size];instances=inst[first:first+a.batch_size]
                    detections=model.predict(list(frames),threshold=.001,include_source_image=False)
                    if not isinstance(detections,list):detections=[detections]
                    for offset,(frame,mask,instance,det) in enumerate(zip(frames,masks,instances,detections)):
                        image_id=len(images)+1
                        images.append(dict(id=image_id,width=448,height=448,camera=camera,frame=first+offset))
                        boxes=instance_boxes(mask,instance,mapping)
                        for box in boxes:annotations.append(dict(id=len(annotations)+1,image_id=image_id,**box))
                        preds=[]
                        for xyxy,score,name in zip(det.xyxy,det.confidence,det.data['class_name']):
                            if name=='__background__':continue
                            assert name in CLASSES,repr(name)
                            x,y,x2,y2=map(float,xyxy)
                            row=dict(image_id=image_id,category_id=CLASSES.index(name)+1,bbox=[x,y,x2-x,y2-y],score=float(score))
                            predictions.append(row);preds.append(row)
                        if first+offset==n//2 and split=='test' and not (a.output/(camera+'_example.npz')).exists():
                            np.savez_compressed(a.output/(camera+'_example.npz'),rgb=frame,
                                ground_truth=json.dumps(boxes),predictions=json.dumps(preds),source=str(path),frame=first+offset)
        saved=dict(split=split,source=str(path),source_signature=signature,images=n*len(CAMERAS))
        pending.append((pool.submit(coco_records,images,annotations,predictions),target,saved))
        if len(pending)>=3:finish(pending.pop(0))
    for item in pending:finish(item)
    pool.shutdown()
    all_records={s:{c:[] for c in CAMERAS} for s in expected}
    for target in sorted(cache.glob('*.pkl')):
        with target.open('rb') as stream:saved=pickle.load(stream)
        for camera,record in saved['records'].items():all_records[saved['split']][camera].append(record)
    valid=[r for rows in all_records['valid'].values() for r in rows]
    # Threshold search only needs IoU .5 counts; avoid repeated AP sorting.
    sweep=[]
    for threshold in np.arange(.05,1,.05):
        tp=fp=ngt=0
        for record in valid:
            for v in record.values():
                selected=v['scores']>=threshold; t=int(v['matches'][0,selected].sum())
                tp+=t;fp+=int(selected.sum())-t;ngt+=v['ngt']
        sweep.append(dict(threshold=float(threshold),F1=2*tp/max(ngt+tp+fp,1)))
    threshold=max(sweep,key=lambda x:x['F1'])['threshold']
    report=dict(config=config,images=expected,validation_threshold_sweep=sweep,selected_threshold=threshold,splits={})
    for split,cameras in all_records.items():
        pooled=[r for rows in cameras.values() for r in rows]
        report['splits'][split]=dict(overall=summarize(pooled,threshold),at_confidence_050=summarize(pooled,.5),
            per_camera={c:summarize(rows,threshold) for c,rows in cameras.items()})
        write_json(a.output/'metrics.json',report)
    write_json(a.output/'status.json',dict(state='complete',files_done=len(files),images_done=done,elapsed_seconds=time.time()-start))
    print(json.dumps({s:v['overall'] for s,v in report['splits'].items()},indent=2),flush=True)


if __name__=='__main__':main()
