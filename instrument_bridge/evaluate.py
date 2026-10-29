"""Visible-instance metrics and separate spawned-table count audit."""
import json
import numpy as np
from .api import CLASSES

def iou_mask(a,b):
 a=np.asarray(a,bool);b=np.asarray(b,bool)
 if a.shape!=b.shape:raise ValueError('Mask shape mismatch')
 inter=np.logical_and(a,b).sum();union=np.logical_or(a,b).sum()
 return float(inter/union) if union else 0.

def iou_box(a,b):
 ax1,ay1,ax2,ay2=a;bx1,by1,bx2,by2=b
 inter=max(0,min(ax2,bx2)-max(ax1,bx1))*max(0,min(ay2,by2)-max(ay1,by1))
 area_a=max(0,ax2-ax1)*max(0,ay2-ay1);area_b=max(0,bx2-bx1)*max(0,by2-by1)
 return inter/max(area_a+area_b-inter,1e-12)

def compare(predictions,pred_masks,gt,threshold=.5):
 """Greedy class-correct IoU matching; P/R/F1, not mAP."""
 out={};tot=[0,0,0]
 for name in CLASSES:
  pi=[i for i,o in enumerate(predictions) if o['class_name']==name];gi=[i for i,o in enumerate(gt) if o['class_name']==name]
  matched=set();tp=0
  for i in sorted(pi,key=lambda i:-predictions[i]['score']):
   candidates=[(iou_mask(pred_masks[i],gt[j]['mask']),j) for j in gi if j not in matched]
   if candidates:
    value,j=max(candidates)
    if value>=threshold:matched.add(j);tp+=1
  fp=len(pi)-tp;fn=len(gi)-tp;tot[0]+=tp;tot[1]+=fp;tot[2]+=fn
  out[name]=dict(visible_gt=len(gi),predicted=len(pi),count_error=len(pi)-len(gi),TP=tp,FP=fp,FN=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),F1=2*tp/max(2*tp+fp+fn,1))
 tp,fp,fn=tot
 return dict(contract='visible_instance_mask_iou50_v1',iou_threshold=threshold,per_class=out,overall=dict(TP=tp,FP=fp,FN=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),F1=2*tp/max(2*tp+fp+fn,1)))

def compare_boxes(predictions,gt,threshold=.5):
 """Class-correct box matching using boxes derived from visible GT masks."""
 ground=[]
 for row in gt:
  ys,xs=np.where(row['mask'])
  if len(xs):ground.append(dict(class_name=row['class_name'],box=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]))
 out={};tot=[0,0,0]
 for name in CLASSES:
  pi=[i for i,o in enumerate(predictions) if o['class_name']==name];gi=[i for i,o in enumerate(ground) if o['class_name']==name]
  matched=set();tp=0
  for i in sorted(pi,key=lambda i:-predictions[i]['score']):
   candidates=[(iou_box(predictions[i]['bbox_xyxy'],ground[j]['box']),j) for j in gi if j not in matched]
   if candidates:
    value,j=max(candidates)
    if value>=threshold:matched.add(j);tp+=1
  fp=len(pi)-tp;fn=len(gi)-tp;tot[0]+=tp;tot[1]+=fp;tot[2]+=fn
  out[name]=dict(visible_gt=len(gi),predicted=len(pi),TP=tp,FP=fp,FN=fn,
                 precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),F1=2*tp/max(2*tp+fp+fn,1))
 tp,fp,fn=tot
 return dict(contract='visible_instance_box_iou50_v1',iou_threshold=threshold,per_class=out,
             overall=dict(TP=tp,FP=fp,FN=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),F1=2*tp/max(2*tp+fp+fn,1)))

def table_spawn_counts(attrs,include_target=True):
 """Count scene-clutter metadata labelled region=table plus active target if on table."""
 scene=json.loads(attrs['scene_clutter']);counts={n:0 for n in CLASSES}
 for row in scene['instances']:
  if row.get('region')=='table' and row['object'] in counts:counts[row['object']]+=1
 if include_target:
  target=str(attrs['target_object']);counts[target]+=1
 return counts
