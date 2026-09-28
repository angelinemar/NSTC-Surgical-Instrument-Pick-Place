"""Visible-box AP50 for the current one-instance-per-class scenes.

101-point interpolated precision; not COCO AP@[.50:.95]. No GT-fed inference.
"""
import numpy as np


def iou_xywh(a,b):
    ax,ay,aw,ah=a;bx,by,bw,bh=b
    intersection=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
    return intersection/max(aw*ah+bw*bh-intersection,1e-12)


def ap50(records):
    metrics={}
    for category in range(1,6):
        truth={(index,j):box for index,record in enumerate(records)
               for j,box in enumerate(record['truth']) if box['category_id']==category}
        predictions=sorted([(float(box['confidence']),index,box['bbox_xywh'])
                            for index,record in enumerate(records) for box in record['predictions']
                            if box['category_id']==category],reverse=True)
        if not truth:
            metrics[str(category)]=None
            continue
        used=set();tp=[]
        for score,index,box in predictions:
            candidates=[(iou_xywh(box,v['bbox']),key) for key,v in truth.items() if key[0]==index and key not in used]
            best=max(candidates,default=(0,None))
            match=best[0]>=.5
            tp.append(int(match))
            if match: used.add(best[1])
        if not tp:
            metrics[str(category)]=0.
            continue
        hits=np.cumsum(tp); precision=hits/np.arange(1,len(tp)+1);recall=hits/len(truth)
        metrics[str(category)]=float(np.mean([precision[recall>=r].max(initial=0) for r in np.linspace(0,1,101)]))
    values=[v for v in metrics.values() if v is not None]
    return dict(ap50_by_category=metrics,mean_ap50=float(np.mean(values)) if values else None,
                evaluated_categories=len(values),definition='visible_boxes_101point_AP50_not_COCO_AP50_95')
