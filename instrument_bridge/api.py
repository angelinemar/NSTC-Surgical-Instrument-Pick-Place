"""Sensor-only detector; no Isaac Sim import and no robot action side effects."""
from pathlib import Path
import json
import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[1]
CLASSES=('scalpel','scissor','love_retractor','kelly','scalpel_type2')
MODEL_PATHS={
 'A':REPO/'models/object_detection/A/checkpoint_best_total.pth',
 'B':REPO/'models/object_detection/B/checkpoint_best_total.pth',
}

def validate_rgb(rgb):
 rgb=np.asarray(rgb)
 if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:raise ValueError('Expected H×W×3 uint8 RGB image')
 return np.ascontiguousarray(rgb)

class InstrumentDetector:
 def __init__(self,model='B',threshold=.5,checkpoint=None):
  if model not in MODEL_PATHS:raise ValueError('Choose model A or B')
  if not 0<=threshold<=1:raise ValueError('threshold must be within [0,1]')
  self.variant=model;self.threshold=float(threshold);self.checkpoint=Path(checkpoint or MODEL_PATHS[model]).resolve()
  if not self.checkpoint.is_file():raise FileNotFoundError(self.checkpoint)
  from training.segmentation_checkpoint import checkpoint_resolution
  self.resolution=checkpoint_resolution(self.checkpoint)
  from rfdetr import RFDETRSegSmall
  import torch
  self.model=RFDETRSegSmall.from_checkpoint(str(self.checkpoint),resolution=self.resolution)
  self.model.inference(compile=False,dtype=torch.float16)

 def predict(self,rgb,include_masks=True,table_roi_xyxy=None):
  rgb=validate_rgb(rgb)
  det=self.model.predict(rgb,threshold=self.threshold,include_source_image=False)
  if det.mask is None:raise RuntimeError('Chosen checkpoint has no instance masks')
  objects=[];masks=[]
  for box,score,name,mask in zip(det.xyxy,det.confidence,det.data['class_name'],det.mask):
   if name=='__background__':continue
   if name not in CLASSES:raise RuntimeError('Unknown class '+str(name))
   m=np.asarray(mask,dtype=bool)
   if m.shape!=rgb.shape[:2]:raise RuntimeError('Mask dimensions differ from RGB')
   x1,y1,x2,y2=(float(v) for v in box)
   obj=dict(class_name=str(name),score=float(score),bbox_xyxy=[x1,y1,x2,y2],center_xy=[(x1+x2)/2,(y1+y2)/2],mask_area_px=int(m.sum()))
   if include_masks:
    from pycocotools import mask as mask_api
    rle=mask_api.encode(np.asfortranarray(m.astype(np.uint8)));rle['counts']=rle['counts'].decode('ascii');obj['mask_rle']=rle
   objects.append(obj);masks.append(m)
  counts={n:sum(o['class_name']==n for o in objects) for n in CLASSES}
  result=dict(model=self.variant,threshold=self.threshold,image_width=rgb.shape[1],image_height=rgb.shape[0],objects=objects,counts=counts)
  if table_roi_xyxy is not None:
   x1,y1,x2,y2=map(float,table_roi_xyxy)
   if not (0<=x1<x2<=rgb.shape[1] and 0<=y1<y2<=rgb.shape[0]):raise ValueError('Invalid table ROI')
   result['table_roi_xyxy']=[x1,y1,x2,y2]
   result['table_counts_estimate']={n:sum(o['class_name']==n and x1<=o['center_xy'][0]<x2 and y1<=o['center_xy'][1]<y2 for o in objects) for n in CLASSES}
   result['table_count_note']='2D center-in-ROI estimate; occluded objects cannot be counted from this image.'
  return result,masks
