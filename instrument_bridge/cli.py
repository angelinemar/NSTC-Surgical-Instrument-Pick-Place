"""CLI for one image or one recorded DP frame; outputs JSON and optional preview."""
import argparse,json
from pathlib import Path
import h5py,numpy as np
from PIL import Image,ImageDraw,ImageFont
from .api import InstrumentDetector,CLASSES
from .evaluate import compare,table_spawn_counts

def ground_truth(h,camera,frame):
 prefix='grip_b' if camera=='wrist' else camera
 instances=h['observations/'+prefix+'_instance'][frame]
 semantic=h['observations/'+prefix+'_semantic'][frame]
 mapping=json.loads(h.attrs['instance_classes'])
 from training.evaluate_dp_cameras import instance_boxes
 boxes=instance_boxes(semantic,instances,mapping)
 return [dict(class_name=CLASSES[b['category_id']-1],instance_id=b['instance_id'],mask=(instances==b['instance_id'])) for b in boxes]

def preview(rgb,result,path):
 img=Image.fromarray(rgb).convert('RGB');draw=ImageDraw.Draw(img)
 for item in result['objects']:
  draw.rectangle(item['bbox_xyxy'],outline='#ffcc00',width=3)
  x,y=item['bbox_xyxy'][:2]
  draw.text((x,max(0,y-19)),f"{item['class_name']} {item['score']:.2f}",fill='white',stroke_width=2,stroke_fill='black')
 img.save(path)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 source=p.add_mutually_exclusive_group(required=True);source.add_argument('--image',type=Path);source.add_argument('--episode',type=Path)
 p.add_argument('--camera',choices=('front','wrist','cam_top','cam_left','cam_right','cam_tray','all'),default='front');p.add_argument('--frame',type=int,default=0)
 p.add_argument('--model',choices=('A','B'),default='B');p.add_argument('--threshold',type=float,default=.5);p.add_argument('--table-roi',nargs=4,type=float,metavar=('X1','Y1','X2','Y2'));p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 if a.camera=='all':
  if a.image:p.error('--camera all requires --episode')
  detector=InstrumentDetector(a.model,a.threshold);aggregate={};per_camera={}
  with h5py.File(a.episode) as h:
   if a.frame<0 or a.frame>=len(h['actions']):raise ValueError('Frame index out of range')
   spawn=table_spawn_counts(h.attrs,include_target=h.attrs['policy_skill']=='pick')
   for camera in ('front','wrist','cam_top','cam_left','cam_right','cam_tray'):
    rgb=h['observations/'+camera+'_rgb'][a.frame];gt=ground_truth(h,camera,a.frame)
    result,masks=detector.predict(rgb,table_roi_xyxy=a.table_roi);result.update(camera=camera,frame=a.frame,source=str(a.episode),visible_ground_truth_counts={n:sum(g['class_name']==n for g in gt) for n in CLASSES})
    result['metrics']=compare(result['objects'],masks,gt);per_camera[camera]=result
    preview(rgb,result,a.output/(camera+'_preview.png'))
  for name in CLASSES:
   tp=sum(v['metrics']['per_class'][name]['TP'] for v in per_camera.values());fp=sum(v['metrics']['per_class'][name]['FP'] for v in per_camera.values());fn=sum(v['metrics']['per_class'][name]['FN'] for v in per_camera.values())
   aggregate[name]=dict(TP=tp,FP=fp,FN=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),F1=2*tp/max(2*tp+fp+fn,1))
  tp=sum(v['TP'] for v in aggregate.values());fp=sum(v['FP'] for v in aggregate.values());fn=sum(v['FN'] for v in aggregate.values())
  summary=dict(model=a.model,checkpoint=str(detector.checkpoint),threshold=a.threshold,episode=str(a.episode),frame=a.frame,spawned_on_table_at_episode_start=spawn,per_camera=per_camera,aggregate_mask_iou50=dict(TP=tp,FP=fp,FN=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),F1=2*tp/max(2*tp+fp+fn,1),per_class=aggregate),note='One frame × six cameras. Camera observations are correlated; this is diagnostic P/R/F1, not dataset-level mAP. Spawn counts include occluded objects. DP annotations may be partial.')
  (a.output/'detections_all_cameras.json').write_text(json.dumps(summary,indent=2))
  print(json.dumps(dict(model=a.model,frame=a.frame,spawned_on_table_at_episode_start=spawn,aggregate_mask_iou50=summary['aggregate_mask_iou50'],per_camera_counts={c:r['counts'] for c,r in per_camera.items()}),indent=2));print('Wrote',a.output/'detections_all_cameras.json');return
 if a.image:
  rgb=np.asarray(Image.open(a.image).convert('RGB'));gt=None;spawn=None;source=str(a.image)
 else:
  with h5py.File(a.episode) as h:
   if a.frame<0 or a.frame>=len(h['actions']):raise ValueError('Frame index out of range')
   rgb=h['observations/'+a.camera+'_rgb'][a.frame];gt=ground_truth(h,a.camera,a.frame)
   spawn=table_spawn_counts(h.attrs,include_target=h.attrs['policy_skill']=='pick');source=str(a.episode)
 detector=InstrumentDetector(a.model,a.threshold)
 result,masks=detector.predict(rgb,table_roi_xyxy=a.table_roi)
 result.update(source=source,camera=a.camera if a.episode else None,frame=a.frame if a.episode else None,checkpoint=str(detector.checkpoint),resolution=detector.resolution)
 if gt is not None:
  result['visible_ground_truth_counts']={n:sum(g['class_name']==n for g in gt) for n in CLASSES}
  result['metrics']=compare(result['objects'],masks,gt)
  result['spawned_on_table_at_episode_start']=spawn
  result['count_note']='Spawn count includes occluded objects; visible ground truth comes from instance masks. DP labels may be incomplete.'
 (a.output/'detections.json').write_text(json.dumps(result,indent=2))
 preview(rgb,result,a.output/'preview.png')
 printable={k:v for k,v in result.items() if k!='objects'}
 print(json.dumps(printable,indent=2))
 print('Wrote',a.output/'detections.json',a.output/'preview.png')
if __name__=='__main__':main()
