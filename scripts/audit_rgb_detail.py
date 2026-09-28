"""Read-only H5 sample audit; outputs diagnostics, never rewrites recordings."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image

def main():
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    files=sorted(args.root.glob('*_policy/*/episode_*.h5'))
    if not files:
        p.error('No policy H5 files found in the selected run folder')
    cameras=('front','wrist','cam_top','cam_left','cam_right','cam_tray')
    for number,path in enumerate(files):
        with h5py.File(path,'r') as h:
            for cam in cameras:
                rgb=h['observations/'+cam+'_rgb']
                sem=h['observations/'+('grip_b' if cam=='wrist' else cam)+'_semantic']
                for index in sorted({0,len(rgb)//2,len(rgb)-1}):
                    a=rgb[index]; labels=sem[index]
                    row=dict(file=str(path),camera=cam,frame=index,shape=list(a.shape),
                        instrument_pixels=int(((labels>=3)&(labels<=7)).sum()),
                        table_fraction=float((labels==8).mean()),
                        rgb_std=float(a.astype(float).std()))
                    target=int(json.loads(h.attrs['semantic_class_ids'])[h.attrs['target_object']])
                    y,x=np.where(labels==target)
                    row['target_pixels']=len(x)
                    row['target_bbox_wh']=[int(x.max()-x.min()+1),int(y.max()-y.min()+1)] if len(x) else [0,0]
                    rows.append(row)
                if number in (0,len(files)//2,len(files)-1):
                    Image.fromarray(rgb[len(rgb)//2]).save(args.output/f'{path.parent.parent.name}_{path.stem}_{cam}.png')
    report=dict(files=len(files),sampled_frames=len(rows),cameras={})
    for cam in cameras:
        selected=[r for r in rows if r['camera']==cam]
        pixels=np.array([r['target_pixels'] for r in selected])
        report['cameras'][cam]=dict(samples=len(selected),target_visible=int((pixels>0).sum()),
            median_target_pixels=float(np.median(pixels)),target_under_25_pixels=int(((pixels>0)&(pixels<25)).sum()),
            mean_table_fraction=float(np.mean([r['table_fraction'] for r in selected])))
    (args.output/'report.json').write_text(json.dumps(dict(summary=report,samples=rows),indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
