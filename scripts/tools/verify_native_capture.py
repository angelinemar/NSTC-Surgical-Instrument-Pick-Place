"""Audit one native capture and write full-resolution RGB/label contact sheets."""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'compat'))
from phase4_storage import require_commit
from training.audit_training_topics import inspect
from validate_feedback_dataset import audit

CAMERAS=('front','wrist','cam_top','cam_left','cam_right','cam_tray')
COLORS=np.array([(20,20,20),(60,150,255),(255,150,40),(80,220,100),
                 (245,220,60),(225,80,210),(80,225,220),(255,90,90)],np.uint8)


def verify(run, output):
    output.mkdir(parents=True,exist_ok=True)
    files=sorted(run.glob('*_policy/*/episode_*.h5'))
    if len(files)!=2:
        raise ValueError('Expected exactly one pick/place pair')
    reports=[]
    for path in files:
        physical=audit(path)
        topics=inspect(path.resolve())
        if not topics['topic_audit_pass']:
            raise ValueError(topics['errors'])
        with h5py.File(path,'r') as h:
            require_commit(path,h)
            assert h.attrs['image_preprocess']=='native_no_resize'
            assert json.loads(h.attrs['crop_top_left_hw'])==[0,0]
            assert json.loads(h.attrs['domain_randomization'])['contract']=='table_aimed_lighting_v1'
            size=h['observations/front_rgb'].shape[1]
            sheet=Image.new('RGB',(3*size,2*(2*size+32)),(24,30,36))
            draw=ImageDraw.Draw(sheet)
            row=dict(skill=str(h.attrs['policy_skill']),frames=len(h['actions']),size=size,
                     audit='PASS',cameras={},randomization=json.loads(h.attrs['domain_randomization']))
            for i,camera in enumerate(CAMERAS):
                semantic='grip_b' if camera=='wrist' else camera
                rgb=h[f'observations/{camera}_rgb'][0]
                mask=h[f'observations/{semantic}_semantic'][0]
                calibration=h['camera_calibration/'+semantic]
                np.testing.assert_array_equal(calibration['intrinsics_output'][:],calibration['intrinsics_raw'][:])
                k=calibration['intrinsics_output'][0]
                assert abs(float(k[0,2])-size/2)<1 and abs(float(k[1,2])-size/2)<1
                gray=rgb.astype(np.float32).mean(2)
                lap=-4*gray[1:-1,1:-1]+gray[:-2,1:-1]+gray[2:,1:-1]+gray[1:-1,:-2]+gray[1:-1,2:]
                row['cameras'][camera]=dict(intrinsics=k.tolist(),first_frame_laplacian_variance=float(lap.var()),
                    class_pixels=np.bincount(mask.ravel(),minlength=8).tolist())
                x=(i%3)*size;y=(i//3)*(2*size+32)
                draw.text((x+8,y+8),camera,fill='white')
                sheet.paste(Image.fromarray(rgb),(x,y+32))
                sheet.paste(Image.fromarray(COLORS[mask]),(x,y+32+size))
            sheet.save(output/(row['skill']+'_native.png'))
            reports.append(row)
    (output/'report.json').write_text(json.dumps(dict(result='PASS',segments=reports,
        scope='one expert demonstration; not learned-policy rollout or generalization certification'),indent=2))
    print(json.dumps([dict(skill=r['skill'],size=r['size'],frames=r['frames'],audit=r['audit']) for r in reports]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.run,a.output)
