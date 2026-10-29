"""Persistent JSONL process for integration from another Python environment.

Input: {"request_id":..., "image_path":"/absolute/frame.png"} or
{"request_id":..., "rgb_png_base64":"..."}. Output: one JSON line.
"""
import argparse,base64,contextlib,io,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from .api import InstrumentDetector

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=('A','B'),default='B');p.add_argument('--threshold',type=float,default=.5);a=p.parse_args()
 with contextlib.redirect_stdout(sys.stderr):detector=InstrumentDetector(a.model,a.threshold)
 for line in sys.stdin:
  request=None
  try:
   request=json.loads(line)
   if ('image_path' in request)==('rgb_png_base64' in request):raise ValueError('Supply exactly one image_path or rgb_png_base64')
   if 'image_path' in request:rgb=np.asarray(Image.open(request['image_path']).convert('RGB'))
   else:rgb=np.asarray(Image.open(io.BytesIO(base64.b64decode(request['rgb_png_base64'],validate=True))).convert('RGB'))
   with contextlib.redirect_stdout(sys.stderr):result,_=detector.predict(rgb,table_roi_xyxy=request.get('table_roi_xyxy'))
   response=dict(request_id=request.get('request_id'),ok=True,**result)
  except Exception as exc:response=dict(request_id=request.get('request_id') if isinstance(request,dict) else None,ok=False,error=type(exc).__name__+': '+str(exc))
  print(json.dumps(response),flush=True)
if __name__=='__main__':main()
