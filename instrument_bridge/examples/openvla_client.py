"""Example client usable from OpenVLA's own Python environment.
Requires only Pillow, numpy and Python standard library there.
"""
import base64,io,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image

class NSTCDetectorClient:
 def __init__(self,model='B',threshold=.5):
  launcher=Path(__file__).resolve().parents[2]/'serve-instruments6.sh'
  self.process=subprocess.Popen([str(launcher),'--model',model,'--threshold',str(threshold)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None,text=True,bufsize=1)
  self.sequence=0
 def predict(self,rgb_uint8_hwc,table_roi_xyxy=None):
  rgb=np.asarray(rgb_uint8_hwc)
  if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:raise ValueError('Expected RGB uint8 H×W×3')
  stream=io.BytesIO();Image.fromarray(rgb).save(stream,format='PNG')
  self.sequence+=1;request={'request_id':self.sequence,'rgb_png_base64':base64.b64encode(stream.getvalue()).decode('ascii')}
  if table_roi_xyxy is not None:request['table_roi_xyxy']=list(table_roi_xyxy)
  self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
  line=self.process.stdout.readline()
  if not line:raise RuntimeError('Detector process exited; inspect its stderr')
  response=json.loads(line)
  if not response.get('ok') or response.get('request_id')!=self.sequence:raise RuntimeError(response.get('error','Detector protocol mismatch'))
  return response
 def close(self):
  self.process.stdin.close();self.process.wait(timeout=10)

# In the robot loop, load once:
# detector=NSTCDetectorClient(model='B')
# result=detector.predict(front_rgb)
# classes_and_counts=result['counts']
# instances=result['objects']
# Choose and lock one instance before asking your policy to act.
