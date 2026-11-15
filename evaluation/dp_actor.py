"""Run policy in its original training venv; binary local pipes, no network service."""
import io,json,sys,contextlib,os
from multiprocessing.connection import Connection

def serve():
 reader=Connection(os.dup(sys.stdin.fileno()),readable=True,writable=False)
 writer=Connection(os.dup(sys.stdout.fileno()),readable=False,writable=True)
 with contextlib.redirect_stdout(sys.stderr):
  import numpy as np,torch
  from training.runtime import PolicyRuntime
  torch.set_num_threads(4)
  models={}
  contracts={}
  for skill,path in zip(('pick','place'),sys.argv[2:4]):
   contract=torch.load(path,map_location='cpu',weights_only=True).get('input_contract')
   if contract=='p4_sensor_task_v2':runtime=PolicyRuntime
   elif contract=='dp_gripper_transition_v2':
    from dp_gripper_v2.runtime import TransitionRuntime
    runtime=TransitionRuntime
   elif contract=='dp_gripper_focused_v3':
    from dp_gripper_v3.runtime import FocusedRuntime
    runtime=FocusedRuntime
   elif contract=='target_relative_dp_scissor_v1':
    from dp_target_relative.learned_policy import Runtime
    runtime=Runtime
   elif contract in ('target_relative_dp_joint_pick_v1','target_relative_dp_joint_place_v1'):
    from dp_target_relative.train_joint import Runtime
    runtime=Runtime
   elif contract.startswith('target_relative_dp_') and contract.endswith('_v1'):
    from dp_target_relative.train_family import Runtime
    runtime=Runtime
   else:raise ValueError(f'Unsupported {skill} checkpoint contract: {contract}')
   models[skill]=runtime(path,device=os.environ.get('DP_EVAL_ACTOR_DEVICE','cuda'))
   contracts[skill]=contract
  torch.manual_seed(int(sys.argv[4]))
  writer.send_bytes(json.dumps({s:dict(skill=m.skill,image_size=m.image_size,controller_contract=m.controller_contract,input_contract=contracts[s]) for s,m in models.items()}).encode())
  active=None;active_target=None;pending=[]
  while True:
   try:payload=reader.recv_bytes()
   except EOFError:break
   with np.load(io.BytesIO(payload),allow_pickle=False) as z:
    skill=str(z['skill']);target=str(z['target']);reset=bool(z['reset'])
    sensors={k:z[k] for k in z.files if k not in ('skill','target','reset')}
   if active_target is not None and target!=active_target and not reset:
    raise ValueError('Target is locked for the current trial; explicit reset required')
   if reset or active!=skill:
    models[skill].set_target(target);pending=[]
   active_target=target
   active=skill
   models[skill].observe(sensors)
   if not pending:
    actions,_=models[skill].predict(inference_steps=100)
    pending=actions[:4].tolist() if contracts[skill].startswith('target_relative_dp_') else actions[:8].tolist()
   writer.send_bytes(json.dumps([pending.pop(0)]).encode())

class Actor:
 def __init__(self,python,pick,place,seed,log):
  import subprocess
  self.log=open(log,'w')
  self.proc=subprocess.Popen([str(python),'-m','evaluation.dp_actor','--serve',str(pick),str(place),str(seed)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log)
  self.reader=Connection(os.dup(self.proc.stdout.fileno()),readable=True,writable=False)
  self.writer=Connection(os.dup(self.proc.stdin.fileno()),readable=False,writable=True)
  if not self.reader.poll(120):
   self.close()
   raise RuntimeError('DP actor failed to initialize; inspect actor.log')
  try:self.metadata=json.loads(self.reader.recv_bytes())
  except Exception:
   self.close()
   raise
 def predict(self,skill,target,sensors,reset):
  import numpy as np
  buf=io.BytesIO();np.savez(buf,skill=skill,target=target,reset=reset,**sensors)
  self.writer.send_bytes(buf.getvalue())
  if not self.reader.poll(120):raise RuntimeError('DP actor timeout')
  return np.asarray(json.loads(self.reader.recv_bytes()),dtype=np.float32)
 def close(self):
  self.proc.terminate()
  try:self.proc.wait(timeout=10)
  except Exception:self.proc.kill();self.proc.wait()
  self.reader.close();self.writer.close()
  self.proc.stdin.close();self.proc.stdout.close();self.log.close()
if __name__=='__main__':serve()
