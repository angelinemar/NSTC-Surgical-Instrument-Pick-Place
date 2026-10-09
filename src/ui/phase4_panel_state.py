"""Read-only dashboard counters. A successful motion is not a saved episode."""
import re

class Dashboard:
    def __init__(self,goal=1):
        self.goal=goal; self.attempt=0; self.saved=0; self.failures=0
        self.phase=None; self.stage='Idle'; self.outcomes={}
    @property
    def episode(self): return min(self.saved+1,self.goal)
    def feed(self,line):
        s=re.sub(r'\x1b\[[0-9;]*m','',line).strip()
        match=re.search(r'ATTEMPT\s+(\d+)\s*\|\s*SUCCESS\s+(\d+)/(\d+)',s)
        if match:
            if self.attempt and self.attempt not in self.outcomes:
                self.outcomes[self.attempt]='incomplete'; self.failures+=1
            self.attempt,self.saved,self.goal=map(int,match.groups())
            self.phase='reset'; self.stage='Spawn / settle'; return
        if s.startswith('[P4 ACTIVE STAGE]'):
            self.stage=s.split(']',1)[1].strip()
            self.phase='reset' if self.stage.endswith(('OPEN_HOVER','MOVE_TO_TARGET')) else 'recording'
            return
        if s.startswith('[REC ') and 'stage=' in s:
            self.stage=s.split('stage=',1)[1]
            self.phase='reset' if self.stage.endswith(('OPEN_HOVER','MOVE_TO_TARGET')) else 'recording'
            return
        if any(t in s for t in ('[PHASE FAIL]','[SPAWN FAIL]','[SENSOR FAIL]','[QUALITY FAIL]')) or ('[QUALITY]' in s and 'FAIL' in s):
            if self.attempt and self.attempt not in self.outcomes:
                self.outcomes[self.attempt]='fail'; self.failures+=1
            self.phase='reset'; self.stage='Discard / retry'; return
        match=re.search(r'\[SAVE SPLIT\]\s*(\d+)/(\d+)',s)
        if match:
            self.saved,self.goal=map(int,match.groups()); self.outcomes[self.attempt]='saved'
            self.phase='end'; self.stage='Episode saved'; return
        if s.startswith('[POSTPROCESS]') or '[RESULT] success=True' in s:
            self.phase='end'; self.stage='QC / saving - not counted yet'; return
        if s.startswith('[P4 RESET]'):
            self.phase='reset'; self.stage='Reset / waiting for environment'
        elif s.startswith('[P4 MANUAL READY]'):
            self.phase='reset'; self.stage='Ready - waiting for Start'
    def finish(self,metrics,exit_code):
        self.saved=int(metrics.get('success_count',self.saved))
        self.failures=int(metrics.get('failure_count',self.failures))
        self.attempt=int(metrics.get('total_attempts',self.attempt))
        self.phase='end'; self.stage='Complete' if exit_code==0 and self.saved>=self.goal else 'Stopped / incomplete - see log'


def rotate_outline(item,pose):
    """Calibrated reference footprint, then requested world-Z yaw; no NumPy."""
    import math
    angle=math.radians(pose['yaw_deg']); c,s=math.cos(angle),math.sin(angle)
    xy=[[pose['x']+c*x-s*y,pose['y']+s*x+c*y] for x,y in item['hull_xy_m']]
    x,y=item['axis_xy']
    return dict(hull_xy=xy,center_xy=[pose['x'],pose['y']],axis_xy=[c*x-s*y,s*x+c*y],source='calibrated prediction')
