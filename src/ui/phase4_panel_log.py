"""Small, dependency-free log filter for the recorder panel."""
import re

class LogFilter:
    def __init__(self): self.stage=''; self.last_reason=''; self.previous=None
    def feed(self,line):
        text=re.sub(r'\x1b\[[0-9;]*m','',line).strip()
        if not text: return None
        if text.startswith('[P4 RENDER QUALITY]'):
            return ('success','Renderer: '+text.split(']',1)[1].strip())
        if text.startswith('[P4 ACTIVE STAGE]'):
            self.stage=text.split(']',1)[1].strip()
            setup=self.stage.endswith(('OPEN_HOVER','MOVE_TO_TARGET'))
            return ('info',('Setup motion (not saved): ' if setup else 'Recording: ')+self.stage)
        if text.startswith('[P4 STAGE]'):
            self.stage=text.split(']',1)[1].strip(); return None
        if text.startswith('Steps ') and self.stage:
            result=('info',self.stage+' | '+text)
        elif any(tag in text for tag in ('[PHASE FAIL]','[SPAWN FAIL]','[SENSOR FAIL]','[QUALITY FAIL]')):
            self.last_reason=text; result=('error',text)
        elif re.match(r'^(?:[A-Za-z_]\w*(?:Error|Exception)|KeyboardInterrupt|SystemExit):',text):
            self.last_reason=text; result=('error',text)
        elif text.startswith('ATTEMPT '): result=('info',text)
        elif '[SAVE SPLIT]' in text: result=('success','SAVED '+text.split('[SAVE SPLIT]',1)[1].strip())
        elif '[RESULT] success=True' in text: result=('info','Physical checks passed; validating images and saving...')
        elif text.startswith(('[P4 COVERAGE]','[P4 RESET]','[P4 MANUAL READY]','[P4 TRAY OCCUPANCY]','[P4 PANEL WARNING]')):
            result=('info',text)
        else: return None
        if result==self.previous: return None
        self.previous=result
        return result

class LogTail:
    def __init__(self): self.offset=0; self.pending=b''
    def read(self,path):
        try:
            with open(path,'rb') as stream:
                if stream.seek(0,2)<self.offset: self.offset=0; self.pending=b''
                stream.seek(self.offset); data=stream.read(131072); self.offset=stream.tell()
        except OSError: return []
        parts=(self.pending+data).split(b'\n'); self.pending=parts.pop()
        return [p.decode('utf-8',errors='replace') for p in parts]
