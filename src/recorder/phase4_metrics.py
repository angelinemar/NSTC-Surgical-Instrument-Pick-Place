"""Count a success only after the episode segments have finished saving."""
import re
from phase3_run_metrics import RunMetrics as BaseMetrics


class RunMetrics(BaseMetrics):
    def observe_line(self, line):
        plain = re.sub(r'\x1b\[[0-9;]*m','',str(line))
        if '[SENSOR FAIL]' in plain:
            self._finish_attempt('sensor_fail',plain.strip())
            return
        category=re.search(r'\[PHASE FAIL\]\s+(pick|place)\s*\|',plain,re.I)
        if category:
            # Do not mistake "displacement" in a PICK failure for PLACE.
            self._finish_attempt(category.group(1).lower()+'_fail',plain.strip())
            return
        if re.search(r'\[RESULT\]\s+success\s*=\s*True',plain):
            self.pending_physical_result = plain.strip()
            return
        if '[SAVE SPLIT]' in plain:
            self._finish_attempt('success',plain.strip()+' | '+getattr(self,'pending_physical_result','physical QC passed'))
            return
        super().observe_line(line)


def require_saved_goal(metrics, args):
    from phase3_run_metrics import _forwarded_value
    if any(v in args for v in ('--camera_tuner','--workspace_tuner','--layout_tuner')):
        return
    requested = int(_forwarded_value(args,('--episodes',),'1'))
    mode = _forwarded_value(args,('--record_mode',),'both')
    selected = ('pick','place') if mode=='both' else (mode,)
    from src.recorder.episode_split import episode_files
    sets = [{p.name for p in episode_files(metrics.out_dir,s,metrics.object_name)} for s in selected]
    count = len(set.intersection(*sets))
    if count < requested:
        raise RuntimeError(f'COLLECTION_INCOMPLETE: saved={count}/{requested}; inspect run_metrics_summary.txt')
