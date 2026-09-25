"""Header inventory only; never misrepresent this as a full pixel/physics audit."""
import argparse
from collections import Counter
import json
from pathlib import Path
import h5py


def inventory(root):
    counts, yaw_bins, sizes, randomized = Counter(), {}, Counter(), Counter()
    files=sorted(Path(root).glob('**/pick_policy/*/episode_*.h5'))
    for path in files:
        with h5py.File(path,'r') as h:
            name=str(h.attrs.get('target_object','unknown'))
            counts[name]+=1
            yaw=float(h.attrs.get('target_yaw_deg',0)) % 360
            yaw_bins.setdefault(name,set()).add(int(yaw//30))
            sizes[str(h['observations/front_rgb'].shape[1:3])]+=1
            randomized[str(json.loads(h.attrs.get('domain_randomization','{}')).get('contract','unrecorded'))]+=1
    return dict(scope='headers_only_pick_segments_no_pixel_or_commit_verification',
                pick_episodes=dict(counts),yaw_30degree_bins={k:sorted(v) for k,v in yaw_bins.items()},
                native_or_saved_sizes=dict(sizes),lighting_provenance=dict(randomized),training_ready=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); report=inventory(a.root)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
