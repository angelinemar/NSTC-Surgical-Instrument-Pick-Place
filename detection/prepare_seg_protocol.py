"""Freeze all recorded DP camera positions as a training exclusion set."""
import argparse,hashlib,json
from pathlib import Path
import h5py
import numpy as np
from detection.segmentation_protocol import SEEDS
from detection.dataset import write_json


def prepare(source,output,repo,resolution=768):
    output.mkdir(parents=True,exist_ok=True)
    layout=json.loads((repo/'env/scene_layout.json').read_text())
    assert layout['robot_rot_wxyz']==[1.,0.,0.,0.], 'Add quaternion conversion for rotated base'
    offset=np.asarray(layout['robot_pos']);points=set();files=0;frames=0
    for path in sorted(source.rglob('episode_*.h5')):
        if '.pending' in path.parts:continue
        with h5py.File(path) as h:
            for camera in h['camera_calibration'].values():
                xyz=camera['position_b'][:]+offset
                if not np.isfinite(xyz).all():raise ValueError('Nonfinite DP camera')
                points.update(map(tuple,np.round(xyz/.02).astype(int)));frames+=len(xyz)
        files+=1
    if not points:raise ValueError('No calibrated DP camera positions')
    reservation=output/'dp_reserved_views.json'
    payload=dict(contract='dp_world_camera_positions_v1',source=str(source.resolve()),files=files,frames=frames,
        base_world_position=offset.tolist(),quantization_m=.02,world_positions=(np.asarray(sorted(points))*.02).tolist())
    if reservation.exists() and json.loads(reservation.read_text())!=payload:raise ValueError('Refusing to replace DP reservation')
    write_json(reservation,payload);sha=hashlib.sha256(reservation.read_bytes()).hexdigest()
    for split,seed in SEEDS.items():
        config=dict(contract='nstc_seg_hemisphere_v1',split=split,seed=seed,resolution=resolution,
            dp_reservation=str(reservation.resolve()),dp_reservation_sha256=sha,dp_exclusion_radius_m=.12,
            views_per_scene=9,materials='recorder_pbr_stainless',minimum_visible_pixels=8)
        path=output/(split+'.json')
        if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Existing protocol differs')
        write_json(path,config)
    return dict(dp_files=files,dp_frames=frames,reserved_positions=len(points),output=str(output))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resolution',type=int,default=768)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output,Path(__file__).resolve().parents[1],a.resolution),indent=2))
