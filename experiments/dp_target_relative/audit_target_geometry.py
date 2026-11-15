"""Measure how far a naive RGB-D mask centroid is from the Scissor grasp label.

Uses saved oracle masks only for diagnosis; never treats them as model input.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


try:
    from .workspace_root import workspace_root
except ImportError:
    from workspace_root import workspace_root
ROOT = workspace_root()
DATA = ROOT / 'datasets/DP datasets/NSTC/DP datasets/scissor/scissor'
CAMERAS = ('front', 'cam_top', 'cam_left', 'cam_right', 'cam_tray')


def estimate(h, cam):
    mask = h[f'supervision_gt/conditioning_mask_{cam}'][0].astype(bool)
    depth = h[f'observations/{cam}_depth'][0].astype(np.float32)
    v, u = np.where(mask & np.isfinite(depth) & (depth > 0))
    if len(u) < 20:
        return None, len(u)
    k = h[f'camera_calibration/{cam}/intrinsics_output'][0]
    pos = h[f'camera_calibration/{cam}/position_b'][0]
    qw, qx, qy, qz = h[f'camera_calibration/{cam}/quaternion_b_ros'][0]
    z = depth[v, u]
    points_cam = np.stack(((u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z), axis=1)
    points = Rotation.from_quat((qx, qy, qz, qw)).apply(points_cam) + pos
    points = points[(points[:, 2] > -.01) & (points[:, 2] < .08)]
    return (np.median(points, axis=0) if len(points) >= 20 else None), len(u)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = {}
    for split in ('train', 'valid', 'test'):
        files = sorted((DATA / split / 'pick_policy/scissor').glob('episode_*.h5'))
        rows = {cam: [] for cam in CAMERAS}
        for path in files:
            with h5py.File(path) as h:
                if h['camera_calibration'].attrs['quaternion_convention'] != 'wxyz_ros_camera_frame':
                    raise ValueError(f'Unexpected camera convention in {path}')
                label = h['supervision_gt/teacher_grasp_pose_b'][:3]
                for cam in CAMERAS:
                    point, pixels = estimate(h, cam)
                    rows[cam].append(dict(episode=path.name, mask_pixels=pixels,
                        xy_error_m=float(np.linalg.norm(point[:2]-label[:2])) if point is not None else None))
        report[split] = dict(episodes=len(files), cameras={})
        for cam, items in rows.items():
            errors = [r['xy_error_m'] for r in items if r['xy_error_m'] is not None]
            report[split]['cameras'][cam] = dict(visible=len(errors), median_xy_error_m=float(np.median(errors)) if errors else None,
                p90_xy_error_m=float(np.quantile(errors, .9)) if errors else None,
                median_mask_pixels=float(np.median([r['mask_pixels'] for r in items])), rows=items)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    for split, data in report.items():
        print(split, data['episodes'], 'episodes')
        for cam, value in data['cameras'].items():
            err = value['median_xy_error_m']
            print(f"  {cam:10} visible {value['visible']:2}/{data['episodes']}: naive XY error median {100*err:.1f} cm" if err is not None else f'  {cam}: no usable mask')


if __name__ == '__main__':
    main()
