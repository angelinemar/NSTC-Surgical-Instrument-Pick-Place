"""Held-out Scissor grasp XY audit using RF-DETR B masks and calibrated depth.

Train-only linear geometry correction uses oracle masks. Validation uses only
RF-DETR predictions as input; GT grasp coordinates are read solely for scoring.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


try:
    from .workspace_root import workspace_root
except ImportError:
    from workspace_root import workspace_root
ROOT = workspace_root()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / 'versions/6.0'))
DATA = ROOT / 'datasets/DP datasets/NSTC/DP datasets/scissor/scissor'


def features(h, mask):
    depth = h['observations/front_depth'][0].astype(float)
    v, u = np.where(mask & np.isfinite(depth) & (depth > 0))
    if len(u) < 20:
        return None
    k = h['camera_calibration/front/intrinsics_output'][0]
    pos = h['camera_calibration/front/position_b'][0]
    qw, qx, qy, qz = h['camera_calibration/front/quaternion_b_ros'][0]
    z = depth[v, u]
    cam = np.stack(((u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z), axis=1)
    points = Rotation.from_quat((qx, qy, qz, qw)).apply(cam) + pos
    xy = points[(points[:, 2] > -.01) & (points[:, 2] < .08), :2]
    if len(xy) < 20:
        return None
    mean = xy.mean(axis=0)
    cov = np.cov(xy.T)
    skew = ((xy-mean)**3).mean(axis=0)
    return np.r_[mean, cov[0, 0], cov[0, 1], cov[1, 1], skew, np.log1p(len(xy))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=.5)
    a = p.parse_args()
    train = sorted((DATA/'train/pick_policy/scissor').glob('episode_*.h5'))
    valid = sorted((DATA/'valid/pick_policy/scissor').glob('episode_*.h5'))
    if len(train) != 70 or len(valid) != 20:
        raise RuntimeError('Episode inventory changed')
    x, y = [], []
    for path in train:
        with h5py.File(path) as h:
            feat = features(h, h['supervision_gt/conditioning_mask_front'][0].astype(bool))
            if feat is not None:
                x.append(feat)
                y.append(h['supervision_gt/teacher_grasp_pose_b'][:2])
    x, y = np.stack(x), np.stack(y)
    x_mean, x_std, y_mean = x.mean(0), np.maximum(x.std(0), 1e-6), y.mean(0)
    x = (x-x_mean)/x_std
    ridge = .1
    weights = np.linalg.solve(x.T@x + ridge*np.eye(x.shape[1]), x.T@(y-y_mean))
    from instrument_bridge.api import InstrumentDetector
    detector = InstrumentDetector(model='B', threshold=a.threshold)
    rows = []
    for i, path in enumerate(valid, 1):
        with h5py.File(path) as h:
            image = h['observations/front_rgb'][0]
            result, masks = detector.predict(image, include_masks=False)
            choices = [(obj, mask) for obj, mask in zip(result['objects'], masks) if obj['class_name'] == 'scissor']
            if choices:
                obj, mask = max(choices, key=lambda x: x[0]['score'])
                feat = features(h, mask)
                if feat is not None:
                    estimate = ((feat-x_mean)/x_std)@weights+y_mean
                    truth = h['supervision_gt/teacher_grasp_pose_b'][:2]
                    err = float(np.linalg.norm(estimate-truth))
                else:
                    err = None
            else:
                obj, err = None, None
            rows.append(dict(episode=path.name, detected=obj is not None,
                             score=obj['score'] if obj else None,
                             mask_pixels=obj['mask_area_px'] if obj else 0,
                             grasp_xy_error_m=err))
        print(f'{i}/{len(valid)} {path.name}: ' + (f'{err*100:.2f} cm' if err is not None else 'no usable Scissor mask'), flush=True)
    errors = [r['grasp_xy_error_m'] for r in rows if r['grasp_xy_error_m'] is not None]
    report = dict(protocol='rfdetr_B_front_mask_depth_train_only_linear_grasp_xy_v1',
                  train_episodes=len(train), valid_episodes=len(valid), usable=len(errors),
                  median_error_m=float(np.median(errors)) if errors else None,
                  p90_error_m=float(np.quantile(errors, .9)) if errors else None,
                  within_1cm=sum(e <= .01 for e in errors), within_2cm=sum(e <= .02 for e in errors),
                  records=rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='records'}, indent=2))


if __name__ == '__main__':
    main()
