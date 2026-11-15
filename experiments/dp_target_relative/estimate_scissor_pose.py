"""Estimate one instrument pose from RF-DETR B mask and calibrated front RGB-D.

The reference shape is assembled from training episodes only. No validation/test
object pose, grasp pose, expert action, or semantic mask is read by this script.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


try:
    from .workspace_root import workspace_root
except ImportError:
    from workspace_root import workspace_root
ROOT = workspace_root()
sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(ROOT/'versions/6.0')]
DATA_ROOT = ROOT/'datasets/DP datasets/NSTC/DP datasets'
CLASSES = ('scalpel', 'scissor', 'love_retractor', 'kelly', 'scalpel_type2')


def rotation_wxyz(q):
    return Rotation.from_quat((q[1], q[2], q[3], q[0]))


def pose_wxyz(position, rotation):
    qx, qy, qz, qw = rotation.as_quat()
    return [float(v) for v in (*position, qw, qx, qy, qz)]


def point_cloud_arrays(depth, k, pos, quat, mask, min_points=20):
    depth = np.asarray(depth, dtype=float)
    v, u = np.where(mask.astype(bool) & np.isfinite(depth) & (depth > 0))
    if len(u) < min_points:
        raise ValueError('Too few visible target pixels')
    z = depth[v, u]
    camera_xyz = np.stack(((u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z), axis=1)
    world_xyz = rotation_wxyz(quat).apply(camera_xyz) + pos
    world_xyz = world_xyz[(world_xyz[:, 2] > -.01) & (world_xyz[:, 2] < .08)]
    if len(world_xyz) < min_points:
        raise ValueError('Too few target surface points near the table')
    return world_xyz


def point_cloud(h, mask):
    if h['camera_calibration'].attrs['quaternion_convention'] != 'wxyz_ros_camera_frame':
        raise ValueError('Unsupported camera quaternion convention')
    return point_cloud_arrays(h['observations/front_depth'][0],
                              h['camera_calibration/front/intrinsics_output'][0],
                              h['camera_calibration/front/position_b'][0],
                              h['camera_calibration/front/quaternion_b_ros'][0], mask)


def reference_shape(instrument='scissor'):
    DATA = DATA_ROOT/instrument/instrument
    train = sorted((DATA/'train/pick_policy'/instrument).glob('episode_*.h5'))
    if len(train) != 70:
        raise RuntimeError(f'{instrument} training inventory changed; re-audit before use')
    with h5py.File(train[0]) as h:
        first = h['debug_gt/selected_object_pose_b'][0]
    ref_pos, ref_rotation = first[:3], rotation_wxyz(first[3:])
    all_xy, root_z = [], []
    for path in train:
        with h5py.File(path) as h:
            if h.attrs['dataset_split'] != 'train' or h.attrs['target_object'] != instrument:
                raise ValueError(f'Non-training {instrument} reference: {path}')
            pose = h['debug_gt/selected_object_pose_b'][0]
            root_z.append(pose[2])
            mask = h['supervision_gt/conditioning_mask_front'][0]
            try:
                points = point_cloud(h, mask)
            except ValueError as exc:
                if 'Too few' not in str(exc):
                    raise
                continue  # tiny/occluded train-only mask cannot define shape
            local = rotation_wxyz(pose[3:]).inv().apply(points-pose[:3])
            reference = ref_rotation.apply(np.stack((local[:, 0], np.zeros(len(local)), local[:, 2]), axis=1))
            all_xy.append(reference[:, :2])
    if len(all_xy) < 20:
        raise RuntimeError(f'Only {len(all_xy)} usable training masks for {instrument} shape registration')
    model = np.concatenate(all_xy)
    model = np.unique(np.round(model/.002).astype(int), axis=0)*.002
    return model, ref_rotation, float(np.median(root_z)), len(all_xy)


def rotate_xy(xy, angle):
    c, s = np.cos(angle), np.sin(angle)
    return xy@np.array(((c, s), (-s, c)))


def register(model, observation):
    observation = observation[::max(1, len(observation)//300)]
    center = np.median(observation, axis=0)
    starts = []
    for angle in np.linspace(-np.pi, np.pi, 72, endpoint=False):
        candidate = rotate_xy(model, angle)
        shift = center-np.median(candidate, axis=0)
        distances = cKDTree(candidate+shift).query(observation)[0]
        starts.append((float(np.median(np.minimum(distances, .02))), angle, shift))
    starts.sort(key=lambda x:x[0])
    best = None
    for _, angle, shift in starts[:5]:
        def loss(x):
            distances = cKDTree(rotate_xy(model, x[0])+x[1:3]).query(observation)[0]
            return float(np.mean(np.minimum(distances, .02)**2))
        fit = minimize(loss, np.r_[angle, shift], method='Nelder-Mead',
                       options={'maxiter':100, 'xatol':.0005, 'fatol':1e-8})
        if best is None or fit.fun < best.fun:
            best = fit
    return float(best.x[0]), best.x[1:3], float(best.fun)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--episode', type=Path, help='Held-out recorded sensor frame')
    source.add_argument('--sensor-npz', type=Path, help='Live RGB-D and calibration captured after simulator reset')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--instrument', choices=CLASSES, default='scissor')
    p.add_argument('--selection', choices=('highest_confidence', 'nearest_tcp'), default='highest_confidence')
    a = p.parse_args()
    if a.episode:
        with h5py.File(a.episode) as h:
            if h.attrs['dataset_split'] not in ('valid', 'test') or h.attrs['target_object'] != a.instrument:
                raise ValueError(f'Expected a held-out {a.instrument} Pick episode')
            rgb = h['observations/front_rgb'][0]
            depth = h['observations/front_depth'][0]
            k = h['camera_calibration/front/intrinsics_output'][0]
            pos = h['camera_calibration/front/position_b'][0]
            quat = h['camera_calibration/front/quaternion_b_ros'][0]
        source_info = dict(episode=str(a.episode.resolve()), camera='front', frame=0,
                           rgb='recorded sensor', depth='recorded sensor', mask='RF-DETR B prediction')
        ee_position_b = None
    else:
        with np.load(a.sensor_npz, allow_pickle=False) as sensor:
            rgb, depth, k, pos, quat = (sensor[key].copy() for key in ('rgb', 'depth', 'intrinsics', 'position_b', 'quaternion_b_ros'))
            ee_position_b = sensor['ee_position_b'].copy() if 'ee_position_b' in sensor else None
        source_info = dict(sensor_bundle=str(a.sensor_npz.resolve()), camera='front',
                           rgb='live simulator sensor', depth='live simulator sensor', mask='RF-DETR B prediction')
    from instrument_bridge.api import InstrumentDetector
    detector = InstrumentDetector(model='B', threshold=a.threshold)
    detections, masks = detector.predict(rgb, include_masks=False)
    candidates = [(obj, mask) for obj, mask in zip(detections['objects'], masks) if obj['class_name']==a.instrument]
    if not candidates:
        raise RuntimeError(f'No {a.instrument} detected in front camera')
    if a.selection == 'nearest_tcp':
        if ee_position_b is None or np.asarray(ee_position_b).shape != (3,):
            raise ValueError('nearest_tcp requires measured ee_position_b in the live sensor bundle')
        ranked = []
        for candidate, candidate_mask in candidates:
            try:
                cloud = point_cloud_arrays(depth, k, pos, quat, candidate_mask, min_points=8)
            except ValueError as exc:
                if 'Too few' not in str(exc):
                    raise
                continue
            distance = float(np.linalg.norm(np.median(cloud[:, :2], axis=0) - ee_position_b[:2]))
            ranked.append((distance, -float(candidate['score']), candidate, candidate_mask, cloud))
        if not ranked:
            raise RuntimeError('No target candidate has enough valid depth points')
        distance, _, obj, mask, observed = min(ranked, key=lambda row: (row[0], row[1]))
        if distance > .12:
            raise RuntimeError(f'No {a.instrument} detection within 12 cm XY of measured TCP')
    else:
        obj, mask = max(candidates, key=lambda item:item[0]['score'])
        observed = point_cloud_arrays(depth, k, pos, quat, mask)
        distance = None
    model, reference_rotation, root_z, reference_count = reference_shape(a.instrument)
    angle, xy, cost = register(model, observed[:, :2])
    target_rotation = Rotation.from_euler('z', angle)*reference_rotation
    result = dict(contract=f'camera_pose_{a.instrument}_v1', source=source_info,
                  pose_b_wxyz=pose_wxyz((xy[0],xy[1],root_z),target_rotation),
                  detector_score=float(obj['score']), mask_pixels=int(obj['mask_area_px']),
                  registration_cost=cost, reference_split='train', reference_training_episodes_used=reference_count,
                  selection=a.selection, measured_tcp_xy_distance_m=distance,
                  note='Camera estimate; no validation/test simulator target pose used')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
