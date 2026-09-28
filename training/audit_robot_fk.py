"""Cross-check recorded robot proprioception against independent URDF kinematics."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def robot_chain(urdf):
    joints = {j.find('child').get('link'): j for j in ET.parse(urdf).getroot().findall('joint')}
    chain, link = [], 'panda_hand'
    while link != 'panda_link0':
        joint = joints[link]
        origin = joint.find('origin')
        transform = np.eye(4)
        if origin is not None:
            transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        axis = joint.find('axis')
        axis = np.fromstring(axis.get('xyz'), sep=' ') if axis is not None else np.array([0., 0., 1.])
        chain.append((joint.get('name'), joint.get('type'), transform, axis))
        link = joint.find('parent').get('link')
    return list(reversed(chain))


def tcp_fk(chain, q):
    transform = np.eye(4)
    for name, kind, origin, axis in chain:
        transform = transform @ origin
        if kind != 'fixed':
            index = int(name.removeprefix('panda_joint')) - 1
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(q[index] * axis).as_matrix()
            transform = transform @ motion
    transform[:3, 3] += transform[:3, :3] @ np.array([0., 0., .1034])
    return transform


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--urdf', type=Path, default=Path('C:/IsaacLab/_isaac_sim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka/lula_franka_gen.urdf'))
    args = p.parse_args()
    chain = robot_chain(args.urdf)
    snapshot = json.loads((args.benchmark/'summary.json').read_text())
    results = []
    for case in snapshot['results']:
        if case['status'] != 'PASS':
            continue
        for path in sorted((args.benchmark/case['case']['id']).glob('*_policy/*/episode_*.h5')):
            with h5py.File(path, 'r') as h:
                observations = h['observations/robot_proprio'][:]
            position_errors, rotation_errors = [], []
            for index in np.unique(np.linspace(0, len(observations)-1, min(64, len(observations))).astype(int)):
                row = observations[index]
                predicted = tcp_fk(chain, row[:7])
                measured_q = row[12:16]
                measured_rotation = Rotation.from_quat(measured_q[[1, 2, 3, 0]])
                position_errors.append(float(np.linalg.norm(predicted[:3, 3] - row[9:12])))
                rotation_errors.append(float((Rotation.from_matrix(predicted[:3, :3]).inv() * measured_rotation).magnitude() * 180 / np.pi))
            results.append(dict(file=str(path), sampled_frames=len(position_errors),
                                max_position_error_m=max(position_errors), max_rotation_error_deg=max(rotation_errors)))
    report = dict(source='independent installed Franka URDF + recorded robot joints only',
                  records=results, files=len(results),
                  max_position_error_m=max((r['max_position_error_m'] for r in results), default=None),
                  max_rotation_error_deg=max((r['max_rotation_error_deg'] for r in results), default=None))
    report['pass'] = bool(results) and report['max_position_error_m'] < .001 and report['max_rotation_error_deg'] < .2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}))
    if not report['pass']:
        raise SystemExit('Kinematic mismatch needs review before training/deployment')


if __name__ == '__main__':
    main()
