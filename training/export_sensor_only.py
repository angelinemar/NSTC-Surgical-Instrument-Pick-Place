"""Export validated demonstrations with an explicit sensor-only input allowlist.

Simulator labels live in a separate perception file. Split at episode level;
pick and place from one episode always receive the same split.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CAMERAS = ('front', 'wrist', 'cam_top', 'cam_left', 'cam_right', 'cam_tray')
INPUTS = ('robot_proprio',) + tuple(c + '_rgb' for c in CAMERAS)


def canonical_quaternions(values):
    values = np.asarray(values, dtype=np.float32).copy()
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms < 1e-6):
        raise ValueError('Invalid quaternion')
    values /= norms
    # Positive scalar fixes the initial hemisphere; subsequent rows stay continuous.
    if values[0, 0] < 0:
        values[0] *= -1
    for i in range(1, len(values)):
        if np.dot(values[i - 1], values[i]) < 0:
            values[i] *= -1
    return values


def assert_sensor_inputs(obs):
    if set(obs.keys()) != set(INPUTS):
        raise ValueError('Policy input keys must exactly equal the sensor allowlist')


def export(source, output, validation_cells):
    # Auditing raw expert files belongs to export, never to the model's forward path.
    sys.path.insert(0, str(ROOT / 'debug/validation/approach_matrix'))
    from audit_training_topics import inspect
    source, output = source.resolve(), output.resolve()
    summary = json.loads((source / 'summary.json').read_text())
    successes = [r for r in summary['results'] if r['status'] == 'PASS']
    if not successes:
        raise ValueError('No independently audited successful pairs')
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(input_allowlist=INPUTS, labels_are_inputs=False,
                    action_contract='absolute robot-base XYZ, wxyz unit quaternion, gripper -1/+1',
                    control_dt_s=0.02, validation_cells=sorted(validation_cells), episodes=[],
                    source_snapshot_complete=summary['complete'], production_ready=False)
    policy_files = {s: h5py.File(output / (s + '.hdf5'), 'w') for s in ('pick', 'place')}
    labels = h5py.File(output / 'perception_labels.hdf5', 'w')
    contracts = set()
    splits = {s: dict(train=[], valid=[]) for s in policy_files}
    try:
        for f in policy_files.values():
            f.create_group('data')
            f.attrs['sensor_only_contract'] = 'p4_sensor_only_v1'
        for index, result in enumerate(successes):
            case = result['case']
            episode = 'demo_%06d' % index
            split = 'valid' if case['cell'] in validation_cells else 'train'
            paths = sorted((source / case['id']).glob('*_policy/*/episode_*.h5'))
            if len(paths) != 2:
                raise ValueError('Incomplete pair: ' + case['id'])
            pair_skills = set()
            for path in paths:
                audit = inspect(path)
                if not audit['topic_audit_pass']:
                    raise ValueError(str(audit['errors']))
                with h5py.File(path, 'r') as src:
                    from phase4_storage import require_commit
                    require_commit(path, src)
                    skill = str(src.attrs['policy_skill'])
                    if skill not in policy_files or skill in pair_skills:
                        raise ValueError('Invalid skill pairing')
                    pair_skills.add(skill)
                    evidence = json.loads(src.attrs['expert_grasp_evidence'])
                    if not (src.attrs['success'] and evidence.get('lift_verified') and evidence.get('place_verified')):
                        raise ValueError('Missing physical success evidence')
                    occupancy = json.loads(src.attrs.get('initial_tray_occupancy', '[]'))
                    target_id = int(src.attrs.get('object_type_id', -1))
                    if (len(occupancy) != 5 or sum(occupancy) != 4 or target_id not in range(5)
                            or occupancy[target_id] != 0):
                        raise ValueError('Command-free policy requires exactly one actionable instrument; use an explicit user task command for ambiguous scenes')
                    if float(src.attrs.get('control_dt_s', 0)) != .02:
                        raise ValueError('Missing or incompatible control interval')
                    contract = tuple(str(src.attrs.get(k, '')) for k in
                                     ('approach_controller', 'motion_timing_contract', 'tcp_jacobian_contract'))
                    contracts.add(contract)
                    if len(contracts) != 1 or not all(contract):
                        raise ValueError('Mixed or missing controller contracts')
                    demo = policy_files[skill]['data'].create_group(episode)
                    actions = src['actions'][:]
                    actions[:, 3:7] = canonical_quaternions(actions[:, 3:7])
                    n = len(actions)
                    demo.attrs['num_samples'] = n
                    obs = demo.create_group('obs')
                    proprio = src['observations/robot_proprio'][:]
                    proprio[:, 12:16] = canonical_quaternions(proprio[:, 12:16])
                    obs.create_dataset('robot_proprio', data=proprio)
                    label_group = labels.create_group(skill + '/' + episode)
                    for camera in CAMERAS:
                        src.copy('observations/' + camera + '_rgb', obs, name=camera + '_rgb')
                        sem_name = 'grip_b_semantic' if camera == 'wrist' else camera + '_semantic'
                        src.copy('observations/' + sem_name, label_group, name=camera)
                    assert_sensor_inputs(obs)
                    demo.create_dataset('actions', data=actions)
                    demo.create_dataset('dones', data=np.arange(n) == n - 1)
                    demo.create_dataset('rewards', data=np.zeros(n, np.float32))
                    splits[skill][split].append(episode)
                    with path.open('rb') as stream:
                        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
                    manifest['episodes'].append(dict(id=episode, skill=skill, split=split,
                        source=str(path.resolve()), case=case,
                        sha256=checksum))
        for skill, f in policy_files.items():
            masks = f.create_group('mask')
            for split, demos in splits[skill].items():
                masks.create_dataset(split, data=np.asarray(demos, dtype='S'))
            f['data'].attrs['total'] = sum(g.attrs['num_samples'] for g in f['data'].values())
        manifest['split_counts'] = {s: {k: len(v) for k, v in groups.items()} for s, groups in splits.items()}
        manifest['controller_contract'] = list(contracts.pop())
        manifest['export_complete'] = True
    finally:
        labels.close()
        for f in policy_files.values():
            f.close()
    # Consumers require this manifest; an interrupted export is never training-ready.
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest['split_counts']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation-cells', type=int, nargs='+', default=[1, 8])
    args = parser.parse_args()
    export(args.source, args.output, set(args.validation_cells))
