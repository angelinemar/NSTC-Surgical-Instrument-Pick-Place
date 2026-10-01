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
TASK_TARGETS = ('scalpel','scissor','love_retractor','kelly','scalpel_type2')


def task_command(name):
    if name not in TASK_TARGETS:
        raise ValueError('Choose an explicit target instrument task')
    return np.eye(len(TASK_TARGETS),dtype=np.float32)[TASK_TARGETS.index(name)]


def copy_policy_images(src, destination, name, mask=False):
    """DP contract stays 224; original native detector pixels remain in raw H5."""
    from PIL import Image
    if src.shape[1:3] == (224, 224):
        src.file.copy(src, destination, name=name)
        return
    if src.shape[1:3] != (448, 448):
        raise ValueError('Unsupported native image dimensions')
    output = destination.create_dataset(name, shape=(len(src), 224, 224) + src.shape[3:],
                                        dtype=src.dtype, compression='gzip', chunks=True)
    method = Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS
    for start in range(0, len(src), 16):
        output[start:start+16] = np.stack([np.asarray(Image.fromarray(frame).resize((224,224), method))
                                         for frame in src[start:start+16]])


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


def source_pairs(source, validation_cells):
    plan = source / 'collection_plan.json'
    if not plan.exists():
        summary = json.loads((source/'summary.json').read_text())
        pairs = []
        for result in summary['results']:
            if result['status'] == 'PASS':
                case = result['case']
                pairs.append((case, 'valid' if case['cell'] in validation_cells else 'train',
                              sorted((source/case['id']).glob('*_policy/*/episode_*.h5'))))
        return pairs, summary['complete'], 'held_out_cells_legacy'
    if not (source/'collection_complete.json').exists():
        raise ValueError('Collection is incomplete; no partial cohort export')
    jobs = json.loads(plan.read_text())
    pairs, seen, seeds = [], set(), {}
    for job in jobs:
        split = job['split']
        if split not in ('train','valid','test') or seeds.setdefault(job['seed'],split) != split:
            raise ValueError('Session seed leakage across splits')
        folder = Path(job['folder']).resolve()
        if not folder.is_relative_to(source) or folder in seen:
            raise ValueError('Invalid or duplicate collection folder')
        seen.add(folder)
        coverage = json.loads((folder/'coverage_report.json').read_text())
        if not coverage.get('complete'):
            raise ValueError('Incomplete physical coverage')
        paths = sorted(folder.glob('pick_policy/*/episode_*.h5'))
        if len(paths) != 10 * job['config']['cycles']:
            raise ValueError('Coverage count mismatch')
        for path in paths:
            place = folder/'place_policy'/job['object']/path.name
            with h5py.File(path,'r') as h:
                case = dict(id=f"{job['object']}_{job['seed']}_{path.stem}", object=job['object'],
                            cell=int(h.attrs['coverage_cell_id']),yaw=float(h.attrs['target_yaw_deg']),
                            session_seed=job['seed'])
            pairs.append((case,split,[path,place]))
    return pairs, True, 'independent_session_seed'


def export(source, output, validation_cells, panel_pairs=None):
    # Auditing raw expert files belongs to export, never to the model's forward path.
    from training.audit_training_topics import inspect
    source, output = source.resolve(), output.resolve()
    pairs, source_complete, split_rule = (source_pairs(source, validation_cells) if panel_pairs is None
                                        else (panel_pairs,True,'committed_episode_or_legacy_session_split'))
    if not pairs:
        raise ValueError('No independently audited successful pairs')
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(input_allowlist=INPUTS, labels_are_inputs=False,
                    task_command_required=True, task_targets=TASK_TARGETS,
                    task_command_source='requested recorder target, supplied explicitly by operator at inference',
                    action_contract='absolute robot-base XYZ, wxyz unit quaternion, gripper -1/+1',
                    control_dt_s=0.02, validation_cells=sorted(validation_cells), episodes=[],
                    source_snapshot_complete=source_complete, split_rule=split_rule, production_ready=False,
                    policy_image_size=224, raw_native_pixels_retained=True)
    policy_files = {s: h5py.File(output / (s + '.hdf5'), 'w') for s in ('pick', 'place')}
    labels = h5py.File(output / 'perception_labels.hdf5', 'w')
    contracts = set()
    splits = {s: dict(train=[], valid=[], test=[]) for s in policy_files}
    seen_sources = set()
    try:
        for f in policy_files.values():
            f.create_group('data')
            f.attrs['sensor_only_contract'] = 'p4_sensor_task_v2'
        for index, (case, split, paths) in enumerate(pairs):
            episode = 'demo_%06d' % index
            expected_skills = set(case.get('saved_skills', ['pick','place']))
            if len(paths) != len(expected_skills):
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
                    if skill not in expected_skills:
                        raise ValueError('Unexpected saved skill')
                    if src.attrs.get('dataset_split','unassigned') not in ('unassigned',split):
                        raise ValueError('Recorded split differs from export split')
                    recorded_seed=json.loads(src.attrs.get('domain_randomization','{}')).get('session_seed')
                    if recorded_seed is not None and recorded_seed!=case.get('session_seed'):
                        raise ValueError('Session seed differs from recording provenance')
                    if skill not in policy_files or skill in pair_skills:
                        raise ValueError('Invalid skill pairing')
                    pair_skills.add(skill)
                    evidence = json.loads(src.attrs['expert_grasp_evidence'])
                    if not (src.attrs['success'] and evidence.get('lift_verified') and evidence.get('place_verified')):
                        raise ValueError('Missing physical success evidence')
                    occupancy = json.loads(src.attrs.get('initial_tray_occupancy', '[]'))
                    target_id = int(src.attrs.get('object_type_id', -1))
                    if (len(occupancy) != 5 or target_id not in range(5)
                            or occupancy[target_id] != 0):
                        raise ValueError('Invalid initial target/tray metadata')
                    target_name = str(src.attrs['target_object'])
                    if TASK_TARGETS[target_id] != target_name:
                        raise ValueError('Requested task target disagrees with recorded target ID')
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
                    demo.create_dataset('task_target', data=task_command(target_name))
                    demo.attrs['task_target_name'] = target_name
                    obs = demo.create_group('obs')
                    proprio = src['observations/robot_proprio'][:]
                    proprio[:, 12:16] = canonical_quaternions(proprio[:, 12:16])
                    obs.create_dataset('robot_proprio', data=proprio)
                    label_group = labels.create_group(skill + '/' + episode)
                    for camera in CAMERAS:
                        copy_policy_images(src['observations/' + camera + '_rgb'], obs, camera + '_rgb')
                        sem_name = 'grip_b_semantic' if camera == 'wrist' else camera + '_semantic'
                        copy_policy_images(src['observations/' + sem_name], label_group, camera, mask=True)
                    assert_sensor_inputs(obs)
                    demo.create_dataset('actions', data=actions)
                    demo.create_dataset('dones', data=np.arange(n) == n - 1)
                    demo.create_dataset('rewards', data=np.zeros(n, np.float32))
                    splits[skill][split].append(episode)
                    with path.open('rb') as stream:
                        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
                    if checksum in seen_sources:
                        raise ValueError('Duplicate source content in policy export')
                    seen_sources.add(checksum)
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
