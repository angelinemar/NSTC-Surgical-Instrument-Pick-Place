"""Verify exported input boundaries and run the deployment API on held-out sensors."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
if __package__:
    from .export_sensor_only import INPUTS, assert_sensor_inputs
    from .runtime import PolicyRuntime
else:
    from export_sensor_only import INPUTS, assert_sensor_inputs
    from runtime import PolicyRuntime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--skill', choices=['pick', 'place'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads((args.dataset / 'manifest.json').read_text())
    if not manifest['export_complete']:
        raise ValueError('Export incomplete')
    splits = {}
    for record in manifest['episodes']:
        previous = splits.setdefault(record['id'], record['split'])
        if previous != record['split']:
            raise ValueError('Pick/place leakage between splits')
    with h5py.File(args.dataset / (args.skill + '.hdf5')) as f:
        train, valid = set(f['mask/train'][:]), set(f['mask/valid'][:])
        if train & valid or not train or not valid:
            raise ValueError('Invalid episode split')
        for demo in f['data'].values():
            assert_sensor_inputs(demo['obs'])
            if set(demo.keys()) != {'obs', 'actions', 'dones', 'rewards'}:
                raise ValueError('Unexpected exported data field')
        key = next(iter(sorted(valid))).decode()
        sensors = {k: f['data/' + key + '/obs/' + k][0] for k in INPUTS}
    blocked_smoke = False
    try:
        PolicyRuntime(args.checkpoint)
    except ValueError as error:
        blocked_smoke = 'Smoke checkpoint' in str(error)
        if not blocked_smoke:
            raise
    policy = PolicyRuntime(args.checkpoint, allow_smoke=True)
    actions, recognition = policy.predict(sensors, inference_steps=10)
    poisoned = dict(sensors, selected_object_pose_b=np.zeros(7))
    try:
        policy.predict(poisoned)
    except ValueError:
        rejected_gt = True
    else:
        raise AssertionError('GT key reached runtime')
    report = dict(policy_observation_keys=list(INPUTS), pair_split_consistent=True,
                  train_episodes=len(train), valid_episodes=len(valid),
                  output_shape=list(actions.shape), recognition_shape=list(recognition.shape),
                  finite_actions=bool(np.isfinite(actions).all()),
                  quaternion_unit=bool(np.allclose(np.linalg.norm(actions[:, 3:7], axis=1), 1, atol=1e-5)),
                  binary_gripper=bool(np.isin(actions[:, 7], [-1, 1]).all()),
                  gt_injection_rejected=rejected_gt, smoke_deployment_blocked=blocked_smoke,
                  live_physical_rollout=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    checks = ('pair_split_consistent', 'finite_actions', 'quaternion_unit',
              'binary_gripper', 'gt_injection_rejected', 'smoke_deployment_blocked')
    if (not all(report[key] for key in checks)
            or report['output_shape'] != [16, 8]
            or report['recognition_shape'] != [6, 224, 224]):
        raise SystemExit('Pipeline verification failed; inspect the saved report')


if __name__ == '__main__':
    main()
