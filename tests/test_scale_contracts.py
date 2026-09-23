import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training'))
from phase4_approach import precision_integral
from phase4_storage import EpisodeTransaction, recover, require_commit
from export_sensor_only import assert_sensor_inputs, INPUTS, canonical_quaternions
from sensor_policy import SensorPolicy, decode_actions


def writer(recorder, ep_idx, segment, out_dir, success, object_name, meta=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(out / f'episode_{ep_idx:06d}.h5', 'w') as h:
        h.attrs.update(meta)
        h.attrs['target_object'] = object_name
        h.create_dataset('actions', data=np.zeros((2, 8)))
    return True


class StorageTests(unittest.TestCase):
    def bound(self, root, skill):
        return inspect.signature(writer).bind(None, 0, skill, str(Path(root) / (skill + '_policy') / 'scissor'), True, 'scissor', {})

    @patch('validate_feedback_dataset.audit')
    def test_pair_is_not_complete_until_both_validate(self, audit):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            tx = EpisodeTransaction(folder, 'scissor', 0, 'both')
            tx.save(writer, self.bound(folder, 'pick'))
            self.assertFalse(tx.commit.exists())
            self.assertFalse(list(Path(folder).glob('*_policy/*/*.h5')))
            tx.save(writer, self.bound(folder, 'place'))
            self.assertTrue(tx.commit.exists())
            self.assertEqual(audit.call_count, 2)
            for file in Path(folder).glob('*_policy/*/*.h5'):
                with h5py.File(file) as h:
                    require_commit(file, h)

    @patch('validate_feedback_dataset.audit')
    def test_crash_during_publish_is_quarantined(self, audit):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            tx = EpisodeTransaction(folder, 'scissor', 0, 'both')
            tx.save(writer, self.bound(folder, 'pick'))
            import phase4_storage
            original = phase4_storage.os.rename
            count = [0]
            def crash(source, destination):
                count[0] += 1
                if count[0] == 2:
                    raise OSError('Injected interruption')
                return original(source, destination)
            with patch('phase4_storage.os.rename', side_effect=crash):
                with self.assertRaises(OSError):
                    tx.save(writer, self.bound(folder, 'place'))
            self.assertFalse(tx.commit.exists())
            orphan = next(Path(folder).glob('*_policy/*/*.h5'))
            with h5py.File(orphan) as h:
                with self.assertRaises(ValueError):
                    require_commit(orphan, h)
            recover(folder)
            self.assertFalse(list(Path(folder).glob('*_policy/*/*.h5')))
            self.assertEqual(len(list((Path(folder) / '.quarantine').rglob('*.h5'))), 2)

    @patch('validate_feedback_dataset.audit')
    def test_single_policy_mode_has_one_committed_file(self, audit):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            tx = EpisodeTransaction(folder, 'scissor', 0, 'pick')
            tx.save(writer, self.bound(folder, 'pick'))
            tx.save(writer, self.bound(folder, 'place'))
            self.assertEqual(len(list(Path(folder).glob('*_policy/*/*.h5'))), 1)
            self.assertTrue(tx.commit.exists())

    @patch('validate_feedback_dataset.audit')
    def test_modified_committed_file_is_rejected(self, audit):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            tx = EpisodeTransaction(folder, 'scissor', 0, 'pick')
            tx.save(writer, self.bound(folder, 'pick'))
            file = next(Path(folder).glob('*_policy/*/*.h5'))
            with h5py.File(file, 'r+') as h:
                h['actions'][0, 0] = 7
            with h5py.File(file) as h:
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    require_commit(file, h)


class SensorBoundaryTests(unittest.TestCase):
    def test_action_chunk_execution_can_update_adjacent_sensor_history(self):
        from collections import deque
        from runtime import PolicyRuntime
        policy = PolicyRuntime.__new__(PolicyRuntime)
        policy.history = deque(maxlen=2)
        sensors = {k: np.zeros((224, 224, 3), dtype=np.uint8) for k in INPUTS if k != 'robot_proprio'}
        sensors['robot_proprio'] = np.zeros(16, dtype=np.float32)
        sensors['robot_proprio'][12] = 1
        for tick in (1, 2, 3):
            sensors['robot_proprio'][0] = tick
            policy.observe(sensors)
        self.assertEqual([row[1][0] for row in policy.history], [2, 3])

    def test_object_pose_id_stage_and_semantics_are_rejected_as_inputs(self):
        clean = dict.fromkeys(INPUTS)
        assert_sensor_inputs(clean)
        for key in ('state', 'object_type_id', 'stage_id', 'selected_object_pose_b', 'front_semantic', 'target_slot_b'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                assert_sensor_inputs(dict(clean, **{key: None}))

    def test_predict_api_has_no_ground_truth_or_target_action_argument(self):
        self.assertEqual(tuple(inspect.signature(SensorPolicy.act).parameters),
                         ('self', 'rgb', 'proprio', 'inference_steps', 'generator'))

    def test_semantic_labels_cannot_condition_diffusion_prediction(self):
        torch.set_num_threads(2)
        model = SensorPolicy()
        batch = dict(rgb=torch.rand(1, 2, 6, 3, 32, 32), proprio=torch.zeros(1, 2, 16),
                     actions=torch.zeros(1, 16, 8), semantic=torch.zeros(1, 6, 32, 32, dtype=torch.long))
        torch.manual_seed(73)
        _, first_dp, first_seg = model.losses(batch)
        batch['semantic'].fill_(7)
        torch.manual_seed(73)
        _, second_dp, second_seg = model.losses(batch)
        torch.testing.assert_close(first_dp, second_dp, rtol=0, atol=0)
        self.assertNotEqual(float(first_seg), float(second_seg))

    def test_quaternion_signs_and_degenerate_prediction(self):
        result = canonical_quaternions([[1, 0, 0, 0], [-1, 0, 0, 0]])
        np.testing.assert_array_equal(result[0], result[1])
        stats = dict(action_center=[0] * 8, action_scale=[1] * 8)
        with self.assertRaises(ValueError):
            decode_actions(torch.zeros(1, 8), stats)


class IntegralTests(unittest.TestCase):
    def test_place_to_pick_clears_integral_and_target_history(self):
        from types import SimpleNamespace as NS
        from phase4_approach import set_precision_skill
        controller = NS(_p4_precision_skill='place', _p4_integral=np.ones((1, 3)), _p4_previous_desired=np.ones((1, 3)))
        env = NS(num_envs=1, action_manager=NS(get_term=lambda _: NS(_ik_controller=controller)))
        set_precision_skill(env, 'pick')
        self.assertIsNone(controller._p4_previous_desired)
        np.testing.assert_array_equal(controller._p4_integral, np.zeros((1, 3)))
        self.assertEqual(controller._p4_precision_skill, 'pick')

    def test_compliance_bias_is_reduced_without_exceeding_bound(self):
        integral = np.zeros(3)
        disturbance = np.array([0., 0., .0029])
        for _ in range(80):
            integral = precision_integral(integral, disturbance - integral, np.zeros(3), True)
        self.assertLess(np.linalg.norm(disturbance - integral), .0001)
        for _ in range(100):
            integral = precision_integral(integral, disturbance, np.zeros(3), True)
        self.assertLessEqual(np.linalg.norm(integral), .004000001)
        np.testing.assert_array_equal(precision_integral(integral, disturbance, np.zeros(3), False), np.zeros(3))


if __name__ == '__main__':
    unittest.main()
