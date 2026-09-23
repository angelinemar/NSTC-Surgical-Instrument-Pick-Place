"""P4 continuation exercises the original stable-poll gate without simulator assets."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase3_settle_contract import settle_all_objects
from phase4_runtime import bounded_settle


class FakeEnv:
    def __init__(self, stable_after=0, invalid=None):
        self.steps = 0
        self.stable_after = stable_after
        self.invalid = invalid
        self.hold = object()

    def step(self, action):
        assert action is self.hold
        self.steps += 1

    def capture(self, *args):
        if self.invalid == 'missing':
            return {'test': {'missing': True}}
        return {'test': dict(position_w=[float('nan') if self.invalid else 0, 0, 0],
                             quaternion_wxyz=[1, 0, 0, 0],
                             linear_velocity_w=[0, 0, 0], angular_velocity_w=[0, 0, 0],
                             linear_speed=0 if self.steps >= self.stable_after else .1)}


class SettleTests(unittest.TestCase):
    def run_settle(self, env):
        with patch('phase3_settle_contract.capture_object_states', side_effect=env.capture):
            report = bounded_settle(settle_all_objects, env, env.hold, str,
                                    min_steps=150, poll_every=10, consecutive_ok=3,
                                    linear_speed_threshold=.01, names=('test',))
        self.assertEqual(report['steps'], env.steps)
        return report

    def test_ready_has_no_extra_wait(self):
        report = self.run_settle(FakeEnv())
        self.assertTrue(report['success'])
        self.assertEqual(report['steps'], 181)
        self.assertNotIn('continuation_steps', report)

    def test_damping_gets_same_gate_continuation(self):
        report = self.run_settle(FakeEnv(stable_after=470))
        self.assertTrue(report['success'])
        self.assertEqual(report['steps'], 491)
        self.assertEqual(report['continuation_steps'], 41)
        self.assertEqual(report['linear_speed_threshold'], .01)

    def test_unstable_is_bounded_failure(self):
        report = self.run_settle(FakeEnv(stable_after=10000))
        self.assertFalse(report['success'])
        self.assertEqual(report['steps'], 750)

    def test_invalid_never_continues_or_succeeds(self):
        for invalid in ('missing', 'nan'):
            with self.subTest(invalid=invalid):
                report = self.run_settle(FakeEnv(invalid=invalid))
                self.assertFalse(report['success'])
                self.assertNotIn('continuation_steps', report)
                self.assertLessEqual(report['steps'], 450)


if __name__ == '__main__':
    unittest.main()
