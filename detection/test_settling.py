import unittest
from detection.settling import settle

class SettlingTests(unittest.TestCase):
    def test_slow_contact_gets_time_and_sustained_checks(self):
        counter=[0]
        def step(): counter[0]+=1
        result=settle(step,lambda:{'scissor':.04 if counter[0]<200 else .001})
        self.assertEqual(result['steps'],220)

    def test_persistent_motion_fails_without_relaxing_limit(self):
        with self.assertRaisesRegex(RuntimeError,'scissor.*0.020000'):
            settle(lambda:None,lambda:{'scissor':.02},maximum_steps=150)

    def test_brief_stop_does_not_pass(self):
        counter=[0]
        def step():counter[0]+=1
        with self.assertRaises(RuntimeError):
            settle(step,lambda:{'scissor':0 if counter[0]==120 else .04},maximum_steps=150)

    def test_nonfinite_motion_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'Invalid'):
            settle(lambda:None,lambda:{'scissor':float('nan')})
