import unittest
from detection.placement import ScenePlacementError,retry_placement

class PlacementTests(unittest.TestCase):
    def test_retry_keeps_only_success_and_reports_rejection(self):
        attempts=[];failures=[]
        def prepare(attempt):
            attempts.append(attempt)
            if attempt<2:raise ScenePlacementError('outside table')
            return {'valid':True}
        result=retry_placement(prepare,lambda attempt,error:failures.append(attempt))
        self.assertEqual(result,{'valid':True})
        self.assertEqual(attempts,[0,1,2]);self.assertEqual(failures,[0,1])

    def test_retries_are_bounded(self):
        failures=[]
        def prepare(attempt):raise ScenePlacementError('unsupported tool')
        with self.assertRaisesRegex(ScenePlacementError,'after 5 attempts'):
            retry_placement(prepare,lambda attempt,error:failures.append(attempt))
        self.assertEqual(len(failures),5)

    def test_other_failures_are_not_swallowed(self):
        def prepare(attempt):raise RuntimeError('camera labels invalid')
        with self.assertRaisesRegex(RuntimeError,'camera labels'):
            retry_placement(prepare,lambda *args:self.fail('Must not retry non-placement errors'))
