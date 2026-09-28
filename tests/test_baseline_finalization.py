"""Finalization must not certify partial, duplicated, or failed coverage."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'debug/validation/approach_matrix'))
from finalize_baseline import OBJECTS, require_coverage


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        cases = [dict(object=obj, cell=cell, yaw=0, id=f'{obj}_{cell}')
                 for obj in OBJECTS for cell in range(10)]
        self.plan = dict(cases=cases)
        self.summary = dict(complete=True, completed=50, planned=50, active=None,
                            results=[dict(case=copy.deepcopy(case), status='PASS', h5_count=2,
                                          metrics=dict(success_count=1, total_attempts=1, failure_count=0))
                                     for case in cases])

    def test_recovered_success_is_accepted(self):
        self.summary['results'][0]['metrics'].update(total_attempts=2, failure_count=1)
        require_coverage(self.summary, self.plan)

    def test_partial_or_failed_coverage_rejected(self):
        for field, value in [('complete', False), ('completed', 49), ('active', {'id': 'running'})]:
            summary = copy.deepcopy(self.summary)
            summary[field] = value
            with self.assertRaises(ValueError):
                require_coverage(summary, self.plan)
        self.summary['results'][0]['status'] = 'FAIL'
        with self.assertRaises(ValueError):
            require_coverage(self.summary, self.plan)

    def test_duplicate_cannot_replace_missing_grid(self):
        self.summary['results'][-1] = copy.deepcopy(self.summary['results'][0])
        with self.assertRaises(ValueError):
            require_coverage(self.summary, self.plan)

    def test_invalid_storage_or_accounting_rejected(self):
        for change in ('h5', 'accounting'):
            summary = copy.deepcopy(self.summary)
            if change == 'h5':
                summary['results'][0]['h5_count'] = 4
            else:
                summary['results'][0]['metrics']['total_attempts'] = 2
            with self.assertRaises(ValueError):
                require_coverage(summary, self.plan)


if __name__ == '__main__':
    unittest.main()
