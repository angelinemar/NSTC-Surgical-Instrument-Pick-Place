"""Recovered cases retain failed-attempt accounting and still require two valid H5."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'debug/validation/approach_matrix'))
from benchmark_all import classify, write_json


class AccountingTests(unittest.TestCase):
    def test_transient_windows_lock_retries_atomic_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'summary.json'
            path.write_text('{"completed": 58}')
            original = Path.replace
            calls = []
            def replace(source, destination):
                calls.append(source)
                if len(calls) < 3:
                    self.assertEqual(json.loads(path.read_text())['completed'], 58)
                    raise PermissionError('Injected Windows sharing violation')
                return original(source, destination)
            with patch.object(Path, 'replace', replace), patch('benchmark_all.time.sleep'):
                write_json(path, {'completed': 59})
            self.assertEqual(json.loads(path.read_text())['completed'], 59)
            self.assertEqual(len(calls), 3)

    def test_permanent_lock_preserves_previous_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'summary.json'
            path.write_text('{"completed": 58}')
            with patch.object(Path, 'replace', side_effect=PermissionError), patch('benchmark_all.time.sleep'):
                with self.assertRaises(PermissionError):
                    write_json(path, {'completed': 59})
            self.assertEqual(json.loads(path.read_text())['completed'], 58)

    def classify_case(self, attempts, saved, failed, files, audit_exit=0):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'run_metrics.json').write_text(json.dumps(dict(
                total_attempts=attempts, success_count=saved, failure_count=failed)))
            for index in range(files):
                file = path / f'skill{index}_policy/scalpel/episode_000000.h5'
                file.parent.mkdir(parents=True)
                file.touch()
            with patch('benchmark_all.subprocess.run', return_value=SimpleNamespace(returncode=audit_exit)):
                return classify(path, 0)

    def test_recovery_is_not_first_attempt_success(self):
        result = self.classify_case(2, 1, 1, 2)
        self.assertEqual(result['status'], 'PASS')
        self.assertTrue(result['recovered_after_failure'])
        self.assertFalse(result['first_attempt_success'])
        self.assertEqual(result['metrics']['failure_count'], 1)

    def test_missing_pair_or_hidden_failure_is_integrity_error(self):
        for args in ((2, 1, 1, 1), (2, 1, 0, 2), (2, 0, 2, 1), (2, 2, 0, 4)):
            with self.subTest(args=args):
                self.assertEqual(self.classify_case(*args)['status'], 'INTEGRITY_ERROR')

    def test_all_attempts_failed_with_zero_saved(self):
        self.assertEqual(self.classify_case(2, 0, 2, 0)['status'], 'FAIL')

    def test_recovery_still_requires_audit(self):
        self.assertEqual(self.classify_case(2, 1, 1, 2, audit_exit=1)['status'], 'AUDIT_FAIL')


if __name__ == '__main__':
    unittest.main()
