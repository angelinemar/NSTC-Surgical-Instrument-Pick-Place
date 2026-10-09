import unittest
from pathlib import Path
import shutil

from src.recorder.phase3_run_metrics import failure_breakdown, reconcile_attempt_timeline, write_attempt_timeline, write_failure_chart, write_failure_reason_line


class FailureSummaryTests(unittest.TestCase):
    def test_groups_numeric_variants_and_sorts_highest_first(self):
        attempts = [
            {"status": "place_fail", "reason": "[PHASE FAIL] place | KELLY_MOVE_TO_TARGET | pose_timeout: error=0.07m angle=24deg; DISCARD"},
            {"status": "place_fail", "reason": "[PHASE FAIL] place | KELLY_MOVE_TO_TARGET | pose_timeout: error=0.05m angle=19deg; DISCARD"},
            {"status": "pick_fail", "reason": "[PHASE FAIL] pick | KELLY_LOWER_PRE | pose_stalled_or_diverging: error=0.08m angle=11deg; DISCARD"},
        ]
        rows = failure_breakdown(attempts)
        self.assertEqual(rows[0]["count"], 2)
        self.assertEqual(rows[0]["stage"], "KELLY_MOVE_TO_TARGET")
        self.assertEqual(rows[0]["reason"], "pose_timeout")

    def test_chart_is_written(self):
        folder = Path("tests/.tmp_failure_summary")
        folder.mkdir(parents=True, exist_ok=True)
        try:
            path = folder / "chart.png"
            write_failure_chart(path, [{"stage": "LOWER_PRE", "reason": "pose_timeout", "count": 2, "percent": 100.0}], "kelly")
            self.assertTrue(path.is_file())
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def test_timeline_is_written(self):
        folder = Path("tests/.tmp_attempt_timeline")
        folder.mkdir(parents=True, exist_ok=True)
        try:
            path = folder / "timeline.png"
            write_attempt_timeline(path, [{"status": "success"}, {"status": "pick_fail"}], "scalpel")
            self.assertTrue(path.is_file())
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def test_reconciles_legacy_aggregate_totals(self):
        data = {"success_count": 2, "failure_count": 1, "attempts": [{"status": "success"}]}
        attempts = reconcile_attempt_timeline(data)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(sum(a["status"] == "success" for a in attempts), 2)

    def test_failure_reason_line_is_written(self):
        folder = Path("tests/.tmp_failure_reason_line")
        folder.mkdir(parents=True, exist_ok=True)
        try:
            path = folder / "reason_line.png"
            write_failure_reason_line(path, [{"stage": "LOWER_PRE", "reason": "pose_timeout", "count": 3, "percent": 100.0}], "kelly")
            self.assertTrue(path.is_file())
        finally:
            shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
