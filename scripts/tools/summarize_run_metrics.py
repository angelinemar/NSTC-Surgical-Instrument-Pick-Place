"""Regenerate compact TXT and PNG failure summaries from existing run_metrics.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.recorder.phase3_run_metrics import failure_breakdown, reconcile_attempt_timeline, write_attempt_timeline, write_failure_chart, write_failure_reason_line


def summarize(path: Path):
    metrics_path = path if path.name == "run_metrics.json" else path / "run_metrics.json"
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = failure_breakdown(data.get("attempts", []))
    data["failure_breakdown"] = rows
    metrics_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    lines = [
        "RECORDER RUN SUMMARY",
        "=" * 96,
        f"Object: {data.get('object', '?')}",
        f"Attempts: {data.get('total_attempts', 0)} | Successful: {data.get('success_count', 0)} | Failed: {data.get('failure_count', 0)}",
        f"Spawn: {data.get('spawn_fail_count', 0)} | Pick: {data.get('pick_fail_count', 0)} | Place: {data.get('place_fail_count', 0)} | Other: {data.get('other_fail_count', 0)}",
        "",
        "FAILURE BREAKDOWN (highest first)",
        "COUNT   PERCENT   STAGE                         REASON",
    ]
    lines += [f"{r['count']:>5}   {r['percent']:>6.1f}%   {r['stage']:<29} {r['reason']}" for r in rows] or ["None"]
    (metrics_path.parent / "run_metrics_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    chart = write_failure_chart(metrics_path.parent / "failure_breakdown.png", rows, data.get("object", "run"))
    reason_line = write_failure_reason_line(metrics_path.parent / "failure_reason_line.png", rows, data.get("object", "run"))
    timeline = write_attempt_timeline(metrics_path.parent / "attempt_timeline.png", reconcile_attempt_timeline(data), data.get("object", "run"))
    print(f"UPDATED {metrics_path.parent} rows={len(rows)} chart={chart} reason_line={reason_line} timeline={timeline}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        summarize(path.resolve())


if __name__ == "__main__":
    main()
