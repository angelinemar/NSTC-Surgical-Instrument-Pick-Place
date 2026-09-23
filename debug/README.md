# Debug and validation

This folder is not an automatically approved production dataset.

- `validation/approach_matrix`: reproducible case runner, resumable matrix, audits and reports.
- `validation/sensor_only_*_runtime.json`: input-boundary and inference smoke evidence.
- `test_runs/<run>/summary.json`: matrix progress and case status.
- `test_runs/<run>/<case>.log`: simulator/controller output.
- `test_runs/<run>/<case>/stage_metrics.jsonl`: actual motion errors/gates.
- `test_runs/<run>/<case>/.commits`: manifests for complete v5 episodes.
- `archive`: historical snapshots, not active imports.
- `tmp`: scratch files, including interrupted test artifacts.
- `output`: historical generated reports.

Root `validation`, `test_runs`, `archive`, `tmp`, `output` are legacy junctions to these folders.
They do not duplicate storage. Do not recursively collect both aliases and canonical paths as
different episodes. The training exporter uses an explicit completed-case manifest instead
of recursively treating every H5 in this directory as training data.
