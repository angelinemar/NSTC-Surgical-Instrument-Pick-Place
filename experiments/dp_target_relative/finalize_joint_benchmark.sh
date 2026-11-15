#!/usr/bin/env bash
set -euo pipefail

cd /home/user/NSTC
base='datasets/DP_evaluation/target_relative_individual_vs_joint_100_20261103'
export MPLCONFIGDIR=/tmp/nstc_joint_benchmark_mpl_20261103
mkdir -p "$MPLCONFIGDIR"

for ((poll=0; poll<2880; poll++)); do
  if python3 - "$base" <<'PY'
import json,sys
from pathlib import Path
base=Path(sys.argv[1])
paths=[base/v/s/'summary.json' for v in ('individual','joint') for s in ('pick','place')]
if len(paths)!=4 or any(not p.exists() for p in paths):sys.exit(1)
if any(not json.loads(p.read_text()).get('complete') for p in paths):sys.exit(1)
PY
  then
    break
  fi
  sleep 30
done

python3 - "$base" <<'PY'
import json,sys
from pathlib import Path
base=Path(sys.argv[1])
paths=[base/v/s/'summary.json' for v in ('individual','joint') for s in ('pick','place')]
if any(not p.exists() or not json.loads(p.read_text()).get('complete') for p in paths):
    raise SystemExit('Timed out before all four 100-trial summaries completed')
PY

.venv-dp/bin/python experiments/dp_target_relative/build_joint_vs_individual_results.py
python3 competition_report/build_checkpoint_registry.py
NSTC-Surgical-Instrument-Pick-Place/.venv-rfdetr/bin/python competition_report/build_report.py
google-chrome --headless --no-sandbox --disable-gpu --disable-dev-shm-usage \
  --no-pdf-header-footer \
  --user-data-dir=/tmp/nstc_joint_benchmark_chrome_20261103 \
  --print-to-pdf=/home/user/NSTC/competition_report/NSTC_surgical_instrument_report.pdf \
  file:///home/user/NSTC/competition_report/report.html
pdfinfo competition_report/NSTC_surgical_instrument_report.pdf | rg 'Pages|Page size'
printf 'FINAL_REPORT_READY %s\n' "$(date -Is)"
