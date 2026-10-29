#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO:$REPO/compat${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${NSTC_RFDETR_PYTHON:-$REPO/.venv-rfdetr/bin/python}"
if [[ ! -x "$PYTHON_BIN" && -x /home/user/NSTC/NSTC-Surgical-Instrument-Pick-Place/.venv-rfdetr/bin/python ]]; then
    PYTHON_BIN=/home/user/NSTC/NSTC-Surgical-Instrument-Pick-Place/.venv-rfdetr/bin/python
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "RF-DETR Python not found. Install the environment in instrument_bridge/README.md or set NSTC_RFDETR_PYTHON." >&2
    exit 2
fi
export NSTC_RFDETR_PYTHON="$PYTHON_BIN"
cd "$REPO"
exec "$PYTHON_BIN" -m instrument_bridge.studio "$@"
