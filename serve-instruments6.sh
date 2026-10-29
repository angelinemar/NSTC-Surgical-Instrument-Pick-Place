#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO:$REPO/compat${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${NSTC_RFDETR_PYTHON:-python3}"
exec "$PYTHON_BIN" -m instrument_bridge.serve "$@"
