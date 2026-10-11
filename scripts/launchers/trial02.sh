#!/usr/bin/env bash
set -euo pipefail
SEG_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SEG_SCRIPT_DIR/runtime6.sh"
cd "$ROOT"
exec "${P4_RFDETR_PYTHON:-$RUNTIME/NSTC-Surgical-Instrument-Pick-Place/.venv-rfdetr/bin/python}" -m detection.trial02 "$@"
