#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export P4_ISAAC_PYTHON="${P4_ISAAC_PYTHON:-$PROJECT_ROOT/../.venv-isaacsim/bin/python}"
if [[ ! -x "$P4_ISAAC_PYTHON" ]]; then
    echo "Isaac Python missing: $P4_ISAAC_PYTHON" >&2
    exit 1
fi
RUNTIME="$(cd -- "$(dirname -- "$P4_ISAAC_PYTHON")/../.." && pwd)"
cd "$PROJECT_ROOT"
COMPAT_LIB="$RUNTIME/.tools/compat/usr/lib/x86_64-linux-gnu"
if [[ -d "$COMPAT_LIB" ]]; then
    export LD_LIBRARY_PATH="$COMPAT_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/compat${PYTHONPATH:+:$PYTHONPATH}"
MODE="${1:-panel}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
    panel|record|tool) exec "$P4_ISAAC_PYTHON" scripts/p4.py "$MODE" "$@" ;;
    check) exec "$P4_ISAAC_PYTHON" scripts/check_structure.py "$@" ;;
    objects) exec "$P4_ISAAC_PYTHON" scripts/p4.py record --list-objects "$@" ;;
    export) exec "$P4_ISAAC_PYTHON" training/export_recordings.py "$@" ;;
    train) exec "$RUNTIME/.venv-dp/bin/python" training/train_sensor_policy.py "$@" ;;
    dp-python) exec "$RUNTIME/.venv-dp/bin/python" "$@" ;;
    rfdetr) exec "${P4_RFDETR_PYTHON:-$RUNTIME/NSTC-Surgical-Instrument-Pick-Place/.venv-rfdetr/bin/python}" training/rfdetr_pipeline.py "$@" ;;
    python) exec "$P4_ISAAC_PYTHON" "$@" ;;
    sim) exec "$(dirname -- "$P4_ISAAC_PYTHON")/isaacsim" "$@" ;;
    *) echo 'Usage: run_linux.sh {panel|record|check|objects|export|train|rfdetr|python|dp-python|sim|tool} [arguments...]' >&2; exit 2 ;;
esac
