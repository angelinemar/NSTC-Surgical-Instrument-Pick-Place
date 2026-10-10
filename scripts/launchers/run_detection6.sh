#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/runtime6.sh"
cd "$ROOT"
MODE="${1:-panel}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
    panel|record|assemble) exec "$P4_ISAAC_PYTHON" -m "detection.$MODE" "$@" ;;
    python) exec "$P4_ISAAC_PYTHON" "$@" ;;
    *) echo 'Usage: run_detection6.sh {panel|record|assemble|python} [arguments...]' >&2; exit 2 ;;
esac
