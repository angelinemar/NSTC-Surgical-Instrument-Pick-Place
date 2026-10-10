#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/runtime6.sh"
export P4_SHUTDOWN_MODE="${P4_SHUTDOWN_MODE:-verified-exit}"
exec "$SCRIPT_DIR/run_linux.sh" "$@"
