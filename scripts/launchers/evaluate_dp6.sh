#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime6.sh"
exec "$P4_ISAAC_PYTHON" -m evaluation.dp_suite "$@"
