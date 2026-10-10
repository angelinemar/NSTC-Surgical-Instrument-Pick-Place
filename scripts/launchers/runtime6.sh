#!/usr/bin/env bash
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${P4_ISAAC_PYTHON:-}" ]]; then
    for base in "$ROOT" "$ROOT/.." "$ROOT/../.."; do
        if [[ -x "$base/.venv-isaacsim6/bin/python" ]]; then
            export P4_ISAAC_PYTHON="$base/.venv-isaacsim6/bin/python"
            break
        fi
    done
fi
if [[ ! -x "${P4_ISAAC_PYTHON:-}" ]]; then
    echo 'Set P4_ISAAC_PYTHON to the Isaac Sim 6.0.1 Python executable.' >&2
    exit 1
fi
RUNTIME="$(cd -- "$(dirname -- "$P4_ISAAC_PYTHON")/../.." && pwd)"
COMPAT="$RUNTIME/.tools/compat/usr/lib/x86_64-linux-gnu"
if [[ -d "$COMPAT" ]]; then
    export LD_LIBRARY_PATH="$COMPAT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH="$ROOT:$ROOT/compat${PYTHONPATH:+:$PYTHONPATH}"
export P4_SIM_VERSION=6.0
export P4_ROBOT_USD="${P4_ROBOT_USD:-https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd}"
