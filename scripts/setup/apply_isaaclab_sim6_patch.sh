#!/usr/bin/env bash
set -euo pipefail
LAB="${1:?Usage: apply_isaaclab_sim6_patch.sh /path/to/IsaacLab-v2.3.2}"
PATCH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/isaaclab-sim6-compat.patch"
EXPECTED=37ddf626871758333d6ed89cf64ad702aef127d0
if [[ "$(git -C "$LAB" rev-parse HEAD)" != "$EXPECTED" ]]; then
    echo "Expected Isaac Lab v2.3.2 commit $EXPECTED; refusing an unverified base." >&2
    exit 1
fi
if git -C "$LAB" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo 'Sim 6 compatibility patch is already applied.'
    exit 0
fi
git -C "$LAB" apply --check "$PATCH"
git -C "$LAB" apply "$PATCH"
echo 'Applied the local Isaac Sim 6.0.1 compatibility port. See docs/ISAAC_VERSIONS.md.'
