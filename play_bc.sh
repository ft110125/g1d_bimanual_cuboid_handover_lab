#!/usr/bin/env bash
# Evaluate a trained 26-D Pink-Abs robomimic checkpoint.
# Usage: bash play_bc.sh <checkpoint> [num_rollouts] [horizon]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/scripts/pipeline_common.sh"

g1d_load_project_env
CHECKPOINT="${1:-}"
NUM_ROLLOUTS="${2:-${G1D_BC_NUM_ROLLOUTS:-10}}"
HORIZON="${3:-${G1D_BC_HORIZON:-1500}}"
TASK="G1D-Bimanual-CuboidHandover-Pink-Abs-PLAY-v0"

[[ -n "$CHECKPOINT" ]] || _g1d_die "Usage: bash play_bc.sh <checkpoint> [num_rollouts] [horizon]"
g1d_require_file "$CHECKPOINT"
g1d_require_positive_integer "num_rollouts" "$NUM_ROLLOUTS"
g1d_require_positive_integer "horizon" "$HORIZON"

EXTRA_ARGS=()
if _g1d_is_true "${G1D_HEADLESS:-0}"; then
    EXTRA_ARGS+=(--headless)
fi

g1d_run_isaaclab_tool \
    "$G1D_ISAACLAB_ROOT/scripts/imitation_learning/robomimic/play.py" \
    --task "$TASK" \
    --checkpoint "$CHECKPOINT" \
    --num_rollouts "$NUM_ROLLOUTS" \
    --horizon "$HORIZON" \
    --enable_pinocchio \
    --device "$G1D_DEVICE" \
    "${EXTRA_ARGS[@]}"
