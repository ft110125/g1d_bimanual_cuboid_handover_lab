#!/usr/bin/env bash
# Replay and validate waist-enabled Pink-Abs demonstrations.
# Usage: bash replay_demos.sh [dataset_file] [num_envs]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/scripts/pipeline_common.sh"

g1d_load_project_env
DATASET_FILE="${1:-${G1D_RAW_DATASET:-$G1D_DATA_DIR/dataset.hdf5}}"
NUM_ENVS="${2:-1}"
TASK="G1D-Bimanual-CuboidHandover-Pink-Abs-Record-v0"

g1d_require_file "$DATASET_FILE"
g1d_require_positive_integer "num_envs" "$NUM_ENVS"

"$G1D_ISAAC_PYTHON" "$G1D_PROJECT_ROOT/scripts/validate_dataset.py" \
    "$DATASET_FILE" --kind raw --action-layout pink-abs

EXTRA_ARGS=()
if _g1d_is_true "${G1D_HEADLESS:-0}"; then
    EXTRA_ARGS+=(--headless)
fi

g1d_run_isaaclab_tool \
    "$G1D_ISAACLAB_ROOT/scripts/tools/replay_demos.py" \
    --task "$TASK" \
    --dataset_file "$DATASET_FILE" \
    --num_envs "$NUM_ENVS" \
    --validate_success_rate \
    --enable_pinocchio \
    --device "$G1D_DEVICE" \
    "${EXTRA_ARGS[@]}"
