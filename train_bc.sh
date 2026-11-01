#!/usr/bin/env bash
# Train a BC-RNN policy on 26-D waist-enabled Pink-Abs demonstrations.
# Usage: bash train_bc.sh [dataset_file] [epochs] [run_name]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/scripts/pipeline_common.sh"

g1d_load_project_env
DEFAULT_DATASET="$G1D_DATA_DIR/generated_dataset.hdf5"
if [[ ! -f "$DEFAULT_DATASET" ]]; then
    DEFAULT_DATASET="$G1D_DATA_DIR/dataset.hdf5"
fi
DATASET_FILE="${1:-${G1D_TRAIN_DATASET:-$DEFAULT_DATASET}}"
EPOCHS="${2:-${G1D_BC_EPOCHS:-}}"
RUN_NAME="${3:-${G1D_BC_RUN_NAME:-g1d_handover_pink_abs_bc_rnn}}"
TASK="G1D-Bimanual-CuboidHandover-Pink-Abs-v0"

g1d_require_file "$DATASET_FILE"
if [[ -n "$EPOCHS" ]]; then
    g1d_require_positive_integer "epochs" "$EPOCHS"
fi

"$G1D_ISAAC_PYTHON" "$G1D_PROJECT_ROOT/scripts/validate_dataset.py" \
    "$DATASET_FILE" --kind training --action-layout pink-abs

ARGS=(
    --task "$TASK"
    --algo bc
    --dataset "$DATASET_FILE"
    --log_dir robomimic
    --name "$RUN_NAME"
)
if [[ -n "$EPOCHS" ]]; then
    ARGS+=(--epochs "$EPOCHS")
fi

g1d_run_isaaclab_tool \
    "$G1D_ISAACLAB_ROOT/scripts/imitation_learning/robomimic/train.py" \
    "${ARGS[@]}"
