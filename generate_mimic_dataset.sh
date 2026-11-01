#!/usr/bin/env bash
# Generate additional waist-enabled demonstrations with Isaac Lab Mimic.
# Usage: bash generate_mimic_dataset.sh [num_trials] [num_envs] [input_file] [output_file]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/scripts/pipeline_common.sh"

g1d_load_project_env
NUM_TRIALS="${1:-${G1D_MIMIC_NUM_TRIALS:-100}}"
NUM_ENVS="${2:-${G1D_MIMIC_NUM_ENVS:-4}}"
INPUT_FILE="${3:-${G1D_ANNOTATED_DATASET:-$G1D_DATA_DIR/annotated_dataset.hdf5}}"
OUTPUT_FILE="${4:-${G1D_GENERATED_DATASET:-$G1D_DATA_DIR/generated_dataset.hdf5}}"
TASK="G1D-Bimanual-CuboidHandover-Pink-Abs-Mimic-v0"

g1d_require_positive_integer "num_trials" "$NUM_TRIALS"
g1d_require_positive_integer "num_envs" "$NUM_ENVS"
g1d_require_file "$INPUT_FILE"
"$G1D_ISAAC_PYTHON" "$G1D_PROJECT_ROOT/scripts/validate_dataset.py" \
    "$INPUT_FILE" --kind annotated --action-layout pink-abs
g1d_prepare_output_file "$OUTPUT_FILE"

EXTRA_ARGS=()
if ! _g1d_is_true "${G1D_MIMIC_GUI:-0}"; then
    EXTRA_ARGS+=(--headless)
fi

g1d_run_isaaclab_tool \
    "$G1D_ISAACLAB_ROOT/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py" \
    --task "$TASK" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --num_envs "$NUM_ENVS" \
    --generation_num_trials "$NUM_TRIALS" \
    --enable_pinocchio \
    --device "$G1D_DEVICE" \
    "${EXTRA_ARGS[@]}"

"$G1D_ISAAC_PYTHON" "$G1D_PROJECT_ROOT/scripts/validate_dataset.py" \
    "$OUTPUT_FILE" --kind generated --action-layout pink-abs \
    --expected-episodes "$NUM_TRIALS"
