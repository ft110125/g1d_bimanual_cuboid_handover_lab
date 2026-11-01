#!/usr/bin/env bash
# Add Pink-Abs Isaac Lab Mimic subtask annotations to human demonstrations.
# Usage: bash annotate_demos.sh [input_file] [output_file]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/scripts/pipeline_common.sh"

g1d_load_project_env
INPUT_FILE="${1:-${G1D_RAW_DATASET:-$G1D_DATA_DIR/dataset.hdf5}}"
OUTPUT_FILE="${2:-${G1D_ANNOTATED_DATASET:-$G1D_DATA_DIR/annotated_dataset.hdf5}}"
TASK="G1D-Bimanual-CuboidHandover-Pink-Abs-Mimic-v0"

g1d_require_file "$INPUT_FILE"
"$G1D_ISAAC_PYTHON" "$G1D_PROJECT_ROOT/scripts/validate_dataset.py" \
    "$INPUT_FILE" --kind raw --action-layout pink-abs
g1d_prepare_output_file "$OUTPUT_FILE"

EXTRA_ARGS=()
if _g1d_is_true "${G1D_HEADLESS:-0}"; then
    EXTRA_ARGS+=(--headless)
fi

# Isaac Lab v2.3.2 may return the number of annotated demonstrations as its
# process status. Validate the output HDF5 instead of interpreting every
# non-zero status as an annotation failure.
set +e
g1d_run_isaaclab_tool \
    "$G1D_ISAACLAB_ROOT/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py" \
    --task "$TASK" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --auto \
    --enable_pinocchio \
    --device "$G1D_DEVICE" \
    "${EXTRA_ARGS[@]}"
ANNOTATE_STATUS=$?
set -e

if [[ ! -s "$OUTPUT_FILE" ]]; then
    echo "ERROR: annotation did not create a non-empty dataset: $OUTPUT_FILE" >&2
    echo "annotate_demos.py exit status: $ANNOTATE_STATUS" >&2
    exit 1
fi

echo "annotate_demos.py exit status: $ANNOTATE_STATUS"
echo "Validating the annotated Pink-Abs dataset..."
"$G1D_ISAAC_PYTHON" "$G1D_PROJECT_ROOT/scripts/validate_dataset.py" \
    "$OUTPUT_FILE" --kind annotated --action-layout pink-abs \
    --reference-dataset "$INPUT_FILE"
