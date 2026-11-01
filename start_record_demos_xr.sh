#!/usr/bin/env bash
# Record waist-enabled right-to-left handover demonstrations with Quest tracking.
# Usage: bash start_record_demos_xr.sh [num_demos] [dataset_file]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/scripts/pipeline_common.sh"

g1d_load_project_env
g1d_require_runtime
g1d_setup_cloudxr

NUM_DEMOS="${1:-${G1D_RECORD_NUM_DEMOS:-10}}"
DATASET_FILE="${2:-${G1D_RAW_DATASET:-$G1D_DATA_DIR/dataset.hdf5}}"
SUCCESS_STEPS="${G1D_RECORD_SUCCESS_STEPS:-10}"
TASK="G1D-Bimanual-CuboidHandover-Pink-Abs-Record-v0"
RECORD_AUTO_START="${G1D_RECORD_AUTO_START:-1}"
RECORD_TOOL="$G1D_ISAACLAB_ROOT/scripts/tools/record_demos.py"

g1d_require_nonnegative_integer "num_demos" "$NUM_DEMOS"
g1d_require_positive_integer "G1D_RECORD_SUCCESS_STEPS" "$SUCCESS_STEPS"
case "$RECORD_AUTO_START" in
    0|1|true|TRUE|false|FALSE|yes|YES|no|NO|on|ON) ;;
    *) _g1d_die "G1D_RECORD_AUTO_START must be a boolean value: $RECORD_AUTO_START" ;;
esac
g1d_require_file "$RECORD_TOOL"

if _g1d_is_true "$RECORD_AUTO_START"; then
    XR_CONTROL_HELP="Recording starts automatically. STOP pauses, START resumes, RESET discards the current attempt."
    echo "Preparing project-local XR auto-start recorder..."
    RECORD_TOOL="$(
        "$G1D_ISAAC_PYTHON" \
            "$G1D_PROJECT_ROOT/scripts/prepare_record_demos_autostart.py" \
            --source "$RECORD_TOOL"
    )"
    [[ -f "$RECORD_TOOL" ]] || _g1d_die "Auto-start recorder was not created: $RECORD_TOOL"
else
    XR_CONTROL_HELP="Recording starts paused. Send START before moving; STOP pauses and RESET discards the current attempt."
fi

# Do not move an existing dataset until every required runtime file and the
# optional auto-start recorder have been validated successfully.
g1d_prepare_output_file "$DATASET_FILE"

# Recording must use Pink-Abs so both wrist targets are solved jointly over
# the arms and the torso_Joint + Yaw_Joint waist redundancy.
export G1D_TELEOP_TASK="$TASK"
export G1D_DEBUG_HANDOVER_PHASES="${G1D_DEBUG_HANDOVER_PHASES:-1}"

cat <<EOF
===========================================================================
G1-D Waist-Enabled Handover Demonstration Recording
===========================================================================
Task:              $TASK
Action layout:     26-D Pink-Abs
                   left wrist 7 + right wrist 7 + right hand 6 + left hand 6
Waist control:     enabled (torso_Joint + Yaw_Joint through Pink IK)
Dataset:           $DATASET_FILE
Target demos:      $NUM_DEMOS (0 means unlimited)
Success hold:      $SUCCESS_STEPS consecutive steps
Timeout:           disabled by Record config and Isaac Lab record_demos.py
XR start mode:     ${RECORD_AUTO_START} (1=automatic, 0=wait for START)
Device:            $G1D_DEVICE
XR runtime:        $XR_RUNTIME_JSON
Recorder:          $RECORD_TOOL
Phase diagnostics: $G1D_DEBUG_HANDOVER_PHASES (1=print phase transitions)

$XR_CONTROL_HELP
Only the ordered sequence right-side pickup -> handover near HANDOVER_POS ->
left-side stable placement is accepted. Finger-joint closure is not used by the
success labels, and no final both-hands-clear phase is required.
===========================================================================
EOF

g1d_run_isaaclab_tool \
    "$RECORD_TOOL" \
    --task "$TASK" \
    --teleop_device handtracking \
    --dataset_file "$DATASET_FILE" \
    --num_demos "$NUM_DEMOS" \
    --num_success_steps "$SUCCESS_STEPS" \
    --enable_pinocchio \
    --device "$G1D_DEVICE"

if [[ ! -s "$DATASET_FILE" ]]; then
    echo "ERROR: recording did not create a non-empty dataset: $DATASET_FILE" >&2
    exit 1
fi

echo "Validating the recorded Pink-Abs dataset..."
VALIDATE_ARGS=("$DATASET_FILE" --kind raw --action-layout pink-abs)
if [[ "$NUM_DEMOS" != "0" ]]; then
    VALIDATE_ARGS+=(--expected-episodes "$NUM_DEMOS")
fi
"$G1D_ISAAC_PYTHON" "$G1D_PROJECT_ROOT/scripts/validate_dataset.py" \
    "${VALIDATE_ARGS[@]}"
