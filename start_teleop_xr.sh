#!/bin/bash
# G1D Bimanual Cuboid Handover — XR Teleoperation Startup Script
#
# Usage: bash start_teleop_xr.sh [num_envs]
#
# This script properly initializes:
# 1. OpenXR Runtime (CloudXR)
# 2. Isaac Lab teleop environment
# 3. Hand tracking from Quest 3

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_ENVS="${1:-1}"
PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"

# ---------------------------------------------------------------------------
# Load .env.teleop (if it exists as a regular file and is not a symlink)
# ---------------------------------------------------------------------------
_ENV_FILE="$SCRIPT_DIR/.env.teleop"
if [ -f "$_ENV_FILE" ] && [ ! -L "$_ENV_FILE" ]; then
    set -a
    . "$_ENV_FILE"
    set +a
fi

# ---------------------------------------------------------------------------
# Defaults for teleop tuning env vars (set only if not already defined)
# ---------------------------------------------------------------------------
: "${G1D_TELEOP_POS_SCALE:=1.0}"
: "${G1D_TELEOP_ROT_SCALE:=1.0}"
: "${G1D_TELEOP_ALPHA_POS:=0.85}"
: "${G1D_TELEOP_ALPHA_ROT:=0.85}"
: "${G1D_TELEOP_POSITION_THRESHOLD:=0.0001}"
: "${G1D_TELEOP_ROTATION_THRESHOLD:=0.002}"
: "${G1D_TELEOP_MAX_POS_STEP:=0.03}"
: "${G1D_TELEOP_MAX_ROT_STEP:=0.25}"
: "${G1D_TELEOP_RIGHT_POSITION_MAP:=x,y,z}"
: "${G1D_TELEOP_RIGHT_ROTATION_MAP:=x,y,z}"
: "${G1D_TELEOP_LEFT_POSITION_MAP:=x,y,z}"
: "${G1D_TELEOP_LEFT_ROTATION_MAP:=x,y,z}"
: "${G1D_TELEOP_RIGHT_EEF_ROT_WXYZ:=1,0,0,0}"
: "${G1D_TELEOP_LEFT_EEF_ROT_WXYZ:=1,0,0,0}"
: "${G1D_TELEOP_DEBUG_EEF:=0}"
: "${G1D_TELEOP_DEBUG_HAND_JOINTS:=1}"
: "${G1D_TELEOP_HAND_MARKER_RADIUS:=0.005}"
: "${G1D_TELEOP_HAND_ALPHA:=0.65}"
: "${G1D_TELEOP_HAND_DEADBAND:=0.01}"
: "${G1D_TELEOP_HAND_MAX_STEP:=0.10}"
: "${G1D_TELEOP_FINGER_FLEX_OPEN_RAD:=0.15}"
: "${G1D_TELEOP_FINGER_FLEX_CLOSED_RAD:=2.40}"
: "${G1D_TELEOP_THUMB_FLEX_OPEN_RAD:=0.10}"
: "${G1D_TELEOP_THUMB_FLEX_CLOSED_RAD:=1.80}"
: "${G1D_TELEOP_THUMB_BASE_OPEN_RAD:=0.20}"
: "${G1D_TELEOP_THUMB_BASE_CLOSED_RAD:=1.20}"
: "${G1D_TELEOP_RIGHT_HAND_INVERT:=0,0,0,0,0,0}"
: "${G1D_TELEOP_LEFT_HAND_INVERT:=0,0,0,0,0,0}"
# AbsPosRelRot task parameters
: "${G1D_TELEOP_ABS_TRACK_JOINT:=wrist}"
: "${G1D_TELEOP_RIGHT_ABS_POSITION_OFFSET_W:=0,0,0}"
: "${G1D_TELEOP_LEFT_ABS_POSITION_OFFSET_W:=0,0,0}"
: "${G1D_TELEOP_ABS_MAX_POSITION_STEP:=0.03}"
: "${G1D_TELEOP_ABS_MAX_ROTATION_STEP:=0.25}"
: "${G1D_TELEOP_ABS_MIN_VALID_POSITION_NORM:=0.05}"
: "${G1D_TELEOP_DEBUG_ABS_TARGET:=1}"
# Pink bimanual absolute wrist task parameters
: "${G1D_TELEOP_XR_ANCHOR_POS:=0,0,0.35}"
: "${G1D_TELEOP_XR_ANCHOR_ROT_WXYZ:=0.7071,0,-0.7071,0}"
: "${G1D_TELEOP_PINK_LEFT_WRIST_CORRECTION_WXYZ:=0.70710678,0,0.70710678,0}"
: "${G1D_TELEOP_PINK_RIGHT_WRIST_CORRECTION_WXYZ:=0,0.70710678,0,-0.70710678}"
: "${G1D_TELEOP_PINK_LEFT_POSITION_OFFSET_W:=0,0,0}"
: "${G1D_TELEOP_PINK_RIGHT_POSITION_OFFSET_W:=0,0,0}"
: "${G1D_TELEOP_PINK_MIN_VALID_POSITION_NORM:=0.05}"
: "${G1D_TELEOP_PINK_POSITION_COST:=8.0}"
: "${G1D_TELEOP_PINK_ORIENTATION_COST:=2.0}"
: "${G1D_TELEOP_PINK_LM_DAMPING:=10.0}"
: "${G1D_TELEOP_PINK_FRAME_GAIN:=0.5}"
: "${G1D_TELEOP_PINK_NULLSPACE_COST:=0.5}"
: "${G1D_TELEOP_PINK_NULLSPACE_LM_DAMPING:=1.0}"
: "${G1D_TELEOP_PINK_NULLSPACE_GAIN:=0.3}"
: "${G1D_TELEOP_PINK_SHOW_WARNINGS:=1}"
: "${G1D_TELEOP_PINK_FAIL_ON_JOINT_LIMIT:=0}"
: "${G1D_TELEOP_DEBUG_PINK_TARGET:=1}"
# Task selection — default: IK-Rel for Cuboid Handover
: "${G1D_TELEOP_TASK:=G1D-Bimanual-CuboidHandover-IK-Rel-v0}"
# Simulation device
: "${G1D_TELEOP_DEVICE:=cuda:0}"
export G1D_TELEOP_DEVICE
export G1D_TELEOP_POS_SCALE G1D_TELEOP_ROT_SCALE
export G1D_TELEOP_ALPHA_POS G1D_TELEOP_ALPHA_ROT
export G1D_TELEOP_POSITION_THRESHOLD G1D_TELEOP_ROTATION_THRESHOLD
export G1D_TELEOP_MAX_POS_STEP G1D_TELEOP_MAX_ROT_STEP
export G1D_TELEOP_RIGHT_POSITION_MAP G1D_TELEOP_RIGHT_ROTATION_MAP
export G1D_TELEOP_LEFT_POSITION_MAP G1D_TELEOP_LEFT_ROTATION_MAP
export G1D_TELEOP_RIGHT_EEF_ROT_WXYZ G1D_TELEOP_LEFT_EEF_ROT_WXYZ
export G1D_TELEOP_DEBUG_EEF
export G1D_TELEOP_DEBUG_HAND_JOINTS G1D_TELEOP_HAND_MARKER_RADIUS
export G1D_TELEOP_HAND_ALPHA G1D_TELEOP_HAND_DEADBAND G1D_TELEOP_HAND_MAX_STEP
export G1D_TELEOP_FINGER_FLEX_OPEN_RAD G1D_TELEOP_FINGER_FLEX_CLOSED_RAD
export G1D_TELEOP_THUMB_FLEX_OPEN_RAD G1D_TELEOP_THUMB_FLEX_CLOSED_RAD
export G1D_TELEOP_THUMB_BASE_OPEN_RAD G1D_TELEOP_THUMB_BASE_CLOSED_RAD
export G1D_TELEOP_RIGHT_HAND_INVERT G1D_TELEOP_LEFT_HAND_INVERT
export G1D_TELEOP_ABS_TRACK_JOINT
export G1D_TELEOP_RIGHT_ABS_POSITION_OFFSET_W G1D_TELEOP_LEFT_ABS_POSITION_OFFSET_W
export G1D_TELEOP_ABS_MAX_POSITION_STEP G1D_TELEOP_ABS_MAX_ROTATION_STEP
export G1D_TELEOP_ABS_MIN_VALID_POSITION_NORM G1D_TELEOP_DEBUG_ABS_TARGET
export G1D_TELEOP_XR_ANCHOR_POS G1D_TELEOP_XR_ANCHOR_ROT_WXYZ
export G1D_TELEOP_PINK_LEFT_WRIST_CORRECTION_WXYZ G1D_TELEOP_PINK_RIGHT_WRIST_CORRECTION_WXYZ
export G1D_TELEOP_PINK_LEFT_POSITION_OFFSET_W G1D_TELEOP_PINK_RIGHT_POSITION_OFFSET_W
export G1D_TELEOP_PINK_MIN_VALID_POSITION_NORM
export G1D_TELEOP_PINK_POSITION_COST G1D_TELEOP_PINK_ORIENTATION_COST
export G1D_TELEOP_PINK_LM_DAMPING G1D_TELEOP_PINK_FRAME_GAIN
export G1D_TELEOP_PINK_NULLSPACE_COST G1D_TELEOP_PINK_NULLSPACE_LM_DAMPING G1D_TELEOP_PINK_NULLSPACE_GAIN
export G1D_TELEOP_PINK_SHOW_WARNINGS G1D_TELEOP_PINK_FAIL_ON_JOINT_LIMIT
export G1D_TELEOP_DEBUG_PINK_TARGET
export G1D_TELEOP_TASK

echo "==========================================================================="
echo "G1D Bimanual Cuboid Handover — XR Teleoperation Launcher"
echo "==========================================================================="
echo ""
echo "  Task: $G1D_TELEOP_TASK"
echo "  Device: $G1D_TELEOP_DEVICE"
echo "  Num Envs: $NUM_ENVS"
echo ""

# Setup OpenXR Runtime (CloudXR)
OPENXR_JSON=""
for path in \
    "/usr/share/openxr/1/openxr_cloudxr.json" \
    "/opt/openxr/share/openxr/1/openxr_cloudxr.json" \
    "/workspace/cloudxr-runtime_v6.2.1/share/openxr/1/openxr_cloudxr.json"; do
    if [ -f "$path" ]; then
        OPENXR_JSON="$path"
        break
    fi
done

if [ -z "$OPENXR_JSON" ]; then
    OPENXR_JSON=$(find /usr /opt /workspace /cloudxr -name "openxr_cloudxr.json" 2>/dev/null | head -1)
    if [ -z "$OPENXR_JSON" ]; then
        echo "Warning: CloudXR OpenXR runtime config not found"
        OPENXR_JSON="/usr/share/openxr/1/openxr_cloudxr.json"
    fi
fi

export XR_RUNTIME_JSON="$OPENXR_JSON"
export NV_CXR_ENABLE_PUSH_DEVICES=1
export XDG_RUNTIME_DIR=/run/user/0

echo "  XR_RUNTIME_JSON=$XR_RUNTIME_JSON"
mkdir -p /run/user/0

# Enable Pinocchio pre-import for Pink tasks only
PINOCCHIO_ARGS=()
if [[ "$G1D_TELEOP_TASK" == *Pink* ]]; then
    PINOCCHIO_ARGS=(--enable_pinocchio)
fi

echo ""
echo "Launching..."
echo ""

cd "$SCRIPT_DIR"

exec $PYTHON scripts/run_isaaclab_tool.py \
    /workspace/isaaclab/scripts/environments/teleoperation/teleop_se3_agent.py \
    --task "$G1D_TELEOP_TASK" \
    --teleop_device handtracking \
    --device "$G1D_TELEOP_DEVICE" \
    --num_envs "$NUM_ENVS" \
    "${PINOCCHIO_ARGS[@]}"
