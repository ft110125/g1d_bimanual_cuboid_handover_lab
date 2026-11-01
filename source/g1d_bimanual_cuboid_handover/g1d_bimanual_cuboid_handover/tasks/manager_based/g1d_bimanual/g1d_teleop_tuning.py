# Copyright (c) 2025-2026, SmartDev.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Teleop tuning parameters loaded from environment variables.

This module uses only the Python standard library (no Isaac Lab, torch, numpy, scipy).
All values are read once at import-time via ``TeleopTuning.from_env()``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_BOOL_TRUE = frozenset({"1", "true", "yes", "on"})
_BOOL_FALSE = frozenset({"0", "false", "no", "off"})


def _parse_bool(name: str, raw: str) -> bool:
    low = raw.strip().lower()
    if low in _BOOL_TRUE:
        return True
    if low in _BOOL_FALSE:
        return False
    raise ValueError(f"{name}: invalid bool value '{raw}' (expected 1/0, true/false, yes/no, on/off)")


def _parse_finite_float(name: str, raw: str) -> float:
    try:
        v = float(raw.strip())
    except ValueError:
        raise ValueError(f"{name}: cannot parse '{raw}' as float") from None
    if not math.isfinite(v):
        raise ValueError(f"{name}: value must be finite, got {v}")
    return v


def _parse_positive_float(name: str, raw: str) -> float:
    v = _parse_finite_float(name, raw)
    if v <= 0.0:
        raise ValueError(f"{name}: value must be > 0, got {v}")
    return v


def _parse_nonneg_float(name: str, raw: str) -> float:
    v = _parse_finite_float(name, raw)
    if v < 0.0:
        raise ValueError(f"{name}: value must be >= 0, got {v}")
    return v


def _parse_alpha(name: str, raw: str) -> float:
    v = _parse_finite_float(name, raw)
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"{name}: alpha must be in [0, 1], got {v}")
    return v


def _parse_axis_map(name: str, raw: str) -> tuple[tuple[int, float], ...]:
    """Parse a signed-axis map like ``x,-z,y`` → ((0,+1), (2,-1), (1,+1)).

    Returns a tuple of 3 ``(axis_index, sign)`` pairs.
    """
    _AXIS = {"x": 0, "y": 1, "z": 2}
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name}: axis map must have exactly 3 elements, got {len(parts)} in '{raw}'")
    result: list[tuple[int, float]] = []
    seen: set[int] = set()
    for p in parts:
        sign = 1.0
        s = p
        if s.startswith("-"):
            sign = -1.0
            s = s[1:]
        elif s.startswith("+"):
            s = s[1:]
        if s not in _AXIS:
            raise ValueError(f"{name}: invalid axis '{p}' in '{raw}' (expected x, y, or z with optional sign)")
        idx = _AXIS[s]
        if idx in seen:
            raise ValueError(f"{name}: duplicate axis '{s}' in '{raw}'")
        seen.add(idx)
        result.append((idx, sign))
    return tuple(result)


def _parse_quaternion(name: str, raw: str) -> tuple[float, float, float, float]:
    """Parse ``w,x,y,z`` quaternion. Normalizes to unit length; rejects zero quaternion."""
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError(f"{name}: quaternion must have 4 elements (w,x,y,z), got {len(parts)} in '{raw}'")
    vals: list[float] = []
    for p in parts:
        try:
            v = float(p)
        except ValueError:
            raise ValueError(f"{name}: cannot parse '{p}' as float in quaternion '{raw}'") from None
        if not math.isfinite(v):
            raise ValueError(f"{name}: quaternion components must be finite, got '{p}' in '{raw}'")
        vals.append(v)
    norm = math.sqrt(sum(v * v for v in vals))
    if norm < 1e-12:
        raise ValueError(f"{name}: quaternion is zero or near-zero in '{raw}'")
    return (vals[0] / norm, vals[1] / norm, vals[2] / norm, vals[3] / norm)


def _parse_invert6(name: str, raw: str) -> tuple[bool, bool, bool, bool, bool, bool]:
    """Parse a comma-separated list of 6 booleans (0 or 1 only)."""
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 6:
        raise ValueError(f"{name}: invert vector must have exactly 6 elements, got {len(parts)} in '{raw}'")
    result: list[bool] = []
    for p in parts:
        if p == "0":
            result.append(False)
        elif p == "1":
            result.append(True)
        else:
            raise ValueError(f"{name}: each element must be 0 or 1, got '{p}' in '{raw}'")
    return (result[0], result[1], result[2], result[3], result[4], result[5])


def _parse_float_triple(name: str, raw: str) -> tuple[float, float, float]:
    """Parse ``a,b,c`` into a tuple of 3 finite floats."""
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name}: must have exactly 3 elements, got {len(parts)} in '{raw}'")
    vals: list[float] = []
    for p in parts:
        try:
            v = float(p)
        except ValueError:
            raise ValueError(f"{name}: cannot parse '{p}' as float in '{raw}'") from None
        if not math.isfinite(v):
            raise ValueError(f"{name}: all values must be finite, got '{p}' in '{raw}'")
        vals.append(v)
    return (vals[0], vals[1], vals[2])


def _parse_track_joint(name: str, raw: str) -> str:
    """Parse and validate a tracked joint name (wrist or palm)."""
    val = raw.strip().lower()
    if val not in ("wrist", "palm"):
        raise ValueError(f"{name}: must be 'wrist' or 'palm', got '{raw}'")
    return val


def _parse_open_closed_pair(name_open: str, name_closed: str, raw_open: str, raw_closed: str) -> tuple[float, float]:
    """Parse and validate an open/closed radian pair (closed must be > open)."""
    v_open = _parse_finite_float(name_open, raw_open)
    v_closed = _parse_finite_float(name_closed, raw_closed)
    if v_closed <= v_open:
        raise ValueError(f"{name_closed} ({v_closed}) must be > {name_open} ({v_open})")
    return v_open, v_closed


# ---------------------------------------------------------------------------
# Frozen dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TeleopTuning:
    """Immutable teleop tuning parameters."""

    pos_scale: float
    rot_scale: float
    alpha_pos: float
    alpha_rot: float
    position_threshold: float
    rotation_threshold: float
    max_pos_step: float
    max_rot_step: float
    right_position_map: tuple[tuple[int, float], ...]
    right_rotation_map: tuple[tuple[int, float], ...]
    left_position_map: tuple[tuple[int, float], ...]
    left_rotation_map: tuple[tuple[int, float], ...]
    right_eef_rot_wxyz: tuple[float, float, float, float]
    left_eef_rot_wxyz: tuple[float, float, float, float]
    debug_eef: bool
    # Hand / Revo2 finger control
    debug_hand_joints: bool
    hand_marker_radius: float
    hand_alpha: float
    hand_deadband: float
    hand_max_step: float
    finger_flex_open_rad: float
    finger_flex_closed_rad: float
    thumb_flex_open_rad: float
    thumb_flex_closed_rad: float
    thumb_base_open_rad: float
    thumb_base_closed_rad: float
    right_hand_invert: tuple[bool, bool, bool, bool, bool, bool]
    left_hand_invert: tuple[bool, bool, bool, bool, bool, bool]
    # AbsPosRelRot task parameters
    abs_track_joint: str
    right_abs_position_offset_w: tuple[float, float, float]
    left_abs_position_offset_w: tuple[float, float, float]
    abs_max_position_step: float
    abs_max_rotation_step: float
    abs_min_valid_position_norm: float
    debug_abs_target: bool
    # Pink bimanual absolute wrist task parameters
    xr_anchor_pos: tuple[float, float, float]
    xr_anchor_rot_wxyz: tuple[float, float, float, float]
    pink_left_wrist_correction_wxyz: tuple[float, float, float, float]
    pink_right_wrist_correction_wxyz: tuple[float, float, float, float]
    pink_left_position_offset_w: tuple[float, float, float]
    pink_right_position_offset_w: tuple[float, float, float]
    pink_min_valid_position_norm: float
    pink_position_cost: float
    pink_orientation_cost: float
    pink_lm_damping: float
    pink_frame_gain: float
    pink_nullspace_cost: float
    pink_nullspace_lm_damping: float
    pink_nullspace_gain: float
    pink_show_warnings: bool
    pink_fail_on_joint_limit: bool
    debug_pink_target: bool

    @classmethod
    def from_env(cls) -> TeleopTuning:
        """Read tuning parameters from environment variables (once)."""
        def _get(key: str, default: str) -> str:
            return os.environ.get(key, default)

        # Parse open/closed pairs
        ff_open, ff_closed = _parse_open_closed_pair(
            "G1D_TELEOP_FINGER_FLEX_OPEN_RAD", "G1D_TELEOP_FINGER_FLEX_CLOSED_RAD",
            _get("G1D_TELEOP_FINGER_FLEX_OPEN_RAD", "0.15"),
            _get("G1D_TELEOP_FINGER_FLEX_CLOSED_RAD", "2.40"),
        )
        tf_open, tf_closed = _parse_open_closed_pair(
            "G1D_TELEOP_THUMB_FLEX_OPEN_RAD", "G1D_TELEOP_THUMB_FLEX_CLOSED_RAD",
            _get("G1D_TELEOP_THUMB_FLEX_OPEN_RAD", "0.10"),
            _get("G1D_TELEOP_THUMB_FLEX_CLOSED_RAD", "1.80"),
        )
        tb_open, tb_closed = _parse_open_closed_pair(
            "G1D_TELEOP_THUMB_BASE_OPEN_RAD", "G1D_TELEOP_THUMB_BASE_CLOSED_RAD",
            _get("G1D_TELEOP_THUMB_BASE_OPEN_RAD", "0.20"),
            _get("G1D_TELEOP_THUMB_BASE_CLOSED_RAD", "1.20"),
        )

        return cls(
            pos_scale=_parse_positive_float("G1D_TELEOP_POS_SCALE", _get("G1D_TELEOP_POS_SCALE", "1.0")),
            rot_scale=_parse_positive_float("G1D_TELEOP_ROT_SCALE", _get("G1D_TELEOP_ROT_SCALE", "1.0")),
            alpha_pos=_parse_alpha("G1D_TELEOP_ALPHA_POS", _get("G1D_TELEOP_ALPHA_POS", "0.85")),
            alpha_rot=_parse_alpha("G1D_TELEOP_ALPHA_ROT", _get("G1D_TELEOP_ALPHA_ROT", "0.85")),
            position_threshold=_parse_nonneg_float("G1D_TELEOP_POSITION_THRESHOLD", _get("G1D_TELEOP_POSITION_THRESHOLD", "0.0001")),
            rotation_threshold=_parse_nonneg_float("G1D_TELEOP_ROTATION_THRESHOLD", _get("G1D_TELEOP_ROTATION_THRESHOLD", "0.002")),
            max_pos_step=_parse_positive_float("G1D_TELEOP_MAX_POS_STEP", _get("G1D_TELEOP_MAX_POS_STEP", "0.03")),
            max_rot_step=_parse_positive_float("G1D_TELEOP_MAX_ROT_STEP", _get("G1D_TELEOP_MAX_ROT_STEP", "0.25")),
            right_position_map=_parse_axis_map("G1D_TELEOP_RIGHT_POSITION_MAP", _get("G1D_TELEOP_RIGHT_POSITION_MAP", "x,y,z")),
            right_rotation_map=_parse_axis_map("G1D_TELEOP_RIGHT_ROTATION_MAP", _get("G1D_TELEOP_RIGHT_ROTATION_MAP", "x,y,z")),
            left_position_map=_parse_axis_map("G1D_TELEOP_LEFT_POSITION_MAP", _get("G1D_TELEOP_LEFT_POSITION_MAP", "x,y,z")),
            left_rotation_map=_parse_axis_map("G1D_TELEOP_LEFT_ROTATION_MAP", _get("G1D_TELEOP_LEFT_ROTATION_MAP", "x,y,z")),
            right_eef_rot_wxyz=_parse_quaternion("G1D_TELEOP_RIGHT_EEF_ROT_WXYZ", _get("G1D_TELEOP_RIGHT_EEF_ROT_WXYZ", "1,0,0,0")),
            left_eef_rot_wxyz=_parse_quaternion("G1D_TELEOP_LEFT_EEF_ROT_WXYZ", _get("G1D_TELEOP_LEFT_EEF_ROT_WXYZ", "1,0,0,0")),
            debug_eef=_parse_bool("G1D_TELEOP_DEBUG_EEF", _get("G1D_TELEOP_DEBUG_EEF", "0")),
            debug_hand_joints=_parse_bool("G1D_TELEOP_DEBUG_HAND_JOINTS", _get("G1D_TELEOP_DEBUG_HAND_JOINTS", "1")),
            hand_marker_radius=_parse_positive_float("G1D_TELEOP_HAND_MARKER_RADIUS", _get("G1D_TELEOP_HAND_MARKER_RADIUS", "0.005")),
            hand_alpha=_parse_alpha("G1D_TELEOP_HAND_ALPHA", _get("G1D_TELEOP_HAND_ALPHA", "0.65")),
            hand_deadband=_parse_alpha("G1D_TELEOP_HAND_DEADBAND", _get("G1D_TELEOP_HAND_DEADBAND", "0.01")),
            hand_max_step=_parse_positive_float("G1D_TELEOP_HAND_MAX_STEP", _get("G1D_TELEOP_HAND_MAX_STEP", "0.10")),
            finger_flex_open_rad=ff_open,
            finger_flex_closed_rad=ff_closed,
            thumb_flex_open_rad=tf_open,
            thumb_flex_closed_rad=tf_closed,
            thumb_base_open_rad=tb_open,
            thumb_base_closed_rad=tb_closed,
            right_hand_invert=_parse_invert6("G1D_TELEOP_RIGHT_HAND_INVERT", _get("G1D_TELEOP_RIGHT_HAND_INVERT", "0,0,0,0,0,0")),
            left_hand_invert=_parse_invert6("G1D_TELEOP_LEFT_HAND_INVERT", _get("G1D_TELEOP_LEFT_HAND_INVERT", "0,0,0,0,0,0")),
            # AbsPosRelRot task parameters
            abs_track_joint=_parse_track_joint("G1D_TELEOP_ABS_TRACK_JOINT", _get("G1D_TELEOP_ABS_TRACK_JOINT", "wrist")),
            right_abs_position_offset_w=_parse_float_triple("G1D_TELEOP_RIGHT_ABS_POSITION_OFFSET_W", _get("G1D_TELEOP_RIGHT_ABS_POSITION_OFFSET_W", "0,0,0")),
            left_abs_position_offset_w=_parse_float_triple("G1D_TELEOP_LEFT_ABS_POSITION_OFFSET_W", _get("G1D_TELEOP_LEFT_ABS_POSITION_OFFSET_W", "0,0,0")),
            abs_max_position_step=_parse_positive_float("G1D_TELEOP_ABS_MAX_POSITION_STEP", _get("G1D_TELEOP_ABS_MAX_POSITION_STEP", "0.03")),
            abs_max_rotation_step=_parse_positive_float("G1D_TELEOP_ABS_MAX_ROTATION_STEP", _get("G1D_TELEOP_ABS_MAX_ROTATION_STEP", "0.25")),
            abs_min_valid_position_norm=_parse_positive_float("G1D_TELEOP_ABS_MIN_VALID_POSITION_NORM", _get("G1D_TELEOP_ABS_MIN_VALID_POSITION_NORM", "0.05")),
            debug_abs_target=_parse_bool("G1D_TELEOP_DEBUG_ABS_TARGET", _get("G1D_TELEOP_DEBUG_ABS_TARGET", "1")),
            # Pink bimanual absolute wrist task
            xr_anchor_pos=_parse_float_triple("G1D_TELEOP_XR_ANCHOR_POS", _get("G1D_TELEOP_XR_ANCHOR_POS", "0,0,0.35")),
            xr_anchor_rot_wxyz=_parse_quaternion("G1D_TELEOP_XR_ANCHOR_ROT_WXYZ", _get("G1D_TELEOP_XR_ANCHOR_ROT_WXYZ", "0.7071,0,-0.7071,0")),
            pink_left_wrist_correction_wxyz=_parse_quaternion("G1D_TELEOP_PINK_LEFT_WRIST_CORRECTION_WXYZ", _get("G1D_TELEOP_PINK_LEFT_WRIST_CORRECTION_WXYZ", "0.70710678,0,0.70710678,0")),
            pink_right_wrist_correction_wxyz=_parse_quaternion("G1D_TELEOP_PINK_RIGHT_WRIST_CORRECTION_WXYZ", _get("G1D_TELEOP_PINK_RIGHT_WRIST_CORRECTION_WXYZ", "0,0.70710678,0,-0.70710678")),
            pink_left_position_offset_w=_parse_float_triple("G1D_TELEOP_PINK_LEFT_POSITION_OFFSET_W", _get("G1D_TELEOP_PINK_LEFT_POSITION_OFFSET_W", "0,0,0")),
            pink_right_position_offset_w=_parse_float_triple("G1D_TELEOP_PINK_RIGHT_POSITION_OFFSET_W", _get("G1D_TELEOP_PINK_RIGHT_POSITION_OFFSET_W", "0,0,0")),
            pink_min_valid_position_norm=_parse_positive_float("G1D_TELEOP_PINK_MIN_VALID_POSITION_NORM", _get("G1D_TELEOP_PINK_MIN_VALID_POSITION_NORM", "0.05")),
            pink_position_cost=_parse_nonneg_float("G1D_TELEOP_PINK_POSITION_COST", _get("G1D_TELEOP_PINK_POSITION_COST", "8.0")),
            pink_orientation_cost=_parse_nonneg_float("G1D_TELEOP_PINK_ORIENTATION_COST", _get("G1D_TELEOP_PINK_ORIENTATION_COST", "2.0")),
            pink_lm_damping=_parse_nonneg_float("G1D_TELEOP_PINK_LM_DAMPING", _get("G1D_TELEOP_PINK_LM_DAMPING", "10.0")),
            pink_frame_gain=_parse_positive_float("G1D_TELEOP_PINK_FRAME_GAIN", _get("G1D_TELEOP_PINK_FRAME_GAIN", "0.5")),
            pink_nullspace_cost=_parse_nonneg_float("G1D_TELEOP_PINK_NULLSPACE_COST", _get("G1D_TELEOP_PINK_NULLSPACE_COST", "0.5")),
            pink_nullspace_lm_damping=_parse_nonneg_float("G1D_TELEOP_PINK_NULLSPACE_LM_DAMPING", _get("G1D_TELEOP_PINK_NULLSPACE_LM_DAMPING", "1.0")),
            pink_nullspace_gain=_parse_positive_float("G1D_TELEOP_PINK_NULLSPACE_GAIN", _get("G1D_TELEOP_PINK_NULLSPACE_GAIN", "0.3")),
            pink_show_warnings=_parse_bool("G1D_TELEOP_PINK_SHOW_WARNINGS", _get("G1D_TELEOP_PINK_SHOW_WARNINGS", "1")),
            pink_fail_on_joint_limit=_parse_bool("G1D_TELEOP_PINK_FAIL_ON_JOINT_LIMIT", _get("G1D_TELEOP_PINK_FAIL_ON_JOINT_LIMIT", "0")),
            debug_pink_target=_parse_bool("G1D_TELEOP_DEBUG_PINK_TARGET", _get("G1D_TELEOP_DEBUG_PINK_TARGET", "1")),
        )
