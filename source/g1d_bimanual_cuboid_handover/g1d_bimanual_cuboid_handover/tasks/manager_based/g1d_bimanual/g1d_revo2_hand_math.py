# Copyright (c) 2025-2026, SmartDev.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure-math module for computing Revo2 hand closure values from OpenXR joint positions.

Dependencies: numpy and standard library ONLY.
No Isaac Lab, torch, scipy, etc.

Output per hand: 6 closure values in [0, 1] where 0 = open, 1 = closed.
Channel order: [thumb_base, thumb_flex, index, middle, ring, pinky].
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# OpenXR joint keys used for each finger
# ---------------------------------------------------------------------------
_LONG_FINGER_KEYS = {
    "index": ["index_metacarpal", "index_proximal", "index_intermediate", "index_distal", "index_tip"],
    "middle": ["middle_metacarpal", "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip"],
    "ring": ["ring_metacarpal", "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip"],
    "pinky": ["little_metacarpal", "little_proximal", "little_intermediate", "little_distal", "little_tip"],
}

_THUMB_FLEX_KEYS = ["thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip"]

_THUMB_BASE_KEYS = [
    "index_metacarpal",
    "little_metacarpal",
    "thumb_metacarpal",
    "thumb_proximal",
]

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _is_finite_vec(v: np.ndarray) -> bool:
    """Check that v is a finite 3-vector."""
    return v.shape == (3,) and np.all(np.isfinite(v))


def _safe_normalize(v: np.ndarray) -> Optional[np.ndarray]:
    """Normalize a 3-vector; return None if zero-length."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return None
    return v / n


def _angle_between(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Angle in radians between two 3-vectors. Returns None if either is zero-length."""
    na = _safe_normalize(a)
    nb = _safe_normalize(b)
    if na is None or nb is None:
        return None
    cos_val = float(np.clip(np.dot(na, nb), -1.0, 1.0))
    return math.acos(cos_val)


# ---------------------------------------------------------------------------
# Finger flex from bone chain
# ---------------------------------------------------------------------------

def _finger_flex_rad(positions: dict[str, np.ndarray], keys: list[str]) -> Optional[float]:
    """Sum of consecutive bone-pair bend angles along a finger chain.

    Given keys [A, B, C, D, E], computes:
        angle(B-A, C-B) + angle(C-B, D-C) + angle(D-C, E-D)

    Returns total bend in radians, or None if any position is invalid.
    """
    pts: list[np.ndarray] = []
    for k in keys:
        p = positions.get(k)
        if p is None or not _is_finite_vec(p):
            return None
        pts.append(p)

    total = 0.0
    vecs = [pts[i + 1] - pts[i] for i in range(len(pts) - 1)]
    for i in range(len(vecs) - 1):
        ang = _angle_between(vecs[i], vecs[i + 1])
        if ang is None:
            return None
        total += ang
    return total


# ---------------------------------------------------------------------------
# Thumb base angle
# ---------------------------------------------------------------------------

def _thumb_base_rad(
    positions: dict[str, np.ndarray],
    is_left: bool,
) -> Optional[float]:
    """Compute thumb opposition angle in 3D.

    The palm thumbward axis points from the little-finger metacarpal
    toward the index-finger metacarpal. The thumb base direction points
    from the thumb metacarpal toward the thumb proximal joint.

    When the thumb is open and extended sideways, both directions are
    approximately aligned and the angle is small. As the thumb moves
    across or in front of the palm, the angle increases.

    ``is_left`` is retained for API compatibility, but no sign
    inversion is required because the index-minus-little axis already
    provides a side-consistent thumbward direction.
    """
    del is_left

    required_keys = (
        "index_metacarpal",
        "little_metacarpal",
        "thumb_metacarpal",
        "thumb_proximal",
    )

    pts: dict[str, np.ndarray] = {}

    for key in required_keys:
        position = positions.get(key)

        if position is None or not _is_finite_vec(position):
            return None

        pts[key] = position

    # Across-palm axis, always pointing toward the index/thumb side.
    palm_thumbward = (
        pts["index_metacarpal"]
        - pts["little_metacarpal"]
    )

    # Direction of the first movable thumb segment.
    thumb_direction = (
        pts["thumb_proximal"]
        - pts["thumb_metacarpal"]
    )

    # Use the full 3-D angle. Do not project onto the palm plane.
    return _angle_between(
        palm_thumbward,
        thumb_direction,
    )


# ---------------------------------------------------------------------------
# Normalize to [0, 1]
# ---------------------------------------------------------------------------

def _normalize_closure(value: float, open_rad: float, closed_rad: float) -> float:
    """Map a raw radian value to [0, 1] closure, clamped."""
    if abs(closed_rad - open_rad) < 1e-12:
        return 0.0
    t = (value - open_rad) / (closed_rad - open_rad)
    return float(np.clip(t, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_hand_closure(
    positions: dict[str, np.ndarray],
    is_left: bool,
    finger_flex_open_rad: float = 0.15,
    finger_flex_closed_rad: float = 2.40,
    thumb_flex_open_rad: float = 0.10,
    thumb_flex_closed_rad: float = 1.80,
    thumb_base_open_rad: float = 0.20,
    thumb_base_closed_rad: float = 1.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 6-channel hand closure from OpenXR joint positions.

    Args:
        positions: dict mapping OpenXR joint name → np.ndarray shape (3,) position.
        is_left: True for left hand, False for right hand.
        *_open_rad / *_closed_rad: reference angles for normalization.

    Returns:
        (closure, valid):
            closure: np.ndarray shape (6,) float64 in [0, 1].
                     Order: [thumb_base, thumb_flex, index, middle, ring, pinky].
            valid: np.ndarray shape (6,) bool. True where the channel was computable.
    """
    closure = np.zeros(6, dtype=np.float64)
    valid = np.zeros(6, dtype=bool)

    # Channel 0: thumb base
    tb = _thumb_base_rad(positions, is_left)
    if tb is not None:
        closure[0] = _normalize_closure(tb, thumb_base_open_rad, thumb_base_closed_rad)
        valid[0] = True

    # Channel 1: thumb flex
    tf = _finger_flex_rad(positions, _THUMB_FLEX_KEYS)
    if tf is not None:
        closure[1] = _normalize_closure(tf, thumb_flex_open_rad, thumb_flex_closed_rad)
        valid[1] = True

    # Channels 2-5: index, middle, ring, pinky
    for i, finger_name in enumerate(["index", "middle", "ring", "pinky"]):
        ff = _finger_flex_rad(positions, _LONG_FINGER_KEYS[finger_name])
        if ff is not None:
            closure[2 + i] = _normalize_closure(ff, finger_flex_open_rad, finger_flex_closed_rad)
            valid[2 + i] = True

    return closure, valid
