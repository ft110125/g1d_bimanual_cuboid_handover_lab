# Copyright (c) 2025-2026, SmartDev.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Absolute bimanual wrist retargeter for Pink IK pipeline.

Reads OpenXR HAND_LEFT and HAND_RIGHT wrist poses and outputs 14D absolute
world pose:
    [left_x, left_y, left_z, left_qw, left_qx, left_qy, left_qz,
     right_x, right_y, right_z, right_qw, right_qx, right_qy, right_qz]

No frame deltas, relative rotation, initial-relative offset, or position scale.
OpenXR device already provides poses in [x, y, z, qw, qx, qy, qz] order.
Wrist orientation correction quaternions are configurable per hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


# Official Isaac Lab Unitree G1 absolute-wrist frame corrections.
# Quaternion order is (w, x, y, z).
OFFICIAL_G1_LEFT_WRIST_CORRECTION_WXYZ = (0.70710678, 0.0, 0.70710678, 0.0)
OFFICIAL_G1_RIGHT_WRIST_CORRECTION_WXYZ = (0.0, 0.70710678, 0.0, -0.70710678)


def _quat_multiply_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (w,x,y,z convention)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


class G1DBimanualAbsWristRetargeter(RetargeterBase):
    """Retargeter that outputs absolute world poses for both wrists (14D).

    Output order: left wrist 7D + right wrist 7D.
    Each 7D = [x, y, z, qw, qx, qy, qz].
    """

    def __init__(self, cfg: G1DBimanualAbsWristRetargeterCfg):
        super().__init__(cfg)
        self._cfg = cfg

        # Normalize correction quaternions here as well as in the environment
        # parser so direct config construction remains safe.
        self._left_correction = self._normalize_correction(
            cfg.left_wrist_correction_wxyz, "left"
        )
        self._right_correction = self._normalize_correction(
            cfg.right_wrist_correction_wxyz, "right"
        )

        # Per-hand position offsets (world frame)
        self._left_offset = np.array(cfg.left_position_offset_w, dtype=np.float64)
        self._right_offset = np.array(cfg.right_position_offset_w, dtype=np.float64)

        self._min_valid_norm = cfg.min_valid_position_norm

        # Per-hand state
        self._left_valid = False
        self._right_valid = False
        self._left_last = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
        self._right_last = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
        # For quaternion sign continuity
        self._left_prev_quat = np.array([1, 0, 0, 0], dtype=np.float64)
        self._right_prev_quat = np.array([1, 0, 0, 0], dtype=np.float64)

    @staticmethod
    def _normalize_correction(
        value: tuple[float, float, float, float], side: str
    ) -> np.ndarray:
        """Return a finite, unit-length wrist correction quaternion."""
        quat = np.asarray(value, dtype=np.float64)
        if quat.shape != (4,) or not np.all(np.isfinite(quat)):
            raise ValueError(
                f"{side} wrist correction must contain four finite values in w,x,y,z order"
            )
        norm = float(np.linalg.norm(quat))
        if norm < 1e-12:
            raise ValueError(f"{side} wrist correction quaternion must be non-zero")
        return quat / norm

    def retarget(self, data: dict) -> torch.Tensor:
        """Convert hand tracking data to 14D absolute wrist pose.

        Returns:
            torch.Tensor: shape (14,) on sim_device.
        """
        left_hand = data.get(DeviceBase.TrackingTarget.HAND_LEFT)
        right_hand = data.get(DeviceBase.TrackingTarget.HAND_RIGHT)

        left_pose = self._extract_wrist(left_hand, "left")
        right_pose = self._extract_wrist(right_hand, "right")

        out = np.concatenate([left_pose, right_pose])
        return torch.tensor(out, dtype=torch.float32, device=self._cfg.sim_device)

    def _extract_wrist(self, hand_data: dict | None, side: str) -> np.ndarray:
        """Extract and validate wrist pose for one hand.

        Returns 7D np.array [x,y,z, qw,qx,qy,qz] or last-valid/zero.
        """
        is_left = side == "left"
        correction = self._left_correction if is_left else self._right_correction
        offset = self._left_offset if is_left else self._right_offset

        if hand_data is None:
            return self._hold(side)

        wrist_pose = hand_data.get("wrist")
        if wrist_pose is None or not isinstance(wrist_pose, np.ndarray) or wrist_pose.shape != (7,):
            return self._hold(side)

        if not np.all(np.isfinite(wrist_pose)):
            return self._hold(side)

        pos = wrist_pose[:3].astype(np.float64)

        # Isaac Lab OpenXR pose is already [x, y, z, qw, qx, qy, qz].
        quat_wxyz = wrist_pose[3:7].astype(np.float64)

        # Validate position norm
        if np.linalg.norm(pos) < self._min_valid_norm:
            return self._hold(side)

        # Validate quaternion
        qnorm = np.linalg.norm(quat_wxyz)
        if qnorm < 1e-12 or not np.all(np.isfinite(quat_wxyz)):
            return self._hold(side)

        # Normalize quaternion
        quat_wxyz = quat_wxyz / qnorm

        # Apply wrist orientation correction: q_corrected = q_raw * q_correction
        quat_corrected = _quat_multiply_np(quat_wxyz, correction)

        # Normalize after multiplication
        cn = np.linalg.norm(quat_corrected)
        if cn < 1e-12:
            return self._hold(side)
        quat_corrected = quat_corrected / cn

        # Quaternion sign continuity (flip if dot with previous < 0)
        prev_quat = self._left_prev_quat if is_left else self._right_prev_quat
        if np.dot(quat_corrected, prev_quat) < 0.0:
            quat_corrected = -quat_corrected

        # Apply position offset
        pos_final = pos + offset

        # Final finite check
        if not (np.all(np.isfinite(pos_final)) and np.all(np.isfinite(quat_corrected))):
            return self._hold(side)

        # Store state
        result = np.concatenate([pos_final, quat_corrected])
        if is_left:
            self._left_valid = True
            self._left_last = result.copy()
            self._left_prev_quat = quat_corrected.copy()
        else:
            self._right_valid = True
            self._right_last = result.copy()
            self._right_prev_quat = quat_corrected.copy()

        return result

    def _hold(self, side: str) -> np.ndarray:
        """Return last valid pose or zero-norm signal for 'never valid'."""
        if side == "left":
            if self._left_valid:
                return self._left_last.copy()
            return np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
        else:
            if self._right_valid:
                return self._right_last.copy()
            return np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64)

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.HAND_TRACKING]


@dataclass
class G1DBimanualAbsWristRetargeterCfg(RetargeterCfg):
    """Configuration for bimanual absolute wrist retargeter."""

    retargeter_type: type[RetargeterBase] = G1DBimanualAbsWristRetargeter

    # Wrist orientation correction quaternions (w,x,y,z).
    # Applied as: q_corrected = q_raw * q_correction.
    # Defaults use the official Unitree G1 absolute-wrist frame conversion.
    # Fine-tune only if the G1-D/Revo2 fixed mount adds a residual rotation.
    left_wrist_correction_wxyz: tuple[float, float, float, float] = (
        OFFICIAL_G1_LEFT_WRIST_CORRECTION_WXYZ
    )
    right_wrist_correction_wxyz: tuple[float, float, float, float] = (
        OFFICIAL_G1_RIGHT_WRIST_CORRECTION_WXYZ
    )

    # World-frame position offsets
    left_position_offset_w: tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_position_offset_w: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Minimum position norm to accept as valid
    min_valid_position_norm: float = 0.05
