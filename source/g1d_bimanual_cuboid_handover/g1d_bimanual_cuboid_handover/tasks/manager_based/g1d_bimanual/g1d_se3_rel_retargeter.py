# Copyright (c) 2025-2026, SmartDev.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Project-local Se3RelRetargeter extension with per-axis mapping and norm clamping.

Inherits from Isaac Lab's Se3RelRetargeter / Se3RelRetargeterCfg.
Output: torch.Tensor shape (6,), order [dx, dy, dz, rx, ry, rz], on sim_device.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import (
    Se3RelRetargeter,
    Se3RelRetargeterCfg,
)


class G1DSe3RelRetargeter(Se3RelRetargeter):
    """Extended Se3RelRetargeter with axis mapping, norm clamping, and safe init."""

    def __init__(self, cfg: G1DSe3RelRetargeterCfg):
        super().__init__(cfg)
        self._g1d_cfg = cfg
        # Override parent thresholds with cfg values
        self._position_threshold = cfg.position_threshold
        self._rotation_threshold = cfg.rotation_threshold
        # State for safe first-sample handling
        self._initialized = False

    def retarget(self, data: dict) -> torch.Tensor:
        """Convert hand joint poses to robot EEF command with axis mapping and clamping.

        Returns:
            torch.Tensor: shape (6,) [dx, dy, dz, rx, ry, rz] on sim_device.
        """
        hand_data = data.get(self.bound_hand)
        if hand_data is None:
            return self._zero_and_reset()

        wrist = hand_data.get("wrist")
        thumb_tip = hand_data.get("thumb_tip")
        index_tip = hand_data.get("index_tip")

        # Reject missing or malformed tracking data
        if wrist is None or thumb_tip is None or index_tip is None:
            return self._zero_and_reset()
        for arr in (wrist, thumb_tip, index_tip):
            if not isinstance(arr, np.ndarray) or arr.shape != (7,):
                return self._zero_and_reset()
            if not np.all(np.isfinite(arr)):
                return self._zero_and_reset()

        # First valid sample: store baseline poses and return zero (prevent init jump)
        if not self._initialized:
            self._previous_wrist = wrist.copy()
            self._previous_thumb_tip = thumb_tip.copy()
            self._previous_index_tip = index_tip.copy()
            self._smoothed_delta_pos = np.zeros(3)
            self._smoothed_delta_rot = np.zeros(3)
            self._initialized = True
            return torch.zeros(6, dtype=torch.float32, device=self._sim_device)

        # Delegate to parent for delta calculation, smoothing, and scaling
        result_tensor = super().retarget(data)
        result_np = result_tensor.cpu().numpy()

        # Check for non-finite output → zero + reset
        if not np.all(np.isfinite(result_np)):
            return self._zero_and_reset()

        pos_raw = result_np[:3]
        rot_raw = result_np[3:6]

        # Apply signed-axis mapping
        cfg = self._g1d_cfg
        pos_mapped = self._apply_axis_map(pos_raw, cfg.position_map)
        rot_mapped = self._apply_axis_map(rot_raw, cfg.rotation_map)

        # Direction-preserving norm clamp
        pos_mapped = self._norm_clamp(pos_mapped, cfg.max_pos_step)
        rot_mapped = self._norm_clamp(rot_mapped, cfg.max_rot_step)

        out = np.concatenate([pos_mapped, rot_mapped])
        if not np.all(np.isfinite(out)):
            return self._zero_and_reset()

        return torch.tensor(out, dtype=torch.float32, device=self._sim_device)

    def _zero_and_reset(self) -> torch.Tensor:
        """Return zero command and reset to uninitialized state."""
        self._initialized = False
        self._smoothed_delta_pos = np.zeros(3)
        self._smoothed_delta_rot = np.zeros(3)
        return torch.zeros(6, dtype=torch.float32, device=self._sim_device)

    @staticmethod
    def _apply_axis_map(
        vec: np.ndarray,
        axis_map: tuple[tuple[int, float], ...],
    ) -> np.ndarray:
        """Remap a 3-vector according to signed-axis map.

        axis_map example: ((0, 1.0), (2, -1.0), (1, 1.0)) means
        output[0] = vec[0]*1.0, output[1] = vec[2]*-1.0, output[2] = vec[1]*1.0
        """
        out = np.empty(3, dtype=vec.dtype)
        for i, (src_axis, sign) in enumerate(axis_map):
            out[i] = vec[src_axis] * sign
        return out

    @staticmethod
    def _norm_clamp(vec: np.ndarray, max_norm: float) -> np.ndarray:
        """Clamp vector magnitude while preserving direction."""
        n = np.linalg.norm(vec)
        if n > max_norm and n > 0.0:
            return vec * (max_norm / n)
        return vec


@dataclass
class G1DSe3RelRetargeterCfg(Se3RelRetargeterCfg):
    """Configuration for G1D project-local Se3RelRetargeter."""

    # Override retargeter_type to point to our subclass
    retargeter_type: type[RetargeterBase] = G1DSe3RelRetargeter

    # Signed-axis maps: tuple of (source_axis_index, sign) for each output axis
    # Identity default: output[0]=input[0], output[1]=input[1], output[2]=input[2]
    position_map: tuple[tuple[int, float], ...] = ((0, 1.0), (1, 1.0), (2, 1.0))
    rotation_map: tuple[tuple[int, float], ...] = ((0, 1.0), (1, 1.0), (2, 1.0))

    # Thresholds (override parent defaults)
    position_threshold: float = 0.0001
    rotation_threshold: float = 0.002

    # Norm clamp limits per step
    max_pos_step: float = 0.03
    max_rot_step: float = 0.25
