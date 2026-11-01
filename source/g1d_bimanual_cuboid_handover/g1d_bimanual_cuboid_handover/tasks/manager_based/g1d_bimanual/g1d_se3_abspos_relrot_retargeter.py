# Copyright (c) 2025-2026, SmartDev.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""AbsPosRelRot retargeter: absolute position + relative rotation for OpenXR hand tracking.

Inherits from G1DSe3RelRetargeter to reuse rotation delta computation, axis mapping,
smoothing, and safe first-sample handling.

Output: torch.Tensor shape (6,), order:
  [abs_pos_w_x, abs_pos_w_y, abs_pos_w_z, rel_rot_x, rel_rot_y, rel_rot_z]

Position is the direct OpenXR joint virtual-world coordinate (+ offset).
Rotation is the frame-to-frame relative delta from the parent class.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase

from .g1d_se3_rel_retargeter import G1DSe3RelRetargeter, G1DSe3RelRetargeterCfg

_VALID_TRACK_JOINTS = ("wrist", "palm")


class G1DSe3AbsPosRelRotRetargeter(G1DSe3RelRetargeter):
    """Retargeter that outputs absolute position (world) + relative rotation (delta).

    Position: tracked OpenXR joint pose[:3] + position_offset_w (no scale, no axis map).
    Rotation: inherited relative delta from G1DSe3RelRetargeter (with axis map, smoothing, clamping).
    """

    def __init__(self, cfg: G1DSe3AbsPosRelRotRetargeterCfg):
        super().__init__(cfg)
        self._abspos_cfg = cfg

        # Validate tracked_position_joint
        if cfg.tracked_position_joint not in _VALID_TRACK_JOINTS:
            raise ValueError(
                f"tracked_position_joint must be one of {_VALID_TRACK_JOINTS}, "
                f"got '{cfg.tracked_position_joint}'"
            )
        # Validate position_offset_w
        if len(cfg.position_offset_w) != 3:
            raise ValueError(f"position_offset_w must have 3 elements, got {len(cfg.position_offset_w)}")
        for i, v in enumerate(cfg.position_offset_w):
            if not math.isfinite(v):
                raise ValueError(f"position_offset_w[{i}] must be finite, got {v}")
        # Validate min_valid_position_norm
        if cfg.min_valid_position_norm <= 0.0:
            raise ValueError(f"min_valid_position_norm must be > 0, got {cfg.min_valid_position_norm}")

        self._offset_w = np.array(cfg.position_offset_w, dtype=np.float64)
        self._min_valid_norm = cfg.min_valid_position_norm
        self._track_joint = cfg.tracked_position_joint

        # Absolute position state
        self._has_valid_position = False
        self._last_valid_abs_pos = np.zeros(3, dtype=np.float64)

    def retarget(self, data: dict) -> torch.Tensor:
        """Convert hand joint poses to [abs_pos_w(3), rel_rot(3)] command.

        Returns:
            torch.Tensor: shape (6,) on sim_device.
        """
        hand_data = data.get(self.bound_hand)
        if hand_data is None:
            return self._abspos_zero_and_reset()

        # --- Extract tracked joint pose for absolute position ---
        tracked_pose = hand_data.get(self._track_joint)
        pos_valid = self._validate_pose(tracked_pose)

        # --- Compute relative rotation via parent ---
        # Parent retarget() returns (6,): [delta_pos(3), delta_rot(3)]
        rel_result = super().retarget(data)
        rel_rot = rel_result[3:6]  # rotation delta from parent (with axis map, smoothing, clamping)

        # --- Absolute position logic ---
        if pos_valid:
            abs_pos_w = tracked_pose[:3].astype(np.float64) + self._offset_w

            if not self._has_valid_position:
                # First valid sample: store position, zero rotation (prevent jump)
                self._has_valid_position = True
                self._last_valid_abs_pos = abs_pos_w.copy()
                rel_rot = torch.zeros(3, dtype=torch.float32, device=self._sim_device)
            else:
                self._last_valid_abs_pos = abs_pos_w.copy()
        else:
            if self._has_valid_position:
                # Tracking lost after valid: hold last position, zero rotation
                abs_pos_w = self._last_valid_abs_pos.copy()
                rel_rot = torch.zeros(3, dtype=torch.float32, device=self._sim_device)
            else:
                # Never had valid position
                return self._abspos_return_zero()

        # Build output
        pos_tensor = torch.tensor(abs_pos_w, dtype=torch.float32, device=self._sim_device)
        out = torch.cat([pos_tensor, rel_rot], dim=0)

        # Final safety: reject non-finite
        if not torch.all(torch.isfinite(out)):
            if self._has_valid_position:
                pos_safe = torch.tensor(self._last_valid_abs_pos, dtype=torch.float32, device=self._sim_device)
                return torch.cat([pos_safe, torch.zeros(3, dtype=torch.float32, device=self._sim_device)])
            return self._abspos_return_zero()

        return out

    def _validate_pose(self, pose) -> bool:
        """Check if a pose is valid for absolute position extraction."""
        if pose is None:
            return False
        if not isinstance(pose, np.ndarray) or pose.shape != (7,):
            return False
        if not np.all(np.isfinite(pose)):
            return False
        pos = pose[:3]
        if np.linalg.norm(pos) < self._min_valid_norm:
            return False
        return True

    def _abspos_zero_and_reset(self) -> torch.Tensor:
        """Return zero and reset all state including parent."""
        self._has_valid_position = False
        self._last_valid_abs_pos = np.zeros(3, dtype=np.float64)
        # Also reset parent state
        super()._zero_and_reset()
        return torch.zeros(6, dtype=torch.float32, device=self._sim_device)

    def _abspos_return_zero(self) -> torch.Tensor:
        """Return zero command without full reset (for 'never had valid' case)."""
        return torch.zeros(6, dtype=torch.float32, device=self._sim_device)

    def reset(self):
        """Reset state for env reset / reconnection."""
        self._has_valid_position = False
        self._last_valid_abs_pos = np.zeros(3, dtype=np.float64)
        self._initialized = False
        self._smoothed_delta_pos = np.zeros(3)
        self._smoothed_delta_rot = np.zeros(3)

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.HAND_TRACKING]


@dataclass
class G1DSe3AbsPosRelRotRetargeterCfg(G1DSe3RelRetargeterCfg):
    """Configuration for AbsPosRelRot retargeter.

    Inherits all G1DSe3RelRetargeterCfg fields (rotation_map, max_rot_step, etc.).
    """

    retargeter_type: type[RetargeterBase] = G1DSe3AbsPosRelRotRetargeter

    # Which OpenXR joint to use for absolute position
    tracked_position_joint: str = "wrist"

    # World-frame position offset applied to the tracked joint
    position_offset_w: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Minimum position norm to accept as valid (reject origin-hugging initial values)
    min_valid_position_norm: float = 0.05
