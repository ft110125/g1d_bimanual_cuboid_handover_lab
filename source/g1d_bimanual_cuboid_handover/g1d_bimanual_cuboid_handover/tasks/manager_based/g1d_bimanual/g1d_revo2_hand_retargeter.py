# Copyright (c) 2025-2026, SmartDev.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Revo2 hand retargeter: converts OpenXR hand joint data to 6-channel Revo2 commands.

Output: torch.Tensor shape (6,) in [-1, 1] on sim_device.
Order: [thumb_base, thumb_flex, index, middle, ring, pinky].
-1 = open, +1 = closed.

Also renders red sphere markers for the 26 OpenXR hand joints when enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg

from .g1d_revo2_hand_math import compute_hand_closure

# HAND_JOINT_NAMES defines the canonical 26-joint order for OpenXR
from isaaclab.devices.openxr.common import HAND_JOINT_NAMES

_NUM_HAND_JOINTS = len(HAND_JOINT_NAMES)  # 26


class G1DRevo2HandRetargeter(RetargeterBase):
    """Retargeter that converts OpenXR hand tracking to 6 Revo2 active channels.

    Features:
    - Per-channel deadband and max-step rate limiting
    - EMA smoothing via alpha
    - Per-channel inversion
    - Last-valid-value hold on tracking loss (per channel)
    - Red sphere marker visualization of raw OpenXR joint positions
    """

    def __init__(self, cfg: G1DRevo2HandRetargeterCfg):
        super().__init__(cfg)
        self._cfg = cfg
        if cfg.bound_hand not in (DeviceBase.TrackingTarget.HAND_LEFT, DeviceBase.TrackingTarget.HAND_RIGHT):
            raise ValueError(f"bound_hand must be HAND_LEFT or HAND_RIGHT, got {cfg.bound_hand}")
        self.bound_hand = cfg.bound_hand
        self._is_left = cfg.bound_hand == DeviceBase.TrackingTarget.HAND_LEFT

        # State: last output, initialized to safe open
        self._last_output = torch.full((6,), -1.0, dtype=torch.float32, device=cfg.sim_device)
        self._initialized = False

        # Marker visualization (lazy init, created on first valid data if enabled)
        self._markers = None
        self._markers_visible = False

    def retarget(self, data: dict) -> torch.Tensor:
        """Convert hand joint data to 6-channel Revo2 command."""
        hand_data = data.get(self.bound_hand)

        if hand_data is None:
            self._hide_markers()
            return self._last_output.clone()

        # Extract positions for math computation
        positions: dict[str, np.ndarray] = {}
        marker_positions: list[np.ndarray] = []
        valid_marker_count = 0

        for jname in HAND_JOINT_NAMES:
            pose = hand_data.get(jname)
            if pose is not None and isinstance(pose, np.ndarray) and pose.shape == (7,):
                pos = pose[:3]
                if np.all(np.isfinite(pos)):
                    positions[jname] = pos
                    marker_positions.append(pos)
                    valid_marker_count += 1
                else:
                    marker_positions.append(None)
            else:
                marker_positions.append(None)

        # Update markers
        if self._cfg.debug_hand_joints:
            self._update_markers(marker_positions, valid_marker_count)

        if not positions:
            self._hide_markers()
            return self._last_output.clone()

        # Compute closure [0, 1] for each channel
        cfg = self._cfg
        closure, valid = compute_hand_closure(
            positions,
            is_left=self._is_left,
            finger_flex_open_rad=cfg.finger_flex_open_rad,
            finger_flex_closed_rad=cfg.finger_flex_closed_rad,
            thumb_flex_open_rad=cfg.thumb_flex_open_rad,
            thumb_flex_closed_rad=cfg.thumb_flex_closed_rad,
            thumb_base_open_rad=cfg.thumb_base_open_rad,
            thumb_base_closed_rad=cfg.thumb_base_closed_rad,
        )

        # Convert [0,1] closure → [-1, +1] (open=-1, closed=+1)
        closure_clamped = np.clip(closure, 0.0, 1.0)
        raw_output = closure_clamped * 2.0 - 1.0  # [0,1] → [-1,1]

        # Apply per-channel inversion
        invert = cfg.left_hand_invert if self._is_left else cfg.right_hand_invert
        for i in range(6):
            if invert[i]:
                raw_output[i] = -raw_output[i]

        # Build output with per-channel validity, deadband, and rate limiting
        output = self._last_output.cpu().numpy().copy()
        for i in range(6):
            if not valid[i]:
                continue  # Hold last value for invalid channels
            target = raw_output[i]
            prev = output[i]
            delta = target - prev

            # Deadband
            if abs(delta) < cfg.hand_deadband:
                continue

            # Rate limit (max step)
            if abs(delta) > cfg.hand_max_step:
                delta = cfg.hand_max_step if delta > 0 else -cfg.hand_max_step

            output[i] = np.clip(prev + delta, -1.0, 1.0)

        # Ensure finite
        if not np.all(np.isfinite(output)):
            return self._last_output.clone()

        result = torch.tensor(output, dtype=torch.float32, device=self._cfg.sim_device)
        self._last_output = result.clone()
        self._initialized = True
        return result

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.HAND_TRACKING]

    # ------------------------------------------------------------------
    # Marker visualization
    # ------------------------------------------------------------------

    def _ensure_markers(self):
        """Lazily create VisualizationMarkers (requires USD stage to be ready)."""
        if self._markers is not None:
            return
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

            side = "Left" if self._is_left else "Right"
            marker_cfg = VisualizationMarkersCfg(
                prim_path=f"/Visuals/G1D/OpenXRHand{side}",
                markers={
                    "joint": sim_utils.SphereCfg(
                        radius=self._cfg.hand_marker_radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            )
            self._markers = VisualizationMarkers(marker_cfg)
            self._markers_visible = True
        except Exception:
            # USD stage may not be ready yet; skip silently
            pass

    def _update_markers(self, marker_positions: list, valid_count: int):
        """Update marker positions, or hide if no valid joints."""
        if valid_count == 0:
            self._hide_markers()
            return

        self._ensure_markers()
        if self._markers is None:
            return

        # Build translations array for valid joints only→ but we need all 26 for consistency
        # Use all 26 positions, replacing invalid with a hidden approach:
        # Actually, VisualizationMarkers dynamically resize. We pass only valid ones.
        valid_pos = []
        for p in marker_positions:
            if p is not None:
                valid_pos.append(p)

        if not valid_pos:
            self._hide_markers()
            return

        if not self._markers_visible:
            self._markers.set_visibility(True)
            self._markers_visible = True

        translations = np.array(valid_pos, dtype=np.float32)
        marker_indices = [0] * len(valid_pos)
        self._markers.visualize(translations=translations, marker_indices=marker_indices)

    def _hide_markers(self):
        """Hide markers on tracking loss."""
        if self._markers is not None and self._markers_visible:
            self._markers.set_visibility(False)
            self._markers_visible = False


@dataclass
class G1DRevo2HandRetargeterCfg(RetargeterCfg):
    """Configuration for G1D Revo2 hand retargeter."""

    retargeter_type: type[RetargeterBase] = G1DRevo2HandRetargeter
    bound_hand: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.HAND_RIGHT

    # Marker visualization
    debug_hand_joints: bool = True
    hand_marker_radius: float = 0.005

    # EMA / smoothing
    hand_alpha: float = 0.65

    # Deadband and rate limiting
    hand_deadband: float = 0.01
    hand_max_step: float = 0.10

    # Normalization reference angles
    finger_flex_open_rad: float = 0.15
    finger_flex_closed_rad: float = 2.40
    thumb_flex_open_rad: float = 0.10
    thumb_flex_closed_rad: float = 1.80
    thumb_base_open_rad: float = 0.20
    thumb_base_closed_rad: float = 1.20

    # Per-channel inversion (thumb_base, thumb_flex, index, middle, ring, pinky)
    right_hand_invert: tuple[bool, bool, bool, bool, bool, bool] = (False, False, False, False, False, False)
    left_hand_invert: tuple[bool, bool, bool, bool, bool, bool] = (False, False, False, False, False, False)
