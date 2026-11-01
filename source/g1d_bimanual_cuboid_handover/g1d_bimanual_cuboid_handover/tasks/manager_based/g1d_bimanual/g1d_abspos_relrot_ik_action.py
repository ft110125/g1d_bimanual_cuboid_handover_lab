# Copyright (c) 2025-2026, SmartDev.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G1D AbsPosRelRot IK ActionTerm.

Takes 6-dim input: [abs_target_pos_world_xyz(3), rel_rot_axis_angle(3)]
Converts to 7-dim absolute pose in robot root frame and sends to Differential IK controller.

Internal controller uses:
  command_type="pose", use_relative_mode=False, ik_method="dls"
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

logger = logging.getLogger(__name__)


class G1DAbsPosRelRotIKAction(DifferentialInverseKinematicsAction):
    """ActionTerm that accepts [abs_pos_world(3), rel_rot(3)] and drives Differential IK in absolute mode.

    Coordinate transform:
      1. abs_target_pos_world -> robot root frame via inverse(T_world_root)
      2. rel_rot axis-angle delta -> applied to current EEF quat in root frame
      3. Combined 7-dim (pos_b, quat_b) -> absolute DifferentialIK controller

    Safety:
      - Rate-limited position tracking (max_position_step per env step)
      - Rate-limited rotation delta (max_rotation_step)
      - Invalid input detection (NaN, Inf, near-origin)
      - Hold-last-target on tracking loss
      - Per-env reset
    """

    cfg: "G1DAbsPosRelRotIKActionCfg"

    def __init__(self, cfg: "G1DAbsPosRelRotIKActionCfg", env: "ManagerBasedEnv"):
        # The parent __init__ creates the DifferentialIKController using cfg.controller.
        # Our cfg.controller is absolute mode (command_type="pose", use_relative_mode=False)
        # which has action_dim=7. But our external action_dim is 6.
        # We need to handle this carefully.

        # Store our parameters before parent init
        self._max_position_step = cfg.max_position_step
        self._max_rotation_step = cfg.max_rotation_step
        self._min_valid_position_norm = cfg.min_valid_position_norm
        self._debug_target = cfg.debug_target
        self._debug_prim_path = cfg.debug_prim_path

        # Call parent init (creates IK controller, resolves joints, etc.)
        super().__init__(cfg, env)

        # Override the raw/processed actions to be 6-dim (our external interface)
        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 6, device=self.device)

        # State buffers for rate limiting and hold
        self._has_valid_target = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._commanded_pos_w = torch.zeros(self.num_envs, 3, device=self.device)  # rate-limited world pos
        self._last_valid_target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Debug visualization (lazy init)
        self._debug_markers = None

    @property
    def action_dim(self) -> int:
        """External action dimension is 6: [abs_pos_w(3), rel_rot(3)]."""
        return 6

    def process_actions(self, actions: torch.Tensor):
        """Process 6-dim [abs_pos_world, rel_rot] into absolute pose command for IK controller.

        Steps:
          1. Validate input shape and values
          2. Determine valid targets vs. hold
          3. Rate-limit position
          4. Rate-limit rotation
          5. Transform world position to root frame
          6. Apply rotation delta to current EEF quat
          7. Send 7-dim absolute pose to DifferentialIK controller
        """
        # Store raw actions
        self._raw_actions[:] = actions

        input_pos_w = actions[:, 0:3]  # absolute world position
        input_rot = actions[:, 3:6]    # relative rotation axis-angle

        # --- Current EEF pose in root frame ---
        ee_pos_b, ee_quat_b = self._compute_frame_pose()

        # --- Current robot root pose in world ---
        root_pos_w = self._asset.data.root_pos_w
        root_quat_w = self._asset.data.root_quat_w

        # --- Compute current EEF world position (for rate limit start) ---
        ee_pos_w = math_utils.quat_apply(root_quat_w, ee_pos_b) + root_pos_w

        # --- Validate input ---
        pos_norm = torch.linalg.norm(input_pos_w, dim=-1)
        pos_finite = torch.all(torch.isfinite(input_pos_w), dim=-1)
        rot_finite = torch.all(torch.isfinite(input_rot), dim=-1)
        pos_valid = pos_finite & (pos_norm > self._min_valid_position_norm)
        all_valid = pos_valid & rot_finite

        # --- Update valid target tracking ---
        newly_valid = all_valid & ~self._has_valid_target  # first valid sample
        continuing_valid = all_valid & self._has_valid_target
        lost_tracking = ~all_valid & self._has_valid_target
        never_valid = ~all_valid & ~self._has_valid_target

        # Store last valid target for hold
        self._last_valid_target_pos_w[all_valid] = input_pos_w[all_valid]

        # --- Determine raw target position (world) ---
        raw_target_pos_w = torch.zeros_like(input_pos_w)

        # For newly valid: initialize commanded position at current EEF world pos
        if newly_valid.any():
            self._commanded_pos_w[newly_valid] = ee_pos_w[newly_valid]
            self._has_valid_target[newly_valid] = True

        # For continuing valid: use input position as raw target
        raw_target_pos_w[continuing_valid] = input_pos_w[continuing_valid]
        raw_target_pos_w[newly_valid] = input_pos_w[newly_valid]

        # For lost tracking: hold last valid position
        raw_target_pos_w[lost_tracking] = self._last_valid_target_pos_w[lost_tracking]

        # For never valid: hold current EEF position
        raw_target_pos_w[never_valid] = ee_pos_w[never_valid]

        # --- Rate-limit position (world frame) ---
        has_target = self._has_valid_target
        if has_target.any():
            delta_pos = raw_target_pos_w[has_target] - self._commanded_pos_w[has_target]
            delta_norm = torch.linalg.norm(delta_pos, dim=-1, keepdim=True).clamp(min=1e-8)
            clamped_norm = torch.clamp(delta_norm, max=self._max_position_step)
            delta_clamped = delta_pos * (clamped_norm / delta_norm)
            self._commanded_pos_w[has_target] = self._commanded_pos_w[has_target] + delta_clamped

        # For envs without valid target, keep commanded at current EEF
        if (~has_target).any():
            self._commanded_pos_w[~has_target] = ee_pos_w[~has_target]

        # --- Rate-limit rotation ---
        safe_rot = input_rot.clone()
        # Zero out rotation for newly valid, lost tracking, and never valid
        safe_rot[newly_valid] = 0.0
        safe_rot[lost_tracking] = 0.0
        safe_rot[never_valid] = 0.0

        rot_norm = torch.linalg.norm(safe_rot, dim=-1, keepdim=True).clamp(min=1e-8)
        rot_clamped_norm = torch.clamp(rot_norm, max=self._max_rotation_step)
        safe_rot = safe_rot * (rot_clamped_norm / rot_norm)

        # --- Transform commanded position: world -> robot root frame ---
        # p_target_b = R_world_root^T * (p_target_w - p_root_w)
        target_pos_b, _ = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, self._commanded_pos_w
        )

        # --- Apply rotation delta to current EEF orientation ---
        # Build delta_pose for apply_delta_pose: [0,0,0, rx,ry,rz]
        zero_pos = torch.zeros(self.num_envs, 3, device=self.device)
        delta_pose_6 = torch.cat([zero_pos, safe_rot], dim=-1)
        _, target_quat_b = math_utils.apply_delta_pose(ee_pos_b, ee_quat_b, delta_pose_6)

        # --- Normalize quaternion ---
        quat_norm = torch.linalg.norm(target_quat_b, dim=-1, keepdim=True).clamp(min=1e-8)
        target_quat_b = target_quat_b / quat_norm

        # --- Final safety: reject non-finite ---
        target_finite = (
            torch.all(torch.isfinite(target_pos_b), dim=-1)
            & torch.all(torch.isfinite(target_quat_b), dim=-1)
        )
        # For non-finite targets, use current EEF pose (hold)
        target_pos_b[~target_finite] = ee_pos_b[~target_finite]
        target_quat_b[~target_finite] = ee_quat_b[~target_finite]

        # --- Send 7-dim absolute pose to DifferentialIK controller ---
        absolute_pose_b = torch.cat([target_pos_b, target_quat_b], dim=-1)  # (N, 7)
        self._ik_controller.set_command(absolute_pose_b)

        # --- Store processed actions for logging ---
        self._processed_actions[:] = actions

        # --- Debug visualization ---
        if self._debug_target:
            self._update_debug_markers(self._commanded_pos_w)

    def apply_actions(self):
        """Compute joint targets using parent's absolute IK pipeline."""
        # Parent's apply_actions computes joint positions from the IK controller's stored target
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset state for specific environments."""
        self._raw_actions[env_ids] = 0.0
        self._has_valid_target[env_ids] = False
        self._commanded_pos_w[env_ids] = 0.0
        self._last_valid_target_pos_w[env_ids] = 0.0

    def _update_debug_markers(self, positions_w: torch.Tensor):
        """Show green sphere markers at rate-limited commanded target positions."""
        if self._debug_markers is None:
            try:
                import isaaclab.sim as sim_utils
                from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

                prim_path = self._debug_prim_path or "/Visuals/G1D/AbsTarget"
                marker_cfg = VisualizationMarkersCfg(
                    prim_path=prim_path,
                    markers={
                        "target": sim_utils.SphereCfg(
                            radius=0.01,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.0, 1.0, 0.0),  # green
                            ),
                        ),
                    },
                )
                self._debug_markers = VisualizationMarkers(marker_cfg)
                self._debug_markers.set_visibility(True)
            except Exception:
                self._debug_target = False  # Disable on failure
                return

        translations = positions_w.detach().cpu().numpy()
        marker_indices = [0] * len(translations)
        self._debug_markers.visualize(translations=translations, marker_indices=marker_indices)


@configclass
class G1DAbsPosRelRotIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for G1D AbsPosRelRot IK action term.

    Inherits from DifferentialInverseKinematicsActionCfg.
    The internal controller MUST be absolute mode:
      DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    """

    class_type: type[ActionTerm] = G1DAbsPosRelRotIKAction

    # Rate limiting
    max_position_step: float = 0.03
    """Max position movement per env step (meters). Limits initial jump and reconnection jumps."""

    max_rotation_step: float = 0.25
    """Max rotation delta norm per env step (radians)."""

    min_valid_position_norm: float = 0.05
    """Minimum norm of position to consider valid (reject origin-hugging OpenXR initial values)."""

    # Debug
    debug_target: bool = False
    """If True, render a green sphere at the rate-limited commanded absolute target."""

    debug_prim_path: str = "/Visuals/G1D/AbsTarget"
    """USD prim path for the debug marker. Must be unique per arm (left/right)."""
