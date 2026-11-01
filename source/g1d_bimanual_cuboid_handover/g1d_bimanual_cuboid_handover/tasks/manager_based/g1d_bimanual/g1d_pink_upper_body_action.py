# Copyright (c) 2025-2026, SmartDev.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G1-D Pink upper body IK action term (arm + waist only, no hand joints).

Inherits from PinkInverseKinematicsAction and fixes the num_hand_joints=0 edge
case (``actions[:, -0:]`` returns all actions instead of empty slice).

action_dim = 14 (left wrist pose 7 + right wrist pose 7).
Outputs joint targets for 16 Pink-controlled joints only.
Hand joints are handled by separate Revo2 action terms.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

logger = logging.getLogger(__name__)


class G1DPinkUpperBodyAction(PinkInverseKinematicsAction):
    """Pink IK action for G1-D upper body (arms 14 + waist 2 = 16 joints).

    Overrides only the hand-joint handling to avoid the ``-0`` slice bug
    when ``num_hand_joints == 0``.  All other Pink IK logic is inherited.

    Safety:
      - Invalid/NaN poses are caught; last safe joint target is held.
      - Per-hand tracking loss: retargeter already holds last-valid per hand,
        so the IK simply tracks the retargeter's output.
      - Pre-first-valid: retargeter outputs (0,0,0) position which has norm=0
        → action_dim poses are set to current robot wrist pose.
      - Debug markers (green spheres) at absolute wrist targets when enabled.
    """

    cfg: "G1DPinkUpperBodyActionCfg"

    def __init__(self, cfg: "G1DPinkUpperBodyActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        # Last safe joint positions (for IK failure fallback)
        self._last_safe_joint_pos = self._asset.data.joint_pos[:, self._controlled_joint_ids].clone()

        # Per-hand valid tracking (position norm > threshold)
        self._left_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._right_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Debug markers (lazy init)
        self._debug_markers_left = None
        self._debug_markers_right = None

        # Startup validation log (once)
        self._log_startup_info()

    # ------------------------------------------------------------------
    # Overrides to fix hand-joint handling
    # ------------------------------------------------------------------

    def _initialize_joint_info(self) -> None:
        """Resolve controlled joints without hand joints."""
        self._isaaclab_controlled_joint_ids, self._isaaclab_controlled_joint_names = self._asset.find_joints(
            self.cfg.pink_controlled_joint_names
        )
        self.cfg.controller.joint_names = self._isaaclab_controlled_joint_names
        self._isaaclab_all_joint_ids = list(range(len(self._asset.data.joint_names)))
        self.cfg.controller.all_joint_names = self._asset.data.joint_names

        # No hand joints
        self._hand_joint_ids = []
        self._hand_joint_names = []

        # Controlled = pink only (16 joints)
        self._controlled_joint_ids = list(self._isaaclab_controlled_joint_ids)
        self._controlled_joint_names = list(self._isaaclab_controlled_joint_names)

    @property
    def hand_joint_dim(self) -> int:
        return 0

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process 14D absolute wrist poses into Pink IK targets.

        Validates poses and substitutes current robot wrist pose for
        invalid/pre-first-valid inputs.
        """
        self._raw_actions[:] = actions

        # Get base link transform
        self.base_link_frame_in_world_rf = self._get_base_link_frame_transform()

        # Validate per-hand: check position norm
        left_pos = actions[:, 0:3]
        right_pos = actions[:, 7:10]
        min_norm = self.cfg.min_valid_position_norm

        left_pos_valid = (
            torch.all(torch.isfinite(left_pos), dim=-1)
            & (torch.linalg.norm(left_pos, dim=-1) > min_norm)
        )
        right_pos_valid = (
            torch.all(torch.isfinite(right_pos), dim=-1)
            & (torch.linalg.norm(right_pos, dim=-1) > min_norm)
        )

        # Check quaternion validity
        left_quat = actions[:, 3:7]
        right_quat = actions[:, 10:14]
        left_quat_valid = (
            torch.all(torch.isfinite(left_quat), dim=-1)
            & (torch.linalg.norm(left_quat, dim=-1) > 1e-6)
        )
        right_quat_valid = (
            torch.all(torch.isfinite(right_quat), dim=-1)
            & (torch.linalg.norm(right_quat, dim=-1) > 1e-6)
        )

        left_valid = left_pos_valid & left_quat_valid
        right_valid = right_pos_valid & right_quat_valid

        self._left_valid = left_valid
        self._right_valid = right_valid

        # Build safe actions: substitute current wrist pose for invalid inputs
        safe_actions = actions.clone()

        # For invalid hands, we need current wrist world pose.
        # The parent's _extract_controlled_frame_poses expects poses in env-local frame.
        # For invalid sides, use current robot wrist pose from articulation data.
        need_left_sub = ~left_valid
        need_right_sub = ~right_valid

        if need_left_sub.any() or need_right_sub.any():
            # Get current wrist body poses from articulation
            articulation_data = self._asset.data
            body_names = articulation_data.body_names

            if need_left_sub.any():
                left_idx = body_names.index(self.cfg.target_eef_link_names.get("left_wrist", "left_wrist_yaw_link"))
                left_body_state = articulation_data.body_link_state_w[:, left_idx, :7]
                # Convert world pos to env-local
                left_pos_env = left_body_state[:, :3] - self._env.scene.env_origins
                left_quat_env = left_body_state[:, 3:7]  # w,x,y,z
                safe_actions[need_left_sub, 0:3] = left_pos_env[need_left_sub]
                safe_actions[need_left_sub, 3:7] = left_quat_env[need_left_sub]

            if need_right_sub.any():
                right_idx = body_names.index(self.cfg.target_eef_link_names.get("right_wrist", "right_wrist_yaw_link"))
                right_body_state = articulation_data.body_link_state_w[:, right_idx, :7]
                right_pos_env = right_body_state[:, :3] - self._env.scene.env_origins
                right_quat_env = right_body_state[:, 3:7]
                safe_actions[need_right_sub, 7:10] = right_pos_env[need_right_sub]
                safe_actions[need_right_sub, 10:14] = right_quat_env[need_right_sub]

        # Extract and transform poses via parent's pipeline
        controlled_frame_poses = self._extract_controlled_frame_poses(safe_actions)
        transformed_poses = self._transform_poses_to_base_link_frame(controlled_frame_poses)
        self._set_task_targets(transformed_poses)

        # Debug visualization
        if self.cfg.debug_pink_target:
            self._update_debug_markers(safe_actions[:, 0:3], safe_actions[:, 7:10])

    def apply_actions(self) -> None:
        """Compute IK solutions and apply joint targets (no hand joints)."""
        ik_joint_positions = self._compute_ik_solutions()

        # Safety: check for non-finite IK results
        ik_finite = torch.all(torch.isfinite(ik_joint_positions), dim=-1)
        if not ik_finite.all():
            # Replace bad solutions with last safe
            ik_joint_positions[~ik_finite] = self._last_safe_joint_pos[~ik_finite]

        # Update last safe
        self._last_safe_joint_pos = ik_joint_positions.clone()

        self._processed_actions = ik_joint_positions

        if self.cfg.enable_gravity_compensation:
            self._apply_gravity_compensation()

        self._asset.set_joint_position_target(self._processed_actions, self._controlled_joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset state for specific environments."""
        if env_ids is None:
            env_ids = slice(None)

        self._raw_actions[env_ids] = 0.0
        self._left_valid[env_ids] = False
        self._right_valid[env_ids] = False
        self._last_safe_joint_pos[env_ids] = (
            self._asset.data.joint_pos[env_ids][:, self._controlled_joint_ids]
        )

    # ------------------------------------------------------------------
    # Startup validation log
    # ------------------------------------------------------------------

    def _log_startup_info(self) -> None:
        """Log controlled joint info once at startup."""
        from g1d_bimanual_cuboid_handover.assets.g1d import (
            LEFT_HAND_ACTIVE_JOINTS,
            RIGHT_HAND_ACTIVE_JOINTS,
        )

        pink_set = set(self._isaaclab_controlled_joint_names)
        revo2_set = set(LEFT_HAND_ACTIVE_JOINTS + RIGHT_HAND_ACTIVE_JOINTS)
        overlap = pink_set & revo2_set

        logger.info(
            "[G1DPinkUpperBodyAction] Startup validation:\n"
            f"  arm dim=14, right hand=6, left hand=6, total=26\n"
            f"  Pink controlled joints={len(self._isaaclab_controlled_joint_names)}\n"
            f"  waist=torso_Joint,Yaw_Joint\n"
            f"  Pink/hand overlap={'none' if not overlap else overlap}\n"
            f"  base link={self.cfg.controller.base_link_name}\n"
            f"  Pink joints: {self._isaaclab_controlled_joint_names}"
        )

        if overlap:
            raise RuntimeError(
                f"Pink controlled joints overlap with Revo2 hand joints: {overlap}. "
                "This will cause conflicting joint targets."
            )

    # ------------------------------------------------------------------
    # Debug markers
    # ------------------------------------------------------------------

    def _update_debug_markers(self, left_pos: torch.Tensor, right_pos: torch.Tensor):
        """Show green markers at absolute wrist target positions."""
        if self._debug_markers_left is None:
            try:
                import isaaclab.sim as sim_utils
                from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

                for attr, path in [
                    ("_debug_markers_left", self.cfg.debug_prim_path_left),
                    ("_debug_markers_right", self.cfg.debug_prim_path_right),
                ]:
                    mcfg = VisualizationMarkersCfg(
                        prim_path=path,
                        markers={
                            "target": sim_utils.SphereCfg(
                                radius=0.01,
                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                            ),
                        },
                    )
                    setattr(self, attr, VisualizationMarkers(mcfg))
                    getattr(self, attr).set_visibility(True)
            except Exception:
                self.cfg.debug_pink_target = False
                return

        left_np = left_pos.detach().cpu().numpy()
        right_np = right_pos.detach().cpu().numpy()
        self._debug_markers_left.visualize(translations=left_np, marker_indices=[0] * len(left_np))
        self._debug_markers_right.visualize(translations=right_np, marker_indices=[0] * len(right_np))


@configclass
class G1DPinkUpperBodyActionCfg(PinkInverseKinematicsActionCfg):
    """Configuration for G1-D Pink upper body IK action (no hand joints).

    Inherits PinkInverseKinematicsActionCfg; sets hand_joint_names to empty
    and disables gravity compensation (G1-D has gravity disabled in sim).
    """

    class_type: type[ActionTerm] = G1DPinkUpperBodyAction

    hand_joint_names: list[str] = []
    """No hand joints — Revo2 fingers are handled by separate action terms."""

    enable_gravity_compensation: bool = False
    """G1-D simulation has disable_gravity=True; no compensation needed."""

    # Validation threshold
    min_valid_position_norm: float = 0.05
    """Minimum position norm to accept wrist pose as valid (reject origin-hugging)."""

    # Debug visualization
    debug_pink_target: bool = False
    """If True, render green sphere markers at absolute wrist targets."""

    debug_prim_path_left: str = "/Visuals/G1D/PinkTargetLeft"
    """USD prim path for left wrist debug marker."""

    debug_prim_path_right: str = "/Visuals/G1D/PinkTargetRight"
    """USD prim path for right wrist debug marker."""
