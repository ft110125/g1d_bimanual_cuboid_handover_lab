# Copyright (c) 2025-2026, SmartDev.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab Mimic wrapper for G1-D Bimanual Cuboid Handover environment.

Supports dual-arm demo collection and data generation with both EEFs.
Tracks handover phase progression via latched signals.
"""

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.sensors import FrameTransformer

from .g1d_bimanual_env_cfg import (
    CUBOID_INITIAL_POS,
    CUBOID_GOAL_POS,
    HANDOVER_POS,
    TABLE_TOP_Z,
    GRASP_LIFT_HEIGHT,
    HANDOVER_REGION_RADIUS,
    HANDOVER_MIN_HEIGHT,
    EEF_NEAR_OBJECT_THRESHOLD,
    MAX_HANDOVER_VELOCITY,
    SUCCESS_XY_THRESHOLD,
    SUCCESS_Z_THRESHOLD,
    SUCCESS_LINEAR_VEL_THRESHOLD,
    SUCCESS_ANGULAR_VEL_THRESHOLD,
    HAND_CLEARANCE_THRESHOLD,
)


class G1DBimanualCuboidHandoverMimicEnv(ManagerBasedRLMimicEnv):
    """Isaac Lab Mimic environment wrapper for G1-D Bimanual Cuboid Handover.

    Phase detection uses approximate criteria based on:
    - EEF-to-object distance (no contact sensor in initial version)
    - Object height above table
    - Object position relative to handover zone and goal
    - Object velocity

    NOTE: Contact sensors are NOT used in this initial version.
    Phase detection relies on proximity + height + velocity heuristics.
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        # Latched phase states per environment (reset on episode reset)
        self._right_grasped: torch.Tensor | None = None
        self._handover_ready: torch.Tensor | None = None
        self._left_grasped: torch.Tensor | None = None
        self._right_released: torch.Tensor | None = None
        self._placed: torch.Tensor | None = None
        self._final_success: torch.Tensor | None = None

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset latched phase states for specified environments."""
        super()._reset_idx(env_ids)
        if self._right_grasped is None:
            # First call — initialize all buffers
            n = self.num_envs
            device = self.device
            self._right_grasped = torch.zeros(n, dtype=torch.bool, device=device)
            self._handover_ready = torch.zeros(n, dtype=torch.bool, device=device)
            self._left_grasped = torch.zeros(n, dtype=torch.bool, device=device)
            self._right_released = torch.zeros(n, dtype=torch.bool, device=device)
            self._placed = torch.zeros(n, dtype=torch.bool, device=device)
            self._final_success = torch.zeros(n, dtype=torch.bool, device=device)
        else:
            env_ids_t = torch.tensor(env_ids, dtype=torch.long, device=self.device)
            self._right_grasped[env_ids_t] = False
            self._handover_ready[env_ids_t] = False
            self._left_grasped[env_ids_t] = False
            self._right_released[env_ids_t] = False
            self._placed[env_ids_t] = False
            self._final_success[env_ids_t] = False

    # ------------------------------------------------------------------
    # Phase detection helpers
    # ------------------------------------------------------------------

    def _get_object_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return object (pos_local, lin_vel, ang_vel)."""
        obj = self.scene["object"]
        pos = obj.data.root_pos_w - self.scene.env_origins
        lin_vel = obj.data.root_lin_vel_w
        ang_vel = obj.data.root_ang_vel_w
        return pos, lin_vel, ang_vel

    def _get_eef_positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (right_eef_pos_local, left_eef_pos_local)."""
        right_ee: FrameTransformer = self.scene["right_ee_frame"]
        left_ee: FrameTransformer = self.scene["left_ee_frame"]
        right_pos = right_ee.data.target_pos_w[:, 0, :] - self.scene.env_origins
        left_pos = left_ee.data.target_pos_w[:, 0, :] - self.scene.env_origins
        return right_pos, left_pos

    def _update_phase_latches(self) -> None:
        """Update all phase latches based on current state. Latches only go True→True."""
        obj_pos, obj_lin_vel, obj_ang_vel = self._get_object_state()
        right_pos, left_pos = self._get_eef_positions()

        obj_speed = torch.norm(obj_lin_vel, dim=-1)

        # Distances
        right_to_obj = torch.norm(right_pos - obj_pos, dim=-1)
        left_to_obj = torch.norm(left_pos - obj_pos, dim=-1)

        # Handover zone distance
        handover_t = torch.tensor(HANDOVER_POS, dtype=torch.float32, device=self.device)
        dist_to_handover = torch.norm(obj_pos - handover_t, dim=-1)

        # Goal distance
        goal_t = torch.tensor(CUBOID_GOAL_POS, dtype=torch.float32, device=self.device)
        xy_err_goal = torch.norm(obj_pos[:, :2] - goal_t[:2], dim=-1)
        z_err_goal = torch.abs(obj_pos[:, 2] - goal_t[2])

        # Phase 1: right_grasped
        # Conditions: right EEF near object AND object lifted above table
        right_near = right_to_obj < EEF_NEAR_OBJECT_THRESHOLD
        obj_lifted = obj_pos[:, 2] > (TABLE_TOP_Z + GRASP_LIFT_HEIGHT)
        new_right_grasped = right_near & obj_lifted
        self._right_grasped = self._right_grasped | new_right_grasped

        # Phase 2: handover_ready
        # Conditions: right_grasped AND object in handover zone AND above min height AND slow
        in_handover_zone = dist_to_handover < HANDOVER_REGION_RADIUS
        above_handover_height = obj_pos[:, 2] > HANDOVER_MIN_HEIGHT
        slow_enough = obj_speed < MAX_HANDOVER_VELOCITY
        new_handover_ready = self._right_grasped & in_handover_zone & above_handover_height & slow_enough
        self._handover_ready = self._handover_ready | new_handover_ready

        # Phase 3: left_grasped
        # Conditions: handover_ready AND left EEF near object AND right still near
        left_near = left_to_obj < EEF_NEAR_OBJECT_THRESHOLD
        new_left_grasped = self._handover_ready & left_near & right_near
        self._left_grasped = self._left_grasped | new_left_grasped

        # Phase 4: right_released
        # Conditions: left_grasped AND right EEF far from object AND object not falling
        right_far = right_to_obj > HAND_CLEARANCE_THRESHOLD
        obj_not_falling = obj_pos[:, 2] > (TABLE_TOP_Z + 0.02)
        new_right_released = self._left_grasped & right_far & left_near & obj_not_falling
        self._right_released = self._right_released | new_right_released

        # Phase 5: placed
        # Conditions: right_released AND object at goal AND on table AND stopped
        at_goal_xy = xy_err_goal < SUCCESS_XY_THRESHOLD
        at_goal_z = z_err_goal < SUCCESS_Z_THRESHOLD
        lin_stopped = obj_speed < SUCCESS_LINEAR_VEL_THRESHOLD
        ang_stopped = torch.norm(obj_ang_vel, dim=-1) < SUCCESS_ANGULAR_VEL_THRESHOLD
        new_placed = self._right_released & at_goal_xy & at_goal_z & lin_stopped & ang_stopped
        self._placed = self._placed | new_placed

        # Phase 6: final_success
        # Conditions: placed AND left hand clear
        left_far = left_to_obj > HAND_CLEARANCE_THRESHOLD
        new_final = self._placed & left_far & lin_stopped & ang_stopped
        self._final_success = self._final_success | new_final

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Return the current EEF pose as a (N, 4, 4) homogeneous matrix.

        Args:
            eef_name: "g1d_right" or "g1d_left"
            env_ids: Environment indices to query.
        """
        if env_ids is None:
            env_ids = slice(None)

        if eef_name == "g1d_right":
            eef_pos = self.obs_buf["policy"]["right_eef_pos"][env_ids]
            eef_quat = self.obs_buf["policy"]["right_eef_quat"][env_ids]
        elif eef_name == "g1d_left":
            eef_pos = self.obs_buf["policy"]["left_eef_pos"][env_ids]
            eef_quat = self.obs_buf["policy"]["left_eef_quat"][env_ids]
        else:
            raise ValueError(f"Unknown eef_name: {eef_name}. Expected 'g1d_right' or 'g1d_left'.")

        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target EEF poses + hand actions into an env.step()-compatible action.

        Action layout (24 dims, sorted attribute order):
            [0:6]   right arm EEF (arm_action)
            [6:12]  right hand 6 channels (gripper_action)
            [12:18] left arm EEF (left_arm_action)
            [18:24] left hand 6 channels (left_gripper_action)
        """
        actions = []

        for eef_name in ["g1d_right", "g1d_left"]:
            target_eef_pose = target_eef_pose_dict[eef_name]
            target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

            curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
            curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

            # Delta position
            delta_position = target_pos - curr_pos

            # Delta rotation → axis-angle
            delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
            delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
            delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

            # Hand action (6 channels)
            hand_action = gripper_action_dict[eef_name]

            # Assemble pose action and optionally add noise
            pose_action = torch.cat([delta_position, delta_rotation], dim=0)
            if action_noise_dict is not None and eef_name in action_noise_dict:
                noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
                pose_action += noise
                pose_action = torch.clamp(pose_action, -1.0, 1.0)

            actions.append(torch.cat([pose_action, hand_action], dim=0))

        return torch.cat(actions, dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inverse of target_eef_pose_to_action: (N, action_dim) → {eef_name: (N, 4, 4)}.

        Action layout (24 dims):
            [0:6]   right arm EEF
            [6:12]  right hand (ignored here)
            [12:18] left arm EEF
            [18:24] left hand (ignored here)
        """
        result = {}

        # Right arm: action[:, 0:6]
        right_delta_pos = action[:, :3]
        right_delta_rot = action[:, 3:6]
        right_curr = self.get_robot_eef_pose("g1d_right", env_ids=None)
        right_curr_pos, right_curr_rot = PoseUtils.unmake_pose(right_curr)
        right_target_pos = right_curr_pos + right_delta_pos
        right_angle = torch.linalg.norm(right_delta_rot, dim=-1, keepdim=True)
        right_axis = right_delta_rot / (right_angle + 1e-8)
        right_delta_quat = PoseUtils.quat_from_angle_axis(right_angle.squeeze(-1), right_axis)
        right_delta_rot_mat = PoseUtils.matrix_from_quat(right_delta_quat)
        right_target_rot = right_delta_rot_mat.matmul(right_curr_rot)
        result["g1d_right"] = PoseUtils.make_pose(right_target_pos, right_target_rot)

        # Left arm: action[:, 12:18]
        left_delta_pos = action[:, 12:15]
        left_delta_rot = action[:, 15:18]
        left_curr = self.get_robot_eef_pose("g1d_left", env_ids=None)
        left_curr_pos, left_curr_rot = PoseUtils.unmake_pose(left_curr)
        left_target_pos = left_curr_pos + left_delta_pos
        left_angle = torch.linalg.norm(left_delta_rot, dim=-1, keepdim=True)
        left_axis = left_delta_rot / (left_angle + 1e-8)
        left_delta_quat = PoseUtils.quat_from_angle_axis(left_angle.squeeze(-1), left_axis)
        left_delta_rot_mat = PoseUtils.matrix_from_quat(left_delta_quat)
        left_target_rot = left_delta_rot_mat.matmul(left_curr_rot)
        result["g1d_left"] = PoseUtils.make_pose(left_target_pos, left_target_rot)

        return result

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract hand actuation part from action sequence.

        Args:
            actions: shape (num_envs, num_steps, 24) or (num_steps, 24).

        Returns:
            dict with 6-channel hand actions per arm.
        """
        return {
            "g1d_right": actions[..., 6:12],
            "g1d_left": actions[..., 18:24],
        }

    def get_subtask_term_signals(self) -> dict[str, torch.Tensor]:
        """Return subtask termination signals for Mimic annotation.

        Updates latched phase states and returns current values.
        Signals:
            - right_grasped: right hand picked up cuboid
            - handover_ready: cuboid in handover zone
            - left_grasped: left hand holding cuboid
            - right_released: right hand released cuboid
            - placed: cuboid at goal position on table
            - final_success: left hand released, task complete
        """
        self._update_phase_latches()
        return {
            "right_grasped": self._right_grasped,
            "handover_ready": self._handover_ready,
            "left_grasped": self._left_grasped,
            "right_released": self._right_released,
            "placed": self._placed,
            "final_success": self._final_success,
        }
