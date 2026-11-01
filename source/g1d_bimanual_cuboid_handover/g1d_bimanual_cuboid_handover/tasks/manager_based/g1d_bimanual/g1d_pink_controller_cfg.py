# Copyright (c) 2025-2026, SmartDev.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pink IK controller configuration for G1-D upper body.

Configures:
  - 2 LocalFrameTasks: left wrist, right wrist (in that order)
  - 1 NullSpacePostureTask: shoulders + waist redundancy
  - Controlled joints: 14 arm + 2 waist = 16 total
  - Base link: AGV_link (fixed base)
  - URDF: same read-only URDF as simulation
"""

from __future__ import annotations

from isaaclab.controllers.pink_ik.local_frame_task import LocalFrameTask
from isaaclab.controllers.pink_ik.null_space_posture_task import NullSpacePostureTask
from isaaclab.controllers.pink_ik.pink_ik_cfg import PinkIKControllerCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg

from g1d_bimanual_cuboid_handover.assets.g1d import (
    G1D_MESH_DIR,
    G1D_URDF_PATH,
    LEFT_ARM_IK_JOINTS,
    LEFT_EEF_BODY,
    RIGHT_ARM_IK_JOINTS,
    RIGHT_EEF_BODY,
    ROBOT_ROOT_BODY,
    WAIST_JOINTS,
)


def build_g1d_pink_controller_cfg(
    position_cost: float = 8.0,
    orientation_cost: float = 2.0,
    lm_damping: float = 10.0,
    frame_gain: float = 0.5,
    nullspace_cost: float = 0.5,
    nullspace_lm_damping: float = 1.0,
    nullspace_gain: float = 0.3,
    show_warnings: bool = True,
    fail_on_joint_limit: bool = False,
) -> PinkIKControllerCfg:
    """Build a PinkIKControllerCfg for G1-D upper body.

    Pink controlled joints (16 total):
        Left arm 7 + Right arm 7 + torso_Joint + Yaw_Joint

    Variable input tasks (in order):
        1. Left wrist LocalFrameTask
        2. Right wrist LocalFrameTask
        3. NullSpacePostureTask (shoulders + waist)

    Returns:
        PinkIKControllerCfg ready for use with G1DPinkUpperBodyAction.
    """
    # Left wrist task (must be first — matches action tensor order)
    left_wrist_task = LocalFrameTask(
        LEFT_EEF_BODY,  # "left_wrist_yaw_link"
        base_link_frame_name=ROBOT_ROOT_BODY,  # "AGV_link"
        position_cost=position_cost,
        orientation_cost=orientation_cost,
        lm_damping=lm_damping,
        gain=frame_gain,
    )

    # Right wrist task (second)
    right_wrist_task = LocalFrameTask(
        RIGHT_EEF_BODY,  # "right_wrist_yaw_link"
        base_link_frame_name=ROBOT_ROOT_BODY,  # "AGV_link"
        position_cost=position_cost,
        orientation_cost=orientation_cost,
        lm_damping=lm_damping,
        gain=frame_gain,
    )

    # Null-space posture task — waist is used as redundancy for dual wrist IK
    nullspace_task = NullSpacePostureTask(
        cost=nullspace_cost,
        lm_damping=nullspace_lm_damping,
        gain=nullspace_gain,
        controlled_frames=[
            LEFT_EEF_BODY,
            RIGHT_EEF_BODY,
        ],
        controlled_joints=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "torso_Joint",
            "Yaw_Joint",
        ],
    )

    return PinkIKControllerCfg(
        urdf_path=G1D_URDF_PATH,
        # PinkKinematicsConfiguration uses Pinocchio RobotWrapper.BuildFromURDF,
        # which still resolves URDF visual/collision meshes. The URDF contains
        # relative references such as ../meshes/AGV_link.STL, so pass the URDF
        # directory as Pinocchio's package/search directory.
        mesh_path=G1D_MESH_DIR,
        articulation_name="robot",
        base_link_name=ROBOT_ROOT_BODY,  # "AGV_link"
        num_hand_joints=0,
        show_ik_warnings=show_warnings,
        fail_on_joint_limit_violation=fail_on_joint_limit,
        variable_input_tasks=[
            left_wrist_task,
            right_wrist_task,
            nullspace_task,
        ],
        fixed_input_tasks=[],
    )


# Pink controlled joint names (16 total): left arm 7 + right arm 7 + waist 2
PINK_CONTROLLED_JOINT_NAMES = LEFT_ARM_IK_JOINTS + RIGHT_ARM_IK_JOINTS + WAIST_JOINTS
