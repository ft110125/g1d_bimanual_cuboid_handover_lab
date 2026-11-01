# Copyright (c) 2025-2026, SmartDev.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unitree G1-D with Revo2 hand ArticulationCfg for Isaac Lab (bimanual).

This module provides:
  - G1D_REVO2_CFG: Standard PD gains (for joint-position control).
  - G1D_REVO2_HIGH_PD_CFG: Stiffer PD gains (for IK tracking / teleop).
  - Joint/body name constants for both arms.

Independent copy for g1d_bimanual_cuboid_handover project.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ASSET_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "Common_Assets")
)
_URDF_PATH = os.path.join(
    _ASSET_DIR, "Robots", "Unitree", "G1-D", "urdf", "g1_d_with_revo2_hand_fixed_base.urdf"
)

# ---------------------------------------------------------------------------
# Joint name lists
# ---------------------------------------------------------------------------

# Right arm: shoulder 3 DOF + elbow 1 DOF = 4 DOF
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

# Right wrist: roll + pitch + yaw = 3 DOF
RIGHT_WRIST_JOINTS = [
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Right hand / Revo2 fingers (11 DOF)
RIGHT_HAND_JOINTS = [
    "right_thumb_metacarpal_joint",
    "right_thumb_proximal_joint",
    "right_thumb_distal_joint",
    "right_index_proximal_joint",
    "right_index_distal_joint",
    "right_middle_proximal_joint",
    "right_middle_distal_joint",
    "right_ring_proximal_joint",
    "right_ring_distal_joint",
    "right_pinky_proximal_joint",
    "right_pinky_distal_joint",
]

# Right arm IK joints: shoulder 3 + elbow 1 + wrist 3 = 7 DOF total
RIGHT_ARM_IK_JOINTS = RIGHT_ARM_JOINTS + RIGHT_WRIST_JOINTS

# Left arm: shoulder 3 DOF + elbow 1 DOF = 4 DOF
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
]

# Left wrist: roll + pitch + yaw = 3 DOF
LEFT_WRIST_JOINTS = [
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

# Left hand / Revo2 fingers (11 DOF)
LEFT_HAND_JOINTS = [
    "left_thumb_metacarpal_joint",
    "left_thumb_proximal_joint",
    "left_thumb_distal_joint",
    "left_index_proximal_joint",
    "left_index_distal_joint",
    "left_middle_proximal_joint",
    "left_middle_distal_joint",
    "left_ring_proximal_joint",
    "left_ring_distal_joint",
    "left_pinky_proximal_joint",
    "left_pinky_distal_joint",
]

# Left arm IK joints: shoulder 3 + elbow 1 + wrist 3 = 7 DOF total
LEFT_ARM_IK_JOINTS = LEFT_ARM_JOINTS + LEFT_WRIST_JOINTS

# ---------------------------------------------------------------------------
# Active hand joints: the 6 independently-controllable Revo2 channels per hand.
# The remaining 5 distal joints are URDF mimic followers and NOT commanded directly.
# ---------------------------------------------------------------------------
RIGHT_HAND_ACTIVE_JOINTS = [
    "right_thumb_metacarpal_joint",
    "right_thumb_proximal_joint",
    "right_index_proximal_joint",
    "right_middle_proximal_joint",
    "right_ring_proximal_joint",
    "right_pinky_proximal_joint",
]

LEFT_HAND_ACTIVE_JOINTS = [
    "left_thumb_metacarpal_joint",
    "left_thumb_proximal_joint",
    "left_index_proximal_joint",
    "left_middle_proximal_joint",
    "left_ring_proximal_joint",
    "left_pinky_proximal_joint",
]

# ---------------------------------------------------------------------------
# Import-time validation
# ---------------------------------------------------------------------------
def _validate_joint_list(name: str, joints: list, expected_len: int) -> None:
    if len(joints) != expected_len:
        raise ValueError(f"{name} must have exactly {expected_len} joints, got {len(joints)}: {joints}")
    if len(set(joints)) != expected_len:
        raise ValueError(f"{name} contains duplicates: {joints}")


_validate_joint_list("RIGHT_ARM_IK_JOINTS", RIGHT_ARM_IK_JOINTS, 7)
_validate_joint_list("LEFT_ARM_IK_JOINTS", LEFT_ARM_IK_JOINTS, 7)
_validate_joint_list("RIGHT_HAND_ACTIVE_JOINTS", RIGHT_HAND_ACTIVE_JOINTS, 6)
_validate_joint_list("LEFT_HAND_ACTIVE_JOINTS", LEFT_HAND_ACTIVE_JOINTS, 6)

# Public URDF path for Pink/Pinocchio (read-only, same URDF as simulation)
G1D_URDF_PATH = _URDF_PATH

# Mesh search directory for Pinocchio's package_dirs.
G1D_MESH_DIR = os.path.dirname(_URDF_PATH)

# Lifting column prismatic joints
LIFT_JOINTS = ["LZ_mt_Joint", "LZ_it_Joint"]

# Waist joints
WAIST_JOINTS = ["torso_Joint", "Yaw_Joint"]

# ---------------------------------------------------------------------------
# Body / Frame names
# ---------------------------------------------------------------------------

# Robot root body (base of the articulation, AGV mobile platform)
ROBOT_ROOT_BODY = "AGV_link"

# Right end-effector body
RIGHT_EEF_BODY = "right_wrist_yaw_link"
RIGHT_EEF_OFFSET = (0.0415, 0.04, 0.0)

# Left end-effector body
LEFT_EEF_BODY = "left_wrist_yaw_link"
LEFT_EEF_OFFSET = (0.0415, -0.04, 0.0)

# ---------------------------------------------------------------------------
# ArticulationCfg — Standard PD
# ---------------------------------------------------------------------------

G1D_REVO2_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_URDF_PATH,
        fix_base=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            # Lifting columns — fully retracted
            "LZ_mt_Joint": 0.0,
            "LZ_it_Joint": 0.0,
            # Waist — neutral
            "torso_Joint": 0.0,
            "Yaw_Joint": 0.0,
            # Left arm — neutral, ready to reach forward
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 1.5708,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            # Right arm — neutral, ready to reach forward
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": -1.5708,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            # Hands — all open
            ".*_thumb_metacarpal_joint": 0.0,
            ".*_thumb_proximal_joint": 0.0,
            ".*_thumb_distal_joint": 0.0,
            ".*_index_proximal_joint": 0.0,
            ".*_index_distal_joint": 0.0,
            ".*_middle_proximal_joint": 0.0,
            ".*_middle_distal_joint": 0.0,
            ".*_ring_proximal_joint": 0.0,
            ".*_ring_distal_joint": 0.0,
            ".*_pinky_proximal_joint": 0.0,
            ".*_pinky_distal_joint": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "lift": ImplicitActuatorCfg(
            joint_names_expr=["LZ_mt_Joint", "LZ_it_Joint"],
            effort_limit_sim=500.0,
            velocity_limit_sim=0.05,
            stiffness=10000.0,
            damping=500.0,
            armature=0.01,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["torso_Joint", "Yaw_Joint"],
            effort_limit_sim={
                "torso_Joint": 30.0,
                "Yaw_Joint": 80.0,
            },
            velocity_limit_sim=5.0,
            stiffness=5000.0,
            damping=100.0,
            armature=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            effort_limit_sim=300.0,
            velocity_limit_sim=100.0,
            stiffness=3000.0,
            damping=100.0,
            armature=0.001,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_thumb_.*_joint",
                ".*_index_.*_joint",
                ".*_middle_.*_joint",
                ".*_ring_.*_joint",
                ".*_pinky_.*_joint",
            ],
            effort_limit_sim=10.0,
            velocity_limit_sim=20.0,
            stiffness=50.0,
            damping=5.0,
            armature=0.001,
        ),
    },
)

# ---------------------------------------------------------------------------
# ArticulationCfg — High PD (for IK tracking / teleoperation)
# ---------------------------------------------------------------------------

G1D_REVO2_HIGH_PD_CFG = G1D_REVO2_CFG.copy()
G1D_REVO2_HIGH_PD_CFG.actuators["arms"] = ImplicitActuatorCfg(
    joint_names_expr=[
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_.*_joint",
    ],
    effort_limit_sim=300.0,
    velocity_limit_sim=100.0,
    stiffness=5000.0,
    damping=200.0,
    armature=0.001,
)
G1D_REVO2_HIGH_PD_CFG.actuators["hands"] = ImplicitActuatorCfg(
    joint_names_expr=[
        ".*_thumb_.*_joint",
        ".*_index_.*_joint",
        ".*_middle_.*_joint",
        ".*_ring_.*_joint",
        ".*_pinky_.*_joint",
    ],
    effort_limit_sim=10.0,
    velocity_limit_sim=20.0,
    stiffness=100.0,
    damping=10.0,
    armature=0.001,
)
