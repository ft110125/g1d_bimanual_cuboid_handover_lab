# Copyright (c) 2025-2026, SmartDev.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G1-D Bimanual Cuboid Handover environment configuration — IK Relative mode with XR teleoperation.

Both arms use DifferentialIK in relative mode.
OpenXR hand tracking (Meta Quest 3 via CloudXR) drives both end-effectors.
Task: Pick up a cuboid with the right hand, hand it over to the left hand,
      then place it at a target location on the left side of the table.
"""

import torch

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from g1d_bimanual_cuboid_handover.tasks.manager_based.g1d_bimanual.g1d_revo2_hand_retargeter import G1DRevo2HandRetargeterCfg
from g1d_bimanual_cuboid_handover.tasks.manager_based.g1d_bimanual.g1d_se3_rel_retargeter import G1DSe3RelRetargeterCfg
from g1d_bimanual_cuboid_handover.tasks.manager_based.g1d_bimanual.g1d_se3_abspos_relrot_retargeter import G1DSe3AbsPosRelRotRetargeterCfg
from g1d_bimanual_cuboid_handover.tasks.manager_based.g1d_bimanual.g1d_abspos_relrot_ik_action import G1DAbsPosRelRotIKActionCfg
from g1d_bimanual_cuboid_handover.tasks.manager_based.g1d_bimanual.g1d_bimanual_abs_wrist_retargeter import G1DBimanualAbsWristRetargeterCfg
from g1d_bimanual_cuboid_handover.tasks.manager_based.g1d_bimanual.g1d_pink_controller_cfg import build_g1d_pink_controller_cfg, PINK_CONTROLLED_JOINT_NAMES
from g1d_bimanual_cuboid_handover.tasks.manager_based.g1d_bimanual.g1d_pink_upper_body_action import G1DPinkUpperBodyActionCfg
from g1d_bimanual_cuboid_handover.tasks.manager_based.g1d_bimanual.g1d_teleop_tuning import TeleopTuning
from isaaclab.devices.openxr.xr_cfg import XrCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import FrameTransformer
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from g1d_bimanual_cuboid_handover.assets.g1d import (
    G1D_REVO2_HIGH_PD_CFG,
    LEFT_ARM_IK_JOINTS,
    LEFT_EEF_BODY,
    LEFT_EEF_OFFSET,
    LEFT_HAND_ACTIVE_JOINTS,
    RIGHT_ARM_IK_JOINTS,
    RIGHT_EEF_BODY,
    RIGHT_EEF_OFFSET,
    RIGHT_HAND_ACTIVE_JOINTS,
    ROBOT_ROOT_BODY,
)

# ===========================================================================
# Task Constants — Cuboid Handover
# ===========================================================================

# Cuboid dimensions (meters): X=depth, Y=width, Z=height
CUBOID_SIZE: tuple[float, float, float] = (0.150, 0.117, 0.060)

# Cuboid physical properties
CUBOID_MASS: float = 0.5  # kg — similar to reference Box

# Table parameters
TABLE_SIZE: tuple[float, float, float] = (0.6, 1.2, 0.85)
TABLE_POS: tuple[float, float, float] = (0.85, 0.0, 0.425)
TABLE_TOP_Z: float = 0.85  # meters

# Cuboid center Z on table = table_top + half_height
_CUBOID_HALF_HEIGHT: float = CUBOID_SIZE[2] / 2.0  # 0.030
_CUBOID_ON_TABLE_Z: float = TABLE_TOP_Z + _CUBOID_HALF_HEIGHT  # 0.88

# Key positions (in robot frame: X=forward, Y=left, Z=up)
CUBOID_INITIAL_POS: tuple[float, float, float] = (0.75, -0.22, _CUBOID_ON_TABLE_Z)
CUBOID_GOAL_POS: tuple[float, float, float] = (0.75, 0.22, _CUBOID_ON_TABLE_Z)
HANDOVER_POS: tuple[float, float, float] = (0.70, 0.00, 1.02)

# Success thresholds
SUCCESS_XY_THRESHOLD: float = 0.05  # m
SUCCESS_Z_THRESHOLD: float = 0.025  # m
SUCCESS_LINEAR_VEL_THRESHOLD: float = 0.08  # m/s
SUCCESS_ANGULAR_VEL_THRESHOLD: float = 0.50  # rad/s
HAND_CLEARANCE_THRESHOLD: float = 0.14  # m (distance from hand to object to consider "released")

# Phase detection thresholds
GRASP_LIFT_HEIGHT: float = 0.05  # m above table to confirm grasp
HANDOVER_REGION_RADIUS: float = 0.08  # m from HANDOVER_POS
HANDOVER_MIN_HEIGHT: float = 0.95  # m — above table for handover
EEF_NEAR_OBJECT_THRESHOLD: float = 0.08  # m — EEF close enough to be "at" object
MAX_HANDOVER_VELOCITY: float = 0.30  # m/s — object moving slowly enough for handover

# Timing
EPISODE_LENGTH_S: float = 35.0  # seconds

# Minimum height before object_dropping termination
MIN_OBJECT_HEIGHT: float = 0.75  # m


# ===========================================================================
# Observation functions for eef_pos / eef_quat (both arms)
# ===========================================================================


def right_ee_frame_pos(
    env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame")
) -> torch.Tensor:
    """Right end-effector position in env-local frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]


def right_ee_frame_quat(
    env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame")
) -> torch.Tensor:
    """Right end-effector quaternion (w, x, y, z) in world frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w[:, 0, :]


def left_ee_frame_pos(
    env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame")
) -> torch.Tensor:
    """Left end-effector position in env-local frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]


def left_ee_frame_quat(
    env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame")
) -> torch.Tensor:
    """Left end-effector quaternion (w, x, y, z) in world frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w[:, 0, :]


# ===========================================================================
# Additional observation terms for handover task
# ===========================================================================


def object_orientation_obs(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Object quaternion (w, x, y, z) in env-local frame."""
    obj = env.scene[object_cfg.name]
    return obj.data.root_quat_w


def object_linear_velocity_obs(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Object linear velocity in world frame."""
    obj = env.scene[object_cfg.name]
    return obj.data.root_lin_vel_w


def object_angular_velocity_obs(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Object angular velocity in world frame."""
    obj = env.scene[object_cfg.name]
    return obj.data.root_ang_vel_w


def target_object_position_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fixed target position for cuboid placement (env-local frame)."""
    goal = torch.tensor(CUBOID_GOAL_POS, dtype=torch.float32, device=env.device)
    return goal.unsqueeze(0).expand(env.num_envs, -1)


def handover_position_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fixed handover position (env-local frame)."""
    pos = torch.tensor(HANDOVER_POS, dtype=torch.float32, device=env.device)
    return pos.unsqueeze(0).expand(env.num_envs, -1)


# ===========================================================================
# Termination: final success (stateless — checks final placement only)
# ===========================================================================


def cuboid_handover_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Stateless success: cuboid at goal, still, both hands clear.

    This is used as the terminal success condition for demo recording.
    The full handover sequence is enforced via Mimic subtask signals.
    """
    obj = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w - env.scene.env_origins
    obj_lin_vel = obj.data.root_lin_vel_w
    obj_ang_vel = obj.data.root_ang_vel_w

    # Goal position check
    goal = torch.tensor(CUBOID_GOAL_POS, dtype=torch.float32, device=env.device)
    xy_err = torch.norm(obj_pos[:, :2] - goal[:2], dim=-1)
    z_err = torch.abs(obj_pos[:, 2] - goal[2])

    # Velocity check
    lin_speed = torch.norm(obj_lin_vel, dim=-1)
    ang_speed = torch.norm(obj_ang_vel, dim=-1)

    # Hand clearance check
    right_ee: FrameTransformer = env.scene["right_ee_frame"]
    left_ee: FrameTransformer = env.scene["left_ee_frame"]
    right_pos = right_ee.data.target_pos_w[:, 0, :] - env.scene.env_origins
    left_pos = left_ee.data.target_pos_w[:, 0, :] - env.scene.env_origins

    right_dist = torch.norm(right_pos - obj_pos, dim=-1)
    left_dist = torch.norm(left_pos - obj_pos, dim=-1)

    success = (
        (xy_err < SUCCESS_XY_THRESHOLD)
        & (z_err < SUCCESS_Z_THRESHOLD)
        & (lin_speed < SUCCESS_LINEAR_VEL_THRESHOLD)
        & (ang_speed < SUCCESS_ANGULAR_VEL_THRESHOLD)
        & (right_dist > HAND_CLEARANCE_THRESHOLD)
        & (left_dist > HAND_CLEARANCE_THRESHOLD)
    )
    return success


# ===========================================================================
# Environment Config — Bimanual IK Relative with XR Teleop
# ===========================================================================


@configclass
class G1DBimanualCuboidHandoverEnvCfg(LiftEnvCfg):
    """G1-D Bimanual Cuboid Handover env config with dual-arm IK relative control and XR teleop.

    Task: Right hand grasps cuboid → lifts → moves to center →
          hands over to left → left places at goal on table.
    """

    # XR anchor — robot first-person view (fixed base robot)
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.35),
        anchor_rot=(0.7071, 0.0, -0.7071, 0.0),
    )

    def __post_init__(self):
        super().__post_init__()

        # ---- Teleop tuning (read from env vars once) ----
        _t = TeleopTuning.from_env()

        # ---- Robot: G1-D (High PD for IK tracking) ----
        self.scene.robot = G1D_REVO2_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ---- Ground plane ----
        self.scene.plane = AssetBaseCfg(
            prim_path="/World/GroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )

        # ---- Table in front of robot ----
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.CuboidCfg(
                size=TABLE_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.2)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS),
        )

        # ---- Object: Cuboid on table (right side, for right-hand grasp) ----
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=CUBOID_INITIAL_POS,
                rot=(1.0, 0.0, 0.0, 0.0),  # Identity — horizontal flat
            ),
            spawn=sim_utils.CuboidCfg(
                size=CUBOID_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=CUBOID_MASS),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.2, 0.6, 0.9),  # Blue — easy to see
                    roughness=0.4,
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=0.8,
                    restitution=0.0,
                ),
            ),
        )

        # ---- Right End-effector frame ----
        marker_cfg_r = FRAME_MARKER_CFG.copy()
        marker_cfg_r.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_r.prim_path = "/Visuals/RightEEFrame"
        self.scene.right_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + ROBOT_ROOT_BODY,
            debug_vis=_t.debug_eef,
            visualizer_cfg=marker_cfg_r,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + RIGHT_EEF_BODY,
                    name="right_end_effector",
                    offset=OffsetCfg(pos=RIGHT_EEF_OFFSET, rot=_t.right_eef_rot_wxyz),
                ),
            ],
        )

        # ---- Left End-effector frame ----
        marker_cfg_l = FRAME_MARKER_CFG.copy()
        marker_cfg_l.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_l.prim_path = "/Visuals/LeftEEFrame"
        self.scene.left_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + ROBOT_ROOT_BODY,
            debug_vis=_t.debug_eef,
            visualizer_cfg=marker_cfg_l,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + LEFT_EEF_BODY,
                    name="left_end_effector",
                    offset=OffsetCfg(pos=LEFT_EEF_OFFSET, rot=_t.left_eef_rot_wxyz),
                ),
            ],
        )

        # Keep "ee_frame" pointing to right arm for LiftEnvCfg compatibility
        marker_cfg_ee = FRAME_MARKER_CFG.copy()
        marker_cfg_ee.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_ee.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + ROBOT_ROOT_BODY,
            debug_vis=_t.debug_eef,
            visualizer_cfg=marker_cfg_ee,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + RIGHT_EEF_BODY,
                    name="end_effector",
                    offset=OffsetCfg(pos=RIGHT_EEF_OFFSET, rot=_t.right_eef_rot_wxyz),
                ),
            ],
        )

        # ---- Actions: Dual-arm IK relative + 6-channel Revo2 hands ----
        # ActionManager sorts attributes alphabetically:
        #   arm_action(6) + gripper_action(6) + left_arm_action(6) + left_gripper_action(6) = 24
        # Final layout: [0:6] right arm, [6:12] right hand, [12:18] left arm, [18:24] left hand

        # Right arm IK (6 dims → 7 joints via Jacobian)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=RIGHT_ARM_IK_JOINTS,
            body_name=RIGHT_EEF_BODY,
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=RIGHT_EEF_OFFSET,
                rot=_t.right_eef_rot_wxyz,
            ),
        )

        # Right hand: 6 active Revo2 channels, EMA smoothed, rescaled to joint limits
        self.actions.gripper_action = mdp.EMAJointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=RIGHT_HAND_ACTIVE_JOINTS,
            preserve_order=True,
            scale=1.0,
            rescale_to_limits=True,
            alpha=_t.hand_alpha,
        )

        # Left arm IK (6 dims → 7 joints via Jacobian)
        self.actions.left_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=LEFT_ARM_IK_JOINTS,
            body_name=LEFT_EEF_BODY,
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=LEFT_EEF_OFFSET,
                rot=_t.left_eef_rot_wxyz,
            ),
        )

        # Left hand: 6 active Revo2 channels, EMA smoothed, rescaled to joint limits
        self.actions.left_gripper_action = mdp.EMAJointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=LEFT_HAND_ACTIVE_JOINTS,
            preserve_order=True,
            scale=1.0,
            rescale_to_limits=True,
            alpha=_t.hand_alpha,
        )

        # ---- Commands: target pose (placement goal — left side of table) ----
        self.commands.object_pose.body_name = ROBOT_ROOT_BODY
        self.commands.object_pose.ranges.pos_x = (
            CUBOID_GOAL_POS[0] - 0.05,
            CUBOID_GOAL_POS[0] + 0.05,
        )
        self.commands.object_pose.ranges.pos_y = (
            CUBOID_GOAL_POS[1] - 0.05,
            CUBOID_GOAL_POS[1] + 0.05,
        )
        self.commands.object_pose.ranges.pos_z = (
            CUBOID_GOAL_POS[2] - 0.02,
            CUBOID_GOAL_POS[2] + 0.02,
        )

        # ---- Observations ----
        @configclass
        class HandoverObservationsCfg:
            @configclass
            class PolicyCfg(ObsGroup):
                joint_pos = ObsTerm(func=mdp.joint_pos_rel)
                joint_vel = ObsTerm(func=mdp.joint_vel_rel)
                object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
                object_orientation = ObsTerm(func=object_orientation_obs)
                object_linear_velocity = ObsTerm(func=object_linear_velocity_obs)
                object_angular_velocity = ObsTerm(func=object_angular_velocity_obs)
                target_object_position = ObsTerm(func=target_object_position_obs)
                handover_position = ObsTerm(func=handover_position_obs)
                actions = ObsTerm(func=mdp.last_action)
                # EEF pose observations for both arms
                right_eef_pos = ObsTerm(func=right_ee_frame_pos)
                right_eef_quat = ObsTerm(func=right_ee_frame_quat)
                left_eef_pos = ObsTerm(func=left_ee_frame_pos)
                left_eef_quat = ObsTerm(func=left_ee_frame_quat)

                def __post_init__(self):
                    self.enable_corruption = False
                    self.concatenate_terms = False

            policy: PolicyCfg = PolicyCfg()

        self.observations = HandoverObservationsCfg()

        # ---- Scene ----
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        # ---- Rewards ----
        # Disable all Box-Pull / Lift inherited rewards — this is a teleop+imitation project
        self.rewards.lifting_object.weight = 0.0
        self.rewards.object_goal_tracking.weight = 0.0
        self.rewards.object_goal_tracking_fine_grained.weight = 0.0
        # Note: No custom RL rewards — primary pipeline is teleop → Mimic → BC

        # ---- Terminations ----
        self.terminations.object_dropping.params["minimum_height"] = MIN_OBJECT_HEIGHT
        self.terminations.success = DoneTerm(
            func=cuboid_handover_success,
        )

        # ---- Events ----
        # Fix initial cuboid position (no randomization for initial development)
        self.events.reset_object_position.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
        }

        # ---- Sim ----
        self.decimation = 4
        self.episode_length_s = EPISODE_LENGTH_S
        self.sim.dt = 0.01

        # ---- XR Teleop Devices (Meta Quest 3 via CloudXR) ----
        # Retargeter order must match action order:
        #   right Se3Rel(6) + right Hand(6) + left Se3Rel(6) + left Hand(6) = 24
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        # Right hand → right arm SE3 relative control (6 dims)
                        G1DSe3RelRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=False,
                            use_wrist_rotation=True,
                            use_wrist_position=True,
                            delta_pos_scale_factor=_t.pos_scale,
                            delta_rot_scale_factor=_t.rot_scale,
                            alpha_pos=_t.alpha_pos,
                            alpha_rot=_t.alpha_rot,
                            enable_visualization=False,
                            sim_device=self.sim.device,
                            position_map=_t.right_position_map,
                            rotation_map=_t.right_rotation_map,
                            position_threshold=_t.position_threshold,
                            rotation_threshold=_t.rotation_threshold,
                            max_pos_step=_t.max_pos_step,
                            max_rot_step=_t.max_rot_step,
                        ),
                        # Right hand tracking → right Revo2 hand (6 channels)
                        G1DRevo2HandRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            sim_device=self.sim.device,
                            debug_hand_joints=_t.debug_hand_joints,
                            hand_marker_radius=_t.hand_marker_radius,
                            hand_alpha=_t.hand_alpha,
                            hand_deadband=_t.hand_deadband,
                            hand_max_step=_t.hand_max_step,
                            finger_flex_open_rad=_t.finger_flex_open_rad,
                            finger_flex_closed_rad=_t.finger_flex_closed_rad,
                            thumb_flex_open_rad=_t.thumb_flex_open_rad,
                            thumb_flex_closed_rad=_t.thumb_flex_closed_rad,
                            thumb_base_open_rad=_t.thumb_base_open_rad,
                            thumb_base_closed_rad=_t.thumb_base_closed_rad,
                            right_hand_invert=_t.right_hand_invert,
                            left_hand_invert=_t.left_hand_invert,
                        ),
                        # Left hand → left arm SE3 relative control (6 dims)
                        G1DSe3RelRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_LEFT,
                            zero_out_xy_rotation=False,
                            use_wrist_rotation=True,
                            use_wrist_position=True,
                            delta_pos_scale_factor=_t.pos_scale,
                            delta_rot_scale_factor=_t.rot_scale,
                            alpha_pos=_t.alpha_pos,
                            alpha_rot=_t.alpha_rot,
                            enable_visualization=False,
                            sim_device=self.sim.device,
                            position_map=_t.left_position_map,
                            rotation_map=_t.left_rotation_map,
                            position_threshold=_t.position_threshold,
                            rotation_threshold=_t.rotation_threshold,
                            max_pos_step=_t.max_pos_step,
                            max_rot_step=_t.max_rot_step,
                        ),
                        # Left hand tracking → left Revo2 hand (6 channels)
                        G1DRevo2HandRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_LEFT,
                            sim_device=self.sim.device,
                            debug_hand_joints=_t.debug_hand_joints,
                            hand_marker_radius=_t.hand_marker_radius,
                            hand_alpha=_t.hand_alpha,
                            hand_deadband=_t.hand_deadband,
                            hand_max_step=_t.hand_max_step,
                            finger_flex_open_rad=_t.finger_flex_open_rad,
                            finger_flex_closed_rad=_t.finger_flex_closed_rad,
                            thumb_flex_open_rad=_t.thumb_flex_open_rad,
                            thumb_flex_closed_rad=_t.thumb_flex_closed_rad,
                            thumb_base_open_rad=_t.thumb_base_open_rad,
                            thumb_base_closed_rad=_t.thumb_base_closed_rad,
                            right_hand_invert=_t.right_hand_invert,
                            left_hand_invert=_t.left_hand_invert,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            },
        )


# ---------------------------------------------------------------------------
# PLAY variant
# ---------------------------------------------------------------------------


@configclass
class G1DBimanualCuboidHandoverEnvCfg_PLAY(G1DBimanualCuboidHandoverEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# Mimic Observations variant (concatenate_terms=False for Mimic wrapper)
# ---------------------------------------------------------------------------


@configclass
class G1DBimanualCuboidHandoverMimicObsCfg(G1DBimanualCuboidHandoverEnvCfg):
    """G1-D Bimanual Cuboid Handover env config with Mimic-compatible observations."""

    def __post_init__(self):
        super().__post_init__()
        # observations already have concatenate_terms=False, which Mimic requires


# ---------------------------------------------------------------------------
# Pink Absolute Bimanual variant — absolute wrist poses → bimanual Pink IK
# ---------------------------------------------------------------------------


@configclass
class G1DBimanualCuboidHandoverPinkAbsEnvCfg(G1DBimanualCuboidHandoverEnvCfg):
    """G1-D Bimanual Cuboid Handover env config with Pink IK absolute wrist control.

    Replaces both arm action terms with a single Pink IK action that controls
    left + right wrists and waist jointly.  Revo2 finger actions are preserved.

    Action layout (26D):
      arm_action           [0:14]   left wrist 7 + right wrist 7 (absolute world pose)
      gripper_action       [14:20]  right Revo2 6ch
      left_gripper_action  [20:26]  left Revo2 6ch
    """

    def __post_init__(self):
        super().__post_init__()

        self.decimation = 4
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = 2

        _t = TeleopTuning.from_env()

        # ---- XR anchor from env vars (static, fixed-base) ----
        self.xr = XrCfg(
            anchor_pos=_t.xr_anchor_pos,
            anchor_rot=_t.xr_anchor_rot_wxyz,
        )

        # ---- Actions: Pink bimanual wrists (14D) + Revo2 hands (6+6) ----
        self.actions.left_arm_action = None

        # Pink IK action (14D = left wrist 7 + right wrist 7)
        self.actions.arm_action = G1DPinkUpperBodyActionCfg(
            asset_name="robot",
            pink_controlled_joint_names=PINK_CONTROLLED_JOINT_NAMES,
            hand_joint_names=[],
            target_eef_link_names={
                "left_wrist": LEFT_EEF_BODY,
                "right_wrist": RIGHT_EEF_BODY,
            },
            controller=build_g1d_pink_controller_cfg(
                position_cost=_t.pink_position_cost,
                orientation_cost=_t.pink_orientation_cost,
                lm_damping=_t.pink_lm_damping,
                frame_gain=_t.pink_frame_gain,
                nullspace_cost=_t.pink_nullspace_cost,
                nullspace_lm_damping=_t.pink_nullspace_lm_damping,
                nullspace_gain=_t.pink_nullspace_gain,
                show_warnings=_t.pink_show_warnings,
                fail_on_joint_limit=_t.pink_fail_on_joint_limit,
            ),
            enable_gravity_compensation=False,
            min_valid_position_norm=_t.pink_min_valid_position_norm,
            debug_pink_target=_t.debug_pink_target,
        )

        # ---- Wrist frame transformers for debug ----
        marker_cfg_lw = FRAME_MARKER_CFG.copy()
        marker_cfg_lw.markers["frame"].scale = (0.08, 0.08, 0.08)
        marker_cfg_lw.prim_path = "/Visuals/PinkLeftWristFrame"
        self.scene.left_wrist_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + ROBOT_ROOT_BODY,
            debug_vis=_t.debug_pink_target,
            visualizer_cfg=marker_cfg_lw,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + LEFT_EEF_BODY,
                    name="left_wrist_origin",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
                ),
            ],
        )

        marker_cfg_rw = FRAME_MARKER_CFG.copy()
        marker_cfg_rw.markers["frame"].scale = (0.08, 0.08, 0.08)
        marker_cfg_rw.prim_path = "/Visuals/PinkRightWristFrame"
        self.scene.right_wrist_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + ROBOT_ROOT_BODY,
            debug_vis=_t.debug_pink_target,
            visualizer_cfg=marker_cfg_rw,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + RIGHT_EEF_BODY,
                    name="right_wrist_origin",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
                ),
            ],
        )

        # ---- XR Teleop Devices: bimanual abs wrist retargeter + Revo2 hands ----
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        G1DBimanualAbsWristRetargeterCfg(
                            sim_device=self.sim.device,
                            left_wrist_correction_wxyz=_t.pink_left_wrist_correction_wxyz,
                            right_wrist_correction_wxyz=_t.pink_right_wrist_correction_wxyz,
                            left_position_offset_w=_t.pink_left_position_offset_w,
                            right_position_offset_w=_t.pink_right_position_offset_w,
                            min_valid_position_norm=_t.pink_min_valid_position_norm,
                        ),
                        G1DRevo2HandRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            sim_device=self.sim.device,
                            debug_hand_joints=_t.debug_hand_joints,
                            hand_marker_radius=_t.hand_marker_radius,
                            hand_alpha=_t.hand_alpha,
                            hand_deadband=_t.hand_deadband,
                            hand_max_step=_t.hand_max_step,
                            finger_flex_open_rad=_t.finger_flex_open_rad,
                            finger_flex_closed_rad=_t.finger_flex_closed_rad,
                            thumb_flex_open_rad=_t.thumb_flex_open_rad,
                            thumb_flex_closed_rad=_t.thumb_flex_closed_rad,
                            thumb_base_open_rad=_t.thumb_base_open_rad,
                            thumb_base_closed_rad=_t.thumb_base_closed_rad,
                            right_hand_invert=_t.right_hand_invert,
                            left_hand_invert=_t.left_hand_invert,
                        ),
                        G1DRevo2HandRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_LEFT,
                            sim_device=self.sim.device,
                            debug_hand_joints=_t.debug_hand_joints,
                            hand_marker_radius=_t.hand_marker_radius,
                            hand_alpha=_t.hand_alpha,
                            hand_deadband=_t.hand_deadband,
                            hand_max_step=_t.hand_max_step,
                            finger_flex_open_rad=_t.finger_flex_open_rad,
                            finger_flex_closed_rad=_t.finger_flex_closed_rad,
                            thumb_flex_open_rad=_t.thumb_flex_open_rad,
                            thumb_flex_closed_rad=_t.thumb_flex_closed_rad,
                            thumb_base_open_rad=_t.thumb_base_open_rad,
                            thumb_base_closed_rad=_t.thumb_base_closed_rad,
                            right_hand_invert=_t.right_hand_invert,
                            left_hand_invert=_t.left_hand_invert,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            },
        )


@configclass
class G1DBimanualCuboidHandoverPinkAbsEnvCfg_PLAY(G1DBimanualCuboidHandoverPinkAbsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
