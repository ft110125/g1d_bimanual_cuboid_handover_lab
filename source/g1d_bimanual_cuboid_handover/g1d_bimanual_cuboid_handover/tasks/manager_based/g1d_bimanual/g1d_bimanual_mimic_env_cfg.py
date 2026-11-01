# Copyright (c) 2025-2026, SmartDev.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab Mimic environment config for G1-D Bimanual Cuboid Handover."""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .g1d_bimanual_env_cfg import G1DBimanualCuboidHandoverMimicObsCfg


@configclass
class G1DBimanualCuboidHandoverMimicEnvCfg(G1DBimanualCuboidHandoverMimicObsCfg, MimicEnvCfg):
    """Combines the G1-D Bimanual Cuboid Handover env config and MimicEnvCfg for data augmentation."""

    def __post_init__(self):
        super().__post_init__()

        # ---- DataGen config ----
        self.datagen_config.name = "demo_src_g1d_bimanual_cuboid_handover"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        # Keep same source demo for all subtasks to preserve handover timing
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 50
        self.datagen_config.seed = 1

        # ---- Subtask configs for right arm ----
        right_subtask_configs = []

        # R0: Right hand reaches, grasps, and lifts the cuboid
        right_subtask_configs.append(
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal="right_grasped",
                subtask_term_offset_range=(5, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.02,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Right hand: reach, grasp, and lift cuboid",
                next_subtask_description="Right hand: move cuboid to handover position",
            )
        )

        # R1: Right hand moves cuboid to handover position (center)
        right_subtask_configs.append(
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal="handover_ready",
                subtask_term_offset_range=(3, 7),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.01,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Right hand: move cuboid to handover position",
                next_subtask_description="Right hand: hold cuboid for left hand",
            )
        )

        # R2: Right hand holds cuboid while left hand grasps
        # Minimal noise during handover for stability
        right_subtask_configs.append(
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal="left_grasped",
                subtask_term_offset_range=(3, 7),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=3,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Right hand: hold cuboid for left hand grasp",
                next_subtask_description="Right hand: release cuboid",
            )
        )

        # R3: Right hand releases cuboid and retreats
        right_subtask_configs.append(
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal="right_released",
                subtask_term_offset_range=(3, 7),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.01,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Right hand: release cuboid and retreat",
                next_subtask_description="Right hand: wait at safe position",
            )
        )

        # R4: Right hand waits at safe position (final subtask)
        right_subtask_configs.append(
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Right hand: wait at safe position",
            )
        )

        self.subtask_configs["g1d_right"] = right_subtask_configs

        # ---- Subtask configs for left arm ----
        left_subtask_configs = []

        # L0: Left hand approaches handover position
        left_subtask_configs.append(
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal="handover_ready",
                subtask_term_offset_range=(3, 7),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.02,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Left hand: approach handover position",
                next_subtask_description="Left hand: grasp cuboid",
            )
        )

        # L1: Left hand grasps cuboid (minimal noise for stability)
        left_subtask_configs.append(
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal="left_grasped",
                subtask_term_offset_range=(3, 7),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=3,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Left hand: grasp cuboid from right hand",
                next_subtask_description="Left hand: hold cuboid while right releases",
            )
        )

        # L2: Left hand holds cuboid while right hand releases
        left_subtask_configs.append(
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal="right_released",
                subtask_term_offset_range=(3, 7),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=3,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Left hand: hold cuboid while right hand releases",
                next_subtask_description="Left hand: move cuboid to goal",
            )
        )

        # L3: Left hand moves cuboid to goal and places it
        left_subtask_configs.append(
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal="placed",
                subtask_term_offset_range=(5, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.01,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Left hand: move cuboid to goal and place on table",
                next_subtask_description="Left hand: release cuboid",
            )
        )

        # L4: Left hand releases cuboid (final subtask)
        left_subtask_configs.append(
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Left hand: release cuboid and retreat",
            )
        )

        self.subtask_configs["g1d_left"] = left_subtask_configs
