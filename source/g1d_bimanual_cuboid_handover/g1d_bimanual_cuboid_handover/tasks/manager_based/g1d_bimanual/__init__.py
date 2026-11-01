# Copyright (c) 2025-2026, SmartDev.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G1-D Bimanual Cuboid Handover tasks — Gymnasium environment registration."""

import gymnasium as gym

from . import agents  # noqa: F401

##
# Register Gym environments.
##

# ---------------------------------------------------------------------------
# Bimanual IK Relative — normal training / teleop
# ---------------------------------------------------------------------------
gym.register(
    id="G1D-Bimanual-CuboidHandover-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1d_bimanual_env_cfg:G1DBimanualCuboidHandoverEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DBimanualPPORunnerCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_handover.json",
    },
)

# ---------------------------------------------------------------------------
# Bimanual IK Relative — PLAY (visualisation / evaluation)
# ---------------------------------------------------------------------------
gym.register(
    id="G1D-Bimanual-CuboidHandover-IK-Rel-PLAY-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1d_bimanual_env_cfg:G1DBimanualCuboidHandoverEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DBimanualPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Bimanual IK Relative — Mimic (data generation / augmentation)
# ---------------------------------------------------------------------------
gym.register(
    id="G1D-Bimanual-CuboidHandover-IK-Rel-Mimic-v0",
    entry_point=f"{__name__}.g1d_bimanual_mimic_env:G1DBimanualCuboidHandoverMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1d_bimanual_mimic_env_cfg:G1DBimanualCuboidHandoverMimicEnvCfg",
    },
)

# ---------------------------------------------------------------------------
# Bimanual Pink Absolute — bimanual Pink IK with absolute wrist poses
# ---------------------------------------------------------------------------
gym.register(
    id="G1D-Bimanual-CuboidHandover-Pink-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1d_bimanual_env_cfg:G1DBimanualCuboidHandoverPinkAbsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DBimanualPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Bimanual Pink Absolute — PLAY (visualisation / evaluation)
# ---------------------------------------------------------------------------
gym.register(
    id="G1D-Bimanual-CuboidHandover-Pink-Abs-PLAY-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1d_bimanual_env_cfg:G1DBimanualCuboidHandoverPinkAbsEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DBimanualPPORunnerCfg",
    },
)
