#!/usr/bin/env python3
# Copyright (c) 2025-2026, SmartDev.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Launch one G1-D environment and verify its imitation-learning contract.

The default test covers the waist-enabled 26-D Pink-Abs recording environment.
A legacy IK-Rel task can still be supplied explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-test the G1-D cuboid-handover imitation environment."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="G1D-Bimanual-CuboidHandover-Pink-Abs-Record-v0",
        help="Registered environment ID to test.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of zero-action simulation steps.",
    )
    parser.add_argument(
        "--enable_pinocchio",
        action="store_true",
        default=False,
        help="Import Pinocchio before launching Isaac Sim. Automatically enabled for Pink tasks.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _project_paths() -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "source" / "g1d_bimanual_cuboid_handover"
    return project_root, source_root


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be a positive integer")

    is_pink = "Pink-Abs" in args.task
    if is_pink or args.enable_pinocchio:
        # Import before AppLauncher so Isaac Lab's Pinocchio build is selected.
        import pinocchio  # noqa: F401

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None

    try:
        _, source_root = _project_paths()
        sys.path.insert(0, str(source_root))

        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import parse_env_cfg

        import g1d_bimanual_cuboid_handover.tasks  # noqa: F401

        cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=1,
            use_fabric=not args.disable_fabric,
        )
        cfg.scene.num_envs = 1
        cfg.scene.env_spacing = 2.5

        if "Record" in args.task and cfg.terminations.time_out is not None:
            raise RuntimeError(
                "The recording environment must disable the time_out termination."
            )

        if is_pink:
            expected_terms = (
                "arm_action",
                "gripper_action",
                "left_gripper_action",
            )
            expected_dims = (14, 6, 6)
            expected_total = 26
            controlled = set(cfg.actions.arm_action.pink_controlled_joint_names)
            missing_waist = {"torso_Joint", "Yaw_Joint"}.difference(controlled)
            if missing_waist:
                raise RuntimeError(
                    "Pink action does not include every waist joint: "
                    f"missing={sorted(missing_waist)}"
                )
        else:
            expected_terms = (
                "arm_action",
                "gripper_action",
                "left_arm_action",
                "left_gripper_action",
            )
            expected_dims = (6, 6, 6, 6)
            expected_total = 24

        env = gym.make(args.task, cfg=cfg).unwrapped

        terms = tuple(env.action_manager.active_terms)
        dims = tuple(env.action_manager.action_term_dim)
        if terms != expected_terms or dims != expected_dims:
            raise RuntimeError(
                "Unexpected action layout: "
                f"terms={terms}, dims={dims}, "
                f"expected_terms={expected_terms}, expected_dims={expected_dims}"
            )
        if env.action_manager.total_action_dim != expected_total:
            raise RuntimeError(
                f"Expected {expected_total} actions, "
                f"got {env.action_manager.total_action_dim}."
            )

        config_path = (
            source_root
            / "g1d_bimanual_cuboid_handover"
            / "tasks"
            / "manager_based"
            / "g1d_bimanual"
            / "agents"
            / "robomimic"
            / "bc_handover.json"
        )
        with config_path.open("r", encoding="utf-8") as stream:
            bc_config = json.load(stream)
        required_obs = set(
            bc_config["observation"]["modalities"]["obs"]["low_dim"]
        )

        obs, _ = env.reset()
        policy_obs = obs.get("policy", {})
        missing_obs = sorted(required_obs.difference(policy_obs.keys()))
        if missing_obs:
            raise RuntimeError(
                f"Missing robomimic policy observations after reset: {missing_obs}"
            )

        if not hasattr(env, "get_handover_phase_signals"):
            raise RuntimeError(
                "The environment does not expose strict handover phase signals."
            )
        signals = env.get_handover_phase_signals()
        expected_signals = {
            "right_grasped",
            "handover_completed",
            "placed",
        }
        if set(signals) != expected_signals:
            raise RuntimeError(f"Unexpected phase signals: {sorted(signals)}")
        if any(bool(value.any().item()) for value in signals.values()):
            raise RuntimeError("Phase signals must be false immediately after reset.")

        actions = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim),
            dtype=torch.float32,
            device=env.device,
        )
        for _ in range(args.steps):
            env.step(actions)

        print("=" * 72)
        print("G1-D cuboid-handover environment smoke test PASSED")
        print(f"task: {args.task}")
        print(f"action terms: {terms}")
        print(f"action dimensions: {dims} (total={expected_total})")
        print(f"waist through Pink IK: {is_pink}")
        print(f"required policy observations: {len(required_obs)}")
        print(f"phase signals: {sorted(expected_signals)}")
        print(f"zero-action steps completed: {args.steps}")
        print("=" * 72)
        return 0
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
