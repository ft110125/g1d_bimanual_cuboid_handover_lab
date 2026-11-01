#!/usr/bin/env python3
# Copyright (c) 2025-2026, SmartDev.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""List registered Gymnasium environments from this project."""

import os
import sys

# Ensure the source package is importable
script_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.normpath(
    os.path.join(script_dir, "..", "source", "g1d_bimanual_cuboid_handover")
)
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)

import g1d_bimanual_cuboid_handover.tasks  # noqa: F401, E402
import gymnasium as gym  # noqa: E402


def main() -> None:
    print("=" * 70)
    print("G1-D Bimanual Cuboid Handover — Registered Environments")
    print("=" * 70)
    prefix = "G1D-Bimanual-CuboidHandover"
    for env_id in sorted(gym.registry.keys()):
        if prefix in env_id:
            print(f"  {env_id}")
    print("=" * 70)


if __name__ == "__main__":
    main()
