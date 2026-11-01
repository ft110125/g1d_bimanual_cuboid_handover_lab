# Copyright (c) 2025-2026, SmartDev.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime environment for ordered three-stage cuboid handover success."""

from isaaclab.envs import ManagerBasedRLEnv

from .g1d_handover_phase import G1DHandoverPhaseMixin


class G1DBimanualCuboidHandoverEnv(G1DHandoverPhaseMixin, ManagerBasedRLEnv):
    """Manager-based environment with three ordered handover latches."""

    pass
