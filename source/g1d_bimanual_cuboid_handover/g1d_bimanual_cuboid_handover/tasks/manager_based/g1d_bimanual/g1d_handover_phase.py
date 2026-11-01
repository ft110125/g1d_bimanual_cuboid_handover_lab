# Copyright (c) 2025-2026, SmartDev.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Three-stage phase tracking for the G1-D cuboid handover task.

The public task contract is intentionally small:

1. ``right_grasped``: the right EEF is near the cuboid while it is lifted.
2. ``handover_completed``: the cuboid reaches ``HANDOVER_POS`` with the left
   EEF nearby, then the right EEF moves clear while the left EEF remains near.
3. ``placed``: after the handover, the cuboid is stably placed at the goal.

Finger-joint closure is deliberately not used by any success or Mimic phase
condition. This avoids coupling task labels to noisy OpenXR finger estimates.
The phase tracker instead uses EEF-to-object distance, object displacement,
object pose/velocity, and ordered phase history.

The handover is deliberately split internally into two moments while retaining
only one public ``handover_completed`` signal:

* first, the cuboid enters the handover region with the left EEF nearby;
* later, the right EEF moves clear while the left EEF remains nearby.

This prevents a missed success when the left hand starts moving the cuboid away
from ``HANDOVER_POS`` before the right EEF has reached the clearance threshold.

The official Isaac Lab demo recorder removes the configured success termination
before creating the environment and evaluates that function manually after each
step. Therefore the ordered history must live on the environment itself rather
than inside a stateful termination term.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import os

import torch

from .g1d_bimanual_env_cfg import (
    CUBOID_GOAL_POS,
    CUBOID_INITIAL_POS,
    EEF_NEAR_OBJECT_THRESHOLD,
    GRASP_LIFT_HEIGHT,
    HAND_CLEARANCE_THRESHOLD,
    HANDOVER_POS,
    HANDOVER_REGION_RADIUS,
    SUCCESS_ANGULAR_VEL_THRESHOLD,
    SUCCESS_LINEAR_VEL_THRESHOLD,
    SUCCESS_XY_THRESHOLD,
    SUCCESS_Z_THRESHOLD,
)

_PHASE_NAMES = (
    "right_grasped",
    "handover_completed",
    "placed",
)


class G1DHandoverPhaseMixin:
    """Cooperative mixin that tracks the ordered three-stage task."""

    def __init__(self, cfg: Any, **kwargs: Any) -> None:
        self._handover_phase_buffers: dict[str, torch.Tensor] = {}
        self._handover_zone_seen: torch.Tensor | None = None
        self._debug_handover_phases = os.getenv(
            "G1D_DEBUG_HANDOVER_PHASES", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._handover_last_update_step: torch.Tensor | None = None
        super().__init__(cfg, **kwargs)
        self._initialize_handover_phase_tracking()

    def _initialize_handover_phase_tracking(self) -> None:
        """Allocate public phase buffers and the private handover-zone latch."""
        self._handover_phase_buffers = {
            name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for name in _PHASE_NAMES
        }
        self._handover_zone_seen = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._handover_last_update_step = torch.full(
            (self.num_envs,),
            -1,
            dtype=torch.long,
            device=self.device,
        )

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        """Reset normal environment state and every handover latch."""
        super()._reset_idx(env_ids)
        if not self._handover_phase_buffers:
            return
        for buffer in self._handover_phase_buffers.values():
            buffer[env_ids] = False
        if self._handover_zone_seen is not None:
            self._handover_zone_seen[env_ids] = False
        if self._handover_last_update_step is not None:
            self._handover_last_update_step[env_ids] = -1

    def step(self, action: torch.Tensor):
        """Step the environment and update phase history after physics advances."""
        result = super().step(action)
        self.update_handover_phase_latches()
        return result

    def _get_object_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obj = self.scene["object"]
        position = obj.data.root_pos_w - self.scene.env_origins
        return position, obj.data.root_lin_vel_w, obj.data.root_ang_vel_w

    def _get_eef_positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        right_frame = self.scene["right_ee_frame"]
        left_frame = self.scene["left_ee_frame"]
        right_position = (
            right_frame.data.target_pos_w[:, 0, :] - self.scene.env_origins
        )
        left_position = (
            left_frame.data.target_pos_w[:, 0, :] - self.scene.env_origins
        )
        return right_position, left_position

    def _debug_transition(
        self,
        name: str,
        transition_mask: torch.Tensor,
        obj_pos: torch.Tensor,
        right_to_obj: torch.Tensor,
        left_to_obj: torch.Tensor,
        dist_to_handover: torch.Tensor,
        xy_err_goal: torch.Tensor,
        z_err_goal: torch.Tensor,
    ) -> None:
        """Print one concise line when a phase becomes true."""
        if not self._debug_handover_phases or not torch.any(transition_mask):
            return
        for env_id in torch.nonzero(transition_mask, as_tuple=False).flatten().tolist():
            pos = obj_pos[env_id].detach().cpu().tolist()
            print(
                f"[G1D][handover][env {env_id}] {name}=True "
                f"object=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
                f"right_dist={right_to_obj[env_id].item():.3f} "
                f"left_dist={left_to_obj[env_id].item():.3f} "
                f"handover_dist={dist_to_handover[env_id].item():.3f} "
                f"goal_xy={xy_err_goal[env_id].item():.3f} "
                f"goal_z={z_err_goal[env_id].item():.3f}",
                flush=True,
            )

    def update_handover_phase_latches(self) -> None:
        """Update the ordered public latches from current physics and EEF state."""
        if (
            not self._handover_phase_buffers
            or self._handover_zone_seen is None
            or self._handover_last_update_step is None
        ):
            return

        # Success and Mimic helpers may query signals more than once in one
        # control step. Process each environment only once per episode step.
        current_step = self.episode_length_buf.to(dtype=torch.long)
        update_mask = current_step != self._handover_last_update_step
        if not torch.any(update_mask):
            return

        obj_pos, obj_lin_vel, obj_ang_vel = self._get_object_state()
        right_pos, left_pos = self._get_eef_positions()

        obj_speed = torch.linalg.vector_norm(obj_lin_vel, dim=-1)
        obj_ang_speed = torch.linalg.vector_norm(obj_ang_vel, dim=-1)
        right_to_obj = torch.linalg.vector_norm(right_pos - obj_pos, dim=-1)
        left_to_obj = torch.linalg.vector_norm(left_pos - obj_pos, dim=-1)

        right_near = right_to_obj < EEF_NEAR_OBJECT_THRESHOLD
        left_near = left_to_obj < EEF_NEAR_OBJECT_THRESHOLD
        right_far = right_to_obj > HAND_CLEARANCE_THRESHOLD

        handover_target = obj_pos.new_tensor(HANDOVER_POS)
        goal_target = obj_pos.new_tensor(CUBOID_GOAL_POS)
        dist_to_handover = torch.linalg.vector_norm(
            obj_pos - handover_target,
            dim=-1,
        )
        xy_err_goal = torch.linalg.vector_norm(
            obj_pos[:, :2] - goal_target[:2],
            dim=-1,
        )
        z_err_goal = torch.abs(obj_pos[:, 2] - goal_target[2])

        # Snapshot history first. Each public phase requires its predecessor to
        # have been true on an earlier control step, preventing annotation
        # boundaries from collapsing onto one frame.
        was_right_grasped = self._handover_phase_buffers["right_grasped"].clone()
        was_handover_zone_seen = self._handover_zone_seen.clone()
        was_handover_completed = self._handover_phase_buffers[
            "handover_completed"
        ].clone()
        was_placed = self._handover_phase_buffers["placed"].clone()

        # Stage 1: right-side pickup. No finger closure is inspected. The
        # combination of right-near and object lift is used as grasp evidence.
        lifted = obj_pos[:, 2] > (CUBOID_INITIAL_POS[2] + GRASP_LIFT_HEIGHT)
        
        if self._debug_handover_phases:
            debug_mask = (
                update_mask
                & ~was_right_grasped
                & ((current_step % 25) == 0)
            )

            for env_id in torch.nonzero(
                debug_mask,
                as_tuple=False,
            ).flatten().tolist():
                lift_amount = (
                    obj_pos[env_id, 2]
                    - CUBOID_INITIAL_POS[2]
                )

                print(
                    f"[G1D][pickup-debug][env {env_id}] "
                    f"right_dist="
                    f"{right_to_obj[env_id].item():.3f} "
                    f"near_threshold="
                    f"{EEF_NEAR_OBJECT_THRESHOLD:.3f} "
                    f"lift="
                    f"{lift_amount.item():.3f} "
                    f"lift_threshold="
                    f"{GRASP_LIFT_HEIGHT:.3f}",
                    flush=True,
                )
        
        right_grasped = was_right_grasped | (right_near & lifted)
        self._handover_phase_buffers["right_grasped"].copy_(
            torch.where(update_mask, right_grasped, was_right_grasped)
        )
        self._debug_transition(
            "right_grasped",
            update_mask & ~was_right_grasped & right_grasped,
            obj_pos,
            right_to_obj,
            left_to_obj,
            dist_to_handover,
            xy_err_goal,
            z_err_goal,
        )

        # Private transfer-contact latch: remember that the right-picked object
        # reached HANDOVER_POS while the left EEF was near it. The object does
        # not need to remain inside the sphere while the right hand retreats.
        in_handover_zone = dist_to_handover < HANDOVER_REGION_RADIUS
        handover_zone_seen = was_handover_zone_seen | (
            was_right_grasped
            & in_handover_zone
            & left_near
        )
        self._handover_zone_seen.copy_(
            torch.where(
                update_mask,
                handover_zone_seen,
                was_handover_zone_seen,
            )
        )
        self._debug_transition(
            "handover_zone_seen",
            update_mask & ~was_handover_zone_seen & handover_zone_seen,
            obj_pos,
            right_to_obj,
            left_to_obj,
            dist_to_handover,
            xy_err_goal,
            z_err_goal,
        )

        # Stage 2: transfer completed. Once handover contact has been observed,
        # the left EEF must remain near while the right EEF moves clear. The
        # cuboid may already be moving away from HANDOVER_POS at this moment.
        handover_completed = was_handover_completed | (
            was_handover_zone_seen
            & left_near
            & right_far
        )
        self._handover_phase_buffers["handover_completed"].copy_(
            torch.where(
                update_mask,
                handover_completed,
                was_handover_completed,
            )
        )
        self._debug_transition(
            "handover_completed",
            update_mask & ~was_handover_completed & handover_completed,
            obj_pos,
            right_to_obj,
            left_to_obj,
            dist_to_handover,
            xy_err_goal,
            z_err_goal,
        )

        # Stage 3: stable placement at the goal. Left-hand attribution was
        # already established by the transfer-completed stage, so the left EEF
        # does not need to remain within 8 cm after the cuboid has been released
        # and settled. This lets users naturally withdraw the left hand.
        at_goal_xy = xy_err_goal < SUCCESS_XY_THRESHOLD
        at_goal_z = z_err_goal < SUCCESS_Z_THRESHOLD
        linear_stopped = obj_speed < SUCCESS_LINEAR_VEL_THRESHOLD
        angular_stopped = obj_ang_speed < SUCCESS_ANGULAR_VEL_THRESHOLD
        placed = was_placed | (
            was_handover_completed
            & at_goal_xy
            & at_goal_z
            & linear_stopped
            & angular_stopped
        )
        self._handover_phase_buffers["placed"].copy_(
            torch.where(update_mask, placed, was_placed)
        )
        self._debug_transition(
            "placed",
            update_mask & ~was_placed & placed,
            obj_pos,
            right_to_obj,
            left_to_obj,
            dist_to_handover,
            xy_err_goal,
            z_err_goal,
        )

        self._handover_last_update_step.copy_(
            torch.where(
                update_mask,
                current_step,
                self._handover_last_update_step,
            )
        )

    def get_handover_phase_signals(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return cloned public phase latches for selected environments."""
        self.update_handover_phase_latches()
        indices: slice | Sequence[int] | torch.Tensor = (
            slice(None) if env_ids is None else env_ids
        )
        return {
            name: buffer[indices].clone()
            for name, buffer in self._handover_phase_buffers.items()
        }
