#!/usr/bin/env python3
"""Validate G1-D handover HDF5 datasets before Mimic or robomimic use.

The project supports both the legacy 24-D IK-Rel layout and the waist-enabled
26-D Pink-Abs layout.  The executable pipeline scripts explicitly select
``pink-abs`` so a legacy dataset cannot be mixed into waist-enabled training.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

try:
    import h5py
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on runtime image
    raise SystemExit(
        "h5py and numpy are required. Run this script with "
        "/workspace/isaaclab/_isaac_sim/python.sh."
    ) from exc

ACTION_LAYOUTS = {
    "ik-rel": {
        "dim": 24,
        "description": "24-D IK-Rel",
    },
    "pink-abs": {
        "dim": 26,
        "description": (
            "26-D Pink-Abs: left wrist 7 + right wrist 7 + "
            "right hand 6 + left hand 6"
        ),
    },
}
MIN_SEQUENCE_LENGTH = 10
PHASE_ORDER = (
    "right_grasped",
    "handover_completed",
    "placed",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bc_config_path() -> Path:
    return (
        _project_root()
        / "source"
        / "g1d_bimanual_cuboid_handover"
        / "g1d_bimanual_cuboid_handover"
        / "tasks"
        / "manager_based"
        / "g1d_bimanual"
        / "agents"
        / "robomimic"
        / "bc_handover.json"
    )


def _required_observation_keys() -> tuple[str, ...]:
    path = _bc_config_path()
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    keys = config["observation"]["modalities"]["obs"]["low_dim"]
    return tuple(str(key) for key in keys)


def _natural_key(name: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


def _is_finite(dataset: h5py.Dataset, chunk_size: int = 4096) -> bool:
    if dataset.size == 0:
        return False
    if dataset.ndim == 0:
        return bool(np.isfinite(dataset[()]).all())
    length = dataset.shape[0]
    for start in range(0, length, chunk_size):
        values = np.asarray(dataset[start : start + chunk_size])
        if not np.isfinite(values).all():
            return False
    return True


def _first_true(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(values)
    return int(indices[0]) if indices.size else None


def _validate_signal(
    episode_name: str,
    signal_name: str,
    dataset: h5py.Dataset,
    action_count: int,
    errors: list[str],
) -> int | None:
    values = np.asarray(dataset).astype(bool).reshape(-1)
    if values.size not in {action_count, action_count + 1}:
        errors.append(
            f"{episode_name}: signal {signal_name!r} length {values.size} "
            f"does not match actions length {action_count}."
        )
    first = _first_true(values)
    if first is None:
        errors.append(f"{episode_name}: signal {signal_name!r} never becomes true.")
        return None
    transitions = np.diff(values.astype(np.int8))
    if np.any(transitions < 0):
        errors.append(
            f"{episode_name}: signal {signal_name!r} is not latched/monotonic."
        )
    return first


def _format_attr(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _episode_count(path: Path) -> int:
    with h5py.File(path, "r") as file:
        data = file.get("data")
        return len(data) if isinstance(data, h5py.Group) else 0


def _validate_pink_quaternions(
    episode_name: str,
    actions: h5py.Dataset,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Check the two absolute wrist quaternions in a Pink-Abs trajectory."""
    values = np.asarray(actions)
    for label, action_slice in (
        ("left wrist", slice(3, 7)),
        ("right wrist", slice(10, 14)),
    ):
        quaternions = values[:, action_slice]
        norms = np.linalg.norm(quaternions, axis=-1)
        if not np.isfinite(norms).all() or np.any(norms < 1.0e-6):
            errors.append(
                f"{episode_name}: {label} Pink quaternion contains a zero or non-finite value."
            )
            continue
        max_error = float(np.max(np.abs(norms - 1.0)))
        if max_error > 0.05:
            warnings.append(
                f"{episode_name}: {label} quaternion norm deviates from one by "
                f"as much as {max_error:.5f}."
            )


def validate_dataset(
    path: Path,
    kind: str,
    action_layout: str,
    expected_episodes: int | None = None,
    reference_dataset: Path | None = None,
) -> int:
    errors: list[str] = []
    warnings: list[str] = []
    required_obs = _required_observation_keys()
    layout = ACTION_LAYOUTS[action_layout]
    expected_action_dim = int(layout["dim"])
    layout_description = str(layout["description"])

    if reference_dataset is not None:
        reference_count = _episode_count(reference_dataset)
        if expected_episodes is not None and expected_episodes != reference_count:
            errors.append(
                "Conflicting episode-count expectations: "
                f"--expected-episodes={expected_episodes}, "
                f"reference dataset has {reference_count}."
            )
        expected_episodes = reference_count

    with h5py.File(path, "r") as file:
        if "data" not in file or not isinstance(file["data"], h5py.Group):
            errors.append("Root group '/data' is missing.")
            episodes: list[str] = []
        else:
            episodes = sorted(file["data"].keys(), key=_natural_key)

        if not episodes:
            errors.append("The dataset contains no episodes.")
        if expected_episodes is not None and len(episodes) != expected_episodes:
            errors.append(
                f"Expected {expected_episodes} episodes, found {len(episodes)}."
            )

        total_steps = 0
        global_action_min = math.inf
        global_action_max = -math.inf

        for episode_name in episodes:
            episode = file["data"][episode_name]
            if "actions" not in episode:
                errors.append(f"{episode_name}: dataset 'actions' is missing.")
                continue

            actions = episode["actions"]
            if actions.ndim != 2:
                errors.append(
                    f"{episode_name}: actions must be rank 2, got shape {actions.shape}."
                )
                continue
            if actions.shape[1] != expected_action_dim:
                errors.append(
                    f"{episode_name}: expected {layout_description}, got action shape "
                    f"{actions.shape}. Do not mix 24-D IK-Rel and 26-D Pink-Abs data."
                )
            if actions.shape[0] < MIN_SEQUENCE_LENGTH:
                errors.append(
                    f"{episode_name}: only {actions.shape[0]} steps; BC-RNN requires "
                    f"at least {MIN_SEQUENCE_LENGTH}."
                )
            if not _is_finite(actions):
                errors.append(f"{episode_name}: actions contain NaN/Inf or are empty.")
            else:
                action_values = np.asarray(actions)
                global_action_min = min(global_action_min, float(action_values.min()))
                global_action_max = max(global_action_max, float(action_values.max()))
                if action_layout == "pink-abs" and actions.shape[1] == 26:
                    _validate_pink_quaternions(
                        episode_name,
                        actions,
                        errors,
                        warnings,
                    )

            action_count = int(actions.shape[0])
            total_steps += action_count

            if "obs" not in episode or not isinstance(episode["obs"], h5py.Group):
                errors.append(f"{episode_name}: observation group 'obs' is missing.")
                continue

            obs = episode["obs"]
            for key in required_obs:
                if key not in obs:
                    errors.append(f"{episode_name}: required observation 'obs/{key}' is missing.")
                    continue
                dataset = obs[key]
                if not isinstance(dataset, h5py.Dataset):
                    errors.append(f"{episode_name}: 'obs/{key}' is not a dataset.")
                    continue
                if dataset.ndim == 0 or dataset.shape[0] not in {action_count, action_count + 1}:
                    errors.append(
                        f"{episode_name}: observation 'obs/{key}' shape {dataset.shape} "
                        f"is incompatible with {action_count} actions."
                    )
                if not _is_finite(dataset):
                    errors.append(f"{episode_name}: observation 'obs/{key}' contains NaN/Inf or is empty.")

            if kind == "annotated":
                signal_path = "datagen_info/subtask_term_signals"
                current: h5py.Group | h5py.Dataset | None = obs
                for component in signal_path.split("/"):
                    if not isinstance(current, h5py.Group) or component not in current:
                        errors.append(
                            f"{episode_name}: annotated signal group "
                            f"'obs/{signal_path}' is missing."
                        )
                        current = None
                        break
                    current = current[component]

                if isinstance(current, h5py.Group):
                    first_indices: list[int] = []
                    for signal_name in PHASE_ORDER:
                        if signal_name not in current:
                            errors.append(
                                f"{episode_name}: annotated signal {signal_name!r} is missing."
                            )
                            continue
                        first = _validate_signal(
                            episode_name,
                            signal_name,
                            current[signal_name],
                            action_count,
                            errors,
                        )
                        if first is not None:
                            first_indices.append(first)
                    if len(first_indices) == len(PHASE_ORDER):
                        if any(a >= b for a, b in zip(first_indices, first_indices[1:])):
                            errors.append(
                                f"{episode_name}: phase boundaries are not strictly ordered: "
                                + ", ".join(
                                    f"{name}={index}"
                                    for name, index in zip(PHASE_ORDER, first_indices)
                                )
                            )

        if global_action_min < -1.001 or global_action_max > 1.001:
            if action_layout == "pink-abs":
                warnings.append(
                    "Pink-Abs contains absolute wrist positions, so action values "
                    "outside [-1, 1] can be valid: "
                    f"min={global_action_min:.5f}, max={global_action_max:.5f}."
                )
            else:
                warnings.append(
                    "IK-Rel action values extend outside [-1, 1]: "
                    f"min={global_action_min:.5f}, max={global_action_max:.5f}."
                )

        print("=" * 72)
        print("G1-D handover dataset validation")
        print("=" * 72)
        print(f"File:         {path}")
        print(f"Kind:         {kind}")
        print(f"Action layout:{layout_description:>53}")
        print(f"Episodes:     {len(episodes)}")
        if expected_episodes is not None:
            print(f"Expected:     {expected_episodes}")
        print(f"Total steps:  {total_steps}")
        if global_action_min != math.inf:
            print(f"Action min:   {global_action_min:.6f}")
            print(f"Action max:   {global_action_max:.6f}")
        if file.attrs:
            print("Root attrs:")
            for key, value in file.attrs.items():
                print(f"  {key}: {_format_attr(value)}")

    for warning in warnings:
        print(f"WARNING: {warning}")

    if errors:
        print("\nValidation FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("\nValidation PASSED.")
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="HDF5 dataset to inspect")
    parser.add_argument(
        "--kind",
        choices=("raw", "annotated", "generated", "training"),
        default="training",
        help="Additional validation appropriate for the pipeline stage",
    )
    parser.add_argument(
        "--action-layout",
        choices=tuple(ACTION_LAYOUTS),
        default="pink-abs",
        help="Expected action representation. Pipeline scripts use pink-abs.",
    )
    parser.add_argument(
        "--expected-episodes",
        type=int,
        default=None,
        help="Require exactly this many episodes.",
    )
    parser.add_argument(
        "--reference-dataset",
        type=Path,
        default=None,
        help="Require the same episode count as this HDF5 dataset.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    path = args.dataset.expanduser().resolve()
    if not path.is_file():
        print(f"ERROR: dataset was not found: {path}", file=sys.stderr)
        return 2
    if args.expected_episodes is not None and args.expected_episodes < 0:
        print("ERROR: --expected-episodes must be non-negative.", file=sys.stderr)
        return 2

    reference_dataset = None
    if args.reference_dataset is not None:
        reference_dataset = args.reference_dataset.expanduser().resolve()
        if not reference_dataset.is_file():
            print(
                f"ERROR: reference dataset was not found: {reference_dataset}",
                file=sys.stderr,
            )
            return 2

    return validate_dataset(
        path,
        args.kind,
        args.action_layout,
        expected_episodes=args.expected_episodes,
        reference_dataset=reference_dataset,
    )


if __name__ == "__main__":
    raise SystemExit(main())
