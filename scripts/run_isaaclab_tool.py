#!/usr/bin/env python3
# Copyright (c) 2025-2026, SmartDev.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Wrapper to run Isaac Lab official scripts with external environment registration.

Usage::

    python scripts/run_isaaclab_tool.py <path_to_isaaclab_script> [script_args...]

Independent copy for g1d_bimanual_cuboid_handover project.
"""

from __future__ import annotations

import inspect
import os
import runpy
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any


def _distribution_version(package_name: str) -> str:
    """Return an installed distribution version without failing startup."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def _version_prefix(value: str) -> tuple[int, int, int] | None:
    """Parse the numeric prefix of a PEP 440-like version string."""
    parts: list[int] = []
    for token in value.split("."):
        digits = ""
        for char in token:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) == 3:
            break
    if not parts:
        return None
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _solve_accepts_primal_start(solve_function: Any, daqp_version: str) -> bool:
    """Return whether ``solve_function`` natively accepts ``primal_start``."""
    try:
        signature = inspect.signature(solve_function)
    except (TypeError, ValueError):
        signature = None

    if signature is not None and "primal_start" in signature.parameters:
        return True

    parsed_version = _version_prefix(daqp_version)
    return parsed_version is not None and parsed_version >= (0, 8, 0)


def _apply_daqp_primal_start_compatibility() -> None:
    """Make qpsolvers 4.12+ interoperate with DAQP 0.7.x offline."""
    if os.environ.get("G1D_DISABLE_DAQP_COMPAT", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        print("[G1D][Pink QP] DAQP compatibility shim disabled by environment", flush=True)
        return

    try:
        import daqp
    except ImportError as exc:
        print(f"[G1D][Pink QP] DAQP import unavailable: {exc}", flush=True)
        return

    qpsolvers_version = _distribution_version("qpsolvers")
    daqp_version = _distribution_version("daqp")
    qpsolvers_parsed = _version_prefix(qpsolvers_version)

    original_solve = daqp.solve
    if getattr(original_solve, "_g1d_primal_start_compat", False):
        return

    native_primal_start = _solve_accepts_primal_start(original_solve, daqp_version)
    qpsolvers_needs_primal_start = qpsolvers_parsed is None or qpsolvers_parsed >= (4, 12, 0)

    print(
        f"[G1D][Pink QP] qpsolvers={qpsolvers_version}, daqp={daqp_version}",
        flush=True,
    )

    if native_primal_start:
        print("[G1D][Pink QP] native primal_start support detected", flush=True)
        return

    if not qpsolvers_needs_primal_start:
        print("[G1D][Pink QP] installed qpsolvers does not require the DAQP warm-start keyword", flush=True)
        return

    def solve_compat(*args: Any, primal_start: Any = None, **kwargs: Any):
        del primal_start
        return original_solve(*args, **kwargs)

    solve_compat.__name__ = getattr(original_solve, "__name__", "solve")
    solve_compat.__qualname__ = getattr(original_solve, "__qualname__", "solve")
    solve_compat.__doc__ = getattr(original_solve, "__doc__", None)
    solve_compat._g1d_primal_start_compat = True  # type: ignore[attr-defined]
    solve_compat._g1d_original_solve = original_solve  # type: ignore[attr-defined]
    daqp.solve = solve_compat

    print(
        "[G1D][Pink QP] installed project-local primal_start compatibility wrapper",
        flush=True,
    )


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_isaaclab_tool.py <script_path> [args...]")
        sys.exit(1)

    # Apply before importing task modules
    _apply_daqp_primal_start_compatibility()

    # Ensure the source package is importable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.normpath(
        os.path.join(script_dir, "..", "source", "g1d_bimanual_cuboid_handover")
    )
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    # Register our external Gym environments
    import g1d_bimanual_cuboid_handover.tasks  # noqa: F401

    # Extract and run the target script
    script_path = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    main()
