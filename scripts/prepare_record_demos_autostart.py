#!/usr/bin/env python3
# Copyright (c) 2025-2026, SmartDev.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Create a temporary auto-start variant of Isaac Lab's demo recorder.

Isaac Lab 2.3.x intentionally starts XR recording in a paused state.  While
paused, its main loop renders the scene but does not call ``env.step()``.  This
means the robot remains motionless until the XR client emits a ``START``
teleoperation command.

This helper copies the installed ``record_demos.py`` to a temporary location
and changes only the initial recording state.  The Isaac Lab installation is
never edited.  START, STOP, RESET, success handling, recorder management and
all other upstream behavior remain unchanged.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path


_ASSIGNMENT_RE = re.compile(
    r"^(?P<indent>[ \t]*)running_recording_instance\s*=\s*not\s+args_cli\.xr\s*$",
    flags=re.MULTILINE,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an auto-start copy of Isaac Lab record_demos.py."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _default_output(source: Path) -> Path:
    version_hint = "unknown"
    try:
        version_file = source.parents[2] / "VERSION"
        if version_file.is_file():
            version_hint = version_file.read_text(encoding="utf-8").strip()
    except OSError:
        pass

    safe_version = re.sub(r"[^A-Za-z0-9_.-]+", "_", version_hint)
    return Path(tempfile.gettempdir()) / f"g1d_record_demos_autostart_{safe_version}.py"


def main() -> int:
    args = _parse_args()
    source = args.source.expanduser().resolve()
    output = (args.output or _default_output(source)).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Isaac Lab recorder was not found: {source}")

    text = source.read_text(encoding="utf-8")
    matches = list(_ASSIGNMENT_RE.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(
            "Could not safely enable XR auto-start. Expected exactly one line "
            "'running_recording_instance = not args_cli.xr' in the installed "
            f"recorder, found {len(matches)}. Source: {source}"
        )

    match = matches[0]
    indent = match.group("indent")
    replacement = (
        f"{indent}# G1D project-local change: keep XR simulation stepping from startup.\n"
        f"{indent}# The upstream default pauses XR until a START command is received.\n"
        f"{indent}running_recording_instance = True\n"
        f"{indent}if args_cli.xr:\n"
        f"{indent}    print(\"[G1D] XR demo recording auto-started; STOP pauses and START resumes.\")"
    )
    patched = _ASSIGNMENT_RE.sub(replacement, text, count=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temp_output.write_text(patched, encoding="utf-8")
    temp_output.chmod(0o755)
    temp_output.replace(output)

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
