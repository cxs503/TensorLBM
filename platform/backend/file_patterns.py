"""Helpers for output file naming compatibility."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_STEP_RE = re.compile(r"^(flow_step|snapshot)_(\d{6})\.png$")


def list_step_images(output_dir: Path) -> list[Path]:
    """Return step images with migration compatibility.

    Prefers canonical ``flow_step_*.png`` when both canonical and legacy
    ``snapshot_*.png`` exist for the same step.
    """
    by_step: dict[int, Path] = {}
    for path in output_dir.rglob("*.png"):
        match = _STEP_RE.match(path.name)
        if match is None:
            continue
        prefix, step_text = match.groups()
        step = int(step_text)
        current = by_step.get(step)
        if current is None:
            by_step[step] = path
            continue
        if prefix == "flow_step" and current.name.startswith("snapshot_"):
            by_step[step] = path
    return [by_step[s] for s in sorted(by_step)]
