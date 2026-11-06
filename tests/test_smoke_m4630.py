"""Smoke test for the m4630 pipeline marker.

The marker module is a sibling file in ``tests/`` (the repository layout policy
forbids new files at the repository root), so this test imports it through a
``sys.path`` entry on its own directory instead of relying on package layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoke_marker_m4630 import PIPELINE_SMOKE_M4630  # noqa: E402


def test_marker_is_mission_number():
    """The marker must carry the mission number as a plain digit string."""
    assert isinstance(PIPELINE_SMOKE_M4630, str)
    assert PIPELINE_SMOKE_M4630.isdigit()
    assert PIPELINE_SMOKE_M4630 == "4630"