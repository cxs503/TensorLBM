"""Tests for the m4630 pipeline smoke marker.

The marker module lives as a sibling file in ``tests/`` (the repository layout
policy forbids new files at the repository root), so it is imported through a
``sys.path`` entry on this file's own directory — the same pattern already used
by other tests in this suite (e.g. ``test_bench_dam_break.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoke_marker_m4630 import PIPELINE_SMOKE_M4630  # noqa: E402


def test_marker_value():
    """The marker constant must equal the mission number as a string."""
    assert PIPELINE_SMOKE_M4630 == "4630"


def test_marker_round_trips_through_the_pipeline():
    """The constant survives file transport: it is a plain ``str`` of digits."""
    assert isinstance(PIPELINE_SMOKE_M4630, str)
    assert PIPELINE_SMOKE_M4630.isdigit()


def test_basic_arithmetic():
    """Sanity check: the harness can execute a trivially true assertion."""
    assert 1 + 1 == 2