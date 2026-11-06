"""Pipeline E2E smoke test for mission m4630."""

from smoke_marker_m4630 import PIPELINE_SMOKE_M4630


def test_marker_value():
    """Assert the smoke marker constant equals the mission number string."""
    assert PIPELINE_SMOKE_M4630 == "4630"


def test_basic_arithmetic():
    """Sanity check: 1 + 1 == 2."""
    assert 1 + 1 == 2
