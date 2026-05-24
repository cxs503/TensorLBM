"""Smoke tests for the hull free-surface benchmark."""
from __future__ import annotations

from tensorlbm.hull_free_surface import HullFreeSurfaceConfig, run_hull_free_surface


def test_hull_free_surface_smoke() -> None:
    result = run_hull_free_surface(HullFreeSurfaceConfig(nx=16, ny=8, nz=8, n_steps=2))
    assert "mean_cd" in result
