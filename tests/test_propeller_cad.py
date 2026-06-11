"""Tests for propeller CAD geometry module."""
from __future__ import annotations

import torch
from tensorlbm.propeller_cad import (
    PropellerGeometryConfig,
    build_propeller_mask,
    propeller_statistics,
    KP505_PRESET,
    GENERIC_PRESET,
)


def test_propeller_config() -> None:
    cfg = KP505_PRESET
    assert cfg.n_blades == 5
    assert cfg.diameter == 48.0
    assert cfg.hub_radius > 0
    assert cfg.hub_radius < cfg.radius


def test_build_propeller_mask_small() -> None:
    cfg = PropellerGeometryConfig(n_blades=4, diameter=32.0)
    mask = build_propeller_mask(nx=60, ny=30, nz=30, cx=21, cy=15, cz=15, config=cfg)
    assert mask.shape == (30, 30, 60)
    solid = mask.sum().item()
    assert solid > 500, f"Only {solid} solid cells"
    assert solid < 5000, f"Too many solid cells: {solid}"


def test_propeller_statistics() -> None:
    cfg = KP505_PRESET
    mask = build_propeller_mask(nx=80, ny=40, nz=40, cx=28, cy=20, cz=20, config=cfg)
    stats = propeller_statistics(cfg, mask)
    assert stats["solid_cells"] > 0
    assert stats["solid_fraction"] > 0.001
    assert stats["estimated_wetted_area"] > 100


def test_masks_at_different_angles() -> None:
    """Ensure different rotation angles produce valid masks."""
    cfg = GENERIC_PRESET
    mask0 = build_propeller_mask(nx=60, ny=30, nz=30, cx=21, cy=15, cz=15, config=cfg, angle_deg=0.0)
    mask45 = build_propeller_mask(nx=60, ny=30, nz=30, cx=21, cy=15, cz=15, config=cfg, angle_deg=45.0)
    s0 = mask0.sum().item()
    s45 = mask45.sum().item()
    # Should be roughly similar (same geometry, rotated)
    assert abs(s0 - s45) / max(s0, 1) < 0.3


__all__ = [
    "test_propeller_config",
    "test_build_propeller_mask_small",
    "test_propeller_statistics",
    "test_masks_at_different_angles",
]
