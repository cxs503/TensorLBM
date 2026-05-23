"""Unit tests for the obstacles module (Wigley hull mask + 3D forces/moments)."""
from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.obstacles import (
    compute_obstacle_forces_3d,
    compute_obstacle_moments_3d,
    wigley_hull_mask,
)


# ---------------------------------------------------------------------------
# wigley_hull_mask
# ---------------------------------------------------------------------------

def test_wigley_hull_mask_shape() -> None:
    mask = wigley_hull_mask(
        nx=40, ny=20, nz=20,
        ix_center=15, iy_center=10, iz_keel=4,
        length_lbm=20, beam_lbm=4, draft_lbm=6,
        device=torch.device("cpu"),
    )
    assert mask.shape == (20, 20, 40)
    assert mask.dtype == torch.bool


def test_wigley_hull_mask_non_empty() -> None:
    """Hull must contain some solid cells."""
    mask = wigley_hull_mask(
        nx=60, ny=30, nz=30,
        ix_center=20, iy_center=15, iz_keel=5,
        length_lbm=24, beam_lbm=6, draft_lbm=8,
        device=torch.device("cpu"),
    )
    assert mask.any(), "Wigley hull mask should contain at least one True cell"


def test_wigley_hull_mask_y_symmetry() -> None:
    """Hull must be symmetric about iy_center (requires odd ny so that iy_center
    is the exact integer midpoint of the grid)."""
    # ny=31 → iy_center=15 is index 15 out of 0..30, exactly at the midpoint.
    mask = wigley_hull_mask(
        nx=60, ny=31, nz=30,
        ix_center=30, iy_center=15, iz_keel=5,
        length_lbm=24, beam_lbm=6, draft_lbm=8,
        device=torch.device("cpu"),
    )
    # Flip along y axis (dim 1) must equal original
    assert torch.equal(mask, mask.flip(1)), "Hull should be symmetric about iy_center"


def test_wigley_hull_mask_cells_only_in_draft() -> None:
    """All solid cells must lie within the draft region."""
    iz_keel, draft = 5, 8
    mask = wigley_hull_mask(
        nx=60, ny=30, nz=30,
        ix_center=30, iy_center=15, iz_keel=iz_keel,
        length_lbm=24, beam_lbm=6, draft_lbm=draft,
        device=torch.device("cpu"),
    )
    # Find z-indices of all solid cells
    solid_z = mask.nonzero()[:, 0]  # dim-0 is z
    assert int(solid_z.min().item()) >= iz_keel
    assert int(solid_z.max().item()) <= iz_keel + draft


def test_wigley_hull_mask_cells_only_in_length() -> None:
    """All solid cells must lie within the ship-length window."""
    ix_center, half_length = 30, 12
    mask = wigley_hull_mask(
        nx=80, ny=30, nz=30,
        ix_center=ix_center, iy_center=15, iz_keel=5,
        length_lbm=2 * half_length, beam_lbm=6, draft_lbm=8,
        device=torch.device("cpu"),
    )
    solid_x = mask.nonzero()[:, 2]  # dim-2 is x
    assert int(solid_x.min().item()) >= ix_center - half_length
    assert int(solid_x.max().item()) <= ix_center + half_length


# ---------------------------------------------------------------------------
# compute_obstacle_forces_3d
# ---------------------------------------------------------------------------

def test_compute_obstacle_forces_3d_zero_f() -> None:
    """Zero distribution must give exactly zero forces."""
    f = torch.zeros(19, 10, 10, 10)
    mask = torch.zeros(10, 10, 10, dtype=torch.bool)
    mask[5, 5, 5] = True
    fx, fy, fz = compute_obstacle_forces_3d(f, mask)
    assert float(fx) == pytest.approx(0.0)
    assert float(fy) == pytest.approx(0.0)
    assert float(fz) == pytest.approx(0.0)


def test_compute_obstacle_forces_3d_empty_mask() -> None:
    """Empty obstacle mask must give exactly zero forces."""
    f = torch.rand(19, 6, 8, 10)
    mask = torch.zeros(6, 8, 10, dtype=torch.bool)
    fx, fy, fz = compute_obstacle_forces_3d(f, mask)
    assert float(fx) == pytest.approx(0.0)
    assert float(fy) == pytest.approx(0.0)
    assert float(fz) == pytest.approx(0.0)


def test_compute_obstacle_forces_3d_scalar_output() -> None:
    """Return values must be 0-dimensional tensors."""
    f = torch.rand(19, 8, 8, 16)
    mask = torch.zeros(8, 8, 16, dtype=torch.bool)
    mask[4, 4, 8] = True
    fx, fy, fz = compute_obstacle_forces_3d(f, mask)
    assert fx.shape == torch.Size([])
    assert fy.shape == torch.Size([])
    assert fz.shape == torch.Size([])


def test_compute_obstacle_forces_3d_finite() -> None:
    """Forces computed from finite f must be finite."""
    mask = wigley_hull_mask(
        nx=30, ny=16, nz=16,
        ix_center=10, iy_center=8, iz_keel=3,
        length_lbm=10, beam_lbm=4, draft_lbm=4,
        device=torch.device("cpu"),
    )
    f = torch.rand(19, 16, 16, 30)
    fx, fy, fz = compute_obstacle_forces_3d(f, mask)
    assert math.isfinite(float(fx))
    assert math.isfinite(float(fy))
    assert math.isfinite(float(fz))


# ---------------------------------------------------------------------------
# compute_obstacle_moments_3d
# ---------------------------------------------------------------------------

def test_compute_obstacle_moments_3d_zero_f() -> None:
    """Zero distribution must give exactly zero moments."""
    f = torch.zeros(19, 8, 8, 8)
    mask = torch.zeros(8, 8, 8, dtype=torch.bool)
    mask[4, 4, 4] = True
    Mx, My, Mz = compute_obstacle_moments_3d(f, mask, 4.0, 4.0, 4.0)
    assert float(Mx) == pytest.approx(0.0)
    assert float(My) == pytest.approx(0.0)
    assert float(Mz) == pytest.approx(0.0)


def test_compute_obstacle_moments_3d_empty_mask() -> None:
    """Empty obstacle mask must give exactly zero moments."""
    f = torch.rand(19, 6, 8, 10)
    mask = torch.zeros(6, 8, 10, dtype=torch.bool)
    Mx, My, Mz = compute_obstacle_moments_3d(f, mask, 3.0, 4.0, 5.0)
    assert float(Mx) == pytest.approx(0.0)
    assert float(My) == pytest.approx(0.0)
    assert float(Mz) == pytest.approx(0.0)


def test_compute_obstacle_moments_3d_scalar_output() -> None:
    """Return values must be 0-dimensional tensors."""
    f = torch.rand(19, 8, 8, 16)
    mask = torch.zeros(8, 8, 16, dtype=torch.bool)
    mask[4, 4, 8] = True
    Mx, My, Mz = compute_obstacle_moments_3d(f, mask, 4.0, 4.0, 8.0)
    assert Mx.shape == torch.Size([])
    assert My.shape == torch.Size([])
    assert Mz.shape == torch.Size([])
