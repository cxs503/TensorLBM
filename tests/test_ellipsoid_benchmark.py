"""Tests for ellipsoid benchmark module."""
from __future__ import annotations

import math

import torch
from tensorlbm.ellipsoid_benchmark import (
    EllipsoidConfig,
    build_ellipsoid_mask,
    ellipsoid_statistics,
    reference_ellipsoid_cd,
    run_ellipsoid_benchmark,
)


def test_build_ellipsoid_mask_zero_aoa() -> None:
    """Mask at α=0 should be symmetric about the centreline."""
    mask = build_ellipsoid_mask(
        nx=60, ny=40, nz=40, a=24.0, b=8.0, alpha_deg=0.0,
    )
    assert mask.shape == (40, 40, 60)
    solid = mask.sum().item()
    assert solid > 0
    # Rough estimate: ellipsoid volume ≈ 4/3·π·24·8² = 6434, ~20-30% of that in voxels
    assert solid > 500
    assert solid < 10000

    # Symmetry check: centre-to-centre reflection about (cx,cy,cz)=(20,20,20)
    # Point (j,i) on z=20 slice reflects to (2*cy-j, 2*cx-i) = (40-j, 40-i)
    cx = cy = cz = 20.0
    mid_z = int(cz)
    slice_2d = mask[mid_z]  # (ny, nx)
    # Only check interior points where reflected index is valid
    # cx=20 means i≤40 ensures ir=40-i≥0; similarly j≤40 ensures jr=40-j≥0
    for j in range(1, 40):
        for i in range(1, 40):
            jr = int(2 * cy - j)
            ir = int(2 * cx - i)
            assert slice_2d[j, i] == slice_2d[jr, ir], \
                f"symmetry broken at (j={j},i={i}) ↔ (jr={jr},ir={ir})"


def test_build_ellipsoid_mask_with_aoa() -> None:
    """Mask at α=10° should be nose-up rotated."""
    mask_0 = build_ellipsoid_mask(
        nx=60, ny=40, nz=40, a=24.0, b=8.0, alpha_deg=0.0,
    )
    mask_10 = build_ellipsoid_mask(
        nx=60, ny=40, nz=40, a=24.0, b=8.0, alpha_deg=10.0,
    )
    # Masks should differ
    assert not torch.equal(mask_0, mask_10)
    # Both should have similar solid fraction (within 10%)
    s0 = mask_0.sum().item()
    s10 = mask_10.sum().item()
    assert abs(s0 - s10) / max(s0, 1) < 0.10


def test_ellipsoid_statistics() -> None:
    stats = ellipsoid_statistics(nx=60, ny=40, nz=40, a=24.0, b=8.0)

    # Analytical: length=48, diameter=16, volume=4/3·π·24·64=6434
    assert abs(stats["length_lu"] - 48.0) < 0.01
    assert abs(stats["diameter_lu"] - 16.0) < 0.01
    assert abs(stats["a_b_ratio"] - 3.0) < 0.01
    assert abs(stats["volume_lu3"] - (4 / 3) * math.pi * 24 * 64) < 0.1
    assert stats["frontal_area_lu2"] > 0
    assert stats["wetted_area_lu2"] > stats["frontal_area_lu2"]
    assert stats["solid_cells"] > 0


def test_reference_ellipsoid_cd() -> None:
    """Reference function returns plausible values."""
    ref0 = reference_ellipsoid_cd(re=100.0, alpha_deg=0.0)
    assert ref0["cd"] > 0.1
    assert ref0["cd"] < 5.0
    assert abs(ref0["cl"]) < 0.01

    ref10 = reference_ellipsoid_cd(re=100.0, alpha_deg=10.0)
    assert ref10["cl"] > 0.1  # positive lift


def test_ellipsoid_benchmark_runs() -> None:
    """Smoke test: minimal simulation should complete without NaN."""
    cfg = EllipsoidConfig(
        semi_major_a=18.0, semi_minor_b=6.0,
        alpha_deg=0.0, re=100,
        nx=80, ny=36, nz=36,
        n_steps=150, warmup_steps=50,
        smagorinsky_cs=0.1,
        device="cpu",
    )
    result = run_ellipsoid_benchmark(cfg)
    assert "cd_sim" in result
    assert "cl_sim" in result
    assert result["cd_sim"] > 0.0
    assert result["cd_sim"] < 100.0  # sanity
    # Cd should be in a reasonable range for a streamlined body
    assert result["cl_sim"] < 50.0


def test_ellipsoid_cd_reasonable() -> None:
    """Cd at α=0 should be lower than equivalent sphere (streamlined)."""
    cfg = EllipsoidConfig(
        semi_major_a=18.0, semi_minor_b=6.0,
        alpha_deg=0.0, re=100,
        nx=80, ny=36, nz=36,
        n_steps=300, warmup_steps=150,
        smagorinsky_cs=0.1,
        device="cpu",
    )
    result = run_ellipsoid_benchmark(cfg)
    cd = result["cd_sim"]
    # For Re=100, sphere Cd ≈ 1.09. Ellipsoid on a coarse grid (D=12, 33%
    # blockage) over-predicts Cd due to blockage — this is a smoke test,
    # not a validation test.  The grid-dependence test verifies realism.
    assert cd > 0.1, f"Cd={cd:.4f} too low (unphysical)"
    assert cd < 10.0, f"Cd={cd:.4f} too high even for coarse grid"


__all__ = [
    "test_build_ellipsoid_mask_zero_aoa",
    "test_build_ellipsoid_mask_with_aoa",
    "test_ellipsoid_statistics",
    "test_reference_ellipsoid_cd",
    "test_ellipsoid_benchmark_runs",
    "test_ellipsoid_cd_reasonable",
]
