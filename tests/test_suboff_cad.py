"""Tests for the SUBOFF submarine CAD module (suboff_cad.py).

Covers:
- Radius profile shape and boundary values.
- Hull mask shape, dtype, and solid cells for all variants.
- build_suboff_mask convenience wrapper and statistics dict.
- suboff_statistics returns expected keys and physical values.
- generate_suboff_previews returns a Figure.
- export_suboff_stl writes a valid ASCII STL file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tensorlbm.suboff_cad import (
    SuboffConfig,
    SuboffHullType,
    build_suboff_mask,
    export_suboff_stl,
    generate_suboff_previews,
    suboff_hull_mask,
    suboff_radius_profile,
    suboff_statistics,
)

CPU = torch.device("cpu")

# Small grid for speed
NX, NY, NZ = 60, 40, 40
LENGTH = 36.0  # lattice units (leaves head-room in the grid)
RADIUS = LENGTH / (2.0 * 8.57)   # R/L = 1/(2*8.57): correct L/D ≈ 8.57


# ---------------------------------------------------------------------------
# Radius profile
# ---------------------------------------------------------------------------


def test_radius_profile_bow_tip() -> None:
    """Radius must be 0 at the bow (xi=0)."""
    r = suboff_radius_profile(np.array([0.0]))
    assert float(r[0]) == pytest.approx(0.0, abs=1e-10)


def test_radius_profile_stern_tip() -> None:
    """Radius must be 0 at the stern (xi=1)."""
    r = suboff_radius_profile(np.array([1.0]))
    assert float(r[0]) == pytest.approx(0.0, abs=1e-6)


def test_radius_profile_parallel_midbody() -> None:
    """Radius must equal 1 throughout the parallel midbody."""
    cfg = SuboffConfig()
    xi_mid = np.linspace(cfg.bow_fraction + 0.01, 1.0 - cfg.stern_fraction - 0.01, 50)
    r = suboff_radius_profile(xi_mid, cfg)
    assert np.allclose(r, 1.0, atol=1e-9)


def test_radius_profile_range() -> None:
    """Radius must be in [0, 1] everywhere."""
    xi = np.linspace(0.0, 1.0, 500)
    r = suboff_radius_profile(xi)
    assert float(r.min()) >= 0.0
    assert float(r.max()) <= 1.0 + 1e-9


def test_radius_profile_monotone_bow() -> None:
    """Radius must be non-decreasing over the bow section."""
    cfg = SuboffConfig()
    xi_bow = np.linspace(0.0, cfg.bow_fraction, 100)
    r = suboff_radius_profile(xi_bow, cfg)
    diffs = np.diff(r)
    assert np.all(diffs >= -1e-9), "Bow radius must be monotonically increasing"


def test_radius_profile_monotone_stern() -> None:
    """Radius must be non-increasing over the stern section."""
    cfg = SuboffConfig()
    xi_stern = np.linspace(1.0 - cfg.stern_fraction, 1.0, 100)
    r = suboff_radius_profile(xi_stern, cfg)
    diffs = np.diff(r)
    assert np.all(diffs <= 1e-9), "Stern radius must be monotonically decreasing"


def test_radius_profile_scalar_input() -> None:
    """suboff_radius_profile must accept a plain float as input."""
    r = suboff_radius_profile(0.5)
    assert float(r) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Hull mask (bare hull only)
# ---------------------------------------------------------------------------


def test_bare_hull_mask_shape() -> None:
    """Bare hull mask must have shape (nz, ny, nx) and bool dtype."""
    mask = suboff_hull_mask(NX, NY, NZ, NX / 2, NY / 2, NZ / 2, LENGTH, RADIUS, CPU)
    assert mask.shape == (NZ, NY, NX)
    assert mask.dtype == torch.bool


def test_bare_hull_mask_solid_cells() -> None:
    """Bare hull must produce a non-trivial solid region."""
    mask = suboff_hull_mask(NX, NY, NZ, NX / 2, NY / 2, NZ / 2, LENGTH, RADIUS, CPU)
    n_solid = int(mask.sum().item())
    assert n_solid > 0, "Bare hull mask has no solid cells"
    assert n_solid < NX * NY * NZ, "Bare hull mask fills the entire domain"


def test_bare_hull_mask_axisymmetric() -> None:
    """Hull must be symmetric about the y-axis and z-axis (top/bottom, port/stbd).

    Use odd NY/NZ so that the integer center index is the exact geometric
    centre and tensor flip is equivalent to mirroring about that centre.
    """
    ny_odd, nz_odd = 41, 41
    cy = ny_odd // 2   # = 20 → flip maps j to 40-j, |j-20| == |40-j-20|
    cz = nz_odd // 2   # = 20
    mask = suboff_hull_mask(NX, ny_odd, nz_odd, NX / 2, float(cy), float(cz), LENGTH, RADIUS, CPU)
    m = mask.cpu()
    # Check port-starboard symmetry (flip y)
    assert torch.equal(m, m.flip(1)), "Hull is not symmetric about the y-axis"
    # Check top-bottom symmetry (flip z)
    assert torch.equal(m, m.flip(0)), "Hull is not symmetric about the z-axis"


# ---------------------------------------------------------------------------
# build_suboff_mask – all variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hull_type", list(SuboffHullType))
def test_build_suboff_mask_shape(hull_type: SuboffHullType) -> None:
    mask, stats = build_suboff_mask(
        hull_type=hull_type,
        nx=NX, ny=NY, nz=NZ,
        length=LENGTH, radius=RADIUS,
    )
    assert mask.shape == (NZ, NY, NX)
    assert mask.dtype == torch.bool
    assert stats["solid_cells"] > 0
    assert stats["fluid_cells"] > 0
    assert stats["total_cells"] == NX * NY * NZ


@pytest.mark.parametrize("hull_type", list(SuboffHullType))
def test_build_suboff_mask_stats_keys(hull_type: SuboffHullType) -> None:
    """build_suboff_mask must return a stats dict with mandatory keys."""
    _, stats = build_suboff_mask(hull_type=hull_type, nx=NX, ny=NY, nz=NZ,
                                  length=LENGTH, radius=RADIUS)
    required = {
        "hull_type", "label", "L_D_ratio", "displacement_lu3",
        "wetted_area_lu2", "prismatic_coefficient",
        "solid_cells", "fluid_cells", "total_cells",
    }
    for key in required:
        assert key in stats, f"Missing key '{key}' in stats for {hull_type}"


def test_build_suboff_mask_full_has_more_solid() -> None:
    """Full model must have more solid cells than the bare hull."""
    _, s_bare = build_suboff_mask("bare_hull", nx=NX, ny=NY, nz=NZ,
                                   length=LENGTH, radius=RADIUS)
    _, s_full = build_suboff_mask("full", nx=NX, ny=NY, nz=NZ,
                                   length=LENGTH, radius=RADIUS)
    assert s_full["solid_cells"] > s_bare["solid_cells"], (
        "Full model should have more solid cells than bare hull"
    )


def test_build_suboff_mask_with_sail_more_than_bare() -> None:
    """With-sail model must have more solid cells than the bare hull."""
    _, s_bare = build_suboff_mask("bare_hull", nx=NX, ny=NY, nz=NZ,
                                   length=LENGTH, radius=RADIUS)
    _, s_sail = build_suboff_mask("with_sail", nx=NX, ny=NY, nz=NZ,
                                   length=LENGTH, radius=RADIUS)
    assert s_sail["solid_cells"] > s_bare["solid_cells"]


def test_build_suboff_mask_default_placement() -> None:
    """Default placement (cx/cy/cz = None) must yield a non-empty mask."""
    mask, stats = build_suboff_mask("bare_hull", nx=NX, ny=NY, nz=NZ)
    assert stats["solid_cells"] > 0


def test_build_suboff_mask_auto_radius() -> None:
    """When radius is None, it is derived from config.r_over_l * length."""
    cfg = SuboffConfig()
    mask, stats = build_suboff_mask(
        "bare_hull", nx=NX, ny=NY, nz=NZ,
        length=LENGTH, radius=None, config=cfg,
    )
    assert stats["radius"] == pytest.approx(cfg.r_over_l * LENGTH, rel=1e-6)
    assert stats["solid_cells"] > 0


# ---------------------------------------------------------------------------
# suboff_statistics
# ---------------------------------------------------------------------------


def test_suboff_statistics_supports_numpy_without_legacy_trapz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Statistics integration must work with NumPy 2.x, which removed trapz."""
    monkeypatch.delattr(np, "trapz", raising=False)

    stats = suboff_statistics("bare_hull", 100.0, 10.0)

    assert stats["displacement_lu3"] > 0.0
    assert stats["wetted_area_lu2"] > 0.0


@pytest.mark.parametrize("hull_type", list(SuboffHullType))
def test_suboff_statistics_keys(hull_type: SuboffHullType) -> None:
    stats = suboff_statistics(hull_type, LENGTH, RADIUS)
    for key in ("hull_type", "label", "L_D_ratio", "r_over_l",
                "bow_fraction", "stern_fraction",
                "displacement_lu3", "wetted_area_lu2", "prismatic_coefficient"):
        assert key in stats, f"Missing key '{key}'"


def test_suboff_statistics_l_d_ratio() -> None:
    """L/D must match length / (2 * radius) = 8.57."""
    # radius = L/(2*8.57) gives diameter = L/8.57 and L/D = 8.57
    r = 100.0 / (2.0 * 8.57)
    stats = suboff_statistics(SuboffHullType.BARE_HULL, 100.0, r)
    assert stats["L_D_ratio"] == pytest.approx(8.57, rel=1e-3)


def test_suboff_statistics_displacement_positive() -> None:
    """Displacement volume must be positive."""
    stats = suboff_statistics("bare_hull", 100.0, 10.0)
    assert stats["displacement_lu3"] > 0


def test_suboff_statistics_prismatic_coefficient_range() -> None:
    """Prismatic coefficient Cp must be in (0, 1)."""
    stats = suboff_statistics("bare_hull", 100.0, 10.0)
    assert 0.0 < stats["prismatic_coefficient"] < 1.0


def test_suboff_statistics_wetted_area_positive() -> None:
    """Wetted surface area must be positive."""
    stats = suboff_statistics("bare_hull", 100.0, 10.0)
    assert stats["wetted_area_lu2"] > 0


# ---------------------------------------------------------------------------
# generate_suboff_previews
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hull_type_str", ["bare_hull", "with_sail", "full"])
def test_generate_suboff_previews_figure(hull_type_str: str) -> None:
    """generate_suboff_previews must return a Figure with 3 axes."""
    import matplotlib.pyplot as plt

    fig = generate_suboff_previews(hull_type_str, length=100.0)
    try:
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 3
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# export_suboff_stl
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hull_type_str", ["bare_hull", "with_sail", "full"])
def test_export_suboff_stl_creates_file(hull_type_str: str, tmp_path: Path) -> None:
    """export_suboff_stl must write a non-empty ASCII STL file."""
    out = export_suboff_stl(
        hull_type_str,
        length=100.0,
        n_axial=20,
        n_circ=16,
        output_path=tmp_path / f"suboff_{hull_type_str}.stl",
    )
    assert out.exists(), "STL file was not created"
    content = out.read_text()
    assert content.startswith("solid suboff_"), "STL header mismatch"
    assert "facet normal" in content, "No facets in STL output"
    assert "endsolid" in content, "STL footer missing"


def test_export_suboff_stl_file_grows_with_full() -> None:
    """Full model STL must be larger than bare hull STL (more triangles)."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bare_stl = export_suboff_stl("bare_hull", length=100.0, n_axial=20, n_circ=16,
                                      output_path=td / "bare.stl")
        full_stl = export_suboff_stl("full", length=100.0, n_axial=20, n_circ=16,
                                      output_path=td / "full.stl")
        assert full_stl.stat().st_size > bare_stl.stat().st_size, (
            "Full model STL should be larger than bare hull STL"
        )


# ---------------------------------------------------------------------------
# Integration: mask → Cp consistency
# ---------------------------------------------------------------------------


def test_cp_coarse_grid_approximate() -> None:
    """Prismatic coefficient from statistics should be physically reasonable."""
    cfg = SuboffConfig()
    stats = suboff_statistics("bare_hull", LENGTH, RADIUS, cfg)
    cp = stats["prismatic_coefficient"]
    # SUBOFF-like hull: Cp ≈ 0.55–0.75
    assert 0.45 < cp < 0.85, f"Cp={cp:.4f} is outside expected range"
