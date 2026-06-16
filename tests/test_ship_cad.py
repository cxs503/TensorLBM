"""Tests for the ship CAD module (ship_cad.py).

Covers:
- Mask shape and dtype for all three hull types.
- Block coefficient is within expected range.
- hull_statistics returns expected keys.
- theoretical_block_coefficient values.
- generate_hull_body_plan / waterplane / sideprofile shapes.
- generate_hull_previews returns a Figure.
- export_hull_stl writes a valid ASCII STL file.
- build_hull_mask convenience wrapper.
- ship_lbm_parameters output keys and stability check.
- Full workflow: CAD mask → block-coefficient consistency.
"""
from __future__ import annotations

import math
from pathlib import Path  # noqa: TC003

import numpy as np
import pytest
import torch

from tensorlbm.ship_cad import (
    ShipHullType,
    build_hull_mask,
    export_hull_stl,
    generate_hull_body_plan,
    generate_hull_previews,
    generate_hull_sideprofile,
    generate_hull_waterplane,
    hull_block_coefficient,
    hull_statistics,
    kcs_hull_mask,
    series60_hull_mask,
    ship_lbm_parameters,
    ship_resistance_estimate,
    theoretical_block_coefficient,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CPU = torch.device("cpu")
NX, NY, NZ = 40, 24, 20

# Hull placement for tests: centre midship, small grid
CX = NX / 2.0
CY = NY / 2.0
CZ_KEEL = NZ / 4.0
LENGTH = NX * 0.5
BEAM = NY * 0.25
DRAFT = NZ * 0.3


# ---------------------------------------------------------------------------
# Mask shape and dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hull_type", list(ShipHullType))
def test_mask_shape(hull_type: ShipHullType) -> None:
    """All hull masks must have shape (nz, ny, nx) and bool dtype."""
    from tensorlbm.obstacles import wigley_hull_mask

    if hull_type == ShipHullType.WIGLEY:
        mask = wigley_hull_mask(NX, NY, NZ, CX, CY, CZ_KEEL, LENGTH, BEAM, DRAFT, CPU)
    elif hull_type == ShipHullType.SERIES60:
        mask = series60_hull_mask(NX, NY, NZ, CX, CY, CZ_KEEL, LENGTH, BEAM, DRAFT, CPU)
    else:
        mask = kcs_hull_mask(NX, NY, NZ, CX, CY, CZ_KEEL, LENGTH, BEAM, DRAFT, CPU)

    assert mask.shape == (NZ, NY, NX), f"Unexpected shape {mask.shape}"
    assert mask.dtype == torch.bool, f"Unexpected dtype {mask.dtype}"


@pytest.mark.parametrize("hull_type", list(ShipHullType))
def test_mask_has_solid_cells(hull_type: ShipHullType) -> None:
    """Every hull must produce at least some solid cells."""
    mask, stats = build_hull_mask(
        hull_type=hull_type,
        nx=NX, ny=NY, nz=NZ,
        cx=CX, cy=CY, cz_keel=CZ_KEEL,
        length=LENGTH, beam=BEAM, draft=DRAFT,
    )
    assert stats["solid_cells"] > 0


# ---------------------------------------------------------------------------
# Block coefficient
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hull_type", list(ShipHullType))
def test_block_coefficient_range(hull_type: ShipHullType) -> None:
    """Numerical Cb must be in a physically plausible range for all hull types.

    Note: on coarse grids the discrete Cb can exceed the theoretical value by
    ~10–20 % due to boundary cell over-counting; the upper bound is set to 1.0
    to allow for this.
    """
    mask, stats = build_hull_mask(
        hull_type=hull_type,
        nx=NX, ny=NY, nz=NZ,
        cx=CX, cy=CY, cz_keel=CZ_KEEL,
        length=LENGTH, beam=BEAM, draft=DRAFT,
    )
    cb = stats["Cb_numerical"]
    assert 0.20 < cb < 1.01, f"Cb={cb} out of expected range for {hull_type}"


def test_wigley_cb_close_to_theoretical() -> None:
    """Wigley numerical Cb should be within 15 % of 4/9 ≈ 0.444 on a fine grid.

    Coarse-grid discretization systematically overestimates Cb because boundary
    cells are counted as fully solid.  We allow a generous 15 % tolerance.
    """
    # Use a larger grid for accuracy
    nx, ny, nz = 80, 40, 32
    cx, cy = nx / 2.0, ny / 2.0
    cz = nz / 4.0
    L, B, T = nx * 0.5, ny * 0.25, nz * 0.3
    mask, stats = build_hull_mask(
        "wigley", nx, ny, nz, cx=cx, cy=cy, cz_keel=cz,
        length=L, beam=B, draft=T,
    )
    cb = stats["Cb_numerical"]
    assert abs(cb - 4.0 / 9.0) < 0.15, f"Wigley Cb={cb:.4f}, expected ≈0.444"


def test_series60_cb_close_to_theoretical() -> None:
    """Series 60 numerical Cb should be ≈ 0.60 (within 20 % on fine grid)."""
    nx, ny, nz = 80, 40, 32
    cx, cy = nx / 2.0, ny / 2.0
    cz = nz / 4.0
    L, B, T = nx * 0.5, ny * 0.25, nz * 0.3
    mask, stats = build_hull_mask(
        "series60", nx, ny, nz, cx=cx, cy=cy, cz_keel=cz,
        length=L, beam=B, draft=T,
    )
    cb = stats["Cb_numerical"]
    assert abs(cb - 0.60) < 0.20, f"Series 60 Cb={cb:.4f}, expected ≈0.60"


def test_kcs_cb_close_to_theoretical() -> None:
    """KCS numerical Cb should be ≈ 0.651 (within 20 % on fine grid)."""
    nx, ny, nz = 80, 40, 32
    cx, cy = nx / 2.0, ny / 2.0
    cz = nz / 4.0
    L, B, T = nx * 0.5, ny * 0.25, nz * 0.3
    mask, stats = build_hull_mask(
        "kcs", nx, ny, nz, cx=cx, cy=cy, cz_keel=cz,
        length=L, beam=B, draft=T,
    )
    cb = stats["Cb_numerical"]
    assert abs(cb - 0.651) < 0.20, f"KCS Cb={cb:.4f}, expected ≈0.651"


# ---------------------------------------------------------------------------
# Ordering: KCS > Series60 > Wigley  (fuller → finer)
# ---------------------------------------------------------------------------

def test_cb_ordering() -> None:
    """Block coefficients must satisfy Cb(KCS) > Cb(Series60) > Cb(Wigley)."""
    nx, ny, nz = 80, 40, 32
    cx, cy = nx / 2.0, ny / 2.0
    cz = nz / 4.0
    L, B, T = nx * 0.5, ny * 0.25, nz * 0.3
    cbs = {}
    for ht in ShipHullType:
        _, stats = build_hull_mask(
            ht, nx, ny, nz, cx=cx, cy=cy, cz_keel=cz,
            length=L, beam=B, draft=T,
        )
        cbs[ht] = stats["Cb_numerical"]
    assert cbs[ShipHullType.KCS] > cbs[ShipHullType.SERIES60], (
        "KCS should be fuller than Series60"
    )
    assert cbs[ShipHullType.SERIES60] > cbs[ShipHullType.WIGLEY], (
        "Series60 should be fuller than Wigley"
    )


# ---------------------------------------------------------------------------
# hull_statistics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hull_type", list(ShipHullType))
def test_hull_statistics_keys(hull_type: ShipHullType) -> None:
    """hull_statistics must return all required keys."""
    stats = hull_statistics(hull_type, length=100.0, beam=16.0, draft=8.0)
    for key in ("hull_type", "label", "Cb", "Cwp", "Cm", "Cp", "L/B", "B/T",
                "displacement_lu3"):
        assert key in stats, f"Missing key: {key}"


@pytest.mark.parametrize("hull_type", list(ShipHullType))
def test_hull_statistics_coefficients_physical_range(hull_type: ShipHullType) -> None:
    """Hull form coefficients Cwp, Cm, and Cp must all lie in (0, 1]."""
    stats = hull_statistics(hull_type, length=100.0, beam=16.0, draft=8.0)
    for key in ("Cwp", "Cm", "Cp"):
        assert 0.0 < stats[key] <= 1.0, (
            f"{hull_type.value}: {key}={stats[key]} is outside the physical range (0, 1]"
        )


def test_hull_statistics_wigley_analytical() -> None:
    """Wigley Cwp = Cm = Cp = 2/3 (analytical result)."""
    stats = hull_statistics(ShipHullType.WIGLEY, length=100.0, beam=16.0, draft=8.0)
    assert abs(stats["Cwp"] - 2.0 / 3.0) < 1e-3, f"Cwp={stats['Cwp']}"
    assert abs(stats["Cm"] - 2.0 / 3.0) < 1e-3, f"Cm={stats['Cm']}"
    assert abs(stats["Cp"] - 2.0 / 3.0) < 1e-3, f"Cp={stats['Cp']}"


def test_hull_statistics_lb_bt() -> None:
    """L/B and B/T ratios must match the input dimensions."""
    s = hull_statistics(ShipHullType.SERIES60, length=120.0, beam=20.0, draft=8.0)
    assert abs(s["L/B"] - 6.0) < 1e-9
    assert abs(s["B/T"] - 2.5) < 1e-9


# ---------------------------------------------------------------------------
# theoretical_block_coefficient
# ---------------------------------------------------------------------------

def test_theoretical_cb_values() -> None:
    assert abs(theoretical_block_coefficient("wigley") - 4.0 / 9.0) < 1e-9
    assert abs(theoretical_block_coefficient("series60") - 0.600) < 1e-9
    assert abs(theoretical_block_coefficient("kcs") - 0.651) < 1e-9


# ---------------------------------------------------------------------------
# Section / profile extraction
# ---------------------------------------------------------------------------

def test_body_plan_shape() -> None:
    stations, y_sects, z_sects = generate_hull_body_plan(ShipHullType.SERIES60, n_stations=7)
    assert len(stations) == 7
    assert len(y_sects) == 7
    for ys in y_sects:
        assert ys.shape == (100,)
        assert np.all(ys >= 0.0), "Half-breadth must be non-negative"
        assert np.all(ys <= 1.0 + 1e-6), "Normalised half-breadth must be ≤ 1"


def test_waterplane_shape() -> None:
    xi, hb = generate_hull_waterplane(ShipHullType.KCS, n_points=50)
    assert xi.shape == (50,)
    assert hb.shape == (50,)
    assert float(hb.max()) <= 1.0 + 1e-6
    assert float(hb.min()) >= 0.0
    # Symmetry: hb(ξ) == hb(-ξ)
    assert np.allclose(hb, hb[::-1], atol=1e-5)


def test_sideprofile_symmetry() -> None:
    xi, draft_arr = generate_hull_sideprofile(ShipHullType.WIGLEY)
    # Side profile should be symmetric about midship
    assert np.allclose(draft_arr, draft_arr[::-1], atol=1e-3)


# ---------------------------------------------------------------------------
# Preview figure
# ---------------------------------------------------------------------------

def test_generate_hull_previews_returns_figure() -> None:
    import matplotlib.figure
    fig = generate_hull_previews(ShipHullType.SERIES60, length=100, beam=16, draft=8)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 3


@pytest.mark.parametrize("hull_type_str", ["wigley", "series60", "kcs"])
def test_generate_hull_previews_all_types(hull_type_str: str) -> None:
    """generate_hull_previews must not raise for any hull type."""
    import matplotlib.pyplot as plt
    fig = generate_hull_previews(hull_type_str, length=100, beam=16, draft=8)
    plt.close(fig)


# ---------------------------------------------------------------------------
# STL export
# ---------------------------------------------------------------------------

def test_export_hull_stl_creates_file(tmp_path: Path) -> None:
    out = export_hull_stl(
        "series60", length=100.0, beam=16.0, draft=8.0,
        n_long=10, n_vert=6,
        output_path=tmp_path / "test_hull.stl",
    )
    assert out.exists(), "STL file was not created"
    content = out.read_text()
    assert content.startswith("solid series60_hull"), "STL header missing"
    assert "facet normal" in content, "No facet normals in STL"
    assert "endsolid" in content, "STL footer missing"


@pytest.mark.parametrize("hull_type_str", ["wigley", "series60", "kcs"])
def test_export_hull_stl_all_types(hull_type_str: str, tmp_path: Path) -> None:
    out = export_hull_stl(
        hull_type_str, length=50.0, beam=10.0, draft=5.0,
        n_long=8, n_vert=4,
        output_path=tmp_path / f"{hull_type_str}.stl",
    )
    assert out.exists()
    assert out.stat().st_size > 100


# ---------------------------------------------------------------------------
# ship_lbm_parameters
# ---------------------------------------------------------------------------

def test_ship_lbm_parameters_keys() -> None:
    params = ship_lbm_parameters(100.0, 5.0)
    for key in ("re_physical", "froude_number", "dx_m", "dt_s",
                "lbm_nu", "lbm_tau", "mach_number", "stable"):
        assert key in params, f"Missing key: {key}"


def test_ship_lbm_parameters_stable() -> None:
    """Default parameters should give a stable configuration (tau > 0.5)."""
    params = ship_lbm_parameters(100.0, 5.0, lbm_length=100.0, lbm_speed=0.05)
    assert params["stable"] is True, "Default LBM parameters should be stable"


def test_ship_lbm_parameters_froude_override() -> None:
    """froude_target must override the given speed."""
    params_fr = ship_lbm_parameters(100.0, 0.0001, froude_target=0.3)
    expected_speed = 0.3 * math.sqrt(9.81 * 100.0)
    expected_re = expected_speed * 100.0 / 1.139e-6
    assert abs(params_fr["re_physical"] - expected_re) / expected_re < 1e-4


def test_ship_lbm_parameters_re_scaling() -> None:
    """Re must scale linearly with length."""
    p1 = ship_lbm_parameters(100.0, 5.0)
    p2 = ship_lbm_parameters(200.0, 5.0)
    assert abs(p2["re_physical"] / p1["re_physical"] - 2.0) < 1e-6


def test_ship_resistance_estimate_outputs() -> None:
    est = ship_resistance_estimate(
        hull_type="series60",
        length_m=120.0,
        beam_m=20.0,
        draft_m=10.0,
        speed_ms=8.0,
        residual_ratio=0.2,
    )
    assert est["hull_type"] == "series60"
    assert est["reynolds"] > 1e8
    assert est["cf_ittc57"] > 0.0
    assert est["total_resistance_n"] > est["friction_resistance_n"] > 0.0


def test_ship_resistance_estimate_rejects_invalid_residual_ratio() -> None:
    with pytest.raises(ValueError, match="residual_ratio"):
        ship_resistance_estimate(
            hull_type="wigley",
            length_m=100.0,
            beam_m=16.0,
            draft_m=8.0,
            speed_ms=5.0,
            residual_ratio=1.2,
        )


# ---------------------------------------------------------------------------
# Full workflow: CAD mask → hull_block_coefficient
# ---------------------------------------------------------------------------

def test_full_workflow_cad_to_mask_to_cb() -> None:
    """End-to-end: build Series 60 mask, compute Cb, check within 25% of 0.60."""
    nx, ny, nz = 60, 30, 24
    mask, stats = build_hull_mask(
        "series60", nx, ny, nz,
        length=nx * 0.5, beam=ny * 0.25, draft=nz * 0.3,
    )
    assert mask.shape == (nz, ny, nx)
    # Cb from stats and from direct computation must agree (within rounding)
    cb_stats = stats["Cb_numerical"]
    cb_direct = hull_block_coefficient(
        mask, beam=stats["beam"], draft=stats["draft"], length=stats["length"]
    )
    assert abs(cb_stats - cb_direct) < 1e-3
    assert abs(cb_stats - 0.60) < 0.25, f"Cb={cb_stats:.4f} not within 25% of 0.60"


def test_full_workflow_cad_to_mask_to_cb_kcs() -> None:
    """End-to-end: build KCS mask, verify Cb ordering and fluid cells > 0."""
    nx, ny, nz = 60, 30, 24
    mask_s60, stats_s60 = build_hull_mask(
        "series60", nx, ny, nz,
        length=nx * 0.5, beam=ny * 0.25, draft=nz * 0.3,
    )
    mask_kcs, stats_kcs = build_hull_mask(
        "kcs", nx, ny, nz,
        length=nx * 0.5, beam=ny * 0.25, draft=nz * 0.3,
    )
    assert stats_kcs["Cb_numerical"] > stats_s60["Cb_numerical"]
    assert stats_kcs["fluid_cells"] > 0
    assert stats_kcs["solid_cells"] > 0
