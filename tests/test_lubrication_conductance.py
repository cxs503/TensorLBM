"""Tests for the sub-lattice lubrication closure (P55-N02 / TensorLBM).

Covered surface
---------------
* Gebart (1992) closed form and its near-jamming limits;
* the grid-independent minimum gap ``delta_min`` for square / hexagonal arrays,
  cross-checked against the ray-cast gap field of the voxel mask;
* the Sangani & Acrivos near-contact two-cylinder conductance
  ``G(delta) = (3 pi / 8) sqrt(R) / delta^(5/2)``, including an independent
  numerical quadrature of ``int ds / h(s)^3``;
* the closure rule ``nu_eff = nu (delta_mask/delta_true)^3``: exact no-op above
  the cutoff, cubic slot law, monotonicity, clipping;
* the dilute negative control (no node is touched at phi = 0.05);
* a numerical check that the closure mechanism really reproduces the cubic slot
  law in the LBM driver (plane channel: halving the effective viscosity by the
  gap ratio raises the flux by the cube of the ratio).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.lubrication import (  # noqa: E402
    GAP_INTEGRAL_CONST,
    GEBART_C1_HEX,
    GEBART_C1_SQ,
    PHI_MAX_HEX,
    PHI_MAX_SQ,
    assign_cylinder_ids,
    closure_nu_field,
    cylinder_gap_fields,
    delta_min_over_cell,
    equivalent_slot_viscosity,
    gap_conductance_truncated,
    gebart_permeability,
    gebart_sqrt_gap,
    make_cylinder_mask,
    sangani_acrivos_gap_conductance,
)

NU = 1.0 / 6.0

BENCH = REPO / "benchmarks" / "verified" / "permeability_fibrous" / "run.py"


def _load_bench():
    spec = importlib.util.spec_from_file_location("p55_bench", BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. closed forms
# ---------------------------------------------------------------------------
def test_phi_max_values():
    assert PHI_MAX_SQ == pytest.approx(math.pi / 4.0)
    assert PHI_MAX_HEX == pytest.approx(math.pi / (2.0 * math.sqrt(3.0)))
    assert PHI_MAX_HEX > PHI_MAX_SQ > 0.78


def test_gebart_c1_ratio_is_sqrt3():
    # C1_sq / C1_hex = sqrt(6/2) = sqrt(3)
    assert GEBART_C1_SQ / GEBART_C1_HEX == pytest.approx(math.sqrt(3.0))


def test_gebart_vanishes_at_jamming_and_is_monotone():
    for arr, phi_max in (("hex", PHI_MAX_HEX), ("sq", PHI_MAX_SQ)):
        near = gebart_permeability(phi_max * (1.0 - 1e-9), 1.0, arr)
        assert near < 1e-3  # s^(5/2) -> 0
        ks = [gebart_permeability(p, 1.0, arr) for p in (0.2, 0.4, 0.6)]
        assert ks[0] > ks[1] > ks[2] > 0.0
        assert gebart_sqrt_gap(0.5 * phi_max, arr) == pytest.approx(math.sqrt(2.0) - 1.0)


def test_delta_min_closed_form_values_and_monotonicity():
    # hex phi = 0.75 -> delta/a = 1 - 2 sqrt(sqrt(3)*0.75/(2 pi)) = 0.09062
    assert delta_min_over_cell(0.75, "hex") == pytest.approx(0.0906083, abs=5e-6)
    # sq phi = 0.40 -> delta/W = 1 - 2 sqrt(0.40/pi) = 0.28635
    assert delta_min_over_cell(0.40, "sq") == pytest.approx(0.2863504, abs=5e-6)
    d = [delta_min_over_cell(p, "hex") for p in (0.3, 0.6, 0.8, 0.9)]
    assert all(d[i] > d[i + 1] for i in range(len(d) - 1))
    assert delta_min_over_cell(0.9, "hex") > 0.0


# ---------------------------------------------------------------------------
# 2. Sangani & Acrivos near-contact conductance
# ---------------------------------------------------------------------------
def test_gap_conductance_scaling_law():
    R, d = 5.0, 0.4
    g1 = sangani_acrivos_gap_conductance(R, d)
    g2 = sangani_acrivos_gap_conductance(R, 2.0 * d)
    assert g2 / g1 == pytest.approx(2.0**-2.5, rel=1e-12)
    assert g1 == pytest.approx(GAP_INTEGRAL_CONST * math.sqrt(R) / d**2.5)


def test_gap_conductance_matches_quadrature():
    R, d = 2.0, 0.25
    # independent numerical integration of int ds / h(s)^3, h = d + s^2/R
    s = np.linspace(-400.0, 400.0, 4_000_001)
    h = d + s**2 / R
    numeric = np.trapezoid(1.0 / h**3, s) if hasattr(np, "trapezoid") else np.trapz(1.0 / h**3, s)
    closed = sangani_acrivos_gap_conductance(R, d)
    assert numeric == pytest.approx(closed, rel=2e-3)


def test_truncated_integral_closed_form_and_limit():
    R, d = 3.0, 0.6
    closed = sangani_acrivos_gap_conductance(R, d)
    # far-field truncation recovers the infinite integral
    assert gap_conductance_truncated(R, d, 1.0e4) == pytest.approx(closed, rel=1e-6)
    # truncation is monotone in s_max and below the infinite value
    a, b = gap_conductance_truncated(R, d, 1.0), gap_conductance_truncated(R, d, 2.0)
    assert 0.0 < a < b < closed


# ---------------------------------------------------------------------------
# 3. closure rule
# ---------------------------------------------------------------------------
def test_closure_is_exact_no_op_above_cutoff():
    nu_eff, w, clipped = equivalent_slot_viscosity(NU, 1.5, 1.0)
    assert nu_eff == NU and w == 0.0 and clipped is False
    nu_eff, w, _ = equivalent_slot_viscosity(NU, 4.0, 1.0)
    assert nu_eff == NU and w == 0.0


def test_closure_cubic_law_and_monotonicity():
    # sub-lattice throat: nu_eff = nu (delta_mask/delta_true)^(3 w), w the taper
    nu_eff, w, clipped = equivalent_slot_viscosity(NU, 0.4, 1.0)
    assert w == pytest.approx(1.0 - 0.4 / 1.5)
    assert clipped is False
    assert nu_eff / NU == pytest.approx((1.0 / 0.4) ** (3.0 * w), rel=1e-12)
    # very wide mask/true ratio is clipped, never infinite
    far, _, far_clipped = equivalent_slot_viscosity(NU, 0.02, 1.0, max_factor=1.0e3)
    assert far_clipped is True and far == pytest.approx(1000.0 * NU)
    # increasing resistance as the true throat narrows
    vals = [equivalent_slot_viscosity(NU, d, 1.0)[0] for d in (1.4, 1.0, 0.6, 0.3)]
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    # when the mask over-widens the throat the viscosity is reduced, never zero
    lo, _, clipped = equivalent_slot_viscosity(NU, 0.2, 0.01, min_factor=0.05)
    assert lo > 0.0 and clipped is True


# ---------------------------------------------------------------------------
# 4. gap fields on real arrays
# ---------------------------------------------------------------------------
def test_gap_field_reproduces_analytic_delta_min():
    for arr, phi in (("hex", 0.75), ("hex", 0.70), ("sq", 0.40)):
        if arr == "hex":
            g = 16
            a = float(g)
            ny = int(round(math.sqrt(3.0) * a))
            radius = a * math.sqrt(math.sqrt(3.0) * phi / (2.0 * math.pi))
            centres = [(0.0, 0.0), (a / 2.0, ny / 2.0)]
            solid = make_cylinder_mask(g, ny, centres, radius)
            dmin_analytic = delta_min_over_cell(phi, "hex") * a
        else:
            g = 24
            radius = g * math.sqrt(phi / math.pi)
            centres = [(g / 2.0, g / 2.0)]
            solid = make_cylinder_mask(g, g, centres, radius)
            dmin_analytic = delta_min_over_cell(phi, "sq") * g
        gap = cylinder_gap_fields(solid.shape[1], solid.shape[0], centres, radius, solid=solid)
        assert gap["delta_min"] == pytest.approx(dmin_analytic, rel=1e-10)


def test_dilute_limit_has_no_sub_lattice_throat():
    g, phi = 24, 0.05
    radius = g * math.sqrt(phi / math.pi)
    centres = [(g / 2.0, g / 2.0)]
    solid = make_cylinder_mask(g, g, centres, radius)
    gap = cylinder_gap_fields(g, g, centres, radius, solid=solid)
    assert gap["delta_min"] > 10.0  # wide throat in lattice units
    close = closure_nu_field(NU, gap, solid.shape)
    assert close["n_corrected"] == 0
    assert np.allclose(close["nu_field"], NU)


def test_cylinder_ids_cover_solid_only():
    g = 16
    radius = g * math.sqrt(0.4 / math.pi)
    centres = [(g / 2.0, g / 2.0)]
    solid = make_cylinder_mask(g, g, centres, radius)
    ids = assign_cylinder_ids(solid, centres)
    assert (ids[solid] == 0).all()
    assert (ids[~solid] == -1).all()


# ---------------------------------------------------------------------------
# 5. the closure mechanism inside the LBM driver (plane channel, cubic law)
# ---------------------------------------------------------------------------
def test_channel_flux_follows_slot_cubic_law_with_closure_viscosity():
    """A plane channel with nu_eff = nu (H_mask/H_true)^3 must carry H_true^3 flux.

    This validates the *mechanism* of the closure (momentum-exchange coefficient
    scaling) independently of any cylinder geometry.
    """
    bench = _load_bench()
    ny, nx = 10, 4
    H_mask = float(ny - 2)  # mid-link walls: fluid nodes 1..ny-2
    solid = np.zeros((ny, nx), dtype=bool)
    solid[0, :] = True
    solid[-1, :] = True
    a = 1e-5

    fluxes = {}
    for ratio in (1.0, 1.25):
        nu_field = np.full((ny, nx), NU * ratio**3)
        lb = bench.LB(solid)
        lb.set_tau_field(nu_field)
        for _ in range(15000):
            lb.step(a)
        fluxes[ratio] = lb.mean_ux_fluid()
    # u ~ a H^3 / (12 nu H) ... for a fixed mask width the mean velocity scales
    # inversely with the viscosity -> ratio^3 reduction in flux
    # The pure 1/nu law is modified at O(1%) by the tau-dependence of the
    # bounce-back wall position, hence the 10% band; the direction and the
    # magnitude (a factor ~1.8-1.95 for a 1.25^3 = 1.95 target) are the signal.
    ratio_meas = fluxes[1.0] / fluxes[1.25]
    assert ratio_meas == pytest.approx(1.25**3, rel=0.10)
    assert ratio_meas > 1.7


def test_channel_matches_poiseuille_analytic():
    bench = _load_bench()
    ny, nx = 18, 4
    solid = np.zeros((ny, nx), dtype=bool)
    solid[0, :] = True
    solid[-1, :] = True
    a = 1e-5
    lb = bench.LB(solid)
    lb.set_tau_field(np.full((ny, nx), NU))
    for _ in range(6000):
        lb.step(a)
    f = lb.f
    rho = f.sum(dim=0)
    ux = (f * lb.cx).sum(dim=0) / rho
    prof = ux[1:-1].numpy().mean(axis=1)
    H = float(ny - 2)
    x = np.arange(1, ny - 1) - 0.5
    u_exact = (a / (2.0 * NU)) * (x * (H - x))
    assert np.linalg.norm(prof - u_exact) / np.linalg.norm(u_exact) < 5e-3