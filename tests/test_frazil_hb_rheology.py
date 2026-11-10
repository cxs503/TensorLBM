"""Tests for the frazil-ice Herschel-Bulkley rheology surface (P519-N04).

Covers

* the concentration -> yield stress closure (gel threshold, anchor, range checks),
* the composition with the W9-1215 physical root (``K = rho nu_w mu_r``, ``n = n(phi)``),
* the dimensional algebra and the SI -> kinematic round trips,
* the analytic shear-rate distributions (consistency with the velocity profiles),
* the similarity construction of the lattice case and the SI <-> lattice mapping,
* a small lattice-Boltzmann smoke run of both the yield-stress and the Newtonian
  degeneration cases.

The full gate-level verification lives in
``benchmarks/frazil_hb_flow_bench.py`` -> ``artifacts/frazil_hb_flow.json``; the
tests here are fast (a few seconds each) and use a deliberately coarse grid.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.frazil_hb_rheology import (
    PHI_GEL,
    PHI_YIELD_REF,
    TAU_Y_REF_PA,
    FrazilHBError,
    dimensional_consistency,
    frazil_hb_state,
    frazil_yield_stress,
    hb_couette_adverse_shear_rate,
    hb_plane_shear_rate,
    lattice_mapping,
    mixture_density,
    physical_flow_case,
    scaling_consistency,
    unit_exponents,
    verification_case,
    yield_fraction,
)
from tensorlbm.frazil_rheology import frazil_consistency_index, frazil_flow_index
from tensorlbm.herschel_bulkley import (
    hb_couette_adverse_profile,
    hb_couette_adverse_wall_velocity,
    hb_plane_velocity_profile,
    validate_rheology,
)
from tensorlbm.frazil_hb_lbm import (
    channel_gap,
    run_frazil_hb_channel,
    shear_rate_channel_consistent,
)
from tensorlbm.herschel_bulkley_lbm import run_hb_channel


# ---------------------------------------------------------------------------
# Yield-stress closure
# ---------------------------------------------------------------------------


def test_yield_stress_is_zero_below_the_gel_point():
    for phi in (0.0, 0.05, 0.1, 0.19, PHI_GEL):
        assert frazil_yield_stress(phi) == 0.0


def test_yield_stress_grows_above_the_gel_point():
    prev = 0.0
    for phi in (0.21, 0.25, 0.30, 0.40, 0.50, 0.55):
        value = frazil_yield_stress(phi)
        assert value > prev
        prev = value
    # power law with p = 2.5 from the gel point
    s = (0.40 - PHI_GEL) / (PHI_YIELD_REF - PHI_GEL)
    assert frazil_yield_stress(0.40) == pytest.approx(TAU_Y_REF_PA * s**2.5, rel=1e-12)


def test_yield_stress_anchor_and_limits():
    assert frazil_yield_stress(PHI_YIELD_REF) == pytest.approx(TAU_Y_REF_PA, rel=1e-12)
    for bad in (-0.1, 0.60, 0.75, float("nan")):
        with pytest.raises(FrazilHBError):
            frazil_yield_stress(bad)
    with pytest.raises(FrazilHBError):
        frazil_yield_stress(0.4, exponent=0.0)
    with pytest.raises(FrazilHBError):
        frazil_yield_stress(0.4, phi_gel=0.7)


def test_mixture_density_bounds_and_monotonicity():
    assert mixture_density(0.0) == pytest.approx(999.8, rel=1e-12)
    assert mixture_density(0.5999) < mixture_density(0.3)  # ice is lighter than water
    with pytest.raises(FrazilHBError):
        mixture_density(0.60)


# ---------------------------------------------------------------------------
# Composition with the W9-1215 physical root
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phi", [0.0, 0.15, 0.30, 0.45, 0.55])
def test_state_uses_the_w9_1215_root(phi):
    state = frazil_hb_state(phi)
    mu_r = float(frazil_consistency_index(phi, mu_water=1.0))
    n = float(frazil_flow_index(phi))
    assert state.mu_r == pytest.approx(mu_r, rel=1e-12)
    assert state.n == pytest.approx(n, rel=1e-12)
    assert state.K == pytest.approx(state.rho * 1.792e-6 * mu_r, rel=1e-12)
    assert state.tau_y_kin == pytest.approx(state.tau_y / state.rho, rel=1e-12)
    assert state.K_kin == pytest.approx(state.K / state.rho, rel=1e-12)


def test_clear_water_is_newtonian():
    state = frazil_hb_state(0.0)
    assert state.tau_y == 0.0
    assert state.n == pytest.approx(1.0, rel=1e-12)
    assert state.mu_r == pytest.approx(1.0, rel=1e-12)


def test_state_rejects_out_of_range_phi():
    with pytest.raises(FrazilHBError):
        frazil_hb_state(0.61)
    with pytest.raises(FrazilHBError):
        frazil_hb_state(-0.01)
    with pytest.raises(FrazilHBError):
        frazil_hb_state(0.4, nu_water=0.0)


# ---------------------------------------------------------------------------
# Dimensional analysis
# ---------------------------------------------------------------------------


def test_unit_exponents_algebra():
    n = 0.7
    assert unit_exponents("tau_kin") == (0.0, 2.0, -2.0)
    assert unit_exponents("rho") == (1.0, -3.0, 0.0)
    # dynamic K = rho * kinematic K  ->  exponents add exactly
    k_dyn = unit_exponents("K", n)
    k_kin = unit_exponents("K_kin", n)
    rho = unit_exponents("rho")
    assert [a + b for a, b in zip(rho, k_kin)] == pytest.approx(list(k_dyn))
    assert k_kin[0] == 0.0 and k_kin[1] == 2.0
    assert k_kin[2] == pytest.approx(-2.0 + n)
    # dynamic tau = rho * kinematic tau
    assert [a + b for a, b in zip(rho, unit_exponents("tau_kin"))] == pytest.approx(
        list(unit_exponents("tau"))
    )
    with pytest.raises(FrazilHBError):
        unit_exponents("nonsense")


@pytest.mark.parametrize("phi", [0.0, 0.25, 0.40, 0.55])
def test_dimensional_consistency_passes(phi):
    state = frazil_hb_state(phi)
    report = dimensional_consistency(state)
    assert report["passed"], report["checks"]
    assert report["worst_residual"] < 1e-12
    for name, check in report["checks"].items():
        assert check["passed"], (name, check)


# ---------------------------------------------------------------------------
# Analytic shear-rate distributions
# ---------------------------------------------------------------------------


def test_plane_shear_rate_is_the_derivative_of_the_velocity_profile():
    rheo = validate_rheology(0.4 * 1.0e-4, 1.0, 0.7)
    H, tau_w = 64.0, 1.0e-4
    y = torch.linspace(-0.499 * H, 0.499 * H, 4001, dtype=torch.float64)
    u = hb_plane_velocity_profile(y, H, tau_w, rheo)
    du = torch.gradient(u, spacing=(y,))[0]
    g = hb_plane_shear_rate(y, H, tau_w, rheo)
    # away from the yield surface (where du/dy is non-smooth) the two agree
    y_c = 0.5 * H * rheo.tau_y / tau_w
    mask = torch.abs(y).abs() > 1.5 * y_c
    assert float((du[mask].abs() - g[mask]).abs().max() / g.max()) < 5e-3


def test_plane_shear_rate_vanishes_in_the_plug():
    rheo = validate_rheology(0.4 * 1.0e-4, 1.0, 0.7)
    y_c = 0.5 * 64.0 * 0.4
    y = torch.linspace(-0.99 * y_c, 0.99 * y_c, 51, dtype=torch.float64)
    g = hb_plane_shear_rate(y, 64.0, 1.0e-4, rheo)
    assert float(g.abs().max()) == 0.0


def test_newtonian_shear_rate_limit_is_linear():
    rheo = validate_rheology(0.0, 0.1, 1.0)
    H, tau_w = 32.0, 1.0e-4
    y = torch.linspace(-0.49 * H, 0.49 * H, 101, dtype=torch.float64)
    g = hb_plane_shear_rate(y, H, tau_w, rheo)
    expected = 2.0 * tau_w * y.abs() / (H * rheo.K)
    assert float((g - expected).abs().max() / expected.max()) < 1e-12


def test_couette_adverse_shear_rate_integrates_to_the_profile():
    rheo = validate_rheology(0.4 * 1.0e-4, 1.0, 0.7)
    h, beta = 64.0, 1.0e-4 / 64.0
    y = torch.linspace(0.0, h, 4001, dtype=torch.float64)
    g = hb_couette_adverse_shear_rate(y, beta, rheo)
    u = hb_couette_adverse_profile(y, beta, h, rheo)
    du = torch.gradient(u, spacing=(y,))[0]
    y_c = rheo.tau_y / beta
    mask = y > 1.5 * y_c
    assert float((du[mask].abs() - g[mask]).abs().max() / g.max()) < 5e-3
    meta = hb_couette_adverse_wall_velocity(beta, h, rheo)
    assert float(g[-1]) == pytest.approx(meta["gamma_dot_wall"], rel=1e-9)


def test_yield_fraction_helper():
    assert yield_fraction(0.0, 1.0) == 0.0
    assert yield_fraction(0.5, 1.0) == 0.5
    assert yield_fraction(2.0, 1.0) == 1.0
    with pytest.raises(FrazilHBError):
        yield_fraction(0.1, 0.0)


# ---------------------------------------------------------------------------
# Physical case -> lattice case
# ---------------------------------------------------------------------------


def test_physical_case_plastic_ratio_is_the_control_parameter():
    case = physical_flow_case(0.40, r_y=0.4)
    assert case["r_y"] == pytest.approx(0.4, rel=1e-12)
    assert case["yielded"]
    # momentum balance a = 2 tau_w/(rho H)
    assert case["tau_w_Pa"] == pytest.approx(
        case["state"].rho * case["a_si"] * case["H"] / 2.0, rel=1e-12
    )
    assert case["y_c_si"] == pytest.approx(0.5 * case["H"] * 0.4, rel=1e-12)


def test_physical_case_flags_an_unyielded_channel():
    case = physical_flow_case(0.40, r_y=None, a_si=1.0e-6)
    assert not case["yielded"]
    assert case["u_max_si"] == 0.0
    assert case["r_y"] >= 1.0


def test_verification_case_preserves_similarity_and_conditions_the_numerics():
    n, r_y = 0.7006, 0.4
    v = verification_case(n, r_y, gap_cells=64)
    assert v["rheo"].n == pytest.approx(n, rel=1e-12)
    assert v["rheo"].tau_y / v["tau_w_lat"] == pytest.approx(r_y, rel=1e-12)
    assert v["u_lat"] == pytest.approx(v["u_lat_target"], rel=1e-6)
    # relaxation times stay in the well-conditioned band
    assert 0.01 < v["tau_minus_half"] < 10.0  # 3*nu at the characteristic shear rate
    assert 0.5 < v["tau_plug"] < 50.0  # BGK relaxation time inside the plug
    assert v["y_c_lat"] == pytest.approx(0.5 * (64 + 1) * r_y, rel=1e-12)
    assert v["H_disc"] == pytest.approx(65.0, rel=1e-12)


def test_verification_case_rejects_bad_input():
    with pytest.raises(FrazilHBError):
        verification_case(0.7, 1.0)
    with pytest.raises(FrazilHBError):
        verification_case(0.7, 0.4, gap_cells=4)
    with pytest.raises(FrazilHBError):
        verification_case(0.7, 0.4, u_lat_target=0.5)
    with pytest.raises(FrazilHBError):
        verification_case(0.7, 0.4, nu_char_target=0.0)


def test_scaling_consistency_is_machine_precise():
    case = physical_flow_case(0.40, r_y=0.4)
    mapping = lattice_mapping(case, gap_cells=64, u_lat_target=0.015)
    ver = verification_case(case["state"].n, case["r_y"], gap_cells=64)
    report = scaling_consistency(case, mapping, ver)
    assert report["passed"], report["checks"]
    assert report["worst_residual"] < 1e-12
    # u_lat is linear in dt, so the target velocity is reproduced
    assert mapping["u_lat"] == pytest.approx(0.015, rel=1e-9)
    # the physical case is *not* resolvable at dx = H/64 (documented limitation)
    assert mapping["tau_minus_half"] < 1e-5


# ---------------------------------------------------------------------------
# Lattice-Boltzmann smoke runs
# ---------------------------------------------------------------------------


def test_channel_gap_is_n_plus_one():
    assert channel_gap(32) == 33.0
    with pytest.raises(ValueError):
        channel_gap(0)


def test_consistent_shear_rate_matches_the_analytic_derivative():
    """The wall-consistent operator must reproduce |du/dy| of the H-B profile.

    The solid rows carry the bounce-back state, so the operator replaces them by
    the plate speeds; the wall plane is one cell from the first fluid node
    (node-based bounce-back), not half a cell.
    """
    rheo = validate_rheology(0.4 * 1.0e-4, 1.0, 0.7)
    n_cells, H = 64, 65.0
    s = torch.arange(n_cells + 2, dtype=torch.float64)
    u = hb_plane_velocity_profile(s - H / 2.0, H, 1.0e-4, rheo)
    field = u.unsqueeze(1).expand(n_cells + 2, 1).contiguous()
    fluid = torch.ones((n_cells + 2, 1), dtype=torch.bool)
    fluid[0, :] = False
    fluid[-1, :] = False
    g = shear_rate_channel_consistent(field, fluid)[1:-1, 0]
    g_a = hb_plane_shear_rate(s[1:-1] - H / 2.0, H, 1.0e-4, rheo)
    y_c = 0.5 * H * rheo.tau_y / 1.0e-4
    mask = (s[1:-1] - H / 2.0).abs() > 1.5 * y_c
    # second-order central difference of the exact profile
    assert float((g[mask] - g_a[mask]).abs().max() / g_a.max()) < 5e-2
    # the wall node itself is differenced against the plate speed (0)
    assert float(g[0]) > 0.0


def test_consistent_shear_rate_uses_the_moving_wall_speed():
    n_cells = 4
    field = torch.zeros((n_cells + 2, 1), dtype=torch.float64)
    fluid = torch.ones((n_cells + 2, 1), dtype=torch.bool)
    fluid[0, :] = False
    fluid[-1, :] = False
    gamma = shear_rate_channel_consistent(field, fluid, 0.0, 0.5)
    # the wall plane is one cell from the adjacent fluid node and the plate moves
    # at 0.5, so the last fluid node sees (0.5 - 0)/2 in the central stencil
    assert float(gamma[n_cells, 0]) == pytest.approx(0.25)
    # away from the moving plate the field is uniform
    assert float(gamma[1, 0]) == pytest.approx(0.0)


@pytest.mark.parametrize("n_cells", [24])
def test_lbm_yield_stress_channel_smoke(n_cells):
    n, r_y = 0.7006179203265004, 0.4
    v = verification_case(n, r_y, gap_cells=n_cells)
    rheo, m = v["rheo"], v["m"]
    a_lat = v["a_lat"]
    y = torch.arange(n_cells + 2, dtype=torch.float64)
    H = float(n_cells) + 1.0
    ana_full = hb_plane_velocity_profile(y - H / 2.0, H, v["tau_w_lat"], rheo)
    res = run_frazil_hb_channel(
        n_cells,
        rheo,
        a_lat,
        nx=1,
        n_steps=6000,
        m=m,
        check_every=500,
        tol=1.0e-9,
        u_init=ana_full,
        dtype=torch.float64,
    )
    u = res["u_profile"].to(torch.float64)[1:-1]
    ana = hb_plane_velocity_profile(
        res["y"].to(torch.float64)[1:-1] - H / 2.0, H, v["tau_w_lat"], rheo
    )
    l2 = float(torch.linalg.norm(u - ana) / torch.linalg.norm(ana))
    # coarse-grid smoke bound; the gated value is N = 256 in the benchmark
    assert l2 < 0.35, l2
    assert float(u.max()) > 0.0


def test_lbm_newtonian_degeneration():
    v = verification_case(1.0, 0.0, gap_cells=24)
    rheo, m = v["rheo"], v["m"]
    a_lat = v["a_lat"]
    y0 = torch.arange(26, dtype=torch.float64)
    u_init = hb_plane_velocity_profile(y0 - 12.5, 25.0, v["tau_w_lat"], rheo)
    res = run_hb_channel(
        24, rheo, a_lat, nx=1, n_steps=6000, m=m, check_every=500,
        tol=1.0e-10, u_init=u_init, dtype=torch.float64,
    )
    y = res["y"].to(torch.float64)[1:-1]
    u = res["u_profile"].to(torch.float64)[1:-1]
    ana = hb_plane_velocity_profile(y - 12.5, 25.0, v["tau_w_lat"], rheo)
    l2 = float(torch.linalg.norm(u - ana) / torch.linalg.norm(ana))
    # coarse-grid smoke bound: the gated value is the N = 128 case of the
    # benchmark (artifacts/frazil_hb_flow.json, gate G4), where the transient has
    # decayed on the grid ladder
    assert l2 < 0.09, l2
    # Newtonian degeneration of the analytic H-B profile: the textbook parabola
    # u = a/(2K) (L^2 - (y-L)^2) with L = H/2 = 12.5, i.e. no plug at all
    H = 25.0
    L = H / 2.0
    tau_w = v["tau_w_lat"]
    parabola = (tau_w * H / (4.0 * rheo.K)) * (1.0 - ((y - L) / L) ** 2)
    assert float((ana - parabola).abs().max() / parabola.max()) < 1e-12


def test_einstein_dilute_limit_is_consistent_with_the_root():
    """At phi -> 0 the W9-1215 root must return mu_r -> 1 (Einstein limit)."""
    state = frazil_hb_state(1.0e-4)
    assert state.mu_r == pytest.approx(1.0 + 2.5 * 1.0e-4, rel=1e-3)
    assert math.isfinite(state.K_kin)