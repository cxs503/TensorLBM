"""Tests for the structure-resolved frazil power-law rheology face.

Fast analytic/structure tests plus one tiny shear-cell smoke case.  The heavy
campaign lives in ``benchmarks/frazil_powerlaw_shear/run_campaign.py``; nothing
here depends on it.
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import math

import numpy as np
import pytest

from tensorlbm.frazil_powerlaw_shear import (
    DEFAULT_DF,
    FrazilShearError,
    bond_force_from_reference,
    build_suspension,
    equilibrium_floc_size,
    evaluate_acceptance,
    fit_power_law,
    fractal_cluster_cca,
    grid_convergence,
    lattice_to_si,
    pure_case,
    relative_viscosity,
)


# ---------------------------------------------------------------------------
# flow-curve fitting
# ---------------------------------------------------------------------------


def test_fit_power_law_recovers_exact_parameters():
    g = np.array([5.0e-4, 1.0e-3, 2.0e-3, 4.0e-3])
    K, n = 0.037, 0.71
    fit = fit_power_law(g, K * g ** n)
    assert fit["n"] == pytest.approx(n, rel=1e-12)
    assert fit["K"] == pytest.approx(K, rel=1e-12)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)
    assert fit["n_points"] == 4


def test_fit_power_law_newtonian_gives_unit_index():
    g = np.array([1.0e-3, 2.0e-3, 4.0e-3])
    fit = fit_power_law(g, 0.2 * g)
    assert fit["n"] == pytest.approx(1.0, abs=1e-12)


def test_fit_power_law_guards():
    with pytest.raises(FrazilShearError):
        fit_power_law([1e-3], [1e-4])                      # single point
    with pytest.raises(FrazilShearError):
        fit_power_law([1e-3, 2e-3], [1e-4])                # length mismatch
    with pytest.raises(FrazilShearError):
        fit_power_law([0.0, 2e-3], [1e-4, 2e-4])           # non-positive rate
    with pytest.raises(FrazilShearError):
        fit_power_law([1e-3, 1e-3], [1e-4, 2e-4])          # degenerate range


def test_fit_power_law_stderr_needs_three_points():
    two = fit_power_law([1e-3, 2e-3], [1e-4, 2.2e-4])
    assert math.isnan(two["n_stderr"])
    many = fit_power_law([1e-3, 2e-3, 4e-3], [1e-4, 2.2e-4, 4.3e-4])
    assert math.isfinite(many["n_stderr"]) and many["n_stderr"] > 0


# ---------------------------------------------------------------------------
# microgeometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_primary", [1, 2, 4, 8])
def test_cluster_has_requested_mass_and_no_overlap(n_primary):
    a = 2.5
    cl = fractal_cluster_cca(n_primary, a, seed=4)
    assert cl.n_primary == n_primary
    assert abs(float(cl.centres.mean(axis=0).sum())) < 1e-9      # centred
    if n_primary > 1:
        d = np.linalg.norm(cl.centres[:, None, :] - cl.centres[None, :, :], axis=2)
        d = d + np.eye(n_primary) * 1e9
        assert d.min() >= 2 * a - 1e-6                            # touching, not overlapping


def test_cluster_fractal_dimension_is_open():
    """An open aggregate must not be a compact ball: df ~ 2, not ~3."""
    cl = fractal_cluster_cca(8, 2.0, seed=11)
    assert 1.6 < cl.df_implied < 2.4
    # compact ball of 8 primaries would have Rg/a = (3/5)^0.5 * 8^(1/3) ~ 1.55
    assert cl.rg_over_a > 1.9


def test_cluster_df_of_single_and_pair_is_undefined():
    assert math.isnan(fractal_cluster_cca(1, 2.0, seed=1).df_implied)
    assert math.isnan(fractal_cluster_cca(2, 2.0, seed=1).df_implied)


def test_cluster_guards():
    with pytest.raises(FrazilShearError):
        fractal_cluster_cca(0, 2.0)
    with pytest.raises(FrazilShearError):
        fractal_cluster_cca(4, 0.0)


# ---------------------------------------------------------------------------
# equilibrium floc size (the single closure assumption)
# ---------------------------------------------------------------------------


def test_bond_force_anchoring_reproduces_reference_size():
    fb = bond_force_from_reference(5.0e-4, 3.0, 0.2, 8, 2.0)
    got = equilibrium_floc_size(5.0e-4, radius=3.0, mu=0.2, bond_force=fb, df=2.0, n_cap=8)
    assert got["n_primary"] == 8
    assert got["raw_n"] == pytest.approx(8.0, rel=1e-9)


def test_equilibrium_floc_size_follows_inverse_sqrt_radius():
    """``R_eq ~ gamma_dot^-1/2`` -- the shear-limited floc-size law."""
    a, mu, n_cap = 3.0, 0.2, 1.0e9        # no clip, to see the raw scaling
    fb = bond_force_from_reference(1.0e-3, a, mu, 4, DEFAULT_DF)
    r1 = equilibrium_floc_size(1.0e-3, radius=a, mu=mu, bond_force=fb, n_cap=n_cap)["radius"]
    r4 = equilibrium_floc_size(4.0e-3, radius=a, mu=mu, bond_force=fb, n_cap=n_cap)["radius"]
    assert r1 / r4 == pytest.approx(2.0, rel=1e-12)


def test_equilibrium_floc_size_clips_and_is_monotone():
    fb = bond_force_from_reference(5.0e-4, 3.0, 0.2, 8, 2.0)
    sizes = [equilibrium_floc_size(g, radius=3.0, mu=0.2, bond_force=fb, df=2.0,
                                   n_cap=8)["n_primary"]
             for g in (5.0e-4, 1.0e-3, 2.0e-3, 8.0e-3)]
    assert sizes == sorted(sizes, reverse=True)
    assert max(sizes) == 8 and min(sizes) == 1
    assert equilibrium_floc_size(8.0e-3, radius=3.0, mu=0.2, bond_force=fb)["regime"] == "dispersed"


def test_equilibrium_floc_size_guards():
    with pytest.raises(FrazilShearError):
        equilibrium_floc_size(0.0, radius=3.0, mu=0.2, bond_force=1e-3)
    with pytest.raises(FrazilShearError):
        bond_force_from_reference(0.0, 3.0, 0.2, 8)


# ---------------------------------------------------------------------------
# suspension construction
# ---------------------------------------------------------------------------


def test_dispersed_placement_hits_target_without_overlap():
    ps, st = build_suspension(24, 24, 25, 0.15, 2.0, n_primary=1, seed=7, wall_margin=3)
    assert st["n_primaries"] > 0
    assert st["phi_lattice"] <= 0.15 + 1e-6
    assert st["phi_lattice"] > 0.10            # random sequential addition reaches it
    centres = np.array([p.meta["primary_centre"] for p in ps])
    d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
    d = d + np.eye(len(centres)) * 1e9
    assert d.min() >= 2 * 2.0 - 1e-9           # no crystal overlap


def test_flocculated_aggregates_are_placed_as_rigid_bodies():
    """Every primary of one aggregate must share the aggregate centre."""
    ps, st = build_suspension(32, 32, 33, 0.06, 2.0, n_primary=4, seed=5, wall_margin=3)
    assert st["n_aggregates"] >= 1
    by_centre: dict = {}
    for p in ps:
        by_centre.setdefault(p.centre, []).append(p)
    sizes = sorted(len(v) for v in by_centre.values())
    assert max(sizes) > 1                      # at least one multi-primary aggregate
    assert all(s % 1 == 0 for s in sizes)
    # aggregates are internally touching-but-disjoint: primaries share one centre
    for centre, group in by_centre.items():
        assert all(p.half_extent == group[0].half_extent for p in group)


def test_hydrodynamic_phi_exceeds_solid_phi_for_open_flocs():
    _, st = build_suspension(32, 32, 33, 0.05, 2.0, n_primary=4, seed=2, wall_margin=3)
    assert st["hydrodynamic_phi"] > st["phi_lattice"]      # trapped water
    assert 1.6 < st["df_implied_mean"] < 2.4


def test_build_suspension_guards():
    with pytest.raises(FrazilShearError):
        build_suspension(16, 16, 6, 0.1, 2.0, wall_margin=3)   # box too small
    with pytest.raises(FrazilShearError):
        build_suspension(16, 16, 17, 1.2, 2.0)


# ---------------------------------------------------------------------------
# post-processing
# ---------------------------------------------------------------------------


def _case(tau, g):
    return {"tau": tau, "gamma_dot": g}


def test_relative_viscosity_is_a_ratio_of_the_same_measurement():
    rv = relative_viscosity(_case(2.0e-4, 2.0e-3), _case(4.0e-4, 2.0e-3))
    assert rv["eta_r"] == pytest.approx(0.5, rel=1e-12)
    with pytest.raises(FrazilShearError):
        relative_viscosity(_case(1.0, 0.0), _case(1.0, 1.0))


def test_grid_convergence_flags():
    lo = {"n": 0.90, "K": 0.10}
    hi = {"n": 0.905, "K": 0.102}
    conv = grid_convergence(lo, hi)
    assert conv["both_below_5pct"] is True
    bad = grid_convergence(lo, {"n": 0.80, "K": 0.10})
    assert bad["both_below_5pct"] is False
    assert bad["n_rel_diff"] == pytest.approx(0.10 / 0.80, rel=1e-12)


def test_lattice_to_si_preserves_the_flow_curve():
    """The SI map must be a change of units only: tau = K gamma^n is preserved."""
    K_lat, n = 0.021, 0.66
    dx, rho, nu_si, nu_lat = 3.0e-4, 1000.0, 1.0e-6, 0.2
    si = lattice_to_si(K_lat, n, dx=dx, rho=rho, nu_si=nu_si, nu_lattice=nu_lat)
    g_lat = 1.5e-3
    tau_lat = K_lat * g_lat ** n
    tau_si_direct = tau_lat * si["tau_scale_Pa"]
    g_si = g_lat / si["dt_s"]
    assert si["K_SI"] * g_si ** n == pytest.approx(tau_si_direct, rel=1e-12)


# ---------------------------------------------------------------------------
# acceptance evaluation
# ---------------------------------------------------------------------------


def _synthetic_acceptance(**over):
    dispersed = {
        "0.0": {"n": 1.0, "K": 0.2, "eta_r": 1.0},
        "0.05": {"n": 1.0, "K": 0.2, "eta_r": 1.125},
        "0.20": {"n": 1.0, "K": 0.2, "eta_r": 1.84, "phi_lattice": 0.20},
    }
    flocculated = {"0.10": {"n": 0.80, "K": 0.3}, "0.20": {"n": 0.78, "K": 0.35}}
    conv = {"n_rel_diff": 0.02, "K_rel_diff": 0.03, "both_below_5pct": True}
    kw = dict(dispersed=dispersed, flocculated=flocculated, convergence=conv)
    kw.update(over)
    return evaluate_acceptance(**kw)


def test_acceptance_all_pass_on_consistent_synthetic_input():
    out = _synthetic_acceptance()
    assert out["all_pass"] is True and out["n_pass"] == 5


def test_acceptance_zero_control_catches_a_bad_negative_control():
    dispersed = {"0.0": {"n": 1.05, "K": 0.2, "eta_r": 1.0},
                 "0.05": {"n": 1.0, "K": 0.2, "eta_r": 1.125},
                 "0.20": {"n": 1.0, "K": 0.2, "eta_r": 1.84, "phi_lattice": 0.20}}
    out = _synthetic_acceptance(dispersed=dispersed)
    assert out["criteria"]["c1_zero_control"]["n_passed"] is False
    assert out["criteria"]["c1_zero_control"]["passed"] is False


def test_acceptance_einstein_and_krieger_dougherty_windows():
    dispersed = {"0.05": {"n": 1.0, "K": 0.2, "eta_r": 1.40},          # 24 % high
                 "0.20": {"n": 1.0, "K": 0.2, "eta_r": 2.40, "phi_lattice": 0.20}}
    out = _synthetic_acceptance(dispersed=dispersed)
    assert out["criteria"]["c2_dilute_einstein"]["passed"] is False
    assert out["criteria"]["c3_krieger_dougherty"]["passed"] is False


def test_acceptance_shear_thinning_needs_thin_at_high_concentration_only():
    thin = {"0.05": {"n": 1.0, "K": 1.0}, "0.10": {"n": 0.90, "K": 1.0}}
    out = _synthetic_acceptance(flocculated=thin)
    assert out["criteria"]["c4_shear_thinning"]["passed"] is True
    thick = {"0.05": {"n": 1.0, "K": 1.0}, "0.10": {"n": 0.99, "K": 1.0}}
    assert _synthetic_acceptance(flocculated=thick)["criteria"]["c4_shear_thinning"]["passed"] is False


def test_acceptance_reports_undefined_criteria_instead_of_passing_them():
    out = evaluate_acceptance(dispersed={}, flocculated={}, convergence=None)
    assert out["all_pass"] is False
    assert out["criteria"]["c5_grid_convergence"]["defined"] is False
    assert out["criteria"]["c2_dilute_einstein"]["passed"] is False


# ---------------------------------------------------------------------------
# tiny shear-cell smoke case (momentum budget of the particle-free control)
# ---------------------------------------------------------------------------


def test_pure_couette_momentum_budget_closes():
    """The reference run must reproduce ``tau = rho nu gamma_dot`` (a few %)."""
    import torch

    # pin the thread count: torch's default pool spins on a loaded host, which
    # turns a 5 s smoke case into minutes without changing a single digit
    torch.set_num_threads(min(2, torch.get_num_threads()))
    r = pure_case(nx=12, ny=12, nz=13, nu=0.2, u_wall=0.01, n_steps=400,
                  n_warmup=250, sample_every=10, wall_band=3)
    ref = 1.0 * r["nu"] * r["gamma_dot"]
    assert r["profile_r2"] > 0.99
    assert abs(r["tau"] / ref - 1.0) < 0.05