"""Tests for :mod:`tensorlbm.scalar_transport_verification` (campaign node p74-n10).

Structure (all sub-second except the three "physics" tests, ~25 s total):

* closed forms and their identities (erf, Taylor-Aris channel/pipe/shear);
* the discrete reference solvers (pipe radial Aris ODE, shear field sampling);
* the lattice erf-diffusion convergence study (order and error level);
* the two estimators of ``D_eff`` used by the face (variance/moment growth and
  Fourier-mode decay) on the cheapest configurations that are still in the
  asymptotic regime;
* mass-conservation drift, and the gate machinery itself on synthetic inputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.scalar_transport_verification import (
    ARIS_K_SHEAR,
    GATE_THRESHOLDS,
    channel_dispersion_run,
    erf_convergence_study,
    erf_diffusion_profile,
    evaluate_gates,
    mass_conservation_probe,
    pipe_aris_discrete,
    shear_aris_closed_form,
    shear_velocity_field,
    simulate_diffusion_1d,
)
from tensorlbm.taylor_aris import ARIS_K_CHANNEL, ARIS_K_PIPE, aris_dispersion_coefficient


# ── closed forms ────────────────────────────────────────────────────────────
def test_erf_profile_has_the_textbook_properties():
    x = np.array([-8.0, 0.0, 8.0])
    c = erf_diffusion_profile(x, t=1.0, diffusivity=0.5)
    assert c[1] == pytest.approx(0.5, abs=1e-15)  # step centre is always 0.5
    assert c[0] == pytest.approx(1.0, abs=1e-15)  # far upstream: undiluted
    assert c[2] == pytest.approx(0.0, abs=1e-15)  # far downstream: still clean
    # self-similarity: c depends on x only through x/sqrt(t)
    a = erf_diffusion_profile(np.array([1.0]), 1.0, 0.5)
    b = erf_diffusion_profile(np.array([2.0]), 4.0, 0.5)
    assert float(a[0]) == pytest.approx(float(b[0]), rel=1e-12)


def test_erf_profile_rejects_degenerate_input():
    with pytest.raises(ValueError):
        erf_diffusion_profile(np.array([0.0]), t=0.0, diffusivity=1.0)
    with pytest.raises(ValueError):
        erf_diffusion_profile(np.array([0.0]), t=1.0, diffusivity=0.0)


def test_shear_aris_constant_is_two_fifteenths():
    """``D_eff = D + (2/15) gamma^2 h^4 / D`` for a linear shear between plates."""
    assert ARIS_K_SHEAR == pytest.approx(2.0 / 15.0, rel=0)
    # independent check of the constant by quadrature of <u'chi>:
    # chi = gamma (h^2 y - y^3/3)/(2D), u' = gamma y
    h, gamma, D = 6.0, 0.3, 0.05
    y = np.linspace(-h, h, 400001)
    chi = gamma * (h**2 * y - y**3 / 3.0) / (2.0 * D)
    term = float(np.trapezoid(gamma * y * chi, y) / (2.0 * h))
    assert term == pytest.approx(gamma**2 * h**4 / D * ARIS_K_SHEAR, rel=1e-8)
    assert shear_aris_closed_form(D, h, gamma) == pytest.approx(D + term, rel=1e-10)


def test_shear_closed_form_reduces_to_molecular_diffusion():
    assert shear_aris_closed_form(0.02, 6.0, 0.0) == pytest.approx(0.02, rel=0)
    with pytest.raises(ValueError):
        shear_aris_closed_form(0.0, 6.0, 1.0)


def test_shear_velocity_field_matches_the_taylor_aris_grid_convention():
    u, half_width = shear_velocity_field(13, 0.5)
    assert u.size == 13
    assert half_width == pytest.approx(6.5)
    assert u[6] == pytest.approx(0.0)          # node on the centreline
    assert u[0] == pytest.approx(-6.0 * 0.5)   # outermost node at -(ny-1)/2
    assert u[-1] == pytest.approx(6.0 * 0.5)
    assert u.mean() == pytest.approx(0.0, abs=1e-15)  # zero cross-section mean
    with pytest.raises(ValueError):
        shear_velocity_field(12, 0.5)


def test_discrete_aris_solve_reproduces_the_shear_closed_form():
    """The Aris ODE on the *lattice nodes* must agree with the closed form.

    This is what caught the (wrong) hand-derived ``1/45`` constant during
    development: the discrete solve disagreed by exactly a factor 6.
    """
    for ny in (9, 13, 17, 33):
        gamma, D = 0.0111, 0.01
        u, h = shear_velocity_field(ny, gamma)
        disc = aris_dispersion_coefficient(u, D)
        closed = shear_aris_closed_form(D, h, gamma)
        assert disc["d_eff"] == pytest.approx(closed, rel=2e-3)


# ── discrete pipe Aris problem (verifies the 1/48 constant) ─────────────────
def test_pipe_aris_discrete_recovers_one_over_48():
    res = pipe_aris_discrete(radius=1.0, u_mean=1.0, diffusivity=0.05, n=200)
    # n=200 leaves an O(dr^2) quadrature residual of ~2.5e-5 on the constant
    assert res["k_measured"] == pytest.approx(ARIS_K_PIPE, rel=1e-4)
    assert res["rel_err"] == pytest.approx(0.0, abs=1e-5)
    assert res["residual"] < 1e-8


def test_pipe_aris_discrete_is_second_order_and_well_posed():
    errs = [
        abs(pipe_aris_discrete(radius=1.0, u_mean=1.0, diffusivity=0.05, n=n)["rel_err"])
        for n in (50, 100, 200)
    ]
    assert errs[0] > errs[1] > errs[2]          # monotone convergence
    slope = math.log(errs[0] / errs[2]) / math.log(4.0)
    assert slope > 3.5                          # observed order (super-convergent)
    with pytest.raises(ValueError):
        pipe_aris_discrete(radius=0.0, u_mean=1.0, diffusivity=0.05)


# ── 1-D lattice erf diffusion ───────────────────────────────────────────────
def test_diffusion_solver_is_second_order_accurate_for_erf():
    study = erf_convergence_study(ns=(32, 64, 128))
    assert study["monotone"]
    assert study["order_fit"] >= 1.8
    assert study["order_fit_r2"] >= 0.98
    assert study["finest_l2_rel_error"] <= 1e-2


def test_diffusion_solver_mass_is_conserved_and_validation_is_strict():
    res = simulate_diffusion_1d(n=64)
    assert res["mass_drift"] <= 1e-12
    assert res["tau"] == pytest.approx(3.0 * res["lattice_diffusivity"] + 0.5)
    with pytest.raises(ValueError):
        simulate_diffusion_1d(n=4)
    with pytest.raises(ValueError):
        simulate_diffusion_1d(n=64, t_start=1.0, t_final=0.5)


def test_diffusion_l2_error_shrinks_by_four_when_dx_is_halved():
    coarse = simulate_diffusion_1d(n=64)["l2_rel_error"]
    fine = simulate_diffusion_1d(n=128)["l2_rel_error"]
    assert coarse / fine == pytest.approx(4.0, rel=0.1)


# ── dispersion meters ───────────────────────────────────────────────────────
def test_negative_control_with_zero_velocity_returns_molecular_diffusion():
    """G6: with u = 0 the meter must return D itself (no spurious dispersion)."""
    res = channel_dispersion_run(
        pe=0.0, ny=9, nx=256, diffusivity=0.02, method="variance", dtype="float64"
    )
    assert GATE_THRESHOLDS["G6_ratio_lo"] <= res["ratio_over_d"] <= GATE_THRESHOLDS["G6_ratio_hi"]
    assert (res["d_eff_discrete"] - res["d_molecular"]) / res["d_molecular"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_moment_meter_reproduces_taylor_aris_dispersion():
    """G2 on the cheapest asymptotic configuration (ny=13, Pe=10)."""
    res = channel_dispersion_run(
        pe=10.0, ny=13, nx=256, diffusivity=0.05, method="variance", dtype="float64"
    )
    assert res["rel_err_analytic"] == pytest.approx(0.0, abs=GATE_THRESHOLDS["G2_rel_err_max"])
    assert res["rel_err_analytic"] < 0.0            # known O(dy^2) under-shoot
    assert res["r2"] >= GATE_THRESHOLDS["G7_r2_min"]
    assert res["sigma_over_nx"] < 1.0 / 6.0         # estimator stays in its domain
    assert res["d_eff"] > res["d_molecular"]        # shear does enhance dispersion


def test_mass_conservation_drift_is_at_round_off():
    res = mass_conservation_probe(
        ny=13, nx=256, diffusivity=0.01, shear_rate=0.0111, steps=2000, dtype="float64"
    )
    assert res["drift_per_1e4_steps"] <= GATE_THRESHOLDS["G5_mass_drift_max"]
    assert res["drift"][-1] >= res["drift"][0]      # monotone accumulation


# ── gate machinery ──────────────────────────────────────────────────────────
def _synthetic_results() -> dict:
    return {
        "erf": {
            "kind": "erf",
            "levels": [
                {"n": n, "dx": dx, "dt": 1e-3, "steps": 10, "l2_rel_error": e, "linf_error": e}
                for n, dx, e in ((64, 0.35, 1.1e-3), (128, 0.177, 2.8e-4), (256, 0.088, 7.1e-5))
            ],
            "dxs": [0.35, 0.177, 0.088],
            "l2_rel_errors": [1.1e-3, 2.8e-4, 7.1e-5],
            "order_fit": 2.0,
            "order_fit_r2": 0.9999,
            "order_two_finest": 1.98,
            "monotone": True,
            "finest_l2_rel_error": 7.1e-5,
        },
        "pipe_aris": {
            "kind": "pipe_aris",
            "n": 200,
            "d_eff": 0.46667,
            "d_eff_closed_form": 0.46667,
            "rel_err": 1e-9,
        },
        **{
            f"chan_ny{ny}_nx{nx}_pe{pe:g}": {
                "kind": "channel",
                "method": "variance",
                "item_id": f"chan_ny{ny}_nx{nx}_pe{pe:g}",
                "ny": ny,
                "nx": nx,
                "pe_r": pe,
                "d_eff": closed * (1.0 + rel),
                "d_eff_analytic": closed,
                "rel_err_analytic": rel,
                "ratio_over_d": closed * (1.0 + rel) / 0.01,
                "diffusivity": 0.01,
                "k_meas": (closed * (1.0 + rel) / 0.01 - 1.0) / pe**2,
                "r2": 1.0,
                "sigma_over_nx": 0.05,
                "steps": 100,
                "half_width": 0.5 * ny,
                "grid": {"ny": ny, "nx": nx, "diffusivity": 0.01},
                "tags": ["G2", "G7", "G8"],
            }
            for ny, nx, pe, closed, rel in (
                (13, 1024, 5.0, 0.0148, -0.015),
                (13, 1024, 10.0, 0.0290, -0.028),
                (13, 1024, 20.0, 0.0862, -0.037),
                (13, 1024, 40.0, 0.3148, -0.040),
                (9, 512, 5.0, 0.0148, -0.030),
                (9, 512, 10.0, 0.0290, -0.056),
                (9, 512, 20.0, 0.0862, -0.075),
                (9, 512, 40.0, 0.3148, -0.083),
            )
        },
        "shear": {
            "kind": "shear",
            "item_id": "shear_pe20",
            "ny": 13,
            "nx": 2048,
            "pe_r": 20.0,
            "d_eff": 0.35,
            "d_eff_analytic": 0.36,
            "rel_err_analytic": -0.028,
            "ratio_over_d": 35.0,
            "diffusivity": 0.01,
            "method": "variance",
            "r2": 1.0,
            "sigma_over_nx": 0.05,
            "steps": 100,
            "half_width": 6.5,
            "grid": {"ny": 13, "nx": 2048, "diffusivity": 0.01},
            "tags": ["G2", "G8"],
        },
        "g3": {
            "kind": "channel",
            "item_id": "pe_small",
            "pe_r": 0.5,
            "d_eff": 0.0100,
            "ratio_over_d": 1.0005,
            "tags": ["G3"],
        },
        "g6": {
            "kind": "channel",
            "item_id": "neg",
            "method": "variance",
            "d_eff": 0.01001,
            "d_molecular": 0.01,
            "ratio_over_d": 1.001,
            "tags": ["G6"],
        },
        "mass": {
            "kind": "mass",
            "dtype": "float64",
            "steps": 10000,
            "checkpoints": [10, 1000, 5000, 10000],
            "drift": [1e-17, 1e-17, 2e-17, 2e-17],
            "drift_final": 2e-17,
            "drift_per_1e4_steps": 2e-17,
            "tags": ["G5"],
        },
        "tau": {
            "kind": "tau_m",
            "sc": 1.0,
            "tau_m": 0.56,
            "tau_d": 0.56,
            "nu": 0.02,
            "d_eff": 0.030,
            "ratio_over_d": 1.5,
            "discrete_aris": 0.030,
            "diffusivity": 0.02,
            "tags": ["G4"],
        },
    }


def test_gate_machinery_passes_a_consistent_synthetic_study():
    gates = evaluate_gates(_synthetic_results())
    for gid in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"):
        assert gid in gates, gid
        assert gates[gid]["pass"], (gid, gates[gid])
    assert gates["all_pass"] is True


def test_gate_machinery_catches_a_failing_taylor_aris_case():
    res = _synthetic_results()
    res["chan_ny13_nx1024_pe20"]["rel_err_analytic"] = -0.09
    gates = evaluate_gates(res)
    assert gates["G2"]["pass"] is False
    assert gates["G2"]["worst"]["rel_err"] == pytest.approx(-0.09)


def test_gate_machinery_catches_a_non_monotone_convergence_series():
    res = _synthetic_results()
    res["erf"]["l2_rel_errors"] = [1.1e-3, 3.0e-4, 3.1e-4]
    res["erf"]["monotone"] = False
    res["erf"]["finest_l2_rel_error"] = 3.1e-4
    gates = evaluate_gates(res)
    assert gates["G7"]["pass"] is False


def test_gate_machinery_excludes_out_of_domain_cells_from_g2():
    """ny=9 rows (and sigma/nx > 1/6 rows) inform the accuracy domain, not G2."""
    res = _synthetic_results()
    gates = evaluate_gates(res)
    excluded = {e["item"] for e in gates["G2"]["excluded"]}
    assert "chan_ny9_nx512_pe40" in excluded
    assert gates["G2"]["pass"] is True
    g8 = {r["ny"]: r for r in gates["G8"]["rows"]}
    assert g8[9]["within_5pct"] is False
    assert gates["G8"]["n_valid"] >= 8
    assert gates["G8"]["n_outside_threshold"] >= 1


def test_gate_machinery_flags_a_mass_leak():
    res = _synthetic_results()
    res["mass"]["drift_per_1e4_steps"] = 1e-6
    gates = evaluate_gates(res)
    assert gates["G5"]["pass"] is False