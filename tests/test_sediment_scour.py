"""Sediment transport / bed-scour verification surface (mission P84W2C-N04).

Criteria (all machine-judgeable):

* C1 incipient-motion curve vs Soulsby & Whitehouse (2005), 8 ``d*`` points, <=15 %
* C2 Rouse exponent vs ``w_s/(kappa u*)``, <5 %, three ``u*`` cases
* C3 sediment mass budget (Exner bed volume + coupled suspended/bed) < 1e-6
* C4 negative controls: no-motion gate, ``w_s -> 0`` uniform, Exner off
* C5 single-pile ``S/D`` in the Melville & Chiew band and the trend with ``D``
* C6 JSON contract + the ship-platform UKC engineering hook
  (see ``tests/test_sediment_ukc_integration.py``)

Run::

    PYTHONPATH=src python3 -m pytest tests/test_sediment_scour.py -q
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tensorlbm.sediment.exner import ExnerConfig, bed_volume, exner_step, mpm_flux
from tensorlbm.sediment.morpho import MorphoConfig, measure_onset_shields
from tensorlbm.sediment.passive_scalar import (
    RouseLBMConfig,
    column_exchange_budget,
    run_rouse_lbm,
)
from tensorlbm.sediment.rouse import (
    KAPPA,
    fit_rouse_exponent,
    rouse_exponent,
    rouse_profile,
)
from tensorlbm.sediment.scour import (
    PileConfig,
    ScourConfig,
    run_single_pile_scour,
    verify_flow_solver,
)
from tensorlbm.sediment.shields import (
    critical_shields,
    dimensionless_grain_size,
    dimensionless_settling_velocity,
    grain_diameter_from_dstar,
    settling_velocity,
)
from tensorlbm.sediment.verification import (
    D_STAR_POINTS,
    SedimentScenario,
    build_results,
    c1_incipient_curve,
    c2_rouse_profiles,
    c3_mass_budget,
    c4_negative_controls,
    c5_pile_scour,
)

SCEN = SedimentScenario()  # 0.385 mm sand, h = 0.15 m


# ---------------------------------------------------------------------------
# Building blocks: units and reference values
# ---------------------------------------------------------------------------
def test_d_star_is_dimensionless_regression() -> None:
    """`d*` must use the cube-root form (the removed formula was off by ~136x).

    For d = 0.1 mm, s = 1.65, nu = 1e-6: d* = 1.855 (Soulsby 1997).  The
    dimensionally inconsistent ``d sqrt((s-1)g)/nu`` returned 252.5.
    """
    d_star = dimensionless_grain_size(1.0e-4, 1.65, 9.81, 1.0e-6)
    assert d_star == pytest.approx(1.855, rel=1e-2)
    assert grain_diameter_from_dstar(d_star, 1.65, 9.81, 1.0e-6) == pytest.approx(
        1.0e-4, rel=1e-10
    )


def test_critical_shields_reference_values() -> None:
    """Soulsby-Whitehouse curve at the classical anchors of the Shields diagram."""
    assert critical_shields(1.0) == pytest.approx(0.1375, abs=2e-3)
    assert critical_shields(10.0) == pytest.approx(0.0331, abs=2e-3)
    assert critical_shields(100.0) == pytest.approx(0.0500, abs=2e-3)
    # the curve is monotone decreasing up to the minimum near d* ~ 10
    vals = [critical_shields(x) for x in np.linspace(0.5, 10.0, 20)]
    assert np.all(np.diff(vals) < 0.0)


def test_settling_stokes_limit() -> None:
    """van Rijn settling velocity must approach Stokes for a fine grain."""
    d, s, g, nu = 5.0e-5, 1.65, 9.81, 1.0e-6
    stokes = (s - 1.0) * g * d * d / (18.0 * nu)
    # van Rijn (1984) is a fit for 100-1000 um sand; at d = 50 um (d* ~ 0.93)
    # it sits ~10 % below the Stokes asymptote, which is the documented spread
    assert settling_velocity(d, s, g, nu) == pytest.approx(stokes, rel=0.15)
    w_star = dimensionless_settling_velocity(stokes, d, s, g, nu)
    assert np.isfinite(w_star) and w_star > 0.0


def test_mpm_gate_is_exact() -> None:
    """Below ``theta_cr`` the MPM flux is exactly zero (no numerical leakage)."""
    cfg = ExnerConfig(d=3.85e-4, theta_cr=0.047)
    assert float(mpm_flux(0.0469999, cfg)) == 0.0
    assert float(mpm_flux(0.047, cfg)) == 0.0
    assert float(mpm_flux(0.048, cfg)) > 0.0


def test_rouse_profile_analytic_roundtrip() -> None:
    """The unbiased fit recovers a known Z from the analytic profile exactly."""
    h, a, Z = 1.0, 0.05, 0.42
    y = np.linspace(0.06, 0.6, 60)
    c = rouse_profile(y, h, a, Z)
    assert fit_rouse_exponent(y, c, h) == pytest.approx(Z, rel=1e-9)


def test_rouse_exponent_definition() -> None:
    assert rouse_exponent(0.03, 0.01) == pytest.approx(0.03 / (KAPPA * 0.01))


# ---------------------------------------------------------------------------
# Flow solver verification (the scour model rides on it)
# ---------------------------------------------------------------------------
def test_poiseuille_flow_solver() -> None:
    """The LBM flow solver reproduces the analytic Poiseuille profile."""
    res = verify_flow_solver(nx=16, ny=33, nu=0.05, body_force=1.0e-5, n_steps=8000)
    assert res["profile_rel_err"] < 0.05
    assert res["u_star_rel_err"] < 0.05


# ---------------------------------------------------------------------------
# C1: incipient-motion curve
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_c1_incipient_motion_curve() -> None:
    """C1: measured onset within 15 % of Soulsby-Whitehouse at all 8 d*."""
    res = c1_incipient_curve(SCEN)
    assert res["n_points"] == 8
    for row in res["points"]:
        assert row["rel_err"] <= 0.15, (
            f"d*={row['d_star']}: rel err {row['rel_err']:.3f} > 15% "
            f"(theta_ref={row['theta_ref']:.4f}, theta_num={row['theta_num']:.4f})"
        )
    assert res["pass"] is True


def test_c1_bisection_detects_wrong_d_star() -> None:
    """The criterion has teeth: the removed `d*` formula fails the 15 % band.

    ``d*_old = d sqrt((s-1)g)/nu`` is dimensionally inconsistent (factor ~136
    too large for a 0.1 mm grain) and moves the critical Shields evaluation
    into the wrong hydraulic regime -- here by >40 % on the 8 C1 points.
    """
    worst = 0.0
    for d_star in D_STAR_POINTS:
        d = grain_diameter_from_dstar(d_star, SCEN.s, SCEN.g, SCEN.nu)
        theta_ref = critical_shields(d_star)
        d_star_old = d * math.sqrt((SCEN.s - 1.0) * SCEN.g) / SCEN.nu
        worst = max(worst, abs(critical_shields(d_star_old) - theta_ref) / theta_ref)
    assert worst > 0.15, f"the old d* formula should be detected (worst {worst:.2%})"


# ---------------------------------------------------------------------------
# C2: Rouse profile
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_c2_rouse_single_case() -> None:
    """C2: LBM Rouse exponent within 5 % of w_s/(kappa u*)."""
    u_star = SCEN.w_s / (KAPPA * 0.6)  # Z = 0.6 (moderate suspension)
    res = run_rouse_lbm(
        RouseLBMConfig(h=SCEN.h, u_star=u_star, w_s=SCEN.w_s, dxi=0.2, D_lat=0.1)
    )
    assert res["Z_ref"] == pytest.approx(SCEN.w_s / (KAPPA * u_star))
    assert res["rel_err"] < 0.05, f"Z_num={res['Z_num']:.4f} vs Z_ref={res['Z_ref']:.4f}"
    assert res["profile_rel_err_vs_analytic"] < 0.10


@pytest.mark.slow
def test_c2_three_shear_velocities() -> None:
    """C2: three u* give three distinct Z, each recovered within 5 %."""
    res = c2_rouse_profiles(SCEN)
    assert len(res["cases"]) == 3
    for case in res["cases"]:
        assert case["rel_err"] < 0.05
    zs = [c["Z_ref"] for c in res["cases"]]
    us = [c["u_star"] for c in res["cases"]]
    assert zs[0] < zs[1] < zs[2]  # larger u* -> smaller Rouse number
    assert res["pass"] is True


# ---------------------------------------------------------------------------
# C3: mass budget
# ---------------------------------------------------------------------------
def test_c3_mass_budget() -> None:
    """C3: bed-volume conservation and coupled suspended+bed drift < 1e-6."""
    res = c3_mass_budget(SCEN, n_steps=800)
    assert res["exner"]["pass"] is True
    assert res["exner"]["mass_drift"] < 1e-6
    assert res["exner"]["bed_change_over_d"] > 1.0  # the bed really evolved
    assert res["coupled_column"]["mass_drift"] < 1e-6
    assert abs(res["coupled_column"]["bed_change"]) > 1e-3  # real exchange
    assert res["pass"] is True


# ---------------------------------------------------------------------------
# C4: negative controls
# ---------------------------------------------------------------------------
def test_c4_negative_controls() -> None:
    """C4 (a) no motion below 0.8 u*_cr, (b) w_s -> 0 uniform, (c) Exner off."""
    res = c4_negative_controls(SCEN)
    assert res["a_no_motion"]["q_s"] < 1e-12
    assert res["a_no_motion"]["bed_identical"] is True
    assert res["b_zero_settling"]["uniformity"] < 1e-3
    assert res["c_exner_off"]["bed_off_change"] == 0.0
    assert res["c_exner_off"]["bed_off_change"] < res["c_exner_off"]["bed_on_change"]


def test_c4_zero_settling_profile_is_uniform() -> None:
    """C4(b) explicitly: w_s = 0 must give a uniform concentration field."""
    res = run_rouse_lbm(RouseLBMConfig(w_s=0.0))
    assert res["profile_uniformity"] < 1e-3
    assert abs(res["Z_num"]) < 1e-6


# ---------------------------------------------------------------------------
# C5: single-pile scour
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_c5_single_pile_scour() -> None:
    """C5: S/D in [1.0, 2.4] and S/D decreases when D doubles at fixed depth."""
    pile = PileConfig(nx=64, ny=96, diameter=8.0, n_steps=6000, n_avg=1000)
    res = c5_pile_scour(pile, ScourConfig(), trend=True)
    assert res["A"] > 1.2, f"near-pile amplification {res['A']:.2f} looks wrong"
    assert 1.0 <= res["S_over_D"] <= 2.4, f"S/D={res['S_over_D']:.2f} outside band"
    assert res["trend"]["pass"] is True
    assert res["trend"]["direction"] == "decrease"


def test_scour_relaxation_matches_closed_form() -> None:
    """The relaxed scour maximum agrees with the analytic equilibrium depth."""
    pile = PileConfig(nx=48, ny=72, diameter=6.0, n_steps=4000, n_avg=800)
    res = run_single_pile_scour(pile, ScourConfig())
    assert res["closed_form_S_over_D"] == pytest.approx(res["S_over_D"], rel=0.02)


# ---------------------------------------------------------------------------
# C6: JSON contract
# ---------------------------------------------------------------------------
def test_c6_json_contract() -> None:
    """C6: build_results emits the contract keys and is JSON-serialisable."""
    cheap = PileConfig(nx=48, ny=72, diameter=6.0, n_steps=3000, n_avg=600)
    results = build_results(
        scen=SCEN,
        c1={
            "points": [{"d_star": 1.0, "theta_ref": 0.1375, "theta_num": 0.1376,
                        "rel_err": 0.001, "converged": True, "bisection_iters": 1,
                        "re_star_cr": 2.0, "d_star_old_formula": 136.0,
                        "old_formula_rel_err": 0.6}],
            "max_rel_err": 0.001,
            "tol": 0.15,
            "pass": True,
        },
        c2={
            "cases": [{"u_star": 0.0097, "u_star_factor": 1.0, "Z_ref": 0.29,
                       "Z_num": 0.291, "rel_err": 0.004, "fit_asymmetry": 0.02,
                       "profile_rel_err_vs_analytic": 0.02, "steps": 10,
                       "n_nodes": 31, "y": [0.05, 0.2], "c_normalised": [1.0, 0.5],
                       "c_analytic": [1.0, 0.5], "y_analytic": [0.05, 0.2]}],
            "max_rel_err": 0.004, "tol": 0.05, "pass": True, "w_s": SCEN.w_s,
            "kappa": KAPPA,
        },
        c3={"exner": {"mass_drift": 1e-14, "bed_change_m": 1e-5,
                      "bed_change_over_d": 2.0, "steps": 10, "dt_s": 1.0,
                      "dt_cfl_s": 10.0, "t_end_s": 10.0, "pass": True},
            "coupled_column": {"mass_drift": 1e-13},
            "pass": True, "tol": 1e-6},
        c4=c4_negative_controls(SCEN),
        c5=None,
        cheap_scour=True,
    )
    for key in ("shields_curve", "rouse_profiles", "bed_elevation",
                "scour_depth", "mass_budget"):
        assert key in results, f"missing key {key}"
    assert len(results["bed_elevation"]["bed_elevation_m"]) == len(
        results["bed_elevation"]["x_over_D"]
    )
    json.dumps(results)  # must be serialisable without a numpy default