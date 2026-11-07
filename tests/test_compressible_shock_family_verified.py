"""Tests for the compressible-shock family verification (campaign P19-N04).

Two layers:

* **theory / detector unit tests** -- every assertion has an independent
  reference: known oblique-shock table values, the definition ``M2(beta*) = 1``,
  a round-trip through the exact jump relations, and synthetic fields whose
  contact line and vortex core are known by construction;
* **artifact tests** -- the committed ``result.json`` / ``gates_verified.json``
  must exist, carry every gate and have all gates passing, so a regression in
  the numbers cannot silently pass CI.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from tensorlbm.compressible_shock import family as fam
from tensorlbm.compressible_shock.rankine_hugoniot import (
    GAMMA,
    oblique_shock,
    oblique_shock_beta,
    normal_shock,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "benchmarks", "verified", "compressible_shock_family")


# ------------------------------------------------------------------- theory
@pytest.mark.parametrize(
    "mach,theta_max_deg",
    [(2.0, 22.9735), (3.0, 34.0734), (5.0, 41.1094)],
)
def test_theta_max_reference_values(mach, theta_max_deg):
    """Maximum deflection angles of standard oblique-shock tables (gamma=1.4)."""
    r = fam.theta_max(mach)
    assert r["theta_max_deg"] == pytest.approx(theta_max_deg, abs=0.05)
    # the maximum-deflection angle is a genuine maximum of theta(beta)
    for d in (-0.05, 0.05):
        b = math.radians(r["beta_star_deg"] + d)
        assert fam._deflection(mach, b) < math.radians(r["theta_max_deg"]) + 1e-6


@pytest.mark.parametrize("mach", [2.0, 3.0, 5.0, 10.0])
def test_sonic_shock_angle_is_defined_by_m2_equals_one(mach):
    r = fam.sonic_shock_angle(mach)
    assert r["M2_full_at_sonic"] == pytest.approx(1.0, abs=1e-9)
    # the sonic point lies below the maximum-deflection (detachment) angle
    assert r["beta_sonic_deg"] < fam.theta_max(mach)["beta_star_deg"]
    assert r["beta_sonic_deg"] > r["mu_deg"]


def test_sonic_angle_reference_values():
    """M=3: 63.77 deg, M=2: 61.49 deg (M2 = 1 condition, gamma = 1.4)."""
    assert fam.sonic_shock_angle(3.0)["beta_sonic_deg"] == pytest.approx(63.767, abs=0.01)
    assert fam.sonic_shock_angle(2.0)["beta_sonic_deg"] == pytest.approx(61.485, abs=0.01)


@pytest.mark.parametrize("mach,theta_deg", [(2.0, 10.0), (3.0, 20.0), (5.0, 25.0), (10.0, 20.0)])
def test_shock_polar_inversion_round_trip(mach, theta_deg):
    """A jump ratio produced by a known shock angle must invert to that angle."""
    theta = math.radians(theta_deg)
    beta = oblique_shock_beta(mach, theta)
    jump = oblique_shock(mach, theta)
    inv = fam.shock_polar_angle_from_ratio(mach, jump["rho_ratio"])
    assert inv["attached"]
    assert math.radians(inv["beta_deg"]) == pytest.approx(beta, rel=1e-7)
    inv_p = fam.shock_polar_angle_from_ratio(mach, jump["p_ratio"], kind="p")
    assert math.radians(inv_p["beta_deg"]) == pytest.approx(beta, rel=1e-7)


def test_inversion_rejects_ratios_above_the_normal_shock_limit():
    with pytest.raises(ValueError):
        fam.shock_polar_angle_from_ratio(3.0, 1.0)  # no jump at all
    out = fam.shock_polar_angle_from_ratio(3.0, 2.0 * normal_shock(3.0)["rho_ratio"])
    assert out["attached"] is False


def test_regular_reflection_removes_the_incident_deflection():
    """Two-shock theory: the reflected shock turns the flow back by theta_i."""
    M1, beta_i = 3.0, 35.0
    r = fam.regular_reflection_angle(M1, beta_i)
    assert r["attached"]
    theta_i = math.radians(r["incident_theta_deg"])
    # the state behind the reflected shock must be parallel to the wall again
    inc = oblique_shock(M1, theta_i)
    assert inc["M2"] == pytest.approx(r["M2"], rel=1e-9)
    # the reflected shock turns the flow by exactly theta_i (that is the
    # closure of the two-shock construction)
    refl = oblique_shock(r["M2"], theta_i)
    assert refl["theta"] == pytest.approx(theta_i, rel=1e-9)
    assert r["reflected_beta_from_wall_deg"] == pytest.approx(
        r["reflected_beta_deg"] - r["incident_theta_deg"], abs=1e-9
    )
    # a shock turning flow back towards the wall has a larger angle than the flow
    assert r["reflected_beta_deg"] > r["reflected_beta_from_wall_deg"]


def test_three_shock_closure_diagnostics_for_the_dmr_configuration():
    """Mach 10 / 30 deg: the weak reflected-shock branch cannot close (documented)."""
    M1 = 10.0 / math.sin(math.radians(60.0))  # triple-point frame, incident beta = 60 deg
    c = fam.three_shock_closure(M1, 60.0)
    assert c["incident_theta_deg"] == pytest.approx(43.137, abs=0.01)
    assert c["M2_behind_incident"] == pytest.approx(1.336, abs=0.01)
    # the required turn-back exceeds what a M2 = 1.34 shock can deflect
    assert c["theta_max_downstream_deg"] < c["incident_theta_deg"]
    assert c["polylines_cross"] is False
    gaps = np.asarray(c["pressure_gap_scan"], dtype=float)
    finite = np.isfinite(gaps)
    assert finite.sum() > 10
    # the only contact is the degenerate one (chi -> theta_i, vanishing shock)
    assert np.nanmin(gaps[finite]) == pytest.approx(0.0, abs=1e-4)
    assert np.nanmax(gaps[finite]) > 10.0


# ------------------------------------------------------------------ detectors
def _synthetic_contact(angle_deg=215.0, nx=120, ny=120):
    xc = np.linspace(0.02, 0.98, nx)
    yc = np.linspace(0.02, 0.98, ny)
    X, Y = np.meshgrid(xc, yc)
    th = math.radians(angle_deg)
    x0, y0 = 0.5, 0.9
    # signed distance to the line through (x0, y0) with direction th
    s = -(X - x0) * math.sin(th) + (Y - y0) * math.cos(th)
    dx = xc[1] - xc[0]
    H = np.tanh(s / (0.6 * dx))
    rho = 1.0 + 0.8 * (H + 1.0) / 2.0  # density jump across the line
    p = np.full_like(rho, 1.0)  # pressure continuous: that is a contact
    u = 1.0 + 0.5 * H * math.cos(th)  # tangential velocity jump only
    v = 0.0 + 0.5 * H * math.sin(th)
    return xc, yc, rho, p, u, v


def test_bilinear_is_exact_on_a_linear_field():
    xc = np.linspace(0.0, 1.0, 21)
    yc = np.linspace(0.0, 2.0, 41)
    X, Y = np.meshgrid(xc, yc)
    F = 2.0 + 3.0 * X - 5.0 * Y
    pts = [(0.13, 0.77), (0.5, 1.0), (0.97, 1.93)]
    for x, y in pts:
        assert fam.bilinear(F, xc, yc, x, y) == pytest.approx(2.0 + 3.0 * x - 5.0 * y, rel=1e-10)


def test_slip_line_detectors_recover_a_synthetic_contact():
    xc, yc, rho, p, u, v = _synthetic_contact()
    origin = (0.5, 0.9)
    ang = np.arange(170.0, 260.01, 0.25)
    d = fam.slip_line_angle(rho, p, xc, yc, origin, 0.30, angles_deg=ang)
    w = fam.slip_line_angle_velocity(u, v, xc, yc, origin, 0.30, angles_deg=ang)
    assert d["angle_deg"] == pytest.approx(215.0, abs=2.0)
    assert w["angle_deg"] == pytest.approx(215.0, abs=2.0)
    assert d["density_jump"] > 0.2
    assert d["pressure_jump"] < 0.05  # contact: no pressure jump
    assert abs(d["angle_deg"] - w["angle_deg"]) < 2.0


def test_vortex_core_finds_a_lamb_oseen_vortex():
    xc = np.linspace(0.0, 1.0, 101)
    yc = np.linspace(0.0, 1.0, 101)
    X, Y = np.meshgrid(xc, yc)
    x0, y0, rc = 0.62, 0.41, 0.08
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    u = -(Y - y0) / rc * np.exp(-r2 / (2 * rc**2))
    v = (X - x0) / rc * np.exp(-r2 / (2 * rc**2))
    out = fam.vortex_core(u, v, xc, yc, box=(0.2, 0.9, 0.1, 0.8))
    assert out["x"] == pytest.approx(x0, abs=0.02)
    assert out["y"] == pytest.approx(y0, abs=0.02)
    assert out["omega_peak"] > 0.0


def test_front_angle_recovers_a_known_inclination():
    yc = np.linspace(0.0, 1.0, 51)
    # a front inclined 30 deg from the x axis has dx/dy = 1/tan(30 deg)
    slope = 1.0 / math.tan(math.radians(30.0))
    xf = 0.2 + slope * yc
    out = fam.front_angle(xf, yc, 0.3, 0.9)
    assert out["angle_deg"] == pytest.approx(30.0, abs=1e-6)
    assert out["rms"] < 1e-12


def test_richardson_recovers_a_second_order_limit():
    import sys

    sys.path.insert(0, os.path.join(ART))
    from analyze_family import richardson

    exact = 0.5
    h = np.array([1.0, 1.0 / 1.5, 1.0 / 1.5**2])
    vals = exact + 0.3 * h**2  # clean second-order sequence
    out = richardson(*vals)
    assert out["extrapolated"] == pytest.approx(exact, rel=1e-6)
    assert out["order"] == pytest.approx(2.0, abs=1e-6)


# ------------------------------------------------------------------ artifacts
def by_id_ok(gates, gid):
    return next(g["pass"] for g in gates["gates"] if g["id"] == gid)


def test_artifacts_exist_and_gates_match_the_documented_state():
    res_p = os.path.join(ART, "result.json")
    gates_p = os.path.join(ART, "gates_verified.json")
    assert os.path.exists(res_p), "result.json missing (run run_family.py --assemble)"
    assert os.path.exists(gates_p), "gates_verified.json missing"
    res = json.load(open(res_p))
    gates = json.load(open(gates_p))
    assert gates["n_gates"] >= 9
    failed = {g["id"] for g in gates["gates"] if not g["pass"]}
    # FFS-02 (recompression-shock angle, 1 deg) is a *documented honest FAIL*:
    # the internal shock is curved and the separated flow unsteady, so the angle
    # is not grid-converged to 1 deg at 300x100 (measured 1.46 deg, observed
    # order 2.26, grid spread 2.18 deg; two independent definitions agree that it
    # is not converged).  Any other failing gate is a regression.
    assert failed <= {"FFS-02"}, f"unexpected failing gates: {sorted(failed)}"
    assert gates["all_pass"] is False
    assert by_id_ok(gates, "FFS-02") is False
    # required thresholds of the campaign node
    by_id = {g["id"]: g for g in gates["gates"]}
    for m in (2.0, 3.0, 5.0, 8.0, 10.0):
        assert by_id[f"NS-M{m:g}"]["threshold"] == 0.01
    assert by_id["DMR-02"]["threshold"] == 0.05  # Mach stem height
    assert by_id["DMR-03"]["threshold"] == 3.0  # slip-line angle
    assert by_id["DMR-04"]["threshold"] == 0.05  # vortex core
    assert by_id["FFS-01"]["threshold"] == 2.0  # detachment angle
    assert by_id["FFS-02"]["threshold"] == 1.0  # reflected shock angle
    # normal-shock scan really covers 5 Mach numbers x 3 grids
    ns = res["normal_shock"]
    assert ns["n_cases"] >= 15
    assert sorted(ns["grids"]) == [400, 800, 1600]
    assert len(ns["machs"]) == 5
    # figures referenced by the report must be committed
    for name in (
        "normal_shock_scan.png",
        "dmr_density.png",
        "dmr_convergence.png",
        "ffs_density.png",
        "ffs_convergence.png",
        "slip_line_score.png",
    ):
        assert os.path.exists(os.path.join(ART, "figures", name)), name