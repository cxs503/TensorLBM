"""Wall-origin identification and three-way friction-velocity closure.

Every test is analytic-recovery or reference-data based: a fit is only trusted
when it reproduces a known wall position / friction velocity, and the
reference-DNS test pins the *measured* model-form error of the log law at
Re_tau = 180 so a regression cannot hide behind it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tensorlbm.validation.wall_origin import (
    fit_wall_origin,
    fit_wall_origin_loglaw,
    fit_wall_origin_spalding,
    fit_wall_origin_viscous,
    fit_wall_origin_wall_stress,
    poiseuille_wall_origin,
    u_tau_from_log_law,
    u_tau_three_way,
    wall_offset_bias,
)

KAPPA = 0.41
B_LOG = 5.0
REPO = Path(__file__).resolve().parents[1]
MKM_CSV = REPO / "benchmarks" / "pending" / "channel_turb_3d" / "dns_ref" / "mkm180.csv"
CALIBRATION = (
    REPO
    / "benchmarks"
    / "pending"
    / "channel_turb_3d"
    / "runs_3d"
    / "laminar_wall_origin_calibration.json"
)


def _viscous_profile(y_origin: float, u_tau: float, nu: float, y: np.ndarray) -> np.ndarray:
    return u_tau**2 / nu * (y - y_origin)


def _log_profile(
    y_origin: float, u_tau: float, nu: float, y: np.ndarray, b: float = B_LOG
) -> np.ndarray:
    """Exact log law with the *test's* universals (module default B_LOG is 5.2)."""
    y_plus = (y - y_origin) * u_tau / nu
    return u_tau * (np.log(y_plus) / KAPPA + b)


# ---------------------------------------------------------------------------
# Analytic recovery
# ---------------------------------------------------------------------------
def test_viscous_fit_recovers_wall_origin_and_friction_velocity() -> None:
    nu, u_tau, y_origin = 5.0e-4, 0.003, 0.1
    y = np.arange(1.0, 7.0)
    u = _viscous_profile(y_origin, u_tau, nu, y)
    fit = fit_wall_origin_viscous(y, u, nu)
    assert fit.y_origin == pytest.approx(y_origin, abs=1.0e-9)
    assert fit.u_tau == pytest.approx(u_tau, rel=1.0e-9)
    assert fit.n_points >= 3
    assert fit.residual_rms < 1.0e-15


def test_wall_stress_fit_recovers_wall_origin_with_fixed_slope() -> None:
    nu, u_tau, y_origin = 5.0e-4, 0.003, 0.645
    y = np.arange(1.0, 8.0)
    u = _viscous_profile(y_origin, u_tau, nu, y)
    fit = fit_wall_origin_wall_stress(y, u, nu, u_tau**2, n_cells=2)
    assert fit.y_origin == pytest.approx(y_origin, abs=1.0e-9)
    assert fit.u_tau == pytest.approx(u_tau, rel=1.0e-12)


def test_log_law_one_dimensional_solve_recovers_friction_velocity() -> None:
    nu, u_tau, y_origin = 5.0e-4, 0.003, 0.645
    y = np.arange(1.0, 63.0)
    u = _log_profile(y_origin, u_tau, nu, y)
    solved = u_tau_from_log_law(y, u, nu, y_origin=y_origin, b=5.0)
    assert solved["mode"] == "level"
    assert solved["u_tau"] == pytest.approx(u_tau, rel=1.0e-3)
    slope = u_tau_from_log_law(y, u, nu, y_origin=y_origin, b=5.0, mode="slope")
    assert slope["u_tau"] == pytest.approx(u_tau, rel=1.0e-9)
    assert slope["b_recovered"] == pytest.approx(5.0, abs=1.0e-6)
    assert solved["n_points"] >= 3
    assert solved["residual_rms"] < 1.0e-6
    # A five per cent wall-origin error must move the one-dimensional solve far
    # less than a five per cent friction-velocity error (the route is stable).
    shifted = u_tau_from_log_law(y, u, nu, y_origin=y_origin + 0.05, b=5.0)
    assert abs(shifted["u_tau"] - u_tau) / u_tau < 0.05


def test_joint_log_law_fit_is_flagged_as_weakly_identified() -> None:
    """A joint (y_w, u_tau) log-law fit is a ridge problem - prove it, do not hide it."""
    nu, u_tau, y_origin = 5.0e-4, 0.003, 0.645
    y = np.arange(1.0, 63.0)
    u = _log_profile(y_origin, u_tau, nu, y)
    fit = fit_wall_origin_loglaw(y, u, nu, u_tau_initial=u_tau)
    degeneracy = fit.diagnostics["degeneracy"]
    # The origin sensitivity is orders of magnitude weaker than the u_tau one.
    ratio = abs(degeneracy["gradient_origin"]) / abs(degeneracy["gradient_u_tau"])
    assert ratio < 1.0e-2
    assert abs(fit.u_tau - u_tau) / u_tau < 0.05


def test_poiseuille_wall_origin_recovers_halfway_walls() -> None:
    nu, body_force, ny = 5.0e-4, 1.0e-9, 33
    y = np.arange(1.0, ny - 1.0)  # fluid samples, wall nodes at y = 0 and ny-1
    wall_low, wall_high = 0.5, ny - 1.5
    height = wall_high - wall_low
    distance = y - wall_low
    u = body_force / (2.0 * nu) * distance * (height - distance)
    measured = poiseuille_wall_origin(y, u, nu, body_force)
    assert measured["wall_position_low"] == pytest.approx(wall_low, abs=1.0e-6)
    assert measured["wall_position_high"] == pytest.approx(wall_high, abs=1.0e-6)
    assert measured["wall_offset_from_first_sample"] == pytest.approx(-0.5, abs=1.0e-6)


def test_wall_offset_bias_quantifies_the_assumed_value_error() -> None:
    """At production resolution a 0.145-cell origin error is a ~0.8 wall-unit shift."""
    nu, u_tau = 5.16666e-4, 0.002770791905384009
    y = np.arange(1.0, 63.0)
    u = _log_profile(0.645, u_tau, nu, y)
    bias = wall_offset_bias(
        y,
        u,
        nu,
        u_tau,
        y_origin_assumed=0.5,
        y_origin_measured=0.645,
        y_plus_window=(30.0, 100.0),
    )
    assert bias["y_plus_shift"] == pytest.approx(0.145 * u_tau / nu, rel=1.0e-9)
    assert 0.5 < bias["y_plus_shift"] < 1.0
    assert bias["u_plus_rms_bias"] > 0.0
    assert bias["exceeds_0p1_cell_trigger"] is True
    # A consistent origin produces (numerically) zero bias.
    null = wall_offset_bias(
        y,
        u,
        nu,
        u_tau,
        y_origin_assumed=0.645,
        y_origin_measured=0.645,
        y_plus_window=(30.0, 100.0),
    )
    assert null["u_plus_max_bias"] < 1.0e-12
    assert null["exceeds_0p1_cell_trigger"] is False


# ---------------------------------------------------------------------------
# Three-way closure
# ---------------------------------------------------------------------------
def test_three_way_closure_is_consistent_for_a_self_consistent_profile() -> None:
    nu, u_tau, half_height, y_origin = 5.0e-4, 0.003, 31.0, 0.645
    body_force = u_tau**2 / half_height
    y = np.arange(1.0, 63.0)
    u = _log_profile(y_origin, u_tau, nu, y)
    closure = u_tau_three_way(
        y=y,
        u=u,
        nu=nu,
        body_force=body_force,
        half_height=half_height,
        wall_shear=u_tau**2,
        y_origin=y_origin,
        b=5.0,
    )
    assert closure["u_tau"]["pressure_gradient"] == pytest.approx(u_tau, rel=1.0e-12)
    assert closure["u_tau"]["wall_shear"] == pytest.approx(u_tau, rel=1.0e-12)
    assert closure["u_tau"]["log_law"] == pytest.approx(u_tau, rel=5.0e-3)
    assert closure["consistent"] is True
    assert closure["max_pairwise_deviation_pct"] < 0.3


def test_three_way_closure_detects_an_eight_percent_shear_deficit() -> None:
    """An 8 % low momentum-exchange wall stress must be reported as inconsistent."""
    nu, u_tau, half_height, y_origin = 5.0e-4, 0.003, 31.0, 0.645
    body_force = u_tau**2 / half_height
    y = np.arange(1.0, 63.0)
    u = _log_profile(y_origin, u_tau, nu, y)
    closure = u_tau_three_way(
        y=y,
        u=u,
        nu=nu,
        body_force=body_force,
        half_height=half_height,
        wall_shear=0.92 * u_tau**2,
        y_origin=y_origin,
        b=5.0,
    )
    deficit = closure["pairwise_deviation_pct"]["pressure_gradient_vs_wall_shear"]
    assert deficit == pytest.approx(100.0 * (1.0 - np.sqrt(0.92)) / (0.5 * (1.0 + np.sqrt(0.92))), rel=1.0e-6)
    assert closure["consistent"] is False


# ---------------------------------------------------------------------------
# Reference DNS (Moser-Kim-Moin 1999, Re_tau = 178.12)
# ---------------------------------------------------------------------------
def _load_mkm() -> tuple[np.ndarray, np.ndarray]:
    y_plus, u_plus = [], []
    with open(MKM_CSV, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            y_plus.append(float(row["y_plus"]))
            u_plus.append(float(row["u_plus"]))
    y_plus = np.asarray(y_plus)
    u_plus = np.asarray(u_plus)
    keep = y_plus > 0.0
    return y_plus[keep], u_plus[keep]


def test_three_way_closure_on_mkm_reference_dns() -> None:
    """Reference-data regression: the closure routes on real DNS at Re_tau = 180.

    In wall units (y = y+, u = u+, nu = 1, a h = 1) all three routes must return
    u_tau = 1.  The pressure/wall-shear pair closes to ~1.5 %, while the
    universal log law (kappa = 0.41, B = 5.0) sits ~3.4 % high: at this Reynolds
    number the log layer is too thin for the universal constants, which is the
    documented model-form error this test pins so it cannot be silently absorbed.
    """
    y_plus, u_plus = _load_mkm()
    closure = u_tau_three_way(
        y=y_plus,
        u=u_plus,
        nu=1.0,
        body_force=1.0,
        half_height=1.0,
        b=5.0,
    )
    routes = closure["u_tau"]
    assert routes["pressure_gradient"] == pytest.approx(1.0, rel=1.0e-12)
    assert routes["wall_shear"] == pytest.approx(0.9855, abs=5.0e-3)
    assert routes["log_law"] == pytest.approx(1.046, abs=1.0e-2)
    # Honest verdict: the three routes do NOT close to 3 % on real DNS with the
    # universal log-law constants.
    assert closure["max_pairwise_deviation_pct"] == pytest.approx(5.95, abs=0.5)
    # The slope variant of the same log law sits 5-6 % higher still: that band is
    # the model-form uncertainty of "a log-law friction velocity" at Re_tau=180.
    assert closure["log_law_slope_variant"]["u_tau"] == pytest.approx(1.106, abs=1.0e-2)
    assert 4.0 < closure["log_law_model_form_band_pct"] < 8.0
    assert closure["consistent"] is False
    # The measured wall origin of the DNS itself is the wall: y_w -> 0.
    assert abs(closure["wall_origin"]["measured"]) < 0.1


def test_spalding_fit_on_mkm_reference_dns_recovers_the_wall() -> None:
    y_plus, u_plus = _load_mkm()
    fit = fit_wall_origin_spalding(y_plus, u_plus, 1.0, u_tau_initial=1.0)
    assert fit.u_tau == pytest.approx(1.05, abs=0.05)
    assert abs(fit.y_origin) < 0.2


def test_auto_method_dispatch() -> None:
    nu, u_tau, y_origin = 5.0e-4, 0.003, 0.645
    y = np.arange(1.0, 63.0)
    # First cell sits at y+ ~ 2.1 -> viscous branch.
    viscous = fit_wall_origin(y, _viscous_profile(y_origin, u_tau, 1.5e-3, y), 1.5e-3)
    assert viscous.method == "viscous"
    # First sample above the sublayer (y_+ ~ 10) -> composite branch.
    coarse_y = np.arange(7.0, 63.0)
    coarse = fit_wall_origin(coarse_y, _log_profile(y_origin, u_tau, nu, coarse_y), nu)
    assert coarse.method == "spalding"
    with pytest.raises(ValueError):
        fit_wall_origin(y, y, nu, method="not_a_method")


# ---------------------------------------------------------------------------
# Regression guard on the measured fabrication calibration
# ---------------------------------------------------------------------------
def test_laminar_calibration_artifact_reports_a_measured_origin() -> None:
    """The hardcoded 0.5 must not come back: the calibration artefact is binding."""
    payload = json.loads(CALIBRATION.read_text())
    rows = payload["runs"]
    assert rows, "calibration artefact is empty"
    for row in rows:
        measured = row["measured"]["wall_origin_estimators"]
        stress_value = measured["wall_stress_local"]
        assert 0.5 < stress_value < 0.75
        assert abs(measured["spread"]) < 0.2
        delta = row["assumed_vs_measured"]["delta_cells"]
        assert delta > 0.05
        # The previously published 0.1 value is not reproducible with the
        # production wall BC, so the artefact must not claim it.
        assert stress_value > 0.3