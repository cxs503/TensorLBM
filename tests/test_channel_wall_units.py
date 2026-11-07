"""Tests for the channel wall-unit reduction (``y_w`` / ``u_tau`` layer).

The first test is the regression that the previous campaign needed: comparing a
*full-height* channel profile against a *half-channel* DNS table inflates the
u+ error by a factor > 2 and must be impossible to do silently.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from tensorlbm.validation import channel_wall_units as cwu

REPO = Path(__file__).resolve().parents[1]
MKM180 = REPO / "benchmarks" / "pending" / "channel_turb_3d" / "dns_ref" / "mkm180.csv"


def load_mkm180() -> tuple[np.ndarray, np.ndarray]:
    y_plus, u_plus = [], []
    with open(MKM180, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            y_plus.append(float(row["y_plus"]))
            u_plus.append(float(row["u_plus"]))
    return np.array(y_plus), np.array(u_plus)


def parabola(ny: int, *, nu: float, body_force: float, y_w: float) -> np.ndarray:
    """Discrete plane-Poiseuille profile with an effective wall at ``y_w``.

    ``s = i - y_w`` is the distance from the effective wall and the opposite
    wall sits at ``s = H`` with ``H = ny - 1 - 2 y_w`` (see
    :func:`tensorlbm.validation.channel_wall_units.effective_height`).
    """
    span = cwu.effective_height(ny, y_w)
    y = np.arange(ny, dtype=float)
    s = y - y_w
    profile = body_force / (2.0 * nu) * s * (span - s)
    profile[s <= 0.0] = 0.0
    profile[s >= span] = 0.0
    return profile


def test_fold_reports_symmetry_and_avoids_clamping():
    """Folding halves the rows and keeps every y+ inside the DNS support.

    Regression for the previous campaign: a *full-height* channel profile was
    interpolated against the *half-channel* DNS table, mapping the second half
    to y+ > Re_tau and comparing it with the centre-line value.  That alone
    inflated the reported u+ error from 5.7 % to 15.2 %.
    """
    y_plus, u_plus = load_mkm180()
    ny, y_w = 200, 0.6447
    rows_folded = (ny - 2) // 2
    dx_plus = 175.0 / rows_folded
    index = np.arange(ny, dtype=float)
    profile = np.zeros(ny)
    half_rows = np.arange(ny // 2) + 0.0
    lower = np.interp((half_rows - y_w) * dx_plus, y_plus, u_plus, left=0.0)
    profile[: ny // 2] = lower
    profile[ny - 1 - np.arange(ny // 2)] = lower  # mirror about the centre line
    profile[: int(np.ceil(y_w))] = 0.0
    y, u, info = cwu.fold_channel_profile(profile, y_w=y_w)
    assert info["rows_folded"] == rows_folded
    assert info["symmetry_index_error"] == 0.0
    folded = cwu.profile_error_vs_reference(y * dx_plus, u, y_plus, u_plus)
    full_y = index[1:-1] - y_w
    unfolded = cwu.profile_error_vs_reference(
        full_y * dx_plus, profile[1:-1], y_plus, u_plus
    )
    assert folded["clamped_fraction"] == 0.0
    assert unfolded["clamped_fraction"] > 0.45
    assert folded["relative_l2"] < 0.02
    assert unfolded["relative_l2"] > 0.10


def test_predecessor_channel_run_reanalysed_with_fold():
    """Re-reduce the archived P20-N01 run: 15.2 % -> 5.7 % against MKM.

    The archived full-height profile is the exact data the previous node
    reported as a 15.19 % u+ error.  Folding it (and using the measured
    ``y_w = 0.6447``) brings it below the 10 % gate, i.e. most of the reported
    gap was post-processing, not physics.
    """
    import json

    path = REPO / "benchmarks" / "pending" / "channel_turb_3d" / "runs_3d" / "L3_result.json"
    if not path.exists():  # pragma: no cover - archived artefact may be pruned
        pytest.skip("archived P20-N01 run not present")
    record = json.loads(path.read_text())
    profile = np.array(record["statistics"]["mean_velocity_profile"])
    derived = record["derived"]
    u_tau = record["configuration"]["u_tau"]
    nu = derived["nu"]
    y_plus, u_plus = load_mkm180()
    y, u, _ = cwu.fold_channel_profile(profile, y_w=0.6447)
    folded = cwu.profile_error_vs_reference(y * u_tau / nu, u / u_tau, y_plus, u_plus)
    full_y = np.arange(1, profile.size - 1) - 0.6447
    unfolded = cwu.profile_error_vs_reference(
        full_y * u_tau / nu, profile[1:-1] / u_tau, y_plus, u_plus
    )
    assert unfolded["relative_l2"] > 0.14
    assert folded["relative_l2"] < 0.07
    assert folded["clamped_fraction"] == 0.0


def test_bulk_velocity_recovers_analytic_parabola_mean():
    ny, nu, acc, y_w = 42, 0.05, 2.0e-5, 0.6447
    profile = parabola(ny, nu=nu, body_force=acc, y_w=y_w)
    u_bulk = cwu.bulk_velocity(profile, y_w=y_w)
    # Plane Poiseuille: U_b = a H^2 / (12 nu) = (2/3) of the *analytic* peak.
    u_peak = acc * cwu.effective_height(ny, y_w) ** 2 / (8.0 * nu)
    assert u_bulk == pytest.approx(2.0 / 3.0 * u_peak, rel=2e-3)
    height = cwu.effective_height(ny, y_w)
    assert u_bulk == pytest.approx(acc * height**2 / (12.0 * nu), rel=2e-3)


def test_wall_origin_inversion_recovers_synthetic_offset():
    for y_w_true in (0.5, 0.6447, 0.75):
        ny, nu, acc = 42, 0.05, 2.0e-5
        profile = parabola(ny, nu=nu, body_force=acc, y_w=y_w_true)
        fit = cwu.measure_wall_origin_poiseuille(profile, nu=nu, body_force=acc)
        assert fit["y_w"] == pytest.approx(y_w_true, abs=2e-3)
        assert fit["residual_max_over_centre"] < 1e-3


def test_u_tau_three_way_is_consistent_on_ideal_data():
    """A converged case (wall shear = momentum balance, log law satisfied)."""
    y_w, nu, u_tau = 0.6447, 2.0e-4, 0.006
    y = np.linspace(1.0, 20.0, 40)
    y_plus = y * u_tau / nu
    u = u_tau * (np.log(y_plus) / cwu.KAPPA + cwu.B_LOG)
    height = 46.0
    body_force = 2.0 * u_tau**2 / height
    result = cwu.u_tau_three_way(
        body_force=body_force,
        height=height,
        wall_shear=u_tau**2,
        y=y,
        u=u,
        nu=nu,
        window=(30.0, 100.0),
    )
    assert result["routes"]["momentum_balance"] == pytest.approx(u_tau, rel=1e-12)
    assert result["max_pairwise_deviation_pct"] < 2.0
    assert result["consistent"] is True


def test_u_tau_three_way_flags_out_of_balance_wall_shear():
    """A 10 % low momentum-exchange wall shear must fail the 2 % gate."""
    y_w, nu, u_tau = 0.6447, 2.0e-4, 0.006
    y = np.linspace(1.0, 20.0, 40)
    u = u_tau * (np.log(y * u_tau / nu) / cwu.KAPPA + cwu.B_LOG)
    height = 46.0
    result = cwu.u_tau_three_way(
        body_force=2.0 * u_tau**2 / height,
        height=height,
        wall_shear=(0.9 * u_tau) ** 2,
        y=y,
        u=u,
        nu=nu,
        window=(30.0, 100.0),
    )
    assert result["consistent"] is False
    assert result["max_pairwise_deviation_pct"] > 5.0
    assert "momentum_balance_vs_wall_shear" in result["deviations_pct"]


def test_log_law_model_form_gap_is_exposed_on_dns():
    """On the DNS itself the level/slope closures differ by several percent.

    This is the *reference noise floor* of criterion "three-way u_tau < 2 %":
    at Re_tau = 180 the one-parameter law of the wall cannot deliver 2 %, so a
    simulation cannot be credited for the gap either.
    """
    y_plus, u_plus = load_mkm180()
    closure = cwu.u_tau_log_law(y_plus[1:], u_plus[1:], 1.0, window=(30.0, 100.0))
    assert 1.0 < closure["level"] < 1.10
    assert closure["model_form_gap_pct"] > 2.0
    assert closure["residual_rms_plus"] < 0.1


def test_total_stress_route_recovers_u_tau_from_synthetic_channel():
    u_tau, nu, height = 0.006, 2.0e-4, 20.0
    y = np.linspace(1.0, 10.0, 40)
    # Log-layer mean profile and the matching Reynolds shear stress.
    dudy = u_tau / (cwu.KAPPA * y)
    u = u_tau * (np.log(y * u_tau / nu) / cwu.KAPPA + cwu.B_LOG)
    uv = nu * np.gradient(u, y) - u_tau**2 * (1.0 - y / height)
    recovered = cwu.u_tau_total_stress(y, u, uv, nu, height, window=(2.0, 8.0))
    assert recovered == pytest.approx(u_tau, rel=5e-2)


def test_friction_coefficient_matches_mkm_dns_and_dean():
    y_plus, u_plus = load_mkm180()
    u_bulk = cwu.bulk_velocity(u_plus, y=y_plus, height=178.12)
    cf = cwu.friction_coefficient(1.0, u_bulk)
    assert cf == pytest.approx(0.00814, rel=1e-2)
    assert u_bulk == pytest.approx(15.68, rel=1e-2)
    assert cwu.dean_channel_friction(178.12) == pytest.approx(cf, rel=5e-2)


def test_friction_coefficient_cf_scaling_with_re_tau():
    """C_f must decay with Re_tau (sanity of the correlation, no experiment)."""
    values = [cwu.dean_channel_friction(re) for re in (180.0, 395.0, 590.0)]
    assert values[0] > values[1] > values[2]
    assert math.isclose(values[1] / values[0], (395.0 / 180.0) ** -0.346, rel_tol=0.15)