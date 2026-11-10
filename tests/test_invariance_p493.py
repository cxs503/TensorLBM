"""Tests for the matured invariance certificate (P493-N02).

These pin the *numerical claims* of the upgraded surface on tiny grids:

* the force-driven periodic channel is second order when the walls sit on the
  lattice midpoint and loses that order (strictly first order for the half-way
  closure) when they do not -- this is the negative control of criterion C6;
* the body-force driven inclined channel keeps the density uniform, which is the
  compressibility gate C8;
* the Galilean decay-rate deviation follows the Ma^2 law (C5) and the
  superposition defect stays bounded (C4);
* the acceptance logic itself accepts a consistent campaign and rejects broken
  data.

Every simulation here is deliberately small (a few seconds); the production
sweeps live in ``scripts/run_invariance_campaign_p493.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.invariance import (
    evaluate_criteria,
    galilean_superposition_residual,
    observed_order,
    periodic_channel_case,
    poiseuille_case,
)
from tensorlbm.invariance.diagnostics import effective_wall_offset, wall_slip


# ---------------------------------------------------------------- C1 / C6 wall
def _periodic(offset, bounce_back="interpolated", widths=(8, 16)):
    errors = []
    for h in widths:
        res = periodic_channel_case(h, offset=offset, bounce_back=bounce_back,
                                    u_mean=0.01, nu=0.1, steps_factor=20.0)
        errors.append(res["l2_error"])
    return errors, observed_order(errors, list(widths))


def test_periodic_channel_second_order_at_lattice_midpoint():
    """Walls at q = 0.5: interpolated and half-way bounce-back coincide (C1)."""
    errors, orders = _periodic(0.0)
    assert min(errors) > 0.0
    assert orders[0] > 1.6, f"observed order {orders[0]:.2f} at q=0.5"


def test_halfway_bounce_back_is_first_order_at_off_midpoint_wall():
    """q = 0.25 with the half-way closure cannot see the wall position (C6)."""
    errors, orders = _periodic(0.25, bounce_back="halfway")
    assert orders[0] < 1.2, f"half-way order {orders[0]:.2f} should be <= 1.2"
    interp, interp_orders = _periodic(0.25)
    ratio = errors[-1] / interp[-1]
    # the ratio grows with resolution (14.5x at H=24 in the campaign); on the
    # tiny unit-test grids it is only ~1.4x
    assert ratio > 1.25, f"error ratio {ratio:.2f} (half-way / interpolated) too small"
    # the interpolated closure improves the coefficient but does not restore
    # second order at q != 0.5; both facts are part of the certificate
    assert interp_orders[0] < 1.5


def test_interpolated_wall_beats_halfway_at_off_midpoint():
    _, interp = _periodic(0.25)
    _, half = _periodic(0.25, bounce_back="halfway")
    assert max(interp) < max(half)


# ------------------------------------------------------------------ C8 / C9
def test_body_force_channel_keeps_density_uniform():
    """Zero pressure gradient drive => compressibility gate passes by orders."""
    body = poiseuille_case(15.0, 8, u_mean=0.01, nu=0.1, steps_factor=12.0,
                           drive="body_force")
    press = poiseuille_case(15.0, 8, u_mean=0.01, nu=0.1, steps_factor=12.0,
                            drive="pressure")
    assert body["density"]["max_abs"] < 1e-3
    assert press["density"]["max_abs"] > body["density"]["max_abs"]
    # flux is imposed exactly at the inlet in both variants
    assert body["flux_error_rel"] < 1e-6


def test_wall_slip_and_offset_diagnostic_recovers_a_known_slip():
    """A synthetic field with a known slip must be measured back."""
    from tensorlbm.invariance.cases import PeriodicChannel, _body_force

    width, nu, u_mean = 16.0, 0.1, 0.01
    geom = PeriodicChannel(8, int(width) + 8, width, 0.0)
    G = _body_force(width, nu, u_mean)
    u_ex = (G / (2 * nu)) * geom.sdf * (width - geom.sdf)
    slip = 1e-4
    ux = u_ex + slip
    zeros = np.zeros_like(ux)
    out = wall_slip(geom, ux, zeros, u_ex, zeros, trim=0.0)
    assert abs(out["mean_slip"] - slip) < 1e-6
    delta = effective_wall_offset(geom, out["mean_slip"], u_mean, nu)
    dudn = G * width / (2 * nu)
    assert abs(delta + slip / dudn) < 1e-6


# --------------------------------------------------------------------- C4/C5
def test_galilean_decay_deviation_follows_ma_squared():
    runs = []
    for periods in (1, 2, 4):
        runs.append(galilean_superposition_residual(
            1.0, periods=periods, n=32, steps=2000, u0=0.02, nu=0.01,
            sample_every=20, fit_start=200))
    base = galilean_superposition_residual(
        0.0, periods=0, n=32, steps=2000, u0=0.02, nu=0.01,
        sample_every=20, fit_start=200)
    lam0 = base["decay_rate_base"]
    pts = [(r["ma"], abs((r["decay_rate"] - lam0) / lam0)) for r in runs]
    assert all(np.isfinite(m) and m > 0 and v > 0 for m, v in pts)
    slope = np.polyfit(np.log([m for m, _ in pts]), np.log([v for _, v in pts]), 1)[0]
    assert 1.5 < slope < 2.5, f"log-log slope {slope:.2f} (expected ~2)"
    # superposition defect: bounded and normalised by the boost
    assert all(0.0 <= r["superposition_residual_relU"] < 1e-2 for r in runs)


def test_superposition_is_exact_when_the_boost_vanishes():
    out = galilean_superposition_residual(
        0.0, periods=0, n=32, steps=200, u0=0.02, nu=0.01,
        sample_every=20, fit_start=50)
    assert out["superposition_residual"] == 0.0
    assert math.isnan(out["superposition_residual_relU"])


# -------------------------------------------------------------- acceptance
def _synthetic_campaign() -> dict:
    widths = [8.0, 12.0, 16.0, 24.0]

    def _entry(err0, power=2.0):
        errs = [err0 * (w / widths[0]) ** -power for w in widths]
        return {"widths": widths, "errors": errs,
                "orders": observed_order(errs, widths), "runs": []}

    periodic = {"interpolated": _entry(2.0e-4)}
    poiseuille = {f"{th:g}": {**_entry(3.0e-4),
                              "wall_offset_cells": [-0.05, -0.05, -0.05, -0.05]}
                  for th in (0.0, 15.0, 30.0, 45.0)}
    cylinder = {re: [{"theta_deg": th, "cd": 1.5 * (1 + 0.001 * th / 45.0),
                      "st": 0.17 * (1 + 0.001 * th / 45.0)}
                     for th in (0.0, 15.0, 30.0, 45.0)]
                for re in ("100", "200")}
    galilean = [{"ma": 0.0, "periods": 0, "decay_rate": 1.0, "decay_rate_base": 1.0,
                 "decay_rel_deviation": 0.0, "superposition_residual_relU": float("nan")}]
    for i, ma in enumerate((0.01, 0.02, 0.04)):
        galilean.append({"ma": ma, "periods": i + 1, "decay_rate": 1.0 + 1e-3 * (ma / 0.01) ** 2,
                         "decay_rate_base": 1.0,
                         "decay_rel_deviation": 1e-3 * (ma / 0.01) ** 2,
                         "superposition_residual_relU": 2e-3})
    return {
        "periodic_channel": periodic,
        "poiseuille": poiseuille,
        "cylinder": cylinder,
        "galilean": galilean,
        "negative_control": {"interpolated": _entry(2.0e-4),
                             "halfway": _entry(2.0e-4, power=1.0)},
        "reproducibility": ["abc", "abc"],
    }


def test_acceptance_all_pass_on_consistent_data():
    report = evaluate_criteria(_synthetic_campaign())
    by_id = {c["id"]: c for c in report["criteria"]}
    assert {"C1", "C1b", "C2", "C3", "C4", "C5", "C6", "C7", "C9"} <= set(by_id)
    for cid, crit in by_id.items():
        assert crit["pass"], f"{cid} unexpectedly failed: {crit}"
    assert report["all_pass"] is True


def test_acceptance_detects_violations():
    res = _synthetic_campaign()
    res["cylinder"]["100"][1]["cd"] *= 1.10          # 10 % drag spread
    res["reproducibility"] = ["abc", "def"]           # not reproducible
    res["periodic_channel"]["interpolated"]["errors"] = [1e-4, 1e-4, 1e-4, 1e-4]
    res["periodic_channel"]["interpolated"]["orders"] = observed_order(
        [1e-4, 1e-4, 1e-4, 1e-4], [8.0, 12.0, 16.0, 24.0])
    report = evaluate_criteria(res)
    by_id = {c["id"]: c for c in report["criteria"]}
    assert by_id["C2"]["pass"] is False
    assert by_id["C7"]["pass"] is False
    assert by_id["C1"]["pass"] is False
    assert report["all_pass"] is False


def test_negative_control_criterion_needs_a_broken_halfway_closure():
    res = _synthetic_campaign()
    widths = [8.0, 12.0, 16.0, 24.0]
    errs = [2.0e-4 * (w / widths[0]) ** -2.0 for w in widths]
    res["negative_control"]["halfway"] = {"widths": widths, "errors": errs,
                                          "orders": observed_order(errs, widths)}
    report = evaluate_criteria(res)
    by_id = {c["id"]: c for c in report["criteria"]}
    assert by_id["C6"]["pass"] is False