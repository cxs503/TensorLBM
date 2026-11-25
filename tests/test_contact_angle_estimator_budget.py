"""Regression tests for the contact-angle error budget (anchor XVAL-LBM-HUANG2007).

These pin the four facts the budget report rests on:

* the synthetic cap generator really produces an arc of known angle, so it can
  serve as ground truth for the measurement operator;
* the angle operator's bias is inside the owner gate but *radius dependent*,
  which is why a coarse calibration cannot be moved to another grid unchanged;
* the delivered calibration sweep contains a genuine plateau (identical angles
  for f_w = 0.60 / 0.62 / 0.64), i.e. the "non-monotone curve" of m22445 is an
  insensitivity, not a fold — asserted from the committed evidence so the claim
  cannot silently drift;
* ``calibrate()`` never labels an unmeasured point as ``ok`` with a NaN angle.
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from tensorlbm.contact_angle_inversion.synthetic import (  # noqa: E402
    cap_geometry,
    sessile_cap_field,
)
from tensorlbm.sc_surface_tension import contact_angle_from_mask  # noqa: E402

MID = 0.5 * (1.957 + 0.1596)
EVID = os.path.join(REPO, "benchmarks", "xval_huang2007_contact_angle_inversion",
                    "estimator_budget", "evidence")


def test_cap_geometry_matches_circle_formulae():
    g = cap_geometry(90.0, 15.0, 1.5)
    assert g["height"] == pytest.approx(15.0, rel=1e-12)
    assert g["half_width"] == pytest.approx(15.0, rel=1e-12)
    g60 = cap_geometry(60.0, 10.0, 0.0)
    assert g60["height"] == pytest.approx(10.0 * (1.0 - 0.5), rel=1e-12)
    assert g60["half_width"] == pytest.approx(10.0 * math.sin(math.radians(60)), rel=1e-12)


def test_synthetic_mask_boundary_is_the_arc():
    """The ``rho = mid`` level set must sit on the analytic circle."""
    th, R = 90.0, 15.0
    f = sessile_cap_field(th, R, nx=128, ny=64, wall_rows=2, interface_width=2.5)
    rho, solid, yc = f["rho"], f["solid"], f["geometry"]["y_c"]
    xs = torch.arange(rho.shape[1], dtype=torch.float64).view(1, -1)
    ys = torch.arange(rho.shape[0], dtype=torch.float64).view(-1, 1)
    r = torch.sqrt((xs - 0.5 * (rho.shape[1] - 1)) ** 2 + (ys - yc) ** 2)
    # cells straddling the arc must bracket mid density
    band = (r - R).abs() < 1.0
    vals = rho[band & (~solid)]
    assert float(vals.min()) < MID < float(vals.max())


@pytest.mark.parametrize("R", [8.0, 15.0])
def test_estimator_bias_is_inside_the_owner_gate(R):
    nx, ny = (64, 32) if R == 8.0 else (128, 64)
    for th in (80.0, 90.0, 110.0):
        f = sessile_cap_field(th, R, nx=nx, ny=ny, wall_rows=2, x_phase=0.25)
        m = contact_angle_from_mask(f["rho"], f["solid"], mid=MID)
        assert m["status"] == "ok", (R, th, m["status"])
        assert abs(m["theta_deg"] - th) / th * 100.0 < 15.0


def test_estimator_bias_depends_on_drop_radius():
    """Coarse calibration carries a bigger offset than the production grid."""
    bias = {}
    for R, (nx, ny) in ((8.0, (64, 32)), (15.0, (128, 64))):
        f = sessile_cap_field(90.0, R, nx=nx, ny=ny, wall_rows=2, x_phase=0.25)
        m = contact_angle_from_mask(f["rho"], f["solid"], mid=MID)
        bias[R] = m["theta_deg"] - 90.0
    assert bias[8.0] > bias[15.0] + 0.5, bias


def _load(name):
    with open(os.path.join(EVID, name)) as fh:
        return json.load(fh)


def test_delivered_sweep_has_an_insensitivity_plateau():
    """f_w = 0.60 / 0.62 / 0.64 return the same angle on the coarse grid."""
    dense = _load("D_dense.json")
    seen = {round(p["f_w"], 2): p["theta_measured"] for p in dense["points"]
            if p.get("status") == "ok" and p["theta_measured"] is not None}
    plateau = [seen[k] for k in (0.60, 0.62, 0.64) if k in seen]
    assert len(plateau) == 3, seen
    assert max(plateau) - min(plateau) < 1e-6, plateau


def test_calibration_points_are_never_ok_with_a_nan_angle():
    for name in ("D_dense.json", "D_dense_fine.json"):
        path = os.path.join(EVID, name)
        if not os.path.exists(path):
            continue
        for p in _load(name)["points"]:
            if p["status"] == "ok":
                assert p["theta_measured"] == p["theta_measured"], (name, p)
                assert math.isfinite(p["theta_measured"]), (name, p)


def test_budget_terms_are_additive_and_dominated_by_the_in_config_part():
    b = _load("D_budget.json")["budget_theta90_deg"]
    assert b["in_config_term"] + b["residual_transfer_term"] == pytest.approx(
        b["total_on_production_grid"], rel=1e-9)
    assert b["shares_pct"]["in_config"] > 50.0, b["shares_pct"]


def test_calibrate_maps_unknown_rejection_to_not_measured(monkeypatch):
    """A rejected measurement must not masquerade as a calibration point."""
    from tensorlbm.contact_angle_inversion import calibration as calmod

    monkeypatch.setattr(calmod, "_measure_theta",
                        lambda *a, **k: {"status": "degenerate-fit"})
    res = calmod.calibrate(fw_values=[0.5], nx=16, ny=16, radius=3.0, steps=2,
                           threads=1)
    assert res.points[0].status == "not_measured"
    assert not math.isfinite(res.points[0].theta_measured)