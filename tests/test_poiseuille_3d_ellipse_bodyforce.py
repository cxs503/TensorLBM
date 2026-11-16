"""Analytic + wiring tests for the body-force elliptic-duct Poiseuille benchmark.

Benchmark: ``benchmarks/pending/poiseuille_3d_ellipse_bodyforce/run.py``.

Covered here (the parts that must be exact before any grid ladder is trusted):

* the closed-form elliptic-duct Poiseuille solution satisfies the governing
  balance ``nu * lap(u) + g = 0`` pointwise (finite-difference cross-check, so a
  wrong coefficient in the runner cannot hide);
* the flow-rate identity ``Q = pi a b Uc / 2`` (equivalently ``u_mean = Uc/2``);
* the two-observable hydraulic inversion ``(Q_total, Uc) -> (a_e, b_e)``
  round-trips to machine precision, with the discriminant identity
  ``S^2 - 4 P^2 = (a_e^2 - b_e^2)^2 >= 0``;
* the digital geometry (fluid-cell count vs ``pi a b``) and the half-way
  bounce-back mask (solid cells only);
* a short solver smoke run accelerates from rest;
* the production ``result.json``, when present, is re-scored from its raw
  ``per_grid`` numbers so the stored verdict cannot drift from the gates.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "pending" / "poiseuille_3d_ellipse_bodyforce"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BENCH))

from run import (  # noqa: E402
    analytic_profile,
    analytic_uc,
    ellipse_setup,
    guo_source_3d,
    inversion_two_observable,
    run_case,
)


@pytest.mark.parametrize("a,b", [(20, 10), (40, 20), (7, 3)])
def test_analytic_satisfies_stokes_balance(a, b):
    """nu*lap(u) + g = 0 must hold pointwise for the closed-form field."""
    g, nu, h = 1.25e-5, 0.1, 1e-3
    Uc = analytic_uc(g, nu, a, b)
    assert Uc > 0
    lap_exact = -2.0 * Uc / a**2 - 2.0 * Uc / b**2
    assert abs(nu * lap_exact + g) < 1e-12

    def u_of(y, z):
        return analytic_profile(g, nu, a, b, np.array([[(y / a) ** 2 + (z / b) ** 2]]))[0, 0]

    worst = 0.0
    for y in np.linspace(-0.4 * a, 0.4 * a, 5):
        for z in np.linspace(-0.4 * b, 0.4 * b, 5):
            fd = (u_of(y + h, z) - 2 * u_of(y, z) + u_of(y - h, z)) / h**2 + (
                u_of(y, z + h) - 2 * u_of(y, z) + u_of(y, z - h)
            ) / h**2
            worst = max(worst, abs(nu * fd + g))
    assert worst < 1e-8, f"Stokes balance violated by {worst}"


@pytest.mark.parametrize("a,b", [(20, 10), (40, 20)])
def test_flow_rate_identity(a, b):
    """Integral of the parabola over the ellipse: Q = pi a b Uc / 2."""
    g, nu = 1.25e-5, 0.1
    Uc = analytic_uc(g, nu, a, b)
    nr = nt = 2000
    r = np.linspace(0.0, 1.0, nr + 1)
    r = 0.5 * (r[1:] + r[:-1])
    th = np.linspace(0.0, 2 * math.pi, nt, endpoint=False)
    R, T = np.meshgrid(r, th, indexing="ij")
    y, z = a * R * np.cos(T), b * R * np.sin(T)
    lam = (y / a) ** 2 + (z / b) ** 2
    u = Uc * (1.0 - lam)
    dA = a * b * R  # Jacobian of the elliptic-polar map
    Q = float(np.sum(u * dA) * (1.0 / nr) * (2 * math.pi / nt))
    assert abs(Q / (math.pi * a * b * Uc / 2.0) - 1.0) < 1e-5


@pytest.mark.parametrize("ae,be", [(20.5, 10.5), (40.4, 20.2), (5.2, 4.9)])
def test_inversion_roundtrip(ae, be):
    g, nu = 1.25e-5, 0.1
    Uc = analytic_uc(g, nu, ae, be)
    Q_total = math.pi * ae * be * Uc / 2.0
    inv = inversion_two_observable(Q_total, Uc, nu, g)
    assert inv["a_e"] == pytest.approx(max(ae, be), rel=1e-12)
    assert inv["b_e"] == pytest.approx(min(ae, be), rel=1e-12)
    assert inv["disc"] == pytest.approx((ae**2 - be**2) ** 2, rel=1e-10)
    assert inv["disc"] >= 0.0


def test_geometry_and_mask():
    a, b = 40, 20
    _, _, _, _, _, lam, fluid2d, wall_mask = ellipse_setup(a, b, 4, torch.device("cpu"))
    n_fluid = int(fluid2d.sum())
    assert abs(n_fluid / (math.pi * a * b) - 1.0) < 0.02
    # bounce-back mask = SOLID cells only (half-way condition at link midpoint)
    assert int(wall_mask[:, :, 0].sum()) == int((lam > 1.0).sum())
    assert wall_mask.shape == (2 * b + 3, 2 * a + 3, 4)


def test_guo_source_moments():
    """Zero-sum source: sum S_q = 0 and sum c_qx S_q = g (exact force)."""
    from tensorlbm.d3q19 import C

    zero = torch.zeros((5, 6, 3), dtype=torch.float64)  # float64: exact moments
    g = 1.25e-5
    S = guo_source_3d(zero, zero, zero, g, 0.8)
    n_cells = zero.numel()
    assert float(S.sum()) == pytest.approx(0.0, abs=1e-17)  # zero-sum
    fx_per_cell = float((C.double()[:, 0].view(19, 1, 1, 1) * S).sum()) / n_cells
    # Guo 2002: the discrete source delivers (1 - 1/(2 tau)) of the nominal
    # force; the other half is carried by the u + F/(2 rho) equilibrium shift.
    assert fx_per_cell == pytest.approx((1.0 - 0.5 / 0.8) * g, rel=1e-12)


def test_smoke_run_accelerates_from_rest():
    r = run_case(
        a=6, b=3, tau=0.8, g=1.25e-5, nx=4, max_steps=900, out_path=None,
        device=torch.device("cpu"), chunk=300, tol=0.0,
    )
    assert r["finite"] and r["u_center"] > 0.0
    tail = [h["u_center"] for h in r["steady_hist_tail"]]
    assert tail[-1] >= tail[0]


RESULT = BENCH / "ladder" / "result.json"


@pytest.mark.skipif(not RESULT.exists(), reason="production ladder not present")
def test_ladder_verdict_is_recomputable():
    d = json.loads(RESULT.read_text())
    rows = d["per_grid"]
    assert len(rows) >= 3
    bin_cal = [r["bin_cal_pct"] for r in rows]
    cell_cal = [r["cell_cal_pct"] for r in rows]
    d_ae = [r["d_ae"] for r in rows]
    d_be = [r["d_be"] for r in rows]
    g1 = all(v <= 3.0 for v in bin_cal)
    g2 = all(bin_cal[i + 1] < bin_cal[i] for i in range(len(bin_cal) - 1))
    g3 = all(cell_cal[i + 1] < cell_cal[i] for i in range(len(cell_cal) - 1))
    g4 = all(0.0 < v <= 1.0 for v in d_ae) and all(0.0 < v <= 1.0 for v in d_be)
    assert d["verified"] == (g1 and g2 and g3 and g4)
