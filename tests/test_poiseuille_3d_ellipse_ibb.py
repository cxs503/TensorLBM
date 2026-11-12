"""Regression guards for the elliptic-duct two-arm wall study (run_ibb.py).

The two-arm study compares the production half-way bounce-back with the sub-grid
SDF 2-point Bouzidi closure on an elliptic duct. Its *verdict* is a measurement
(reported in docs/POISEUILLE_3D_ELLIPSE_IBB_TWO_ARM_20261113.md), so the tests
here assert the invariants that make that measurement trustworthy:

* the signed distance used by the IBB arm has the required sign convention and
  is close (first-order Taylor, O(d^2)) to the exact Euclidean distance next to
  the wall — the only place the Bouzidi weight reads it;
* the analytic reference (White's elliptic-duct solution) is self-consistent:
  ``ellipse_g`` inverts the centre-line velocity law exactly and the flux law
  agrees with the pressure-gradient form;
* both wall arms produce finite flow fields with a positive axial velocity
  inside the fluid mask, i.e. a difference between the arms is a boundary-model
  difference, not a broken simulation.

Fast by construction: a = 6 (b = 3), a few hundred steps per arm.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "pending" / "poiseuille_3d_ellipse"))

from run_ibb import ellipse_g, ellipse_setup, run_arm  # noqa: E402


def _ellipse_points(a: float, b: float, yc: float, zc: float) -> np.ndarray:
    """Brute-force point set of the continuous ellipse boundary."""
    t = np.linspace(0.0, 2.0 * math.pi, 20000, endpoint=False)
    return np.stack([yc + a * np.cos(t), zc + b * np.sin(t)], axis=1)


def test_sdf_sign_and_near_wall_distance():
    a, b = 12, 6
    sdf, fluid, solid, lam, grid = ellipse_setup(a, b, nx=4)
    sdf2d = sdf[:, :, 0].numpy()
    fluid2d = fluid[:, :, 0].numpy()

    # sign convention required by interpolated_bounce_back()
    assert np.all(sdf2d[fluid2d] > 0.0)
    assert np.all(sdf2d[~fluid2d] < 0.0)
    # the mask is the digital ellipse of the nominal semi-axes
    assert np.array_equal(fluid2d, (lam.numpy() <= 1.0))

    # cut-link band = fluid cells with a solid 4-neighbour
    band = np.zeros_like(fluid2d)
    band[1:-1, 1:-1] = fluid2d[1:-1, 1:-1] & (
        ~fluid2d[:-2, 1:-1] | ~fluid2d[2:, 1:-1] | ~fluid2d[1:-1, :-2] | ~fluid2d[1:-1, 2:]
    )
    pts = _ellipse_points(a, b, grid["yc"], grid["zc"])
    zz, yy = np.nonzero(band)
    d_exact = np.array(
        [np.min(np.hypot(pts[:, 0] - y, pts[:, 1] - z)) for z, y in zip(zz, yy)]
    )
    err = np.abs(sdf2d[band] - d_exact)
    # sdf = -h/|grad h| is exact to O(d^2); one lattice unit from the wall the
    # Taylor remainder stays below ~0.1 cell for the 2:1 ellipse.
    assert err.max() < 0.12, f"near-wall SDF error {err.max():.4f} cells too large"


def test_analytic_reference_is_self_consistent():
    a, b, nu = 20.0, 10.0, 0.1
    u_max = 0.02
    g = ellipse_g(int(a), int(b), nu, u_max)
    u_back = g * a * a * b * b / (2.0 * nu * (a * a + b * b))
    assert abs(u_back - u_max) < 1e-12
    # White, Viscous Fluid Flow: Q = pi a b U_max / 2 = pi a^3 b^3 G / (4 mu (a^2+b^2))
    q_flux = math.pi * a * b * u_max / 2.0
    q_grad = math.pi * a**3 * b**3 * g / (4.0 * nu * (a * a + b * b))
    assert abs(q_flux - q_grad) / q_flux < 1e-12


@pytest.mark.parametrize("wall", ["halfway", "ibb"])
def test_both_arms_are_finite_and_drive_fluid_only(wall):
    met, ux, lam = run_arm(
        6, 3, wall, tau=0.8, u_max_target=0.02, nx=4, max_steps=400, acc_steps=40, tol=0.0
    )
    fluid = lam <= 1.0
    assert np.all(np.isfinite(ux))
    assert np.all(ux[fluid] > 0.0), "body-force driven flow must be positive along x"
    for key in ("theta_nom_pct", "theta_sQ_pct", "s_Q", "Q"):
        assert math.isfinite(met[key])
    # both closures must sit within a couple of lattice units of the nominal wall
    assert abs(met["delta_a_eff"]) < 2.5
    assert abs(met["delta_b_eff"]) < 2.5
    assert met["wall"] == wall and met["a"] == 6 and met["b"] == 3