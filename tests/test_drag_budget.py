"""Exact-algebra tests for the control-volume (momentum-budget) drag estimator.

Every case here has a closed-form answer, so the tests assert *invariants*
(exact values, sign conventions, face-by-face additivity), not merely that the
function runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.drag_budget import momentum_balance_drag

U = 1.0
SIDE = 20.0


def _uniform(ny: int = 20, nx: int = 30, ux: float = U, uy: float = 0.0, rho: float = 1.0):
    rhouux = np.full((ny, nx), rho * ux * ux)
    rhouxuy = np.full((ny, nx), rho * ux * uy)
    p = np.full((ny, nx), rho / 3.0)
    return rhouux, rhouxuy, p


def test_uniform_stream_has_zero_drag():
    """A uniform mean stream carries the same momentum on every face."""
    rhouux, rhouxuy, p = _uniform()
    out = momentum_balance_drag(rhouux, rhouxuy, p, U, SIDE)
    assert out["cd"] == pytest.approx(0.0, abs=1e-12)
    assert out["faces"]["left"] == pytest.approx(-out["faces"]["right"])


def test_pressure_only_drop_gives_analytic_cd():
    """Uniform velocity + pressure drop dp over the CV -> Cd = dp*H/(0.5 U^2 D)."""
    ny, nx = 20, 30
    rhouux, rhouxuy, p = _uniform(ny, nx)
    p[:, 0] = 1.0 / 3.0 + 0.25      # left face sees p0 + dp/2
    p[:, -1] = 1.0 / 3.0 - 0.25     # right face sees p0 - dp/2
    dp = 0.5
    out = momentum_balance_drag(rhouux, rhouxuy, p, U, SIDE)
    expected = dp * ny / (0.5 * U * U * SIDE)
    assert out["cd"] == pytest.approx(expected, rel=1e-12)
    assert out["faces"]["bottom"] == pytest.approx(0.0)
    assert out["faces"]["top"] == pytest.approx(0.0)


def test_wake_deficit_gives_analytic_cd():
    """Outlet momentum deficit of width D and depth a*U -> Cd = 2 a (2-a) D/H.

    Physics: only the deficit band contributes, so the answer is known in
    closed form and the sign tells us the body pushes the fluid backwards.
    """
    ny, nx = 20, 30
    a = 0.5
    rhouux, rhouxuy, p = _uniform(ny, nx)
    rhouux[ny // 2 - 10: ny // 2 + 10, -1] = (U * (1.0 - a)) ** 2  # full-width deficit of D cells
    p[:, -1] = 1.0 / 3.0
    out = momentum_balance_drag(rhouux, rhouxuy, p, U, SIDE)
    expected = 2.0 * a * (2.0 - a) * ny / SIDE
    assert out["cd"] == pytest.approx(expected, rel=1e-12)
    assert out["cd"] > 0.0


def test_transverse_flux_cancels_for_symmetric_inflow():
    """Equal inflow/outflow across top and bottom carries no net x-momentum."""
    ny, nx = 20, 30
    rhouux, rhouxuy, p = _uniform(ny, nx, uy=0.3)
    out = momentum_balance_drag(rhouux, rhouxuy, p, U, SIDE)
    assert out["faces"]["top"] == pytest.approx(-out["faces"]["bottom"])
    assert out["cd"] == pytest.approx(0.0, abs=1e-12)


def test_net_flux_and_faces_are_additive():
    ny, nx = 20, 30
    rng = np.random.default_rng(0)
    rhouux = rng.random((ny, nx))
    rhouxuy = rng.random((ny, nx)) - 0.5
    p = rng.random((ny, nx))
    out = momentum_balance_drag(rhouux, rhouxuy, p, U, SIDE)
    faces = out["faces"]
    assert out["net_flux"] == pytest.approx(sum(faces.values()), rel=1e-12)
    assert out["drag"] == pytest.approx(-out["net_flux"], rel=1e-12)


def test_input_validation():
    with pytest.raises(ValueError):
        momentum_balance_drag(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)), U, SIDE)
    with pytest.raises(ValueError):
        momentum_balance_drag(np.zeros((4, 4)), np.zeros((5, 4)), np.zeros((4, 4)), U, SIDE)
    with pytest.raises(ValueError):
        momentum_balance_drag(np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4)), 0.0, SIDE)
