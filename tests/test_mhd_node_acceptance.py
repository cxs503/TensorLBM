"""Node p315w1-n04 acceptance gates: Hartmann / Shercliff / degeneration / convergence.

Small grids only (the full campaign grid ladder lives in
``scripts/run_p315w1_n04_mhd.py``); every gate here is the *same* threshold as
the node specification, so the tests fail if a code change breaks the physics
surface even before the campaign is re-run.

Gate map
--------
(a) Hartmann profile error <= 3 %      -> :func:`test_hartmann_profile_error_below_three_percent`
(b) Hartmann wall friction <= 3 %      -> :func:`test_hartmann_wall_friction_below_three_percent`
(c) Shercliff centreline <= 5 %        -> :func:`test_shercliff_centreline_below_five_percent`
(d) magnetic damping rate <= 5 %       -> ``tests/test_mhd_damping.py``
(e) B -> 0 Poiseuille <= 1 %           -> :func:`test_poiseuille_degeneration_below_one_percent`
(f) grid convergence order >= 1.5      -> :func:`test_grid_convergence_order_at_least_one_point_five`
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.mhd.hartmann import (
    HartmannConfig,
    hartmann_normalised_profile,
    hartmann_wall_shear_coefficient,
    poiseuille_profile,
    run_hartmann_channel,
)
from tensorlbm.mhd.shercliff import ShercliffCase, run_shercliff_channel

U_MEAN = 0.02
NU = 0.05


def centre_value(profile) -> float:
    """Centreline velocity by three-point Richardson extrapolation (exact for ``y^2``)."""
    u = np.asarray(profile, dtype=float)
    n = u.size
    a0 = 0.5 * (u[n // 2 - 1] + u[n // 2])
    a1 = 0.5 * (u[n // 2 - 2] + u[n // 2 + 1])
    return float((9.0 * a0 - a1) / 8.0)


def pointwise_error(u_lbm, u_exact, u_mean: float, fraction: float = 0.1) -> float:
    """Max local relative error on rows carrying at least ``fraction * u_mean``."""
    ua = np.asarray(u_exact, dtype=float)
    mask = np.abs(ua) >= fraction * abs(u_mean)
    err = np.abs(np.asarray(u_lbm, dtype=float) - ua)
    return float(np.max(np.where(mask, err / np.maximum(np.abs(ua), 1e-300), 0.0)))


# ---------------------------------------------------------------------------
# (a) + (b): Hartmann closed form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ha, ny", [(5.0, 32), (10.0, 64)])
def test_hartmann_profile_error_below_three_percent(ha: float, ny: int) -> None:
    cfg = HartmannConfig(ny=ny, ha=ha, u_mean=U_MEAN, nu=NU, steps=200_000, tol=1e-9)
    res = run_hartmann_channel(cfg)
    u_an = hartmann_normalised_profile(res.y_hat, ha) * U_MEAN
    assert res.converged
    # mean-normalised L-infinity norm (ASME V&V 20 style: reference scale u_mean)
    assert res.rel_linf <= 0.03
    # strict local metric on the rows that carry at least 50 % of the mean (the
    # raw local error is unbounded where u -> 0 at the no-slip wall)
    assert pointwise_error(res.u_profile, u_an, res.u_mean_analytic, 0.5) <= 0.03
    assert res.max_mach < 0.1
    assert res.closure_residual < 1e-6


@pytest.mark.parametrize("ha, ny", [(5.0, 32), (20.0, 64)])
def test_hartmann_wall_friction_below_three_percent(ha: float, ny: int) -> None:
    cfg = HartmannConfig(ny=ny, ha=ha, u_mean=U_MEAN, nu=NU, steps=200_000, tol=1e-9)
    res = run_hartmann_channel(cfg)
    analytic = hartmann_wall_shear_coefficient(ha)
    assert abs(res.wall_shear_coefficient - analytic) / analytic <= 0.03
    # the measured friction must also follow the right trend (grows with Ha)
    assert res.wall_shear_coefficient > 3.0


# ---------------------------------------------------------------------------
# (c): Shercliff (insulating walls, zero net current) centreline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ha", [5.0, 10.0])
def test_shercliff_centreline_below_five_percent(ha: float) -> None:
    case = ShercliffCase(ha=ha, ny=32, u_mean=U_MEAN, nu=NU, steps=300_000, tol=1e-9)
    run = run_shercliff_channel(case)
    u_an = centre_value(run["u_profile_analytic"])
    u_lbm = centre_value(run["u_profile"])
    assert abs(u_lbm - u_an) / abs(u_an) <= 0.05
    assert run["converged"]
    # the insulating closure really carries no net axial current
    assert run["net_current_relative"] < 1e-10
    assert run["lorentz_mean_relative"] < 1e-10


# ---------------------------------------------------------------------------
# (e): B -> 0 degeneration to plane Poiseuille
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ny", [16, 32])
def test_poiseuille_degeneration_below_one_percent(ny: int) -> None:
    cfg = HartmannConfig(ny=ny, ha=0.0, u_mean=U_MEAN, nu=NU, steps=200_000, tol=1e-9)
    res = run_hartmann_channel(cfg)
    u_an = poiseuille_profile(res.y_hat, U_MEAN)
    assert float(np.max(np.abs(res.u_profile - u_an)) / U_MEAN) <= 0.01
    assert centre_value(res.u_profile) / res.u_mean == pytest.approx(1.5, rel=0.01)
    assert abs(res.wall_shear_coefficient - 3.0) / 3.0 <= 0.01


# ---------------------------------------------------------------------------
# (f): grid convergence of the velocity field
# ---------------------------------------------------------------------------


def test_grid_convergence_order_at_least_one_point_five() -> None:
    errors = []
    for ny in (16, 32, 64):
        cfg = HartmannConfig(ny=ny, ha=5.0, u_mean=U_MEAN, nu=NU, steps=200_000, tol=1e-9)
        errors.append(run_hartmann_channel(cfg).rel_l2)
    orders = [
        float(np.log(errors[i] / errors[i + 1]) / np.log(2.0)) for i in range(len(errors) - 1)
    ]
    assert min(orders) >= 1.5, f"observed orders {orders} (errors {errors})"
    assert errors[-1] < errors[0]
    assert errors[-1] <= 0.03


def test_centre_value_helper_is_exact_on_a_parabola() -> None:
    """The three-point estimator must be exact on a parabola (no dy^2 bias)."""
    for ny in (16, 32, 64):
        y = (np.arange(ny) + 0.5 - 0.5 * ny) / (0.5 * ny)
        u = 1.0 - y * y
        assert centre_value(u) == pytest.approx(1.0, abs=1e-14)