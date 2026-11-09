"""Regression tests for the wall-modelled LES fixes of mission m7797.

Two defects are pinned here:

* ``coles_profile`` multiplied the Coles wake function by ``4 Pi / kappa``
  instead of ``2 Pi / kappa`` (double wake -> boundary-layer edge velocity and
  skin friction ``c_f = 2/Ue+^2`` low by ~10 %, criterion (c) failed at 11.4 %);
* the wall-modelled channel sampled its wall model on the bounce-back wall row
  and half a cell too close to the wall, which collapsed the recovered friction
  velocity to 0.60 of the imposed force balance.

Everything here is cheap (no channel time integration) so it can run in CI.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.wall_model_les.channel import ChannelWMLESConfig, _stress_coupling
from tensorlbm.wall_model_les.flat_plate import (
    PI_WAKE,
    coles_profile,
    ittc_1957_local,
    run_flat_plate,
)
from tensorlbm.wall_model_les.laws import B_DEFAULT, KAPPA_DEFAULT, spalding_u_plus


# ---------------------------------------------------------------------------
# Coles wake function normalisation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pi_wake", [0.30, 0.45, 0.55])
def test_coles_wake_edge_increment_is_two_pi_over_kappa(pi_wake):
    """At the boundary-layer edge the wake contributes exactly ``2 Pi / kappa``."""
    delta_plus = 2000.0
    y_plus, u_plus, ue_plus, _cf, _theta = coles_profile(
        delta_plus=delta_plus, pi_wake=pi_wake, n_points=4001
    )
    law_at_edge = float(spalding_u_plus(torch.tensor(delta_plus)))
    increment = ue_plus - law_at_edge
    assert increment == pytest.approx(2.0 * pi_wake / KAPPA_DEFAULT, rel=2.0e-3)
    # the historical (wrong) value was twice that
    assert abs(increment - 4.0 * pi_wake / KAPPA_DEFAULT) > 0.5 * increment


def test_coles_profile_shape_factor_is_physical():
    """``H = delta*/theta`` of the log-wake family must stay near 1.3."""
    out = run_flat_plate(re_theta_start=1.0e4, re_theta_end=5.0e4, n_steps=12, n_profile_points=2001)
    assert 1.25 <= out.shape_factor.min() <= 1.40
    assert 1.25 <= out.shape_factor.max() <= 1.40
    # monotone decrease of H along the plate (ZPG turbulent boundary layer)
    assert out.shape_factor[-1] <= out.shape_factor[0]


def test_flat_plate_skin_friction_matches_ittc_1957():
    """Criterion (c): max local c_f error vs the ITTC-1957 local line < 5 %."""
    out = run_flat_plate(re_theta_start=1.0e4, re_theta_end=5.0e4, n_steps=24, n_profile_points=1501)
    metrics = out.metrics()
    assert metrics["max_rel_error_vs_ittc_local"] < 0.05
    # and the failure that the fix removed
    assert metrics["max_rel_error_vs_ittc_local"] < 0.114


def test_ittc_local_is_the_derivative_of_the_mean_line():
    """``c_f = C_F + Re dC_F/dRe`` (sanity of the comparison partner)."""
    re = 6.0e6
    cf_local = float(ittc_1957_local(re))
    cf_mean = 0.075 / (math.log10(re) - 2.0) ** 2
    assert cf_local < cf_mean
    assert 0.75 < cf_local / cf_mean < 1.0


# ---------------------------------------------------------------------------
# channel wall geometry
# ---------------------------------------------------------------------------
def test_measured_wall_origin_moves_the_exchange_node():
    legacy = ChannelWMLESConfig(ny=49, wall_origin_offset=0.5)
    measured = ChannelWMLESConfig(ny=49, wall_origin_offset=1.0)
    assert (legacy.j_wall_node, legacy.j_exchange) == (0, 1)
    assert (measured.j_wall_node, measured.j_exchange) == (1, 2)
    # the first physical node is half a cell from the measured wall in both
    # the historical assumption places the 'first fluid node' *on* the wall
    assert legacy.y_of_node(legacy.j_wall_node) == pytest.approx(0.0)
    assert measured.y_of_node(measured.j_wall_node) == pytest.approx(0.5)
    # the exchange node must not sit inside the wall layer
    assert measured.y_of_node(measured.j_exchange) >= 1.5 - 1e-12


def test_wall_origin_offset_is_validated():
    with pytest.raises(ValueError):
        ChannelWMLESConfig(ny=49, wall_origin_offset=0.2)
    with pytest.raises(ValueError):
        ChannelWMLESConfig(ny=49, wall_coupling="nonsense")


# ---------------------------------------------------------------------------
# wall-stress coupling sign
# ---------------------------------------------------------------------------
def _quiescent_state(nx=4, nz=3, ny=5, u0=0.0):
    rho = torch.ones((nz, 1, nx), dtype=torch.float64)
    ux = torch.full((nz, 1, nx), u0, dtype=torch.float64)
    uy = torch.zeros_like(ux)
    uz = torch.zeros_like(ux)
    from tensorlbm.d3q19 import equilibrium3d

    f = equilibrium3d(rho.expand(nz, ny, nx).clone(), ux.expand(nz, ny, nx).clone(),
                      uy.expand(nz, ny, nx).clone(), uz.expand(nz, ny, nx).clone())
    return f, rho, ux, uy, uz


@pytest.mark.parametrize("u_tau", [0.005, 0.02])
def test_stress_coupling_sign_and_fixed_point(u_tau):
    """The numerical wall flux is corrected toward ``rho u_tau^2``.

    A wall-adjacent row that is *too slow* (``tau_num < tau_w``) must gain
    streamwise momentum, and one that is too fast must lose it: the sign of the
    feedback decides whether the wall model converges at all.
    """
    nu = 2.0e-3
    # u0 too small -> tau_num < tau_w -> the coupling removes momentum
    # u0 too large -> tau_num > tau_w -> the coupling adds momentum
    for u0, expect_rise in ((0.001, False), (0.5, True)):
        f, rho, ux, uy, uz = _quiescent_state(u0=u0)
        before = float(ux.mean())
        diag = _stress_coupling(f, rho, ux, uy, uz, 0, torch.tensor(u_tau), nu)
        from tensorlbm.d3q19 import macroscopic3d

        _rho, ux_new, _uy, _uz = macroscopic3d(f)
        after = float(ux_new.mean())
        if expect_rise:
            assert diag["du"] > 0.0
            assert after > before
        else:
            assert diag["du"] < 0.0
            assert after < before


def test_stress_coupling_matches_model_stress_at_fixed_point():
    """At the fixed point ``tau_num == rho u_tau^2`` the coupling is inactive."""
    nu = 2.0e-3
    u_tau = 0.01
    u0 = nu * 0.0 + u_tau**2 * 0.5 / nu  # u0 giving tau_num = nu u0 / 0.5 = u_tau^2
    f, rho, ux, uy, uz = _quiescent_state(u0=u0)
    diag = _stress_coupling(f, rho, ux, uy, uz, 0, torch.tensor(u_tau), nu)
    assert abs(diag["tau_num"] - u_tau**2) < 1e-6 * max(u_tau**2, 1e-12) + 1e-12
    assert abs(diag["du"]) < 1e-9