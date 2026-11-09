"""Tests for :mod:`tensorlbm.taylor_aris` — Taylor–Aris dispersion verification face.

The tests are split into three groups:

* **analytic plumbing** (closed forms, discrete Aris auxiliary problem) — pure
  numpy, sub-second;
* **lattice scalar solver invariants** (mass conservation, zero-flux walls,
  calibration of the dispersion meter at ``Pe = 0``) — a few seconds;
* **physics** (Taylor dispersion in plane Poiseuille flow reproduced against the
  discrete Aris reference) — the shortest configuration that is still in the
  asymptotic regime.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.taylor_aris import (
    ARIS_K_CHANNEL,
    ARIS_K_PIPE,
    DispersionConfig,
    aris_dispersion_coefficient,
    poiseuille_channel_velocity,
    poiseuille_pipe_velocity,
    scalar_collide,
    scalar_equilibrium,
    scalar_stream,
    simulate_channel_dispersion,
    taylor_aris_channel,
    taylor_aris_pipe,
)

# ---------------------------------------------------------------------------
# analytic plumbing
# ---------------------------------------------------------------------------


def test_aris_constants_are_the_classical_values():
    """2/105 (parallel plates) and 1/48 (circular pipe)."""
    assert ARIS_K_CHANNEL == pytest.approx(2.0 / 105.0)
    assert ARIS_K_PIPE == pytest.approx(1.0 / 48.0)


def test_closed_form_reduces_to_molecular_diffusion_without_flow():
    assert taylor_aris_channel(0.01, 4.0, 0.0) == pytest.approx(0.01)
    assert taylor_aris_pipe(0.01, 4.0, 0.0) == pytest.approx(0.01)


def test_closed_form_scales_as_expected():
    """D_eff - D = K a^2 U^2 / D; quadrupling U quadruples the enhancement."""
    D, a, U = 0.02, 4.0, 0.03
    base = taylor_aris_channel(D, a, U) - D
    assert taylor_aris_channel(D, a, 2 * U) - D == pytest.approx(4.0 * base)
    assert taylor_aris_channel(2 * D, a, U) - 2 * D == pytest.approx(0.5 * base)
    # pipe coefficient is 105/(2*48) = 1.09375 times the channel one at equal a
    assert (taylor_aris_pipe(D, a, U) - D) / base == pytest.approx(
        (1.0 / 48.0) / (2.0 / 105.0)
    )


def test_closed_form_rejects_non_positive_diffusivity():
    with pytest.raises(ValueError):
        taylor_aris_channel(0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        taylor_aris_pipe(-1.0, 1.0, 1.0)


def test_velocity_profiles_have_zero_wall_values_and_parabolic_shape():
    h, U = 4.5, 0.04
    y = np.arange(9) - 4.0
    u = poiseuille_channel_velocity(y, h, U)
    assert u[4] == pytest.approx(1.5 * U)  # centreline
    assert u[0] == pytest.approx(u[-1])
    assert u[0] == pytest.approx(1.5 * U * (1.0 - (4.0 / h) ** 2))
    # exact discrete mean of the parabola is not U, but close to it
    assert float(u.mean()) == pytest.approx(U, rel=0.11)
    plug = poiseuille_channel_velocity(y, h, U, profile="plug")
    assert np.allclose(plug, U)


def test_pipe_velocity_profile():
    a, U = 2.0, 0.05
    r = np.array([0.0, 1.0, 2.0])
    u = poiseuille_pipe_velocity(r, a, U)
    assert u[0] == pytest.approx(2.0 * U)
    assert u[-1] == pytest.approx(0.0)
    assert np.allclose(poiseuille_pipe_velocity(r, a, U, profile="plug"), U)


@pytest.mark.parametrize("n", [17, 33, 65, 129])
def test_discrete_aris_problem_reproduces_the_channel_closed_form(n):
    """The discrete Aris integral converges to ``2 h^2 U^2 / (105 D)``."""
    h, U, D = 0.5 * n, 0.04, 0.02
    y = np.arange(n) - 0.5 * (n - 1)
    u = poiseuille_channel_velocity(y, h, U)
    res = aris_dispersion_coefficient(u, D)
    assert res["residual"] < 1e-10  # solver quality
    assert res["taylor_term"] > 0.0  # D_eff = D + <u' chi> must enhance
    rel = (res["d_eff"] - taylor_aris_channel(D, h, U)) / taylor_aris_channel(D, h, U)
    tol = 5e-3 if n < 33 else 1e-4
    assert abs(rel) < tol


def test_discrete_aris_problem_is_second_order_accurate():
    """Error of the discrete Aris coefficient must drop by ~4x per grid doubling."""
    D, U = 0.02, 0.04
    errs = []
    for n in (17, 33, 65):
        h = 0.5 * n
        y = np.arange(n) - 0.5 * (n - 1)
        u = poiseuille_channel_velocity(y, h, U)
        d = aris_dispersion_coefficient(u, D)["d_eff"]
        errs.append(abs(d - taylor_aris_channel(D, h, U)))
    assert errs[1] < errs[0]
    assert errs[2] / errs[1] < 0.35  # measured ~0.24


def test_aris_requires_enough_nodes_and_positive_D():
    with pytest.raises(ValueError):
        aris_dispersion_coefficient([0.0, 1.0], 0.01)


# ---------------------------------------------------------------------------
# lattice scalar solver invariants
# ---------------------------------------------------------------------------


def _scalar_state(nx: int, ny: int, dtype=torch.float64):
    x = torch.arange(nx, dtype=dtype)
    phi = (1.0 + 0.4 * torch.cos(2.0 * math.pi * x / nx)).view(1, nx).expand(ny, nx).contiguous()
    ux = torch.zeros((ny, nx), dtype=dtype)
    uy = torch.zeros_like(ux)
    g = scalar_equilibrium(phi, ux, uy)
    return g, phi, ux, uy


def test_scalar_stream_conserves_mass_with_zero_flux_walls():
    """Half-way bounce-back at top/bottom must not leak scalar mass."""
    nx, ny, D = 32, 9, 0.02
    g, phi, ux, uy = _scalar_state(nx, ny)
    wall = torch.zeros(ny, dtype=torch.bool)
    wall[0] = wall[-1] = True
    m0 = float(g.sum())
    for _ in range(200):
        g = scalar_collide(g, g.sum(dim=0), ux, uy, tau_d=3.0 * D + 0.5)
        g = scalar_stream(g, wall)
    assert float(g.sum()) == pytest.approx(m0, rel=1e-12)


def test_scalar_stream_is_periodic_in_x():
    nx, ny = 16, 5
    g = torch.zeros((9, ny, nx), dtype=torch.float64)
    g[1, :, 0] = 1.0  # a +x population on every row at x=0
    out = scalar_stream(g)
    # after streaming it must sit at x = 1 on every row, nothing left behind
    assert torch.allclose(out[1, :, 1], torch.ones(ny, dtype=torch.float64))
    assert float(out[1, :, 0].abs().sum()) == 0.0
    assert float(out.sum()) == pytest.approx(float(g.sum()))


def test_scalar_diffusion_decays_a_cosine_mode_at_the_analytic_rate():
    """Pure diffusion: rate = D k^2 with D = c_s^2 (tau - 1/2)."""
    nx, ny, D = 32, 9, 0.02
    g, phi, ux, uy = _scalar_state(nx, ny)
    k = 2.0 * math.pi / nx
    x = torch.arange(nx, dtype=torch.float64)

    def amp(g):
        px = g.sum(dim=0).mean(dim=0)
        return float(2.0 * (px.double() * torch.cos(k * x)).sum() / nx)

    a0 = amp(g)
    for _ in range(400):
        g = scalar_collide(g, g.sum(dim=0), ux, uy, tau_d=3.0 * D + 0.5)
        g = scalar_stream(g)
    a1 = amp(g)
    rate = -math.log(a1 / a0) / (400 * k**2)
    # O(k^2) lattice dispersion correction of the D2Q9 diffusion operator:
    # k = 2 pi / 32 gives +0.8 % (measured), i.e. the tolerance below.
    assert rate == pytest.approx(D, rel=1e-2)


# ---------------------------------------------------------------------------
# dispersion meter
# ---------------------------------------------------------------------------


def test_meter_returns_molecular_diffusion_for_pure_diffusion():
    """C5 calibration: with U = 0 the meter must return D (no Taylor term)."""
    # t_min = 3 h^2 / D = 937 steps for h = 2.5, D = 0.02, so 2500 steps leave
    # a long converged window to fit.
    res = simulate_channel_dispersion(
        DispersionConfig(
            nx=64, ny=5, diffusivity=0.02, u_mean=0.0, steps=2500, report_every=50
        )
    )
    assert res["pe"] == 0.0
    # measured -0.4 % at ny = 9 and -1.4 % at ny = 5: the wall reflection of the
    # scalar populations perturbs the near-wall diffusivity slightly, so the
    # tolerance is the resolution limit rather than machine precision.
    assert res["rel_err_vs_discrete"] == pytest.approx(0.0, abs=3e-2)
    assert res["mass_drift"] < 1e-3
    assert res["transient_resolved"]


def test_meter_separates_numerical_diffusion_in_plug_flow():
    """Uniform advection has no shear, so D_eff must stay at D (<=2 %)."""
    res = simulate_channel_dispersion(
        DispersionConfig(
            nx=64,
            ny=9,
            diffusivity=0.02,
            u_mean=0.04,
            profile="plug",
            steps=4000,
            report_every=100,
        )
    )
    assert res["d_eff_analytic"] == pytest.approx(0.02)
    assert abs(res["rel_err_vs_discrete"]) < 0.02


def test_meter_reproduces_taylor_dispersion_at_pe_10():
    """Pe = U h / D ~ 10 with only five nodes across the channel.

    The residual of the meter is dominated by the transverse resolution (the
    shear profile is sampled on ny nodes while the zero-flux wall reflection
    truncates the wall row): measured -13.6 % at ny = 5, -5.1 % at ny = 9 and
    -1.4 % at ny = 17 for the same Pe.  This test therefore only pins the
    *order* of the effect at the cheapest resolution; the campaign runs the
    converged (ny = 17) configuration.
    """
    res = simulate_channel_dispersion(
        DispersionConfig(
            nx=32,
            ny=5,
            diffusivity=0.01,
            u_mean=0.04,
            steps=6000,
            report_every=100,
            dtype="float64",
        )
    )
    assert 8.0 < res["pe"] < 12.0
    assert res["taylor_term_discrete"] > 0.0
    assert abs(res["rel_err_vs_discrete"]) < 0.15
    assert res["rate_scatter"] < 1e-6
    # the enhancement must be positive and of the right order of magnitude
    assert 0.8 < res["d_eff"] / res["d_eff_discrete"] < 1.2


def test_meter_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        simulate_channel_dispersion(DispersionConfig(ny=8, steps=10))
    with pytest.raises(ValueError):
        simulate_channel_dispersion(DispersionConfig(ny=9, diffusivity=0.0, steps=10))


def test_moving_frame_keeps_the_plume_centred():
    """The whole-cell Galilean shift must keep the plume away from the domain edge."""
    res = simulate_channel_dispersion(
        DispersionConfig(
            nx=128,
            ny=9,
            diffusivity=0.02,
            u_mean=0.05,
            method="variance",
            init_width=2.0,
            steps=4000,
            report_every=200,
        )
    )
    # 4000 steps * 0.05 = 200 cells of drift > nx: without the roll the plume
    # would have crossed the periodic boundary several times and the variance
    # would be wildly non-monotone.
    hist = np.array(res["history"]["q"], dtype=float)
    assert np.all(np.diff(hist) > 0.0)
    assert res["sigma_over_nx"] < 0.25
    assert abs(res["rel_err_vs_discrete"]) < 0.12  # ny = 9 resolution limit