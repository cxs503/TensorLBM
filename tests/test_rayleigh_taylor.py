"""Tests for the Boussinesq Rayleigh--Taylor growth-rate module."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorlbm.rayleigh_taylor import (  # noqa: E402
    RTConfig,
    analytic_growth_rate,
    fit_growth_rate,
    fit_growth_rate_twomode,
    init_rt_state,
    interface_column,
    interface_zero_crossing,
    mode_amplitude,
    mode_amplitude_zc,
    rt_step,
    run_rayleigh_taylor,
    viscous_growth_rate,
)


def test_analytic_growth_rate_formula() -> None:
    a, g, lam = 1.0 / 3.0, 1.0e-2, 64.0
    k = 2.0 * math.pi / lam
    assert analytic_growth_rate(a, g, lam) == pytest.approx(math.sqrt(a * g * k), rel=1e-12)


def test_viscous_growth_rate_reduces_to_inviscid() -> None:
    a, g, lam = 1.0 / 3.0, 1.0e-2, 128.0
    inv = analytic_growth_rate(a, g, lam)
    visc = viscous_growth_rate(a, g, lam, nu=0.0)
    assert visc == pytest.approx(inv, rel=1e-12)
    # viscosity always damps the instability
    assert viscous_growth_rate(a, g, lam, 0.05) < inv


def test_initial_state_mode_amplitude_and_bounds() -> None:
    cfg = RTConfig(lam=32, n_steps=1)
    f, g, phi = init_rt_state(cfg)
    assert f.shape == (9, cfg.ny, cfg.nx)
    assert float(phi.min()) > -1.005 and float(phi.max()) < 1.005
    # the cos(kx) perturbation has amplitude eta0 = lam/100
    eta0 = cfg.eta0_frac * cfg.lam
    assert mode_amplitude_zc(phi, cfg.lam) == pytest.approx(eta0, abs=0.08 * eta0)
    assert mode_amplitude(phi, cfg.lam) == pytest.approx(eta0, abs=0.04 * eta0)


def test_interface_column_is_conserved_moment() -> None:
    cfg = RTConfig(lam=32, n_steps=1)
    _, _, phi = init_rt_state(cfg)
    # heavy fluid occupies the upper half at rest: N(x) ~ ny/2
    n = interface_column(phi)
    assert float(n.mean()) == pytest.approx(0.5 * cfg.ny, abs=0.5)


def test_zero_crossing_interface_is_finite() -> None:
    cfg = RTConfig(lam=32, n_steps=1)
    _, _, phi = init_rt_state(cfg)
    y_int = interface_zero_crossing(phi)
    assert not bool(torch.isnan(y_int).any())
    assert float(y_int.mean()) == pytest.approx(0.5 * cfg.ny, abs=0.5)


def test_rt_step_conserves_phase_mass() -> None:
    cfg = RTConfig(lam=32, n_steps=1)
    f, g, phi = init_rt_state(cfg)
    m0 = float(g[:, 1:-1, :].sum().item())
    for _ in range(20):
        f, g = rt_step(f, g, cfg, phi=phi)
        phi = g.sum(dim=0)
    m1 = float(g[:, 1:-1, :].sum().item())
    assert abs(m1 / m0 - 1.0) < 1e-6


def test_fit_growth_rate_recovers_exponential() -> None:
    sigma = 0.02
    times = [float(i) for i in range(1, 201)]
    amps = [0.5 * math.exp(sigma * t) for t in times]
    fit = fit_growth_rate(times, amps, 1.0, 5.0)
    assert fit.sigma == pytest.approx(sigma, rel=1e-6)
    assert fit.r2 > 0.999


def test_fit_growth_rate_twomode_recovers_cosh() -> None:
    sigma = 0.015
    times = [float(i) for i in range(1, 200)]
    amps = [0.4 * math.cosh(sigma * t) for t in times]
    fit = fit_growth_rate_twomode(times, amps, (0.5 * sigma, 2.0 * sigma))
    assert fit.sigma == pytest.approx(sigma, rel=2e-2)


def test_fit_growth_rate_rejects_degenerate_window() -> None:
    with pytest.raises(ValueError):
        fit_growth_rate([2.0, 2.0, 2.0, 2.0], [1.0, 2.0, 3.0, 4.0], 0.0, 10.0)
    with pytest.raises(ValueError):
        fit_growth_rate_twomode([1.0, 2.0], [1.0, 2.0], (0.01, 0.02))


def test_short_run_returns_finite_growth_rate() -> None:
    cfg = RTConfig(lam=32, n_steps=120, layers=2.0, sample_every=2)
    res = run_rayleigh_taylor(cfg)
    assert math.isfinite(res["sigma_sim"])
    assert res["sigma_sim"] > 0.0
    assert res["phase_mass_drift_pct"] < 0.1
    assert res["sigma_analytic"] == pytest.approx(
        analytic_growth_rate(cfg.atwood, cfg.g, cfg.lam), rel=1e-12
    )


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        RTConfig(lam=4).validate()
    with pytest.raises(ValueError):
        RTConfig(tau=0.4).validate()
    with pytest.raises(ValueError):
        RTConfig(atwood=1.5).validate()

# ---------------------------------------------------------------------------
# Bounded Allen-Cahn interface transport (round 2, m18081)
# ---------------------------------------------------------------------------


def test_ac2d_matches_ac3d_kernel() -> None:
    """The 2-D AD stepper is the 3-D kernel restricted to ``nz = 1`` (bit-exact)."""
    from tensorlbm.allen_cahn_lbm import step_allen_cahn_field
    from tensorlbm.rayleigh_taylor import step_allen_cahn_2d

    torch.manual_seed(0)
    phi = 0.5 * torch.randn(8, 12, dtype=torch.float64)
    ux = 0.05 * torch.randn(8, 12, dtype=torch.float64)
    uy = 0.05 * torch.randn(8, 12, dtype=torch.float64)
    kw = dict(beta=1.0e-3, kappa=2.0e-3, mobility=5.0, boundary="periodic")
    got = step_allen_cahn_2d(phi, ux, uy, **kw)
    ref = step_allen_cahn_field(
        phi.unsqueeze(0),
        beta=kw["beta"], kappa=kw["kappa"], mobility=kw["mobility"],
        u=(ux.unsqueeze(0), uy.unsqueeze(0), torch.zeros(1, 8, 12, dtype=torch.float64)),
        boundary="periodic",
    )[0]
    assert torch.allclose(got, ref, atol=1e-15)


def test_ac_parameters_recover_requested_interface() -> None:
    """``(sigma, W)`` -> ``(beta, kappa)`` round-trips through the AC relations."""
    from tensorlbm.rayleigh_taylor import ac_parameters

    cfg = RTConfig(phase_scheme="allen_cahn", ac_sigma=3.0e-4, ac_width=2.0)
    beta, kappa = ac_parameters(cfg)
    assert math.sqrt(2.0 * kappa / beta) == pytest.approx(cfg.ac_width, rel=1e-12)
    assert (2.0 * math.sqrt(2.0) / 3.0) * math.sqrt(beta * kappa) == pytest.approx(
        cfg.ac_sigma, rel=1e-12)


def test_ac_scheme_keeps_phase_bounded() -> None:
    """The legacy population scheme inflates ``|phi|``; the AC scheme does not."""
    from tensorlbm.rayleigh_taylor import interface_zero_crossing

    cfg = RTConfig(lam=32, layers=6.0, g=2.0e-3, atwood=0.5, tau=0.515, tau_phi=0.501,
                   n_steps=1, phase_scheme="allen_cahn", ac_sigma=3.0e-4,
                   ac_width=2.0, ac_mobility=20.0)
    f, g, phi = init_rt_state(cfg)
    for _ in range(400):
        f, g = rt_step(f, g, cfg, phi=phi)
        phi = g.sum(dim=0)
    assert float(phi.abs().max()) <= 1.0 + 1e-12
    assert bool(torch.isfinite(phi).all())
    # interface still a level set: one crossing per column, amplitude growing
    zc = interface_zero_crossing(phi)
    assert bool(torch.isfinite(zc).all())
    assert float(zc.max() - zc.min()) > 1.0


def test_reference_velocity_is_boussinesq_symmetric() -> None:
    """Reference velocities: C_2D = 1, bubble == spike, geometric-mean relation."""
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                           / "benchmarks" / "case_pool" / "rayleigh_taylor"))
    import run_case as rc

    assert rc.C_2D == 1.0
    a, g, lam = 0.5, 2.0e-4, 64.0
    ub = rc.boussinesq_velocity(a, g, lam)
    assert ub == pytest.approx(math.sqrt(2.0 * a * g / (2.0 * math.pi / lam)), rel=1e-12)
    gb = rc.goncharov_velocity(a, g, lam, "b")
    gs = rc.goncharov_velocity(a, g, lam, "s")
    # sqrt(u_b^G u_s^G) = sqrt(2 A g/(C k)) (1-A^2)^(-1/4): the Boussinesq value
    # is the geometric mean of the two two-fluid branches up to that factor.
    assert ub == pytest.approx(math.sqrt(gb * gs) * (1.0 - a * a) ** 0.25, rel=1e-12)
    # the product invariant u_b u_s = 2 A g /(C k) is Atwood-independent
    for a2 in (0.1, 0.25, 1.0 / 3.0):
        assert rc.boussinesq_velocity(a2, g, lam) ** 2 * (2.0 * math.pi / lam) / (2.0 * a2 * g) == (
            pytest.approx(1.0 / rc.C_2D, rel=1e-12))
