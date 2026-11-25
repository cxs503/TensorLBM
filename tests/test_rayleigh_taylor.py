"""Tests for the Boussinesq Rayleigh--Taylor growth-rate module."""

from __future__ import annotations

import math
import runpy
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
    """The locked primary and alternate conventions both preserve symmetry."""
    driver = Path(__file__).resolve().parents[1] / "benchmarks/case_pool/rayleigh_taylor/run_case.py"
    reference = runpy.run_path(str(driver))
    assert reference["C_2D"] == 3.0
    assert reference["C_2D_ALT"] == 1.0

    g, lam = 2.0e-4, 64.0
    k = 2.0 * math.pi / lam
    goncharov = reference["goncharov_velocity"]
    for atwood in (0.1, 0.25, 1.0 / 3.0, 0.5):
        for c, velocity in (
            (3.0, reference["boussinesq_velocity"]),
            (1.0, reference["boussinesq_velocity_alt"]),
        ):
            u_symmetric = velocity(atwood, g, lam)
            assert u_symmetric == pytest.approx(math.sqrt(2.0 * atwood * g / (c * k)), rel=1e-12)
            bubble = goncharov(atwood, g, lam, "b", c=c)
            spike = goncharov(atwood, g, lam, "s", c=c)
            assert u_symmetric == pytest.approx(
                math.sqrt(bubble * spike) * (1.0 - atwood * atwood) ** 0.25, rel=1e-12
            )


# ---------------------------------------------------------------------------
# CASE-RT-INSTABILITY-ATWOOD: Atwood = 0.5 growth-rate verification
# ---------------------------------------------------------------------------


def test_atwood_half_growth_rate_positive_and_bounded() -> None:
    """CASE-RT-INSTABILITY-ATWOOD: A = 0.5 single-mode RT growth rate sanity check.

    The full case-pool suite (``benchmarks/case_pool/rayleigh_taylor_atwood05``)
    verifies the pre-registered gate |sigma_sim / sigma_visc - 1| <= 15 % with
    the reference configuration (lambda=64, g=2e-4, 2500+ steps).  This test is
    a *lightweight* sanity check that runs in < 60 s and verifies:

    1. The measured growth rate is positive and finite.
    2. The Allen-Cahn interface stays bounded (|phi| <= 1).
    3. The error vs the viscous reference is of the right sign and order
       (systematic low bias ~10-16 %, consistent with the full suite).

    The 15 % gate is enforced by the case-pool suite, not this unit test.
    """
    atwood = 0.5
    lam = 32
    g = 2.0e-4
    tau = 0.515
    nu = (tau - 0.5) / 3.0
    sigma_visc = viscous_growth_rate(atwood, g, lam, nu)

    cfg = RTConfig(
        lam=lam,
        layers=8.0,
        g=g,
        atwood=atwood,
        tau=tau,
        tau_phi=0.501,
        n_steps=600,
        sample_every=3,
        a_lo_frac=0.015,
        a_hi_frac=0.12,
        stop_a_frac=0.30,
        seed_eigenmode=True,
        phase_scheme="allen_cahn",
        ac_sigma=3.0e-4,
        ac_width=2.0,
        ac_mobility=20.0,
    )
    res = run_rayleigh_taylor(cfg)

    # Core invariants
    assert math.isfinite(res["sigma_sim"]), "sigma_sim must be finite"
    assert res["sigma_sim"] > 0.0, "growth rate must be positive"

    # Allen-Cahn must keep phi bounded
    assert res.get("phi_max", 1.0) <= 1.0 + 1e-6, "phi must stay bounded in [-1, 1]"
    assert res.get("phi_min", -1.0) >= -1.0 - 1e-6, "phi must stay bounded in [-1, 1]"

    # Error vs viscous reference: systematic low bias is expected
    # (full suite: -10.1 % at lambda=64; this short run: ~-16 %)
    err_pct = (res["sigma_sim"] / sigma_visc - 1.0) * 100.0
    assert err_pct < 0, f"expected low bias, got {err_pct:+.1f}%"
    assert err_pct > -25, f"error {err_pct:.1f}% exceeds sanity bound of -25%"
