"""Acceptance tests for the VOF free-surface solver and its analytic ladder.

The tests are deliberately cheap (each run is a few dozen grid cells for a few
dozen steps): they guard the *mechanism* - conservative transport, a
divergence-free projection, hydrostatic equilibrium and the analytic anchors -
while the quantitative ladder numbers are produced by
``scripts/p128w3_vof_ladder.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.vof_free_surface.ladder import (
    capillary_damping_lamb,
    dam_break_ritter_profile,
    stokes2_free_surface,
)
from tensorlbm.vof_free_surface.solver import VOFChannelFlow, VOFConfig, still_water_level


def _config(**kwargs) -> VOFConfig:
    defaults = dict(
        nx=48,
        ny=36,
        length_x=1.2,
        length_y=0.9,
        sigma=1.0e-9,
        surface_tension=False,
        cfl=0.3,
    )
    defaults.update(kwargs)
    return VOFConfig(**defaults)


def _dam_break_fraction(cfg: VOFConfig, x0: float = 0.6, h0: float = 0.30) -> np.ndarray:
    xc = (np.arange(cfg.nx) + 0.5) * cfg.dx
    yc = (np.arange(cfg.ny) + 0.5) * cfg.dy
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    return ((X < x0) & (Y < h0)).astype(float)


def test_ritter_solution_closed_form() -> None:
    """Ritter: reservoir depth, rarefaction parabola and front speed 2 c0."""
    h0, g, x0 = 0.10, 9.81, 0.5
    t = 0.2
    c0 = math.sqrt(g * h0)
    x = np.array([x0 - c0 * t - 1.0e-6, x0 + c0 * t, x0 + 2.0 * c0 * t])
    h, u, front = dam_break_ritter_profile(x, t, x0=x0, h0=h0, gravity=g)
    assert h[0] == pytest.approx(h0, rel=1e-6)
    assert h[1] == pytest.approx((2.0 * c0 - c0 * t / t) ** 2 / (9.0 * g), rel=1e-6)
    assert h[2] == pytest.approx(0.0, abs=1e-12)
    assert front == pytest.approx(x0 + 2.0 * c0 * t, rel=1e-12)
    assert u[1] > 0.0


def test_lamb_capillary_dispersion_and_decay() -> None:
    """Lamb (1932): omega^2 = g k + sigma k^3/rho and decay rate 2 nu k^2."""
    wavelength, nu, rho, sigma, g = 0.015, 2.0e-5, 1000.0, 0.0728, 9.81
    omega, decay = capillary_damping_lamb(
        wavelength, nu, gravity=g, rho=rho, sigma=sigma
    )
    k = 2.0 * math.pi / wavelength
    assert omega**2 == pytest.approx(g * k + sigma * k**3 / rho, rel=1e-12)
    assert decay == pytest.approx(2.0 * nu * k * k, rel=1e-12)
    # capillary waves are dominated by surface tension only below ~1.7 cm
    assert sigma * k**3 / rho > g * k


def test_stokes2_deep_water_limit() -> None:
    """Deep-water second-order bound wave reduces to eta = a cos + (k a^2/2) cos 2."""
    a, k, h, g = 0.05, 2.0 * math.pi / 2.0, 20.0, 9.81
    x = np.linspace(0.0, 2.0 * math.pi / k, 64, endpoint=False)
    eta, omega, amp2 = stokes2_free_surface(x, 0.0, a=a, k=k, h=h, gravity=g)
    assert amp2 == pytest.approx(0.5 * k * a * a, rel=1e-3)
    assert omega == pytest.approx(math.sqrt(g * k), rel=1e-4)
    assert float(np.mean(eta)) == pytest.approx(0.0, abs=1e-12)


def test_poisson_preconditioner_inverts_constant_density_operator() -> None:
    """The FFT/DCT solve must be the exact inverse of the discrete operator."""
    cfg = _config()
    solver = VOFChannelFlow(cfg, np.ones((cfg.nx, cfg.ny)))
    dt = 1.0e-3
    ax = np.full((cfg.nx, cfg.ny), dt / 1000.0)
    ay = np.zeros((cfg.nx, cfg.ny + 1))
    ay[:, 1:-1] = dt / 1000.0
    rng = np.random.default_rng(0)
    rhs = rng.standard_normal((cfg.nx, cfg.ny))
    rhs -= rhs.mean()
    q = solver._poisson_fft(rhs)
    residual = solver._apply_pressure_operator(q, ax, ay) * 1000.0 / dt - rhs
    assert np.abs(residual).max() < 1e-10 * max(np.abs(rhs).max(), 1.0)


def test_projection_is_divergence_free_and_conserves_mass() -> None:
    """After a step: discrete divergence-free velocity and conserved liquid volume."""
    cfg = _config()
    solver = VOFChannelFlow(cfg, _dam_break_fraction(cfg))
    volume0 = solver.volume()
    for _ in range(25):
        solver.step()
    div = np.abs(solver.divergence(solver.u, solver.v)).max()
    scale = solver.max_speed() / cfg.dx
    assert div < 1e-3 * scale
    assert abs(solver.volume() - volume0) / volume0 < 1e-2
    assert np.isfinite(solver.F).all()
    assert solver.F.min() >= 0.0 and solver.F.max() <= 1.0


def test_hydrostatic_equilibrium_is_preserved() -> None:
    """A basin at rest must stay at rest (gravity/Neumann-pressure consistency)."""
    cfg = _config(bottom_noslip=False)
    depth = 0.45
    xc = (np.arange(cfg.nx) + 0.5) * cfg.dx
    yc = (np.arange(cfg.ny) + 0.5) * cfg.dy
    F = np.broadcast_to((yc[:, None] < depth).T, (cfg.nx, cfg.ny)).astype(float).copy()
    solver = VOFChannelFlow(cfg, F)
    heigh0 = still_water_level(solver.F, cfg.dy)
    for _ in range(10):
        solver.step()
    assert solver.max_speed() < 1e-2
    assert np.abs(still_water_level(solver.F, cfg.dy) - heigh0).max() < 1e-3


def test_dam_break_front_advances_and_mass_is_kept() -> None:
    """The continuous-release front must move downstream and stay finite."""
    cfg = _config(nx=96, ny=54, length_x=1.44, length_y=0.81, cfl=0.3)
    solver = VOFChannelFlow(cfg, _dam_break_fraction(cfg, x0=0.72, h0=0.27))
    xc = (np.arange(cfg.nx) + 0.5) * cfg.dx
    station = int(np.argmin(np.abs(xc - 0.85)))
    depths = []
    for target in (0.03, 0.08, 0.13):
        solver.run(target)
        depths.append(float(solver.surface_height()[station]))
    # the release must reach and progressively deepen the downstream station
    assert depths[0] < depths[1] < depths[2]
    assert depths[2] > 0.5 * cfg.dy
    assert abs(solver.mass_error()) < 1e-2
    assert solver.max_speed() < 5.0