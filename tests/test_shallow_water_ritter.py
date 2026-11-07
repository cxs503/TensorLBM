"""Tests for the shallow-water dam break / Ritter self-similar solution."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorlbm.shallow_water import (  # noqa: E402
    ShallowWaterConfig,
    front_from_mass,
    front_position,
    init_dam_break,
    ritter_depth,
    ritter_front,
    ritter_velocity,
    run_ritter_dam_break,
    swe_step,
)


def test_ritter_front_law() -> None:
    h0, g = 1.0, 1.0
    for t in (0.5, 1.0, 5.0):
        assert ritter_front(t, h0, g) == pytest.approx(2.0 * math.sqrt(g * h0) * t)


def test_ritter_depth_boundaries() -> None:
    h0, g, t = 1.0, 1.0, 2.0
    assert ritter_depth(-3.0, t, h0, g) == pytest.approx(h0)          # undisturbed
    assert ritter_depth(4.0, t, h0, g) == pytest.approx(0.0)          # beyond front
    # at the dam section x = 0: h = 4 c0^2 t^2 / (9 g t^2) = 4 h0 / 9
    assert ritter_depth(0.0, t, h0, g) == pytest.approx(4.0 / 9.0, rel=1e-12)
    # monotone decreasing in x inside the rarefaction
    xs = [0.0, 0.5, 1.0, 1.5, 3.9]
    hs = [ritter_depth(x, t, h0, g) for x in xs]
    assert all(hs[i] >= hs[i + 1] for i in range(len(hs) - 1))


def test_ritter_velocity_bounds() -> None:
    h0, g, t = 1.0, 1.0, 1.0
    c0 = math.sqrt(g * h0)
    assert ritter_velocity(-2.0, t, h0, g) == 0.0
    assert ritter_velocity(2.5, t, h0, g) == 0.0
    assert ritter_velocity(0.0, t, h0, g) == pytest.approx(2.0 / 3.0 * c0)


def test_displaced_volume_matches_ritter_mass_law() -> None:
    """The integral form x_f = 27 V / (4 h0) reproduces the analytic front."""
    h0, g = 1.0, 1.0
    t = 3.0
    n = 20000
    x_front = ritter_front(t, h0, g)
    xs = [x_front * i / n for i in range(n)]
    volume = sum(ritter_depth(x, t, h0, g) for x in xs) * (x_front / n)
    assert front_from_mass(volume, h0) == pytest.approx(x_front, rel=1e-3)


def test_initial_state() -> None:
    cfg = ShallowWaterConfig(nx=100, dam_x=0.0)
    U = init_dam_break(cfg)
    assert U.shape == (2, cfg.nx)
    assert float(U[1].abs().max().item()) == 0.0
    upstream = U[0][: 20]
    assert float(upstream.min().item()) == pytest.approx(cfg.h0)


def test_swe_step_conserves_mass() -> None:
    cfg = ShallowWaterConfig(nx=200)
    U = init_dam_break(cfg)
    m0 = float(U[0].sum().item()) * cfg.dx
    dt = cfg.cfl * cfg.dx / (2.0 * cfg.c0)
    for _ in range(50):
        U = swe_step(U, cfg, dt)
    m1 = float(U[0].sum().item()) * cfg.dx
    assert abs(m1 / m0 - 1.0) < 1e-10


def test_swe_step_is_finite_and_non_negative() -> None:
    cfg = ShallowWaterConfig(nx=200)
    U = init_dam_break(cfg)
    dt = cfg.cfl * cfg.dx / (2.0 * cfg.c0)
    for _ in range(100):
        U = swe_step(U, cfg, dt)
    assert bool((U[0] >= 0.0).all())
    assert bool((U[0] <= cfg.h0 * 1.05).all())


def test_dam_break_run_front_advances() -> None:
    cfg = ShallowWaterConfig(nx=800, t_end=2.0, sample_every=100)
    run = run_ritter_dam_break(cfg)
    assert len(run.time) == len(run.front)
    # the front advances monotonically
    assert all(run.front[i] <= run.front[i + 1] + 1e-9 for i in range(len(run.front) - 1))
    # the geometric tip tracker is diffusion limited at a coarse grid (see the
    # benchmark report); the integral (mass based) front is the accurate one
    assert abs(run.front[-1] / run.front_analytic[-1] - 1.0) < 0.25
    # the mass-based front (integral Ritter law) is accurate
    x_mass = front_from_mass(run.volume_downstream[-1], cfg.h0)
    assert abs(x_mass / run.front_analytic[-1] - 1.0) < 0.08


def test_front_position_zero_when_dry() -> None:
    cfg = ShallowWaterConfig(nx=50)
    import torch

    h = torch.zeros(cfg.nx, dtype=torch.float64)
    assert front_position(h, cfg) == cfg.x_min


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        ShallowWaterConfig(nx=8).validate()
    with pytest.raises(ValueError):
        ShallowWaterConfig(dam_x=99.0).validate()
    with pytest.raises(ValueError):
        ShallowWaterConfig(cfl=2.0).validate()