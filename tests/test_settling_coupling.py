"""Tests for the two-way LBM/particle momentum coupling (``settling_coupling``)."""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d2q9 import macroscopic
from tensorlbm.frazil_suspension import hindered_settling_velocity
from tensorlbm.settling_coupling import (
    CS2,
    Disk,
    DiskLatticeCoupling,
    closure_terminal_velocity,
    measure_added_mass,
    momentum_ledger_probe,
)

torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Momentum ledger
# ---------------------------------------------------------------------------


def test_momentum_ledger_closes_in_channel():
    """Momentum given to disk + walls equals the fluid momentum change."""
    res = momentum_ledger_probe(ny=64, nx=64, radius=6.0, steps=120, channel=True)
    assert res["residual_relative"] < 1e-12, res["residual_relative"]
    # the ledger must be non-trivial: the walls really do take momentum, so the
    # momentum left in the fluid is strictly smaller than the disk impulse
    f_disk = abs(float(res["impulse_particle_to_fluid"][1]))
    f_wall = abs(float(res["impulse_wall_to_fluid"][1]))
    dp = abs(float(res["dp_fluid"][1]))
    assert f_wall > 0.1 * f_disk
    assert dp < f_disk


def test_momentum_ledger_closes_periodic():
    res = momentum_ledger_probe(ny=64, nx=64, radius=6.0, steps=120, channel=False)
    assert res["residual_relative"] < 1e-12
    assert float(res["impulse_wall_to_fluid"].norm()) == pytest.approx(0.0, abs=1e-12)


def test_momentum_ledger_includes_body_force():
    """Guo body force is a momentum source that the ledger must account for."""
    res = momentum_ledger_probe(
        ny=64, nx=64, radius=6.0, steps=120, body_force=(1e-5, 0.0)
    )
    assert res["residual_relative"] < 1e-12
    assert abs(float(res["impulse_body"][0])) > 1.0


def test_ledger_closes_with_zero_feedback():
    """beta = 0 (one-way: fluid sees a stationary obstacle) still closes."""
    res = momentum_ledger_probe(ny=48, nx=48, radius=5.0, steps=80, beta=0.0)
    assert res["residual_relative"] < 1e-12


def test_no_particle_to_fluid_channel_at_zero_feedback():
    """beta = 0 must make the wall velocity -- and hence the drag -- vanish."""
    lat_a = DiskLatticeCoupling(64, 64)
    lat_a.set_disks([Disk(center=(32.0, 32.0), radius=6.0, velocity=(0.0, -0.02))])
    lat_b = DiskLatticeCoupling(64, 64)
    lat_b.set_disks([Disk(center=(32.0, 32.0), radius=6.0, velocity=(0.0, +0.02))])
    fa = torch.zeros(2, dtype=torch.float64)
    fb = torch.zeros(2, dtype=torch.float64)
    for _ in range(20):
        fa = fa + lat_a.step(beta=0.0)["force"]
        fb = fb + lat_b.step(beta=0.0)["force"]
    # the obstacle does not know how fast it is "moving" when beta = 0
    assert torch.allclose(fa, fb, atol=1e-12)


def test_static_obstacle_conserves_mass():
    """With a stationary wall (u_w = 0) the fluid mass must be conserved."""
    lat = DiskLatticeCoupling(64, 64)
    lat.set_disks([Disk(center=(32.0, 32.0), radius=6.0)])
    fluid = ~lat.mask
    m0 = float((lat.f.sum(dim=0) * fluid).sum())
    for _ in range(200):
        lat.step(beta=1.0)
    m1 = float((lat.f.sum(dim=0) * fluid).sum())
    assert abs(m1 - m0) / m0 < 1e-13


# ---------------------------------------------------------------------------
# Moving-wall boundary condition against the analytic Stokes layer
# ---------------------------------------------------------------------------


def test_moving_wall_stokes_second_problem():
    """Amplitude/phase of the Stokes layer driven by a moving wall.

    ``u(y,t) = U0 exp(-y/delta) sin(omega t - y/delta)``, ``delta = sqrt(2 nu /
    omega)``.  A factor-of-two error in the moving-wall momentum term (the
    classic way to get this wrong) would show up as a factor two in the
    amplitude at the first fluid node.
    """
    ny, nx, tau = 96, 8, 0.8
    nu = (tau - 0.5) / 3.0
    lat = DiskLatticeCoupling(ny, nx, tau=tau)
    walls = torch.zeros((ny, nx), dtype=torch.bool)
    walls[0, :] = True
    walls[-1, :] = True
    lat.set_disks([], walls=walls)
    u0, period = 0.01, 200.0
    omega = 2.0 * math.pi / period
    delta = math.sqrt(2.0 * nu / omega)
    rows = [1, 2, 3, 4, 5]
    samples = {j: [] for j in rows}
    for k in range(1, int(5 * period)):
        w = torch.zeros((ny, nx), dtype=torch.float64)
        w[0, :] = u0 * math.sin(omega * k)
        lat.set_wall_velocity(w, torch.zeros_like(w))
        lat.step()
        if k > int(4 * period):
            _, ux, _ = macroscopic(lat.f)
            for j in rows:
                samples[j].append(float(ux[j].mean()))
    for j in rows:
        t = torch.arange(len(samples[j]), dtype=torch.float64)
        f = torch.tensor(samples[j], dtype=torch.float64)
        A = torch.stack([torch.sin(omega * t), torch.cos(omega * t), torch.ones_like(t)], 1)
        sol = torch.linalg.lstsq(A, f.unsqueeze(1)).solution.squeeze(1)
        amp = float((sol[0] ** 2 + sol[1] ** 2).sqrt()) / u0
        exact = math.exp(-(0.5 + (j - 1)) / delta)
        assert abs(amp - exact) / exact < 0.05, (j, amp, exact)


# ---------------------------------------------------------------------------
# Added inertia
# ---------------------------------------------------------------------------


def test_added_inertia_is_positive_and_finite():
    """The forced-oscillation regression must return a physical inertia."""
    out = measure_added_mass(ny=64, nx=64, radius=6.0, period=200.0, n_periods=3)
    assert math.isfinite(out["added_mass_coefficient"])
    assert 0.3 < out["added_mass_coefficient"] < 5.0
    assert out["fit_relative_residual"] < 0.5
    assert out["cross_force_rms"] < 1e-10  # no spurious transverse force


# ---------------------------------------------------------------------------
# One-way degradation (m7094 certification)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c", [0.1, 0.2, 0.3])
def test_closure_degradation_reproduces_certified_hindered_velocity(c):
    """beta = 0 must reproduce the m7094 Richardson-Zaki number to 1e-12."""
    out = closure_terminal_velocity(c, exponent=3.55, max_steps=200000)
    assert out["relative_error"] < 1e-12, out["relative_error"]
    expected = float(hindered_settling_velocity(c, 1.0, exponent=3.55))
    assert out["velocity_ratio"] == pytest.approx(expected, rel=1e-12)


def test_closure_rejects_out_of_range_concentration():
    with pytest.raises(ValueError):
        closure_terminal_velocity(0.7, exponent=3.55)
    with pytest.raises(ValueError):
        DiskLatticeCoupling(8, 8, tau=0.4)


def test_weights_and_cs2_consistent():
    """Sanity: the lattice constants used by the module are the D2Q9 ones."""
    from tensorlbm.d2q9 import C, W

    assert abs(CS2 - 1.0 / 3.0) < 1e-15
    assert abs(float(W.sum()) - 1.0) < 1e-12
    assert int((C * C).sum(-1).max()) == 2