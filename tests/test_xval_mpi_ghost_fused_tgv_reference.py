"""Reference-side contracts for XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED.

The round-1 revision of this cross-validation compared the decomposed field
against a 3D Taylor-Green mode that decayed with the *z-invariant/2D* exponent
``2 nu k^2`` instead of ``3 nu k^2``, which alone accounted for the whole ~25%
rel-L2 error and made the anchor look like a physics failure.  These tests pin
the reference so the mistake cannot come back:

1. ``lap(u) = -3 k^2 u`` for the mode used here (finite-difference check, no
   solver involved);
2. ``TGV.decay`` uses ``3 nu k^2`` and the z-invariant ``2 nu k^2`` rate is kept
   only as a named, non-default quantity;
3. the *measured* decay rate of the solver (log-slope of the projected
   amplitude, no reference used) sits within 8% of ``3 nu k^2`` and more than
   20% away from ``2 nu k^2``, at two amplitudes;
4. the ``|u|_2 / n^1.5`` normalisation used by the figure code is ``U0 f / 2``.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.xval_mpi_ghost_fused.campaign import stream_periodic
from tensorlbm.xval_mpi_ghost_fused.case import TGV, equilibrium3d, macroscopics


def _mode(n: int) -> torch.Tensor:
    x = torch.arange(n, dtype=torch.float64) * (2.0 * math.pi / n)
    s, c = torch.sin(x), torch.cos(x)
    return s.view(1, 1, -1) * c.view(1, -1, 1) * c.view(-1, 1, 1)


def _laplacian(field: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(field)
    for axis in range(3):
        out = out + torch.roll(field, 1, axis) - 2.0 * field + torch.roll(field, -1, axis)
    return out


@pytest.mark.parametrize("n", [8, 16, 32])
def test_mode_is_laplacian_eigenfunction_of_three_axes(n: int) -> None:
    """The exponent question is *how many* axes carry a -k^2 factor: it is 3, not 2.

    The central difference is the exact discrete symbol
    ``-4 sin^2(k/2)`` per axis, so the mode eigenvalue is ``-12 sin^2(k/2)``,
    which tends to ``-3 k^2`` as ``n`` grows.  Either way the ratio to ``-k^2``
    is ~3 (2.85 at n=8, 2.96 at n=16, 2.99 at n=32) and never 2.
    """
    u = _mode(n)
    lap = _laplacian(u)
    mask = u.abs() > 1e-8
    ratio = lap[mask] / u[mask]
    k = 2.0 * math.pi / n
    assert torch.allclose(ratio, torch.full_like(ratio, -12.0 * math.sin(k / 2.0) ** 2),
                          rtol=1e-12, atol=1e-14)
    physical = float(ratio.mean()) / (-k * k)
    assert 2.8 < physical <= 3.0, physical
    assert not (1.9 < physical < 2.1)


def test_decay_rate_is_three_nu_k_squared() -> None:
    case = TGV(n=16, tau=0.65)
    k, nu = case.k, case.nu
    assert case.laplacian_eigenvalue == pytest.approx(-3.0 * k * k, rel=1e-15)
    assert case.decay_rate == pytest.approx(3.0 * nu * k * k, rel=1e-15)
    assert case.decay_rate_2d == pytest.approx(2.0 * nu * k * k, rel=1e-15)
    assert case.decay(0.0) == pytest.approx(1.0)
    assert case.decay(100.0) == pytest.approx(math.exp(-3.0 * nu * k * k * 100.0), rel=1e-15)


def test_analytic_velocity_decays_with_three_nu_k_squared() -> None:
    case = TGV(n=16, tau=0.65, dtype=torch.float64)
    ux0, uy0, _ = case.analytic_velocity(0.0)
    ux1, uy1, _ = case.analytic_velocity(50.0)
    ratio = float(ux1[0, 0, 1].double() / ux0[0, 0, 1].double())
    assert ratio == pytest.approx(math.exp(-3.0 * case.nu * case.k**2 * 50.0), rel=1e-12)


def test_series_normalisation_is_half_u0() -> None:
    """||u||_2 / n^1.5 of the analytic field (through its equilibrium) is U0 f / 2."""
    for n in (8, 16):
        case = TGV(n=n, u0=0.04)
        rho = case.initial_density()
        ux, uy, uz = case.analytic_velocity(0.0)
        field = equilibrium3d(rho, ux, uy, uz)
        _, mx, my, mz = macroscopics(field)
        series = float(torch.linalg.vector_norm(torch.stack([mx, my, mz])) / n**1.5)
        assert series == pytest.approx(case.u0 * 0.5, rel=1e-6)


def _fit_rate(n: int, steps: int, u0: float) -> tuple[float, float, float]:
    torch.set_num_threads(1)
    case = TGV(n=n, u0=u0, tau=0.65, dtype=torch.float64)
    x = torch.arange(n, dtype=torch.float64) * (2.0 * math.pi / n)
    s, c = torch.sin(x), torch.cos(x)
    m1 = s.view(1, 1, -1) * c.view(1, -1, 1) * c.view(-1, 1, 1)
    m2 = -c.view(1, 1, -1) * s.view(1, -1, 1) * c.view(-1, 1, 1)
    n1 = float((m1 * m1).sum())
    n2 = float((m2 * m2).sum())
    f = case.initial_field()
    samples = []
    for step in range(1, steps + 1):
        f = stream_periodic(case.collide(f))
        if step % 10 == 0:
            _, ux, uy, _ = macroscopics(f, torch.float64)
            a1 = float((ux * m1).sum()) / n1
            a2 = float((uy * m2).sum()) / n2
            samples.append((step, 0.5 * (a1 + a2)))
    pts = [(t, math.log(a)) for t, a in samples if t >= steps // 2 and a > 0.0]
    mt = sum(t for t, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    slope = sum((t - mt) * (y - my) for t, y in pts) / sum((t - mt) ** 2 for t, _ in pts)
    return -slope, case.decay_rate, case.decay_rate_2d


@pytest.mark.parametrize("u0", [0.04, 0.004])
def test_measured_decay_rate_matches_3d_exponent_not_2d(u0: float) -> None:
    """Solver-only measurement: no reference solution enters this test."""
    fitted, rate3, rate2 = _fit_rate(n=16, steps=200, u0=u0)
    assert abs(fitted - rate3) / rate3 < 0.08, (fitted, rate3, rate2)
    assert abs(fitted - rate2) / rate2 > 0.20, (fitted, rate3, rate2)