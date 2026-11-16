"""Stokes-layer thickness: analytic identity vs an independent LBM measurement.

Why this test exists
--------------------
``oscillatory_cylinder_inertia.stokes_cylinder_coefficients`` quotes a resolution
parameter ``delta_over_R`` (= acoustic-Stokes layer thickness / cylinder radius) and
feeds the same number into the exact unsteady-Stokes reference ``Gamma(beta)`` through
``k = R sqrt(omega/nu)``.  Both mappings must satisfy

    delta = sqrt(2 nu / omega),        (Stokes' second problem)
    k      = R sqrt(omega / nu) = sqrt(2) R / delta,
    beta   = D^2 / (nu T),  T = 2 pi / omega
    =>  delta/R = 2 / sqrt(pi beta),   k = sqrt(pi beta / 2).

The library shipped ``delta_over_R = 2 sqrt(2/(pi beta))``, i.e. exactly ``sqrt(2)``
too thick, and then used ``k = 1/delta_over_R``, i.e. ``sqrt(2)`` too small — which
overstates ``Re Gamma`` by 5-19 % in the ``beta = 25..100`` band and overstates the
quadrature reference ``-Im Gamma`` by up to a factor 2.1.

This test does not take the identity on trust.  It measures the decay length of the
oscillating velocity next to a no-slip wall **with the LBM itself** (no Bessel
functions, no analytic solution), and compares that measured ``delta`` with both the
corrected formula and the former one.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.boundary.boundaries import bounce_back_cells
from tensorlbm.boussinesq import apply_guo_forcing
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.oscillatory_cylinder_inertia import stokes_cylinder_coefficients
from tensorlbm.solver import stream


def _stokes_second_problem(nx=4, ny=96, wall_rows=8, delta_target=6.0, period=200,
                           spin=5, measure=3, u0=0.01, dtype=torch.float64):
    """Oscillating body force over a flat no-slip wall; returns the measured delta.

    Stokes' second problem with a body force ``a(t) = a0 cos(omega t)`` on the fluid
    and ``u = 0`` at the wall has the exact solution

        u(y,t) = (a0/omega) [ sin(omega t) - e^{-y/delta} sin(omega t - y/delta) ];

    the *amplitude* envelope of the first harmonic therefore decays like
    ``e^{-y/delta}`` toward the wall.  Fitting the slope of ``ln A(y)`` gives ``delta``
    without ever evaluating that solution.
    """
    omega = 2.0 * math.pi / period
    a0 = omega * u0
    nu = delta_target ** 2 * omega / 2.0          # delta = sqrt(2 nu / omega)

    mask = torch.zeros((ny, nx), dtype=torch.bool)
    mask[:wall_rows, :] = True
    fluid = ~mask

    rho0 = torch.ones((ny, nx), dtype=dtype)
    zero = torch.zeros_like(rho0)
    f = bounce_back_cells(equilibrium(rho0, zero, zero), mask)
    Fx = torch.zeros((ny, nx), dtype=dtype)
    Fy = torch.zeros_like(Fx)

    n_steps = (spin + measure) * period
    prof = np.zeros((0, ny))
    ts = np.zeros((0,), dtype=float)
    for step in range(1, n_steps + 1):
        t = float(step)
        Fx[fluid] = a0 * math.cos(omega * t)
        old_solid = f[:, mask].clone()
        f = apply_guo_forcing(f, torch.stack([Fx, Fy]), 3.0 * nu + 0.5)
        f[:, mask] = old_solid
        f = stream(f)
        if step > spin * period:
            _rho, ux, _uy = macroscopic(f)
            prof = np.vstack([prof, ux[:, 0].detach().cpu().numpy()])
            ts = np.append(ts, t)
        f = bounce_back_cells(f, mask)
    return prof, ts, omega, nu, wall_rows


def _harmonic(prof: np.ndarray, ts: np.ndarray, omega: float) -> np.ndarray:
    """Complex first-harmonic amplitude ``U(y)`` of ``u(y,t)`` at every row.

    ``u = Re{U e^{i omega t}}`` with the basis ``[cos, sin]`` gives ``U = p + i q``.
    """
    A = np.stack([np.cos(omega * ts), np.sin(omega * ts)], axis=1)
    coef, *_ = np.linalg.lstsq(A, prof, rcond=None)
    return coef[0] + 1j * coef[1]


@pytest.mark.slow
def test_measured_stokes_layer_matches_sqrt_2_nu_over_omega():
    """LBM decay length next to a no-slip wall == sqrt(2 nu / omega) (not sqrt(2) x)."""
    delta_target = 6.0
    prof, ts, omega, nu, wall_rows = _stokes_second_problem(delta_target=delta_target)
    U = _harmonic(prof, ts, omega)

    # The exact first harmonic is U(y) = (a0/omega)[1 + i e^{-(1+i) y/delta}], so the
    # *deficit* D(y) = U(plateau) - U(y) = -i (a0/omega) e^{-(1+i)y/delta} has modulus
    # e^{-y/delta} and phase -y/delta: two offset-free read-outs of the same length.
    y = np.arange(len(U), dtype=float)
    plateau = complex(np.median(U[y >= wall_rows + 6.0 * delta_target].real),
                      np.median(U[y >= wall_rows + 6.0 * delta_target].imag))
    D = plateau - U
    band = (y >= wall_rows + 0.5) & (y <= wall_rows + 4.0 * delta_target) & (np.abs(D) > 1e-6)
    slope, _intercept = np.polyfit(y[band], np.log(np.abs(D[band])), 1)
    delta_amp = -1.0 / slope

    pslope, _ = np.polyfit(y[band], np.unwrap(np.angle(D[band])), 1)
    delta_phase = abs(1.0 / pslope)   # sign carries the rotation direction only

    delta_exact = math.sqrt(2.0 * nu / omega)
    err_amp = abs(delta_amp - delta_exact) / delta_exact
    err_phase = abs(delta_phase - delta_exact) / delta_exact
    assert err_amp < 0.05, f"amplitude: delta={delta_amp:.3f} vs {delta_exact:.3f}"
    assert err_phase < 0.10, f"phase: delta={delta_phase:.3f} vs {delta_exact:.3f}"

    # the *reported* resolution parameter must agree with the measurement
    for D in (16.0, 32.0, 64.0):
        beta = D * D / (nu * (2.0 * math.pi / omega))
        row = stokes_cylinder_coefficients(beta)
        delta_formula = row["delta_over_R"] * D / 2.0
        assert abs(delta_formula - delta_exact) / delta_exact < 0.02, (
            f"D={D}: reported delta/R={row['delta_over_R']:.4f} -> "
            f"{delta_formula:.3f} cells vs measured {delta_amp:.3f}"
        )


def test_beta_to_k_and_delta_mappings_are_consistent():
    """Analytic identities behind the unsteady-Stokes reference (no fitting involved)."""
    for beta in (6.25, 25.0, 50.0, 100.0, 400.0, 5000.0):
        row = stokes_cylinder_coefficients(beta)
        # delta/R = sqrt(2 nu/omega)/R with nu = D^2/(beta T), T = 2 pi/omega
        assert row["delta_over_R"] == pytest.approx(2.0 / math.sqrt(math.pi * beta), rel=1e-12)
        # k = R sqrt(omega/nu) = sqrt(pi beta / 2)
        assert row["k"] == pytest.approx(math.sqrt(math.pi * beta / 2.0), rel=1e-12)
        # the two are linked by k = sqrt(2) / (delta/R) (Stokes' second problem)
        assert row["k"] * row["delta_over_R"] == pytest.approx(math.sqrt(2.0), rel=1e-12)


def test_gamma_large_k_asymptote_is_one_plus_two_over_k():
    """Independent asymptote: Re Gamma -> 1 + 2/k (not 1 + 2 sqrt(2)/k) as k -> inf.

    The expansion follows from K_1(z)/K_0(z) = 1 + 1/(2z) + O(z^-2) for |z| >> 1
    applied to Gamma = 1 + 4 K_1(z)/(z K_0(z)), z = (1+i) k:

        Gamma -> 1 + 4/z + 2/z^2 + ... ,   4/z = (2/k)(1 - i),  2/z^2 = -i/k^2,

    hence Re Gamma -> 1 + 2/k and -Im Gamma -> 2/k + 1/k^2.  Note the in-phase
    asymptote is the one that matters for the added-mass reference: it is 1 + sqrt(2)
    delta/R, not 1 + 2 sqrt(2) delta/R.
    """
    from scipy.special import kv

    for k in (20.0, 50.0, 200.0):
        z = complex(1.0, 1.0) * k
        gamma = 1.0 + 4.0 * kv(1, z) / (z * kv(0, z))
        assert abs(gamma.real - (1.0 + 2.0 / k)) < 0.5 / k ** 2
        assert abs(-gamma.imag - (2.0 / k + 1.0 / k ** 2)) < 1.0 / k ** 3