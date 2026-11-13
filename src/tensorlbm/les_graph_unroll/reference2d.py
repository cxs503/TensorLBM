"""Reference (fine-grid, no sub-grid model) D2Q9 solver and initial conditions.

The reference solution used for calibration/evaluation is the *same* BGK-LBM
kernel run on a much finer grid with the same **physical** viscosity and the
same analytic initial condition.  Grid refinement therefore changes only the
lattice viscosity, ``nu_lat = nu_phys * dt / dx^2`` with ``dx = 2*pi/nx``:

    tau(grid) = 0.5 + 3 * nu_phys / dx^2

so all grids solve the same physical problem (same Reynolds number).  The
reference is a DNS *proxy*; ``reference_convergence`` quantifies its own
resolution error by comparing two successive grids.

Nothing here is differentiable — this is the "truth" side of the
cross-validation and is deliberately written in numpy so that no autograd
machinery can leak into the reference.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "C",
    "W9",
    "equilibrium_np",
    "macroscopic_np",
    "run_reference",
    "tgv_ic",
    "fourier_ic",
    "box_filter",
    "kinetic_energy",
    "tau_for_re",
]

C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], float)
W9 = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], float)
_EX = C[:, 0].reshape(9, 1, 1)
_EY = C[:, 1].reshape(9, 1, 1)
_W = W9.reshape(9, 1, 1)


def equilibrium_np(rho, ux, uy):
    eu = _EX * ux + _EY * uy
    usq = ux * ux + uy * uy
    return _W * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq)


def macroscopic_np(f):
    rho = f.sum(axis=0)
    ux = (f * _EX).sum(axis=0) / rho
    uy = (f * _EY).sum(axis=0) / rho
    return rho, ux, uy


def tau_for_re(nx: int, u_ref: float, re: float, length: float = 1.0) -> float:
    """Lattice ``tau`` for physical ``Re = u_ref * length / nu_phys`` on an ``nx`` grid."""
    nu_phys = u_ref * length / re
    dx = 2.0 * math.pi / nx
    nu_lat = nu_phys / (dx * dx)
    return 0.5 + 3.0 * nu_lat


def _grid(nx: int):
    x = np.arange(nx) * (2.0 * math.pi / nx)
    return x.reshape(nx, 1), x.reshape(1, nx)


def tgv_ic(nx: int, u0: float = 0.1):
    """2D Taylor-Green vortex: ``u = u0 sin x cos y, v = -u0 cos x sin y``."""
    x, y = _grid(nx)
    X = np.broadcast_to(x, (nx, nx))
    Y = np.broadcast_to(y, (nx, nx))
    ux = u0 * np.sin(X) * np.cos(Y)
    uy = -u0 * np.cos(X) * np.sin(Y)
    return _ic(ux, uy)


#: Fixed phases of the synthetic-turbulence initial condition (the same
#: *analytic* field is sampled on every grid, so grid refinement changes the
#: discretisation, never the initial condition).
FOURIER_MODES = (
    # (kx, ky, amplitude, phase)
    (1, 1, 0.55, 0.20),
    (2, -1, 0.42, 1.10),
    (-1, 2, 0.38, 2.35),
    (2, 2, 0.30, 3.05),
    (3, -2, 0.24, 0.75),
    (-3, 1, 0.20, 4.20),
)


def fourier_ic(nx: int, u0: float = 0.1):
    """Solenoidal synthetic-turbulence field: sum of fixed Fourier modes.

    ``u = curl(psi e_z)`` with ``psi = sum_k (A_k/k) sin(k.x + phi_k)``, which
    is discretely divergence free and identical (up to sampling) on every
    grid.  ``u0`` rescales the field to the requested rms amplitude.
    """
    x, y = _grid(nx)
    X = np.broadcast_to(x, (nx, nx))
    Y = np.broadcast_to(y, (1, nx))
    ux = np.zeros((nx, nx))
    uy = np.zeros((nx, nx))
    for kx, ky, amp, phase in FOURIER_MODES:
        k2 = kx * kx + ky * ky
        ph = amp / math.sqrt(k2) * np.cos(kx * X + ky * Y + phase)
        ux += ky * ph
        uy += -kx * ph
    rms = math.sqrt(float(np.mean(ux * ux + uy * uy)))
    ux *= u0 / rms
    uy *= u0 / rms
    return _ic(ux, uy)


def _ic(ux, uy):
    rho = np.ones_like(ux)
    return equilibrium_np(rho, ux, uy), ux, uy


def run_reference(f0, n_steps: int, tau: float, *, snapshot_steps=()):
    """Plain BGK reference run (no sub-grid model).  Returns ``(f, {step: (ux, uy)})``."""
    f = np.array(f0, dtype=np.float64, copy=True)
    wanted = set(int(s) for s in snapshot_steps)
    snaps = {}
    for k in range(1, int(n_steps) + 1):
        rho, ux, uy = macroscopic_np(f)
        feq = equilibrium_np(rho, ux, uy)
        fp = f - (f - feq) / tau
        f = np.empty_like(fp)
        for i in range(9):
            f[i] = np.roll(fp[i], shift=(int(C[i, 0]), int(C[i, 1])), axis=(0, 1))
        if k in wanted:
            r, u1, u2 = macroscopic_np(f)
            snaps[k] = (u1.copy(), u2.copy())
    return f, snaps


def box_filter(field: np.ndarray, ratio: int) -> np.ndarray:
    """2x2 (``ratio x ratio``) block average: the explicit LES filter."""
    nx = field.shape[0]
    n = nx // ratio
    view = field.reshape(n, ratio, n, ratio)
    return view.mean(axis=(1, 3))


def kinetic_energy(ux: np.ndarray, uy: np.ndarray) -> float:
    return 0.5 * float(np.mean(ux * ux + uy * uy))