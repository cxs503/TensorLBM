"""D2Q9 BGK lattice-Boltzmann forward model (numpy, float64).

Self-contained 2D incompressible LBM used by the adjoint / BPTT /
out-of-core (OOTC) gradient machinery.  Conventions follow the standard
BGK scheme (e.g. Krueger et al., "The Lattice Boltzmann Method",
Springer 2017, ch. 2):

    f_i^{n+1}(x) = f_i^n(x - e_i) - (1/tau) [ f_i^n(x - e_i) - feq_i(x - e_i) ]

with periodic boundaries.  All functions are deterministic (no RNG) so
that a recomputed trajectory is bit-identical to the stored one, which
is what makes the OOTC gradient exactly equal to the BPTT reference.
"""

from __future__ import annotations

import numpy as np

# D2Q9 lattice velocities (lattice units) and weights.
EX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.float64)
EY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.float64)
W = np.array([4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
              1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0], dtype=np.float64)

__all__ = ["EX", "EY", "W", "macro", "equilibrium", "collide", "stream",
           "step", "initial_pulse"]


def macro(f: np.ndarray):
    """Macroscopic (rho, ux, uy) from D2Q9 distributions f (9, ny, nx)."""
    rho = f.sum(axis=0)
    ux = (f * EX[:, None, None]).sum(axis=0) / rho
    uy = (f * EY[:, None, None]).sum(axis=0) / rho
    return rho, ux, uy


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Second-order D2Q9 equilibrium distribution (9, ny, nx)."""
    uc = EX[:, None, None] * ux[None, :, :] + EY[:, None, None] * uy[None, :, :]
    u2 = ux * ux + uy * uy
    return W[:, None, None] * rho[None, :, :] * (
        1.0 + 3.0 * uc + 4.5 * uc * uc - 1.5 * u2[None, :, :])


def collide(f: np.ndarray, tau: float) -> np.ndarray:
    """BGK collision: f* = f - (f - feq)/tau."""
    rho, ux, uy = macro(f)
    feq = equilibrium(rho, ux, uy)
    return f - (f - feq) / tau


def stream(f: np.ndarray) -> np.ndarray:
    """Periodic streaming: out_i(x) = f_i(x - e_i)."""
    out = np.empty_like(f)
    for i in range(9):
        out[i] = np.roll(f[i], shift=(EX[i], EY[i]), axis=(0, 1))
    return out


def step(f: np.ndarray, tau: float) -> np.ndarray:
    """One full BGK LBM step (collision then streaming)."""
    return stream(collide(f, tau))


def initial_pulse(ny: int, nx: int, amp: float = 0.05, sigma: float = 16.0,
                 cy: int | None = None, cx: int | None = None) -> np.ndarray:
    """Deterministic initial condition: rest equilibrium + Gaussian velocity
    pulse (no RNG, so trajectories are reproducible bit-for-bit)."""
    if cy is None:
        cy = ny // 2
    if cx is None:
        cx = nx // 2
    y, x = np.mgrid[0:ny, 0:nx]
    r2 = (y - cy) ** 2 + (x - cx) ** 2
    ux = amp * np.exp(-r2 / (2.0 * sigma * sigma))
    uy = np.zeros_like(ux)
    rho = np.ones((ny, nx), dtype=np.float64)
    return equilibrium(rho, ux, uy)
