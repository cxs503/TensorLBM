"""Linearised potential-flow wave tank with an elastic (massless) plate cover.

Numerical arm of campaign node ``p338w3-n05``.  The initial/boundary value
problem solved is

    Laplace(phi) = 0                in  -h < z < 0,  x periodic
    dphi/dz = 0                     at z = -h                (rigid bottom)
    phi = phi_s(x,t)                at z = 0                 (Dirichlet data)
    eta_t = dphi/dz                 at z = 0                 (kinematic)
    phi_s,t = -g eta - (D/rho_w) eta_xxxx                    (dynamic, plate)

Discretisation (finite volume / finite difference on a structured grid):

* horizontal second derivative: 4th-order periodic central stencil
  ``[-1, 16, -30, 16, -1] / (12 dx^2)``;
* vertical: 2nd-order central, Neumann bottom via mirror ghost node,
  surface ``dphi/dz`` by a 4th-order one-sided stencil;
* the plate term ``eta_xxxx`` is the horizontal 2nd-derivative operator applied
  twice (periodic, 4th order);
* the Laplace problem is solved **exactly** every time step by a sparse LU of the
  constant elliptic matrix (factorised once, ``permc_spec='NATURAL'`` to preserve
  band structure), so the elliptic operator carries no iteration error;
* time integration: staggered leapfrog (``eta`` at half steps, ``phi_s`` at
  integer steps), second order, stable for ``omega dt < 2``.

Unknown ordering is ``index = i * nz + j`` (vertical index fastest), which makes
the matrix banded with half-bandwidth ``2*nz``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class TankConfig:
    """Geometry and resolution of the numerical tank (SI units)."""

    length: float = 600.0     # periodic domain length [m]
    depth: float = 20.0       # still-water depth [m]
    nx: int = 512             # horizontal cells
    nz: int = 32              # vertical cells
    rho_w: float = 1025.0
    g: float = 9.81

    @property
    def dx(self) -> float:
        return self.length / self.nx

    @property
    def dz(self) -> float:
        return self.depth / self.nz


class FlexuralWaveTank:
    """Periodic linear potential-flow tank with an elastic plate surface."""

    def __init__(self, config: TankConfig, rigidity: float):
        self.cfg = config
        self.D = float(rigidity)
        self.nx, self.nz = config.nx, config.nz
        self.dx, self.dz = config.dx, config.dz
        self.x = np.arange(self.nx) * self.dx
        self._factorise()

    # ------------------------------------------------------------------ setup
    def _factorise(self) -> None:
        nx, nz, dx, dz = self.nx, self.nz, self.dx, self.dz
        n = nx * nz
        rows, cols, vals = [], [], []

        def node(i, j):
            return (i % nx) * nz + j

        # 4th-order periodic stencil for d^2/dx^2
        hcoef = {-2: -1.0 / 12.0, -1: 4.0 / 3.0, 0: -5.0 / 2.0,
                 1: 4.0 / 3.0, 2: -1.0 / 12.0}
        for j in range(nz):
            for i in range(nx):
                r = node(i, j)
                for off, c in hcoef.items():                 # horizontal part
                    rows.append(r)
                    cols.append(node(i + off, j))
                    vals.append(c / dx**2)
                if j == 0:                                    # Neumann via mirror ghost
                    rows += [r, r]
                    cols += [node(i, 1), node(i, 0)]
                    vals += [2.0 / dz**2, -2.0 / dz**2]
                elif j < nz - 1:                              # interior
                    rows += [r, r, r]
                    cols += [node(i, j + 1), node(i, j), node(i, j - 1)]
                    vals += [1.0 / dz**2, -2.0 / dz**2, 1.0 / dz**2]
                else:                                         # surface-adjacent node
                    rows += [r, r]
                    cols += [node(i, j), node(i, j - 1)]
                    vals += [-2.0 / dz**2, 1.0 / dz**2]
        A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
        self._A = A
        self._lu = spla.splu(A, permc_spec="NATURAL", diag_pivot_thresh=0.0)

    # -------------------------------------------------------------- operators
    def surface_velocity(self, phi_s: np.ndarray) -> np.ndarray:
        """``w = dphi/dz`` at ``z = 0`` for surface potential ``phi_s``."""
        nx, nz, dz = self.nx, self.nz, self.dz
        rhs = np.zeros(nx * nz)
        rhs[np.arange(nx) * nz + (nz - 1)] = -np.asarray(phi_s, dtype=float) / dz**2
        u = self._lu.solve(rhs).reshape(nx, nz)
        # one-sided 2nd-order derivative at the surface (values at 0, -dz, -2dz)
        # one-sided 4th-order derivative at the surface (values at 0, -dz, ..., -4dz)
        return (25.0 * phi_s - 48.0 * u[:, nz - 1] + 36.0 * u[:, nz - 2]
                - 16.0 * u[:, nz - 3] + 3.0 * u[:, nz - 4]) / (12.0 * self.dz)

    def _d2x(self, a: np.ndarray) -> np.ndarray:
        dx = self.dx
        return (-np.roll(a, 2) + 16.0 * np.roll(a, 1) - 30.0 * a
                + 16.0 * np.roll(a, -1) - np.roll(a, -2)) / (12.0 * dx**2)

    def d4x(self, a: np.ndarray) -> np.ndarray:
        """``d^4/dx^4`` (periodic, 4th-order)."""
        return self._d2x(self._d2x(a))

    # ------------------------------------------------------------ integration
    def run(self, eta0: np.ndarray, dt: float, nsteps: int,
            record_every: int = 1, phi_s0: np.ndarray | None = None,
            store_field: bool = False):
        """March the linear system; return ``(t, eta_samples, eta_field_or_None)``.

        ``eta_samples`` has shape ``(nsamples, nx)``: the full surface elevation
        sampled every ``record_every`` steps.  The same array is returned as
        ``eta_field`` when ``store_field`` is set, otherwise ``None``.
        """
        nx, g = self.nx, self.cfg.g
        rho_w = self.cfg.rho_w
        eta = np.array(eta0, dtype=float)          # eta at half steps
        phi_s = np.zeros(nx) if phi_s0 is None else np.array(phi_s0, dtype=float)
        plate = self.D / rho_w
        ts, out = [], []
        for n in range(nsteps):
            w = self.surface_velocity(phi_s)
            eta = eta + dt * w
            phi_s = phi_s + dt * (-g * eta - plate * self.d4x(eta))
            if n % record_every == 0:
                ts.append((n + 1) * dt)
                out.append(eta.copy())
        arr = np.asarray(out)
        return np.asarray(ts), arr, (arr if store_field else None)


def initial_mode(x: np.ndarray, k: float, amplitude: float = 0.01,
                 phase: float = 0.0) -> np.ndarray:
    """Single-mode initial elevation ``eta = a cos(k x + phase)``."""
    return amplitude * np.cos(k * x + phase)


def initial_packet(x: np.ndarray, k0: float, x0: float, sigma: float,
                   amplitude: float = 0.01) -> np.ndarray:
    """Gaussian wave packet centred on ``x0`` with carrier ``k0``."""
    env = np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    return amplitude * env * np.cos(k0 * (x - x0))