"""Fourier x Chebyshev-clustered-FD DNS engine for plane turbulent channel flow.

Why this module exists
----------------------
The lattice-Boltzmann route to a DNS-grade channel needs ``dy+ ~ 1`` everywhere
(a uniform lattice has ``dx+ = dy+ = dz+``), i.e. ``h ~ Re_tau`` nodes for the
half height alone.  At ``Re_tau = 180`` that is a ~200 x 400 x 400 lattice and at
``Re_tau = 590`` well over 10^9 cells; two earlier campaigns ran the native LBM
kernel into divergence before reaching any post-warmup statistics (see
``docs/p223w3_n02_turbulent_channel_dns.md`` and
``docs/reports/P20-N01-channel3d-mkm1999.md``).  This module is the
non-uniform-grid engine that removes that geometric constraint: spectral
(Fourier) in the homogeneous directions, second-order finite differences on a
cosine-clustered wall-normal grid, so that ``dy+`` near the wall can be ~0.1
while the centre-line spacing stays a few wall units.

Numerical method
----------------
* Non-dimensionalisation: half height ``h``, friction velocity ``u_tau``, hence
  ``Re_tau = u_tau h / nu`` and ``nu = 1/Re_tau``; the driving body force is
  ``a = u_tau^2 / h`` so that the exact streamwise momentum balance of the
  statistically stationary flow is ``tau_w = rho a h`` -> ``u_tau = 1``.
* Velocity-pressure formulation, rotational form of the convection term
  ``N = -omega x u`` (the ``grad(|u|^2/2)`` part is absorbed in the pressure).
* De-aliasing: 3/2 zero padding in both Fourier directions, with the Nyquist
  modes of the physical grid dropped (2/3 rule).
* Time integration: second-order Adams-Bashforth for the nonlinear term +
  Crank-Nicolson for the wall-normal viscous operator, which is solved
  implicitly (tridiagonal, per Fourier mode).  Explicit viscous treatment is
  impossible here: the clustered grid has ``nu dt / dy_min^2`` >> 1.
* Pressure: per-mode 1D Poisson in ``y`` with homogeneous Neumann conditions at
  both walls (ghost-node elimination); the singular ``k = 0`` mode is solved by
  double integration instead of a tridiagonal system.

Verification anchors (see ``tests/test_spectral_channel.py``)
-------------------------------------------------------------
* forced flow converges to the exact Poiseuille parabola
  ``u(y) = a y (2h - y) / (2 nu)``,
* discrete continuity at truncation level,
* wall shear reproduces the momentum-balance value ``rho a h``,
* the wall-unit / spectrum reductions are self-tested on the Moser-Kim-Mansour
  reference itself (gate self-test in ``tensorlbm.validation.mkm1999``).

Everything here is plain ``numpy``/``scipy`` (no GPU, no compiled extension), so
the engine runs unchanged on the x86 control plane and on the loong64 cluster
nodes.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import irfft2, rfft2

__all__ = [
    "SpectralChannelConfig",
    "SpectralChannel",
    "poiseuille_exact",
    "log_law",
    "reichardt",
]

_TRAPZ = getattr(np, "trapezoid", None) or getattr(np, "trapz")

# ---------------------------------------------------------------------------
# analytic references
# ---------------------------------------------------------------------------


def poiseuille_exact(y, height: float, accel: float, nu: float):
    """Exact plane-Poiseuille profile for a body-force driven channel."""
    y = np.asarray(y, dtype=float)
    return accel * y * (height - y) / (2.0 * nu)


def log_law(y_plus, kappa: float = 0.41, b: float = 5.2):
    """von Karman log law (outer reference; meaningless below y+ ~ 30)."""
    y_plus = np.asarray(y_plus, dtype=float)
    return np.log(y_plus) / kappa + b


def reichardt(y_plus, kappa: float = 0.41):
    """Reichardt (1951) composite law, ~1 % against channel DNS in u+."""
    y_plus = np.asarray(y_plus, dtype=float)
    return (
        np.log1p(kappa * y_plus) / kappa
        + 7.8 * (1.0 - np.exp(-y_plus / 11.0) - (y_plus / 11.0) * np.exp(-y_plus / 3.0))
    )


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class SpectralChannelConfig:
    """Discretisation + campaign parameters (all non-dimensional)."""

    nx: int = 96
    ny: int = 129
    nz: int = 96
    lx: float = 2.0 * math.pi
    lz: float = math.pi
    h: float = 1.0
    re_tau: float = 180.0
    u_tau: float = 1.0
    dt: float = 5.0e-4
    nsteps: int = 2000
    stat_every: int = 5
    stat_start: int = 0
    #: mean-channel initial profile ("log" | "parabola")
    init_profile: str = "log"
    #: amplitude of the (tapered) random perturbation, units of u_tau
    noise: float = 0.10
    seed: int = 20261111
    #: FFT worker threads (scipy pocketfft)
    fft_workers: int = 1
    #: "cn" (Crank-Nicolson, default) or "be" (backward Euler) for viscosity
    viscous_scheme: str = "cn"
    #: spectral hyperviscosity exp(-chi (k-kc)_+^4 dt); 0 = pure DNS
    spectral_viscosity: float = 0.0
    checkpoint_every: int = 0
    checkpoint_path: str | None = None
    restart_from: str | None = None
    label: str = "case"

    @property
    def nu(self) -> float:
        return self.u_tau * self.h / self.re_tau

    @property
    def height(self) -> float:
        return 2.0 * self.h

    @property
    def accel(self) -> float:
        """Body force that makes ``tau_w = rho a h = rho u_tau^2``."""
        return self.u_tau**2 / self.h

    @property
    def eddy_turnover(self) -> float:
        """One large-eddy turnover time (``h/u_tau``)."""
        return self.h / self.u_tau

    def validate(self) -> None:
        if self.viscous_scheme not in ("cn", "be"):
            raise ValueError(f"viscous_scheme must be 'cn' or 'be', got {self.viscous_scheme!r}")
        if self.init_profile not in ("log", "parabola"):
            raise ValueError(f"init_profile must be 'log' or 'parabola', got {self.init_profile!r}")
        if self.nx % 2 or self.nz % 2:
            raise ValueError("nx and nz must be even for the two-level rfft layout")
        if self.ny < 8:
            raise ValueError("ny must be >= 8")
        if self.dt <= 0.0 or self.re_tau <= 0.0 or self.h <= 0.0:
            raise ValueError("dt, re_tau and h must be positive")

    # -- derived grid metrics (wall units) ---------------------------------
    def grid_metrics(self) -> dict:
        eta = np.arange(self.ny) / (self.ny - 1.0)
        y = self.h * (1.0 - np.cos(math.pi * eta))
        dy = np.diff(y)
        scale = self.u_tau / self.nu
        return {
            "dx_plus": self.lx / self.nx * scale,
            "dz_plus": self.lz / self.nz * scale,
            "dy_min_plus": float(dy.min() * scale),
            "dy_max_plus": float(dy.max() * scale),
            "re_tau": self.re_tau,
            "cells": self.nx * self.ny * self.nz,
        }


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------


@dataclass
class SpectralChannel:
    """Channel DNS state + operators (see module docstring for the method)."""

    config: SpectralChannelConfig
    y: np.ndarray = field(init=False)
    u: np.ndarray = field(init=False)
    nprev: np.ndarray = field(init=False)
    step_index: int = field(default=0, init=False)
    stats: dict = field(default_factory=dict, init=False)

    # -- construction -------------------------------------------------------
    def __post_init__(self) -> None:
        self.config.validate()
        c = self.config
        eta = np.arange(c.ny) / (c.ny - 1.0)
        self.y = c.h * (1.0 - np.cos(math.pi * eta))
        self._build_fd_weights()
        self._build_wavenumbers()
        self.u = self._initial_velocity()
        self.nprev = np.zeros(self._shape_spec, dtype=complex)
        self.stats = self._empty_stats()
        if c.restart_from:
            self._load_state(c.restart_from)

    @property
    def _shape_phys(self) -> tuple[int, int, int, int]:
        return (3, self.config.ny, self.config.nx, self.config.nz)

    @property
    def _shape_spec(self) -> tuple[int, int, int, int]:
        return (3, self.config.ny, self.config.nx, self.config.nz // 2 + 1)

    @property
    def _nk(self) -> tuple[int, int]:
        """Spectral layout: (signed kx, non-negative kz)."""
        return (self.config.nx, self.config.nz // 2 + 1)

    def _build_fd_weights(self) -> None:
        """Three-point (2nd order on a non-uniform grid) d/dy coefficients.

        Only interior rows (1 .. ny-2) are unknowns; the wall rows carry the
        Dirichlet condition ``u = 0``.
        """
        y = self.y
        h1 = y[1:-1] - y[:-2]
        h2 = y[2:] - y[1:-1]
        self._cm = -h2 / (h1 * (h1 + h2))          # f_{i-1}
        self._c0 = (h2 - h1) / (h1 * h2)           # f_i
        self._cp = h1 / (h2 * (h1 + h2))           # f_{i+1}
        self._d2m = 2.0 / (h1 * (h1 + h2))
        self._d20 = -2.0 / (h1 * h2)
        self._d2p = 2.0 / (h2 * (h1 + h2))
        self.dy_min = float(np.min(np.diff(y)))
        self.dy_max = float(np.max(np.diff(y)))

    def _build_wavenumbers(self) -> None:
        """Signed streamwise wavenumbers (full complex axis) x non-negative kz."""
        c = self.config
        m = np.fft.fftfreq(c.nx) * c.nx
        self.kx = 2.0 * math.pi * np.rint(m).astype(int) / c.lx
        self.kz = 2.0 * math.pi * np.arange(c.nz // 2 + 1) / c.lz
        self.k2 = (self.kx[:, None] ** 2 + self.kz[None, :] ** 2)   # (nx, nzk)
        self.ikex = 1j * self.kx[:, None] * np.ones((1, self.kz.size))
        self.ikez = 1j * self.kz[None, :] * np.ones((self.kx.size, 1))

    # -- transforms ---------------------------------------------------------
    def _fwd(self, a: np.ndarray) -> np.ndarray:
        """real (3,ny,nx,nz) -> complex (3,ny,nx,nz/2+1): signed kx x kz>=0."""
        return rfft2(a, axes=(2, 3), workers=self.config.fft_workers)

    def _inv(self, A: np.ndarray) -> np.ndarray:
        c = self.config
        return irfft2(A, s=(c.nx, c.nz), axes=(2, 3), workers=c.fft_workers)

    # -- dealiasing (3/2 padding) -------------------------------------------
    @property
    def _pad_shape(self) -> tuple[int, int]:
        return (3 * self.config.nx // 2, 3 * self.config.nz // 2)

    @property
    def _pad_spec_shape(self) -> tuple[int, int]:
        """(nx_padded, nz_padded//2+1)."""
        nx_p, nz_p = self._pad_shape
        return (nx_p, nz_p // 2 + 1)

    def _pad(self, A: np.ndarray) -> np.ndarray:
        """Zero-pad a (ny,nx,nzk) spectrum to the 3/2 grid, Nyquist modes dropped.

        The kx axis is a full complex axis in fft order (0..n/2, -n/2..-1), so
        the negative modes live at the tail and are copied to the tail of the
        padded axis; the kz axis is halved and simply truncated to nz/2 modes.
        """
        ny, nx, nzk = A.shape
        nxp, nzkp = self._pad_spec_shape
        keep = A.copy()
        keep[:, nx // 2, :] = 0.0        # 2/3 rule: drop the kx Nyquist
        keep[:, :, -1] = 0.0             # ... and the kz Nyquist
        out = np.zeros((ny, nxp, nzkp), dtype=complex)
        half = nx // 2                   # positive modes 0 .. nx/2-1
        out[:, :half, :nzk] = keep[:, :half, :]
        out[:, nxp - (nx - half - 1):, :nzk] = keep[:, half + 1:, :]
        return out

    def _inv_pad(self, A: np.ndarray) -> np.ndarray:
        nx_p, nz_p = self._pad_shape
        return irfft2(A, s=(nx_p, nz_p), axes=(1, 2), workers=self.config.fft_workers)

    def _fwd_pad(self, a: np.ndarray) -> np.ndarray:
        """Transform a padded physical field back, truncating to the kept modes."""
        F = rfft2(a, axes=(1, 2), workers=self.config.fft_workers)
        ny = a.shape[0]
        nx, nzk = self._nk
        out = np.zeros((ny, nx, nzk), dtype=complex)
        half = nx // 2
        out[:, :half, :] = F[:, :half, :nzk]
        out[:, half + 1:, :] = F[:, F.shape[1] - (nx - half - 1):, :nzk]
        return out

    # -- differential operators --------------------------------------------
    def _dy(self, F: np.ndarray) -> np.ndarray:
        """d/dy on interior nodes (works for real or complex (ny,nkx,nkz))."""
        out = np.zeros_like(F)
        out[1:-1] = (self._cm[:, None, None] * F[:-2]
                     + self._c0[:, None, None] * F[1:-1]
                     + self._cp[:, None, None] * F[2:])
        return out

    def _laplacian_y(self, F: np.ndarray) -> np.ndarray:
        """(d2/dy2 - k^2) F on interior nodes, zero on the walls."""
        out = np.zeros_like(F)
        out[1:-1] = (self._d2m[:, None, None] * F[:-2]
                     + (self._d20[:, None, None] - self.k2[None, :, :]) * F[1:-1]
                     + self._d2p[:, None, None] * F[2:])
        return out

    def _decay(self, F: np.ndarray) -> np.ndarray:
        """Optional spectral hyperviscosity applied to the velocity spectrum."""
        chi = self.config.spectral_viscosity
        if chi <= 0.0:
            return F
        kmag = np.sqrt(self.k2)
        kcut = 0.75 * math.pi * self.config.nx / self.config.lx
        damp = np.exp(-chi * np.maximum(kmag - kcut, 0.0) ** 4 * self.config.dt)
        return F * damp[None, :, :]

    # -- tridiagonal machinery ---------------------------------------------
    def _thomas(self, lower, diag, upper, rhs):
        """Vectorised Thomas algorithm along axis 0, batched over trailing axes."""
        n = diag.shape[0]
        d = np.array(diag, copy=True)
        r = np.array(rhs, copy=True)
        for i in range(1, n):
            m = lower[i] / d[i - 1]
            d[i] = d[i] - m * upper[i - 1]
            r[i] = r[i] - m * r[i - 1]
        x = np.empty_like(r)
        x[-1] = r[-1] / d[-1]
        for i in range(n - 2, -1, -1):
            x[i] = (r[i] - upper[i] * x[i + 1]) / d[i]
        return x

    def _solve_helmholtz(self, rhs: np.ndarray, a: float) -> np.ndarray:
        """Solve (1 - a (d2/dy2 - k^2)) x = rhs with Dirichlet walls."""
        n = self.config.ny - 2
        nk = self.k2.size
        diag = np.broadcast_to((1.0 - a * self._d20)[:, None] + a * self.k2.reshape(1, -1),
                               (n, nk))
        lower = np.zeros(n)
        upper = np.zeros(n)
        lower[1:] = -a * self._d2m[1:]
        upper[:-1] = -a * self._d2p[:-1]
        sol = self._thomas(lower, diag, upper, rhs[1:-1].reshape(n, -1))
        out = np.zeros_like(rhs)
        out[1:-1] = sol.reshape(rhs[1:-1].shape)
        return out

    def _solve_pressure(self, g: np.ndarray) -> np.ndarray:
        """Solve ``(d2/dy2 - k^2) p = g`` with homogeneous Neumann walls.

        KNOWN LIMITATION (frozen by ``tests/test_spectral_channel.py``): the
        compact Laplacian solved here is *not* the operator the collocated
        discretisation actually inverts (``div(grad) = A A - k^2`` with ``A`` the
        three-point d/dy).  The projection therefore leaves a residual
        divergence whose odd/even (checkerboard) part is invisible to ``A``; on a
        random, strongly non-solenoidal field the residual grows instead of
        vanishing and a turbulent trajectory blows up within O(10) steps.  The
        exact-composition route ``A^2 - k^2 I = (A - kI)(A + kI)`` was
        implemented and measured to be numerically unusable (Thomas pivots of a
        near-skew-symmetric factor); the fix is a staggered wall-normal velocity
        (Kim & Moin 1985 arrangement), see the campaign report for the evidence.
        Smoothed initial fields are projected to round-off, which is why the
        laminar verification anchors below pass exactly.
        """
        ny = self.config.ny
        n = ny - 2
        out = np.zeros_like(g)
        k2f = self.k2.reshape(-1)
        nz_mask = k2f > 0.0
        if np.any(nz_mask):
            diag = np.empty((n, int(nz_mask.sum())))
            lower = np.zeros(n)
            upper = np.zeros(n)
            diag[0] = self._d2m[0] * 4.0 / 3.0 + self._d20[0] - k2f[nz_mask]
            upper[0] = self._d2p[0] - self._d2m[0] / 3.0
            diag[1:-1] = self._d20[1:-1, None] - k2f[nz_mask][None, :]
            lower[1:-1] = self._d2m[1:-1]
            upper[1:-1] = self._d2p[1:-1]
            lower[-1] = self._d2m[-1] - self._d2p[-1] / 3.0
            diag[-1] = self._d20[-1] + 4.0 * self._d2p[-1] / 3.0 - k2f[nz_mask]
            upper[-1] = 0.0
            rhs = g[1:-1].reshape(n, -1)[:, nz_mask]
            sol = self._thomas(lower, diag, upper, rhs)
            tmp = out[1:-1].reshape(n, -1)
            tmp[:, nz_mask] = sol
            out[1:-1] = tmp.reshape(out[1:-1].shape)
        if np.any(~nz_mask):
            y = self.y
            rhs0 = g.reshape(ny, -1)[:, ~nz_mask]
            w = np.zeros_like(rhs0)          # w = dp/dy with w(0) = 0 (Neumann)
            for i in range(1, ny):
                w[i] = w[i - 1] + 0.5 * (rhs0[i] + rhs0[i - 1]) * (y[i] - y[i - 1])
            p = np.zeros_like(rhs0)
            for i in range(1, ny):
                p[i] = p[i - 1] + 0.5 * (w[i] + w[i - 1]) * (y[i] - y[i - 1])
            out.reshape(ny, -1)[:, ~nz_mask] = p
        return out

    # -- initial condition --------------------------------------------------
    def _initial_velocity(self) -> np.ndarray:
        c = self.config
        y = self.y
        if c.init_profile == "parabola":
            U = poiseuille_exact(y, c.height, c.accel, c.nu)
        else:
            yp = np.maximum(y, 1e-12) * c.u_tau / c.nu
            U = np.where(yp < 5.0, yp, reichardt(yp)) * c.u_tau
            U = 0.5 * (U + U[::-1])
        U[0] = U[-1] = 0.0
        u = np.zeros(self._shape_phys)
        u[0] = U[:, None, None]
        if c.noise > 0.0:
            rng = np.random.default_rng(c.seed)
            taper = np.sin(math.pi * y / c.height)[:, None, None]
            for comp in range(3):
                n = rng.standard_normal((c.ny, c.nx, c.nz)) * (c.noise * c.u_tau)
                n *= taper
                n -= n.mean(axis=(1, 2), keepdims=True)
                u[comp] += n
                u[comp, 0] = 0.0
                u[comp, -1] = 0.0
        return u

    # -- nonlinear term -----------------------------------------------------
    def _nonlinear(self, u: np.ndarray) -> np.ndarray:
        """-omega x u in spectral space (rotational form, 3/2 de-aliased)."""
        F = self._fwd(u)
        dux_dy = self._dy(F[0])
        duy_dy = self._dy(F[1])
        duz_dy = self._dy(F[2])
        wx = self._pad(self.ikez[None] * F[1] - duz_dy)
        wy = self._pad(self.ikex[None] * F[2] - self.ikez[None] * F[0])
        wz = self._pad(dux_dy - self.ikex[None] * F[1])
        ux_p = self._inv_pad(self._pad(F[0]))
        uy_p = self._inv_pad(self._pad(F[1]))
        uz_p = self._inv_pad(self._pad(F[2]))
        wx_p = self._inv_pad(wx)
        wy_p = self._inv_pad(wy)
        wz_p = self._inv_pad(wz)
        nx = self._fwd_pad(-(wy_p * uz_p - wz_p * uy_p))
        ny = self._fwd_pad(-(wz_p * ux_p - wx_p * uz_p))
        nz = self._fwd_pad(-(wx_p * uy_p - wy_p * ux_p))
        out = np.zeros_like(F)
        out[0], out[1], out[2] = nx, ny, nz
        out[:, :, 0, 0] = 0.0     # the mean is driven by the body force only
        return out

    # -- one time step ------------------------------------------------------
    def step(self) -> None:
        c = self.config
        F = self._fwd(self.u)
        Nn = self._nonlinear(self.u)
        rhs = F + c.dt * (1.5 * Nn - 0.5 * self.nprev)
        # streamwise body force: the (0,0) spectral coefficient of a uniform
        # physical field phi is nx*nz*phi, hence the normalisation factor
        rhs[0, :, 0, 0] += c.dt * c.accel * (c.nx * c.nz)
        a_cn = 0.5 * c.nu * c.dt
        a_imp = a_cn if c.viscous_scheme == "cn" else c.nu * c.dt
        Fstar = np.empty_like(F)
        for i in range(3):
            r = rhs[i]
            if c.viscous_scheme == "cn":
                r = r + a_cn * self._laplacian_y(F[i])
            Fstar[i] = self._solve_helmholtz(self._decay(r), a_imp)
        div = (self.ikex[None] * Fstar[0] + self._dy(Fstar[1]) + self.ikez[None] * Fstar[2]) / c.dt
        # div has shape (ny,nkx,nkz): treat it as one "component" for the solver
        p = self._solve_pressure(div)
        Fnew = Fstar.copy()
        Fnew[0] -= c.dt * self.ikex[None] * p
        Fnew[1] -= c.dt * self._dy(p)
        Fnew[2] -= c.dt * self.ikez[None] * p
        self.u = self._inv(Fnew)
        self.nprev = Nn
        self.step_index += 1

    # -- diagnostics --------------------------------------------------------
    def divergence(self) -> float:
        F = self._fwd(self.u)
        d = self.ikex[None] * F[0] + self._dy(F[1]) + self.ikez[None] * F[2]
        d[:, 0, 0] = 0.0
        return float(np.max(np.abs(d)))

    def mean_profile(self) -> np.ndarray:
        return self.u[0].mean(axis=(1, 2))

    def wall_shear(self) -> float:
        """rho nu du/dy at the walls; momentum balance requires rho a h."""
        c = self.config
        U = self.mean_profile()
        y = self.y
        h1 = y[1] - y[0]
        h2 = y[2] - y[1]
        d0 = (h1 + h2) / (h1 * h2) * U[1] - h1 / (h2 * (h1 + h2)) * U[2]
        g1 = y[-1] - y[-2]
        g2 = y[-2] - y[-3]
        dn = (g1 + g2) / (g1 * g2) * U[-2] - g1 / (g2 * (g1 + g2)) * U[-3]
        return 0.5 * c.nu * (d0 + dn)

    def _empty_stats(self) -> dict:
        ny = self.config.ny
        nkx, nkz = self._nk
        stats: dict[str, Any] = {"samples": 0, "steps": 0}
        for name in ("u", "v", "w", "uu", "vv", "ww", "uv", "uw", "vw"):
            stats[name] = np.zeros(ny)
        stats["spec_u"] = np.zeros((nkx, nkz))
        stats["spec_v"] = np.zeros((nkx, nkz))
        stats["spec_w"] = np.zeros((nkx, nkz))
        stats["tke_hist"] = []
        return stats

    def sample(self) -> None:
        u = self.u
        s = self.stats
        s["samples"] += 1
        s["steps"] = self.step_index
        s["u"] += u[0].mean(axis=(1, 2))
        s["v"] += u[1].mean(axis=(1, 2))
        s["w"] += u[2].mean(axis=(1, 2))
        s["uu"] += (u[0] ** 2).mean(axis=(1, 2))
        s["vv"] += (u[1] ** 2).mean(axis=(1, 2))
        s["ww"] += (u[2] ** 2).mean(axis=(1, 2))
        s["uv"] += (u[0] * u[1]).mean(axis=(1, 2))
        s["uw"] += (u[0] * u[2]).mean(axis=(1, 2))
        s["vw"] += (u[1] * u[2]).mean(axis=(1, 2))
        F = self._fwd(u)
        w = np.gradient(self.y)[:, None, None]
        s["spec_u"] += (np.abs(F[0]) ** 2 * w).sum(axis=0)
        s["spec_v"] += (np.abs(F[1]) ** 2 * w).sum(axis=0)
        s["spec_w"] += (np.abs(F[2]) ** 2 * w).sum(axis=0)
        if (self.step_index % 200) == 0:
            mean_u = u[0].mean(axis=(1, 2))
            fluct = u[0] - mean_u[:, None, None]
            s["tke_hist"].append(
                (self.step_index,
                 float(0.5 * np.mean(fluct ** 2 + u[1] ** 2 + u[2] ** 2)))
            )

    # -- driver -------------------------------------------------------------
    def run(self, nsteps: int | None = None) -> dict:
        c = self.config
        nsteps = c.nsteps if nsteps is None else nsteps
        t0 = time.time()
        for _ in range(nsteps):
            self.step()
            if (self.step_index >= c.stat_start) and (self.step_index % c.stat_every) == 0:
                self.sample()
            if c.checkpoint_every and (self.step_index % c.checkpoint_every) == 0 \
                    and c.checkpoint_path:
                self.save_state(f"{c.checkpoint_path}.step{self.step_index}.npz")
        wall = time.time() - t0
        n_cells = c.nx * c.ny * c.nz
        return {
            "label": c.label,
            "steps": nsteps,
            "wall_seconds": wall,
            "mcell_steps_per_s": n_cells * nsteps / wall / 1e6,
            "step_index": self.step_index,
            "divergence_max": self.divergence(),
            "wall_shear": self.wall_shear(),
            "tau_w_momentum_balance": c.accel * c.h,
            "max_speed": float(np.max(np.abs(self.u[0]))),
            "samples": self.stats["samples"],
        }

    # -- state -------------------------------------------------------------
    def save_state(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            u=self.u,
            nprev=self.nprev,
            step_index=np.array([self.step_index]),
            stats=np.array([json.dumps(self._stats_to_json())]),
        )

    def _stats_to_json(self) -> dict:
        out = {}
        for k, v in self.stats.items():
            out[k] = v.tolist() if isinstance(v, np.ndarray) else v
        return out

    def _load_state(self, path: str | Path) -> None:
        data = np.load(path, allow_pickle=False)
        self.u = data["u"]
        self.nprev = data["nprev"]
        self.step_index = int(data["step_index"][0])
        raw = json.loads(str(data["stats"][0]))
        for k, v in raw.items():
            self.stats[k] = np.asarray(v) if isinstance(v, list) else v

    # -- reductions --------------------------------------------------------
    def profiles(self) -> dict:
        """Time-averaged profiles in wall units with the Reynolds stresses."""
        c = self.config
        s = self.stats
        if s["samples"] == 0:
            raise RuntimeError("no statistics samples were taken")
        n = float(s["samples"])
        U = s["u"] / n
        V = s["v"] / n
        W = s["w"] / n
        uu = s["uu"] / n - U**2
        vv = s["vv"] / n - V**2
        ww = s["ww"] / n - W**2
        uv = s["uv"] / n - U * V
        half = c.ny // 2 + 1
        fold = lambda a: 0.5 * (a[:half] + a[::-1][:half])
        y = self.y[:half]
        y_plus = y * c.u_tau / c.nu
        U_f = fold(U)
        rs = {name: fold(v) / c.u_tau**2 for name, v in
              (("uu", uu), ("vv", vv), ("ww", ww), ("uv", uv))}
        dU = np.gradient(U_f, y)
        total = c.nu * dU - rs["uv"] * c.u_tau**2
        bulk = float(_TRAPZ(U_f, y) / c.h)
        return {
            "y": y.tolist(),
            "y_plus": y_plus.tolist(),
            "u_plus": (U_f / c.u_tau).tolist(),
            "re_stress_plus": {k: v.tolist() for k, v in rs.items()},
            "total_shear_over_utau2": (total / c.u_tau**2).tolist(),
            "bulk_velocity": bulk,
            "bulk_plus": bulk / c.u_tau,
            "nu": c.nu,
            "u_tau": c.u_tau,
            "re_tau": c.re_tau,
            "samples": s["samples"],
        }

    def spectra(self) -> dict:
        """Shell-averaged 1D spectra of the three velocity components."""
        c = self.config
        s = self.stats
        n = float(max(s["samples"], 1))
        KX, KZ = np.meshgrid(self.kx, self.kz, indexing="ij")
        kmag = np.sqrt(KX**2 + KZ**2)
        weight = np.ones_like(kmag)
        weight[:, 1:] = 2.0        # +kz and -kz both contribute (kz axis halved)
        dk = 2.0 * math.pi / max(c.lx, c.lz)
        nbins = max(2, int(kmag.max() / dk) + 1)
        edges = dk * np.arange(nbins + 1)
        idx = np.clip(np.digitize(kmag.ravel(), edges) - 1, 0, nbins - 1)
        out: dict[str, Any] = {"k": (0.5 * (edges[1:] + edges[:-1])).tolist(),
                               "kx": self.kx.tolist(), "kz": self.kz.tolist()}
        for key in ("E_u", "E_v", "E_w"):
            P = s["spec_" + key[-1]] / n
            out[key] = np.bincount(idx, weights=(P * weight).ravel(),
                                   minlength=nbins).tolist()
        return out


def module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()