"""Circulant (Fourier-diagonalised) solver for the *same* discrete system.

The finite-difference / finite-volume elliptic operator of
:class:`tensorlbm.wave_ice_dispersion.tank.FlexuralWaveTank` is circulant in ``x``
(periodic, constant coefficients), so the discrete 2D system block-diagonalises
exactly under the DFT: solving it by one FFT plus ``nx`` independent vertical
tridiagonal solves is the **exact direct solution of the identical discrete
equations**, not a different discretisation.  ``tests/test_wave_ice_dispersion.py``
checks that both solution paths agree to round-off.

The horizontal symbol of the 4th-order stencil is kept explicitly::

    kappa2_j = (5/2 - 8/3 cos(theta_j) + 1/6 cos(2 theta_j)) / dx^2
    kappa4_j = kappa2_j^2

so the finite-difference error in ``x`` survives (``kappa_j != theta_j/dx``).
Time integration stays the same staggered leapfrog; on top of it the module
exposes the exact discrete eigenvalue ``sqrt(s_d (g + D kappa4 / rho_w))`` so the
total error of the march can be split into its spatial and temporal parts.
"""

from __future__ import annotations

import numpy as np

from .tank import TankConfig


def _thomas(a, b, c, d):
    """Vectorised Thomas algorithm; coefficients have shape ``(nmodes,)``."""
    n = len(b)
    cp = np.zeros_like(b)
    dp = np.zeros_like(b)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.zeros_like(b)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


class CirculantTank:
    """Same discrete scheme as :class:`FlexuralWaveTank`, solved in Fourier space."""

    def __init__(self, config: TankConfig, rigidity: float):
        self.cfg = config
        self.D = float(rigidity)
        self.nx, self.nz = config.nx, config.nz
        self.dx, self.dz = config.dx, config.dz
        self.x = np.arange(self.nx) * self.dx
        theta = 2.0 * np.pi * np.arange(self.nx // 2 + 1) / self.nx   # rfft modes
        self.theta = theta
        self.kappa2 = (2.5 - (8.0 / 3.0) * np.cos(theta)
                       + (1.0 / 6.0) * np.cos(2.0 * theta)) / self.dx**2
        self.kappa_eff = np.sqrt(np.maximum(self.kappa2, 0.0))
        self.kappa4 = self.kappa2**2
        self.s_dtn = self._dtn_symbol()

    # ------------------------------------------------------------------ setup
    def _dtn_symbol(self) -> np.ndarray:
        """Discrete Dirichlet-to-Neumann symbol ``s_d(kappa) = w / phi_s``."""
        nz, dz = self.nz, self.dz
        kap2 = self.kappa2
        a = np.zeros((nz, kap2.size))
        b = np.zeros((nz, kap2.size))
        c = np.zeros((nz, kap2.size))
        d = np.zeros((nz, kap2.size))
        for j in range(nz):
            b[j] = -2.0 / dz**2 - kap2
            a[j] = 1.0 / dz**2
            c[j] = 1.0 / dz**2
        a[0] = 0.0
        c[0] = 2.0 / dz**2                     # Neumann bottom, mirror ghost
        d[nz - 1] = -1.0 / dz**2               # unit Dirichlet datum at z = 0
        phi = _thomas(a, b, c, d)
        w = (25.0 - 48.0 * phi[nz - 1] + 36.0 * phi[nz - 2]
             - 16.0 * phi[nz - 3] + 3.0 * phi[nz - 4]) / (12.0 * dz)
        return w

    def omega_discrete(self) -> np.ndarray:
        """Exact eigenvalue of the discrete system (rad/s) for every DFT mode."""
        g, rho_w = self.cfg.g, self.cfg.rho_w
        return np.sqrt(np.abs(self.s_dtn * (g + self.D * self.kappa4 / rho_w)))

    def omega_stability(self) -> float:
        """Largest discrete frequency (sets the leapfrog stability limit)."""
        return float(np.max(self.omega_discrete()))

    # -------------------------------------------------------------- operators
    @staticmethod
    def _to_hat(field: np.ndarray) -> np.ndarray:
        return np.fft.rfft(field)

    def _to_real(self, hat: np.ndarray) -> np.ndarray:
        return np.fft.irfft(hat, n=self.nx)

    # ------------------------------------------------------------ integration
    def run(self, eta0: np.ndarray, dt: float, nsteps: int,
            record_every: int = 1, record_field: bool = True):
        """March the discrete system; return ``(t, eta_samples)``."""
        g, rho_w = self.cfg.g, self.cfg.rho_w
        plate = self.D / rho_w
        eta = self._to_hat(eta0)
        phi = np.zeros_like(eta)
        ts, out = [], []
        for n in range(nsteps):
            eta = eta + dt * (self.s_dtn * phi)
            phi = phi + dt * (-(g + plate * self.kappa4) * eta)
            if n % record_every == 0:
                ts.append((n + 1) * dt)
                out.append(self._to_real(eta) if record_field else None)
        if not record_field:                     # only probe history is needed
            return np.asarray(ts), None
        return np.asarray(ts), np.asarray(out)

    def run_probes(self, eta0: np.ndarray, dt: float, nsteps: int,
                   probes: np.ndarray, record_every: int = 1):
        """March and return the elevation at the requested node indices only."""
        g, rho_w = self.cfg.g, self.cfg.rho_w
        plate = self.D / rho_w
        eta = self._to_hat(eta0)
        phi = np.zeros_like(eta)
        ts, out = [], []
        for n in range(nsteps):
            eta = eta + dt * (self.s_dtn * phi)
            phi = phi + dt * (-(g + plate * self.kappa4) * eta)
            if n % record_every == 0:
                ts.append((n + 1) * dt)
                out.append(np.fft.irfft(eta, n=self.nx)[probes])
        return np.asarray(ts), np.asarray(out)