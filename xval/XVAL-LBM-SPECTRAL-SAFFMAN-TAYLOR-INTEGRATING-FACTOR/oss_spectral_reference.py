"""Independent NumPy reference implementation of the OSS spectral
Saffman--Taylor / viscous-fingering scheme.

This module is a **line-by-line translation of the algorithm** in

    ebinan92/Fingering_dynamics / spectrum_method / main.py   (commit 150a573)
    (Fourier pseudo-spectral Hele-Shaw miscible fingering, integrating factor +
     RK4 bootstrap + Adams--Bashforth 2 + trapezoidal corrector)

written so that the cross-validation study can be reproduced without a network
clone of the OSS repository.  Every formula, operation order and array shape
follows the OSS ``Compute`` class; the only additions are (a) an explicit RNG
seed and (b) a function that exposes the state, because the OSS driver only
plots its final field.

If the OSS clone *is* available (``OSS_FINGERING_DIR``), the study prefers
executing the original ``Compute`` class from the clone and uses this module
only as a shape/step-order cross-check (torch port vs numpy translation).

Formulation reference (cited in the OSS header):
C. T. Tan & G. M. Homsy, Phys. Fluids 31, 2339 (1988).
"""

from __future__ import annotations

import numpy as np


class OSSCompute:
    """NumPy transcription of ``spectrum_method/main.py::Compute``."""

    def __init__(self, ny=128, nx=256, aspect=2.0, pe=500.0, viscosity_ratio=0.0, dt=0.1):
        self.ny, self.nx = ny, nx
        self.A, self.Pe, self.R, self.dt = aspect, pe, viscosity_ratio, dt
        # wavenumbers exactly as OSS: km from nx points over [0, A*Pe], kn over [0, Pe]
        a = np.arange(0, nx // 2) * 2 * np.pi / (aspect * pe)
        b = np.arange(-nx // 2 + 1, 0) * 2 * np.pi / (aspect * pe)
        self.km = np.append(a, np.append(0, b))
        c = np.arange(0, ny // 2) * 2 * np.pi / pe
        d = np.arange(-ny // 2 + 1, 0) * 2 * np.pi / pe
        self.kn = np.append(c, np.append(0, d))
        self.km_km = np.tile(self.km, (ny, 1))
        self.kn_kn = np.tile(self.kn, (nx, 1)).T
        self.km2kn2 = self.km_km ** 2 + self.kn_kn ** 2
        self.x = np.array([(m / (nx / 2)) * 0.5 * aspect * pe for m in range(nx)])
        self.y = np.array([(n / ny) * pe for n in range(ny)])
        self.c = np.zeros((ny, nx))
        self.t = 0.0
        self.stage = 0
        self.j_hat_prev = None

    # ---- state handling -------------------------------------------------
    def set_c(self, c):
        self.c = np.array(c, dtype=np.float64)
        self.c_hat = np.fft.fft2(self.c)
        self.phi_x = np.zeros_like(self.c)
        self.phi_y = np.zeros_like(self.c)
        self._update_derivs()
        self.t, self.stage, self.j_hat_prev = 0.0, 0, None

    def _update_derivs(self):
        self.c_x = np.fft.ifft2(self.c_hat * self.km_km * 1.0j)
        self.c_y = np.fft.ifft2(self.c_hat * self.kn_kn * 1.0j)

    def updateC(self, c):
        self.c = c
        self.c_hat = np.fft.fft2(c)
        self._update_derivs()

    def updateC_hat(self, k):
        self.c_hat = self.c_hat + k
        self._update_derivs()

    # ---- OSS building blocks -------------------------------------------
    def getPhi_hat(self):
        temp1 = self.R * self.c_hat * 1.0j * self.kn_kn
        temp2 = 1.0j * self.R * (self.km_km * self.c_x + self.kn_kn * self.c_y) + self.km2kn2
        return -np.divide(temp1, temp2, out=np.zeros_like(temp1), where=self.km2kn2 != 0)

    def updatePhi_xAndPhi_y(self, phi_hat):
        self.phi_x = np.fft.ifft2(phi_hat * self.km_km * 1.0j).real
        self.phi_y = np.fft.ifft2(phi_hat * self.kn_kn * 1.0j).real

    def getJ_hat(self):
        J = self.phi_y * self.c_x - self.phi_x * self.c_y
        return np.fft.fft2(J)

    def updateAndgetJ_hat(self, k):
        self.updateC_hat(k)
        self.updatePhi_xAndPhi_y(self.getPhi_hat())
        return self.getJ_hat()

    def getC_barbyRunge(self):
        c_hat = np.copy(self.c_hat)
        phi_x = np.copy(self.phi_x)
        phi_y = np.copy(self.phi_y)
        k1 = -self.updateAndgetJ_hat(0) * self.dt
        self.c_hat = c_hat
        k2 = -self.updateAndgetJ_hat(1 / 2 * k1) * self.dt
        self.c_hat = c_hat
        k3 = -self.updateAndgetJ_hat(1 / 2 * k2) * self.dt
        self.c_hat = c_hat
        k4 = -self.updateAndgetJ_hat(k3) * self.dt
        self.c_hat = c_hat
        self.phi_x, self.phi_y = phi_x, phi_y
        c_tilda = c_hat + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return c_tilda * np.exp(-self.km2kn2 * self.dt)

    def getC_barbyAdams(self, J_hat, J_hat_before):
        c_tilda = self.dt * (-1.5 * J_hat + 0.5 * J_hat_before) + self.c_hat
        return c_tilda * np.exp(-self.km2kn2 * self.dt)

    def getNextC(self, J_hat, J_bar, c_hat, c_bar):
        return c_hat - self.dt * ((J_bar + J_hat) / 2 + self.km2kn2 * (c_bar + c_hat) / 2)

    # ---- OSS main loop --------------------------------------------------
    def step(self):
        c_hat = self.c_hat
        J_hat = self.getJ_hat()
        if self.stage == 0 or self.j_hat_prev is None:
            c_bar = self.getC_barbyRunge()
        else:
            c_bar = self.getC_barbyAdams(J_hat, self.j_hat_prev)
        self.updateC(np.fft.ifft2(c_bar).real)
        phi_bar = self.getPhi_hat()
        self.updatePhi_xAndPhi_y(phi_bar)
        J_bar = self.getJ_hat()
        nextc_hat = self.getNextC(J_hat, J_bar, c_hat, c_bar)
        nextc = np.fft.ifft2(nextc_hat).real
        self.updateC(nextc)
        phi_hat = self.getPhi_hat()
        self.updatePhi_xAndPhi_y(phi_hat)
        self.j_hat_prev = J_hat
        self.stage += 1
        self.t += self.dt

    def run(self, n_steps):
        for _ in range(n_steps):
            self.step()


def oss_random_front_ic(ny, nx, seed):
    """OSS initial condition: ``np.hstack([rand(ny, nx/2), zeros(ny, nx/2)])``."""
    rng = np.random.RandomState(seed)
    half = nx // 2
    return np.hstack([rng.rand(ny, half), np.zeros((ny, half))])


def oss_single_mode_ic(ny, nx, aspect, pe, x0, width, amplitude, n_mode):
    """Smooth analytic IC (not used by OSS; used for the convergence study)."""
    x = np.arange(nx) * (aspect * pe / nx)
    y = np.arange(ny) * (pe / ny)
    base = 0.5 * (1.0 - np.tanh((x - x0) / width))
    env = 1.0 / np.cosh(np.clip((x - x0) / width, -50, 50))
    pert = np.cos(2.0 * np.pi * n_mode * y / pe)
    return base[None, :] + amplitude * pert[:, None] * env[None, :]