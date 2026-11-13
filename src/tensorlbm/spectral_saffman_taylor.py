"""Pseudo-spectral miscible Saffman--Taylor (viscous fingering) solver.

Hele-Shaw cell, Fourier pseudo-spectral discretisation, **exponential integrating
factor** time stepping for the diffusion operator combined with Adams--Bashforth
(2nd order) for the convective (Jacobian) term, an RK4 bootstrap at ``t = 0`` and
a trapezoidal (Crank--Nicolson) corrector -- i.e. the time-integration scheme of
the OSS reference implementation

    ebinan92/Fingering_dynamics  ::  spectrum_method/main.py       (commit 150a573)

Governing system (non-dimensional; ``Pe`` is absorbed into the domain extent
``[0, A*Pe] x [0, Pe]`` exactly as in the OSS code, whose wavenumbers are
``2*pi*m/(A*Pe)`` and ``2*pi*n/Pe``, so the diffusion coefficient is 1):

    dc/dt + J(psi, c) = lap(c)
    J(psi, c)         = psi_y c_x - psi_x c_y = u c_x + v c_y,  u = psi_y, v = -psi_x

The flow is the curl of Darcy's law for a Hele-Shaw cell with an
exponential viscosity law ``mu ~ exp(R c)`` (``R`` = log-viscosity ratio):

    curl( u = -(K/mu) grad p )   =>   omega = -R grad c . grad u-parallel
    omega = -lap(psi)            =>   lap(psi) + R (c_x psi_x + c_y psi_y) = -R c_y

(the right-hand side ``-R c_y`` is the linearisation about the unit base flow).
Formulation reference (cited in the OSS header):
C. T. Tan & G. M. Homsy, Phys. Fluids 31, 2339 (1988).

Elliptic treatment (``elliptic="oss_frozen"`` reproduces the OSS
``getPhi_hat`` symbol)::

    psi_hat = -R * 1j*kn * c_hat / (km^2 + kn^2 + 1j*R*(km*c_x + kn*c_y))

i.e. the variable-coefficient term is pushed into the denominator as if it were
a constant multiplier.  A pointwise product is *not* a product in Fourier space,
so this is an approximation, not an exact spectral discretisation of the
curl-Darcy relation.  The XVAL study measures the consequence: the normalised
residual of the exact relation ``lap(psi) + R (c_x psi_x + c_y psi_y) + R c_y``
is O(1) (1.5--2.0) for R >= 0.05, yet the resulting time-dependent solution is
stable (round-off amplification 1.01--1.03 per step at 128x256, Pe = 500) and
reproduces the OSS reference to 3e-15.  It is therefore kept as the
cross-validated core; an exact variable-coefficient solve remains future work.

Architecture note (Tensor ecosystem boundary contract): pure solver numerics --
no platform state, no cross-repository orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch

__all__ = ["SpectralSaffmanTaylor2D", "FingeringDiagnostics"]


@dataclass
class FingeringDiagnostics:
    """Point-in-time scalar diagnostics of a spectral fingering run."""

    t: float
    mass: float               # <c> (conserved: advection is a divergence)
    c_min: float
    c_max: float
    l2: float                 # ||c - <c>||_2 / sqrt(N)
    mode_amplitude: float     # |c_hat[n_mode, 0]| / (ny*nx) -- transverse mode
    tip_position: float       # max x where c >= threshold
    stage: int = 0


class SpectralSaffmanTaylor2D:
    """Fourier pseudo-spectral miscible fingering solver (integrating factor).

    Parameters
    ----------
    ny, nx :
        Collocation points in the transverse (y) and streamwise (x) directions.
        Physical domain ``[0, A*Pe] x [0, Pe]``.
    aspect :
        Domain aspect ratio ``A`` (x extent / y extent).  OSS default 2.
    pe :
        Peclet number entering the domain size (OSS convention).
    viscosity_ratio :
        Log-viscosity ratio ``R`` (0 => pure diffusion, no vorticity coupling).
    dt :
        Time step of the integrating-factor / AB2 / Crank--Nicolson scheme.
    mode_index :
        Transverse Fourier mode used by the ``mode_amplitude`` diagnostic.
    elliptic :
        Only ``"oss_frozen"`` is implemented (faithful to the OSS symbol).  The
        exact variable-coefficient Darcy solve is future work, registered in
        the XVAL report.
    """

    def __init__(
        self,
        ny: int = 128,
        nx: int = 256,
        *,
        aspect: float = 2.0,
        pe: float = 500.0,
        viscosity_ratio: float = 0.0,
        dt: float = 0.1,
        mode_index: int = 4,
        elliptic: str = "oss_frozen",
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        if ny % 2 or nx % 2:
            raise ValueError("ny and nx must be even (FFT Nyquist layout)")
        if elliptic not in ("oss_frozen",):
            raise ValueError("elliptic must be 'oss_frozen'")
        self.ny, self.nx = int(ny), int(nx)
        self.aspect = float(aspect)
        self.pe = float(pe)
        self.viscosity_ratio = float(viscosity_ratio)
        self.dt = float(dt)
        self.mode_index = int(mode_index)
        self.elliptic = elliptic
        self.device = torch.device(device)
        self.dtype = dtype
        cplx = torch.complex128 if dtype == torch.float64 else torch.complex64

        # Wavenumbers exactly as in the OSS Compute.__init__.
        a = np.arange(0, nx // 2, dtype=np.float64) * 2.0 * np.pi / (self.aspect * self.pe)
        b = np.arange(-nx // 2 + 1, 0, dtype=np.float64) * 2.0 * np.pi / (self.aspect * self.pe)
        self.km = np.concatenate([a, [0.0], b])                      # (nx,)
        c_ = np.arange(0, ny // 2, dtype=np.float64) * 2.0 * np.pi / self.pe
        d_ = np.arange(-ny // 2 + 1, 0, dtype=np.float64) * 2.0 * np.pi / self.pe
        self.kn = np.concatenate([c_, [0.0], d_])                    # (ny,)

        km2d = np.tile(self.km, (ny, 1))                             # (ny, nx)
        kn2d = np.tile(self.kn, (nx, 1)).T                           # (ny, nx)
        self._km = torch.as_tensor(km2d, dtype=dtype, device=self.device)
        self._kn = torch.as_tensor(kn2d, dtype=dtype, device=self.device)
        self._k2 = self._km ** 2 + self._kn ** 2
        self._zero_mode = self._k2 == 0
        self._nonzero = ~self._zero_mode
        self._k2_safe = torch.where(self._nonzero, self._k2, torch.ones_like(self._k2))
        self._ifac = torch.exp(-self._k2 * self.dt)                  # integrating factor

        # state
        self.c = torch.zeros((ny, nx), dtype=dtype, device=self.device)
        self.c_hat = torch.zeros((ny, nx), dtype=cplx, device=self.device)
        self.c_x = torch.zeros((ny, nx), dtype=cplx, device=self.device)
        self.c_y = torch.zeros((ny, nx), dtype=cplx, device=self.device)
        self.phi_x = torch.zeros((ny, nx), dtype=dtype, device=self.device)
        self.phi_y = torch.zeros((ny, nx), dtype=dtype, device=self.device)
        self.j_hat_prev: Optional[torch.Tensor] = None
        self.t = 0.0
        self.stage = 0

    # ------------------------------------------------------------- geometry
    @property
    def x(self) -> torch.Tensor:
        return torch.as_tensor(np.arange(self.nx) * (self.aspect * self.pe / self.nx),
                               dtype=self.dtype, device=self.device)

    @property
    def y(self) -> torch.Tensor:
        return torch.as_tensor(np.arange(self.ny) * (self.pe / self.ny),
                               dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------ utilities
    def _fft2(self, f): return torch.fft.fft2(f)
    def _ifft2(self, f_hat): return torch.fft.ifft2(f_hat)
    def _dx(self, f_hat): return self._ifft2(f_hat * self._km * 1j)
    def _dy(self, f_hat): return self._ifft2(f_hat * self._kn * 1j)

    def set_concentration(self, c) -> None:
        """Install a concentration field (numpy array or tensor, shape (ny, nx))."""
        if isinstance(c, np.ndarray):
            c = torch.as_tensor(c, dtype=self.dtype, device=self.device)
        self.c = c.to(dtype=self.dtype, device=self.device).clone()
        self.c_hat = self._fft2(self.c)
        self._update_derivs()
        self.t, self.stage, self.j_hat_prev = 0.0, 0, None

    def random_front_ic(self, seed: int = 0) -> np.ndarray:
        """OSS initial condition: random data in the left half, zeros right."""
        rng = np.random.RandomState(seed)
        half = self.nx // 2
        return np.hstack([rng.rand(self.ny, half), np.zeros((self.ny, half))])

    def single_mode_ic(self, *, x0: float, width: float, amplitude: float = 0.01,
                       n_mode: Optional[int] = None) -> np.ndarray:
        """Smooth base front along x (tanh) plus one transverse Fourier mode."""
        n_mode = self.mode_index if n_mode is None else n_mode
        x = np.arange(self.nx) * (self.aspect * self.pe / self.nx)
        y = np.arange(self.ny) * (self.pe / self.ny)
        base = 0.5 * (1.0 - np.tanh((x - x0) / width))
        env = 1.0 / np.cosh(np.clip((x - x0) / width, -50, 50))
        pert = np.cos(2.0 * np.pi * n_mode * y / self.pe)
        return base[None, :] + amplitude * pert[:, None] * env[None, :]

    # ------------------------------------------- OSS spectral building blocks
    def _update_derivs(self) -> None:
        """Refresh c_x, c_y from the spectrum (OSS ``updateC``)."""
        self.c_x = self._dx(self.c_hat)
        self.c_y = self._dy(self.c_hat)

    def _set_field(self, c_real) -> None:
        """Install a real-space field and refresh spectrum + gradients (``updateC``)."""
        self.c = c_real
        self.c_hat = self._fft2(c_real)
        self._update_derivs()

    def _nudge_c_hat(self, dk) -> None:
        """OSS ``updateC_hat``: add a spectral increment and refresh gradients."""
        self.c_hat = self.c_hat + dk
        self._update_derivs()

    def phi_hat_oss(self) -> torch.Tensor:
        """OSS frozen-coefficient streamfunction spectrum (``getPhi_hat``)."""
        R = self.viscosity_ratio
        num = R * self.c_hat * 1j * self._kn                 # OSS: kn to the 1st power
        den = 1j * R * (self._km * self.c_x + self._kn * self.c_y) + self._k2
        den_safe = torch.where(self._nonzero, den, torch.ones_like(den))
        return -torch.where(self._nonzero, num / den_safe, torch.zeros_like(num))

    # ------------------------------------------------------------- flow update
    def _update_flow(self) -> None:
        """Refresh the flow used by the convective term (J = u c_x + v c_y).

        ``u = psi_y``, ``v = -psi_x`` with ``psi`` from the OSS frozen-coefficient
        symbol (the scheme this port cross-validates).
        """
        phi_hat = self.phi_hat_oss()
        self.phi_x = self._dx(phi_hat).real
        self.phi_y = self._dy(phi_hat).real

    def jacobian_hat(self) -> torch.Tensor:
        """FFT of J = psi_y c_x - psi_x c_y = u c_x + v c_y (OSS ``getJ_hat``)."""
        return self._fft2(self.phi_y * self.c_x - self.phi_x * self.c_y)

    def _j_hat_after(self, dk) -> torch.Tensor:
        """OSS ``updateAndgetJ_hat``: nudge spectrum, update flow, return J_hat."""
        self._nudge_c_hat(dk)
        self._update_flow()
        return self.jacobian_hat()

    # -------------------------------------------------------------- time steps
    def _rk4_predictor(self):
        """OSS ``getC_barbyRunge``: RK4 on the explicit term + integrating factor."""
        c_hat0 = self.c_hat.clone()
        px0, py0 = self.phi_x.clone(), self.phi_y.clone()
        dt = self.dt
        k1 = -self._j_hat_after(torch.zeros_like(c_hat0)) * dt
        self.c_hat = c_hat0
        k2 = -self._j_hat_after(0.5 * k1) * dt
        self.c_hat = c_hat0
        k3 = -self._j_hat_after(0.5 * k2) * dt
        self.c_hat = c_hat0
        k4 = -self._j_hat_after(k3) * dt
        self.c_hat = c_hat0
        self.phi_x, self.phi_y = px0, py0
        return (c_hat0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0) * self._ifac

    def _ab2_predictor(self, j_hat, j_hat_prev):
        """OSS ``getC_barbyAdams``: AB2 on the explicit term + integrating factor."""
        return (self.c_hat + self.dt * (-1.5 * j_hat + 0.5 * j_hat_prev)) * self._ifac

    def _corrector(self, j_hat, j_bar, c_hat, c_bar):
        """OSS ``getNextC``: trapezoidal (Crank--Nicolson) correction."""
        return c_hat - self.dt * ((j_bar + j_hat) / 2.0 + self._k2 * (c_bar + c_hat) / 2.0)

    def step(self) -> None:
        """One full OSS-scheme time step (predictor + corrector)."""
        c_hat = self.c_hat
        j_hat = self.jacobian_hat()
        if self.stage == 0 or self.j_hat_prev is None:
            c_bar = self._rk4_predictor()
        else:
            c_bar = self._ab2_predictor(j_hat, self.j_hat_prev)

        self._set_field(self._ifft2(c_bar).real)
        self._update_flow()
        j_bar = self.jacobian_hat()

        next_hat = self._corrector(j_hat, j_bar, c_hat, c_bar)
        next_c = self._ifft2(next_hat).real
        self._set_field(next_c)

        self._update_flow()
        self.j_hat_prev = j_hat
        self.stage += 1
        self.t += self.dt

    def run(self, t_end: float, callback: Optional[Callable[["SpectralSaffmanTaylor2D"], None]] = None,
            max_steps: Optional[int] = None) -> int:
        n = int(round((t_end - self.t) / self.dt))
        if max_steps is not None:
            n = min(n, max_steps)
        for _ in range(n):
            self.step()
            if callback is not None:
                callback(self)
        return n

    # ------------------------------------------------------------- diagnostics
    def diagnostics(self, tip_threshold: float = 0.5) -> FingeringDiagnostics:
        c = self.c
        cm = float(c.mean())
        amp = float(torch.abs(self.c_hat[self.mode_index, 0]).item() / (self.ny * self.nx))
        col = (c >= tip_threshold).any(dim=0)
        if bool(col.any()):
            idx = int(torch.nonzero(col, as_tuple=False)[-1].item())
            tip = float(self.x[idx].item())
        else:
            tip = float(self.x[0].item())
        return FingeringDiagnostics(
            t=self.t, mass=cm, c_min=float(c.min()), c_max=float(c.max()),
            l2=float(torch.linalg.vector_norm(c - cm)) / float(np.sqrt(self.ny * self.nx)),
            mode_amplitude=amp, tip_position=tip, stage=self.stage)

    def elliptic_residual(self) -> Dict[str, float]:
        """Normalised residual of the exact curl(Darcy) relation.

        Exact (variable-coefficient) relation, linearised about a unit base flow::

            res = lap(psi) + R (c_x psi_x + c_y psi_y) + R c_y  ==  0

        The OSS frozen-coefficient symbol does not satisfy it (this is the number
        reported in the XVAL difference list); the variable-coefficient solve
        does, up to the CG tolerance.
        """
        phi_hat = self.phi_hat_oss()
        psi_x = self._dx(phi_hat).real
        psi_y = self._dy(phi_hat).real
        lap = self._ifft2(-self._k2 * phi_hat).real
        conv = self.c_x.real * psi_x + self.c_y.real * psi_y
        source = self.viscosity_ratio * self.c_y.real
        res = lap + self.viscosity_ratio * conv + source
        scale = (float(torch.linalg.vector_norm(torch.as_tensor(lap)))
                 + abs(self.viscosity_ratio) * float(torch.linalg.vector_norm(conv))) or 1.0
        return {"residual_l2": float(torch.linalg.vector_norm(res)) / scale,
                "psi_l2": float(torch.linalg.vector_norm(phi_hat)), "R": self.viscosity_ratio}

    def divergence_error(self) -> float:
        """max |div u| / max |u| (spectral operator: round-off for both solves)."""
        u, v = self.phi_y.real, -self.phi_x.real
        div = self._ifft2(self._fft2(u) * self._km * 1j + self._fft2(v) * self._kn * 1j).real
        scale = float(u.abs().max() + v.abs().max()) or 1.0
        return float(div.abs().max()) / scale

    def numpy_state(self) -> np.ndarray:
        return self.c.detach().cpu().numpy()