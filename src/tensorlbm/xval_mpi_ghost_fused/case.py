"""D3Q19 Taylor-Green vortex case used to cross-validate the ghost exchange.

Physical reference (R2 anchor correction)
-----------------------------------------
The reference is the leading-order (Stokes / linear-decay) Taylor-Green mode of
the incompressible Navier-Stokes equations on a ``2*pi`` periodic box::

    u =  U0 sin(kx) cos(ky) cos(kz) e^{-3 nu k^2 t}
    v = -U0 cos(kx) sin(ky) cos(kz) e^{-3 nu k^2 t}
    w =  0,                               k = 2 pi / n
    p =  p0 + (rho0 U0^2/16) (cos 2kx + cos 2ky)(cos 2kz + 2)

**Correction (m18155 audit finding 3).** Round 1 shipped ``e^{-2 nu k^2 t}``,
i.e. the *two-dimensional* TGV exponent.  Every velocity component of the mode
above is a product of one normal mode in each of the three axes, so its
Laplacian eigenvalue is

    lambda = -(k_x^2 + k_y^2 + k_z^2) = -3 k^2      (continuum, k_d = k)

and the Stokes solution of ``du/dt = nu lap(u)`` decays as ``e^{-nu 3 k^2 t}``
for the *velocity amplitude* (``e^{-6 nu k^2 t}`` for the kinetic energy).  The
``2 nu k^2`` exponent under-predicted the decay by ``e^{+nu k^2 t}`` and that
mis-attributed the residual: at n=64, tau=0.65, 400 steps it inflated the
reported relative L2 error from ~17 % to 24.9 %.  On the lattice the same
derivation is kept, only with the *discrete* eigenvalue
``sum_d 2 (1 - cos k_d)`` (``= 6 (1 - cos k)`` here), which differs from
``3 k^2`` by ``k^2/12`` (0.08 % at n=64, 0.32 % at n=32) -- see
:meth:`TGV.laplacian_eigenvalue`.

Regime of validity
------------------
The reference is linear, so it is only a fair physical anchor while the
nonlinear terms stay small, i.e. for a small "dissipation-scale" Reynolds
number ``re_k = U0 / (nu k)``.  ``TGV.re_k`` reports it; the R2 campaign carries
a predeclared ``U0`` ladder (0.04 -> 0.01 -> 0.0025, ``re_k`` 8.1 -> 2.0 -> 0.51)
so that the transport fidelity claim is made in the linear regime instead of
being absorbed by O(U0^2) vortex stretching.  The O(U0^2) harmonics and the
acoustic transient of the true TGV are *not* in the reference: whatever they
contribute stays inside the reported error, it is never tuned away.

The field layout matches the transport: ``(19, nz, ny, nx)`` with ``x`` on the
last axis (the slab axis), ``y`` and ``z`` fully periodic inside every rank.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch

from ..d3q19 import C, W, W_EXACT64


def weights_for(dtype: torch.dtype) -> torch.Tensor:
    source = W_EXACT64 if dtype == torch.float64 else W
    return source.to(dtype)


def lattice(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return C.to(dtype)


def equilibrium3d(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """D3Q19 equilibrium on the (19, nz, ny, nx) layout."""
    c = C.to(dtype)
    w = weights_for(dtype).view(19, *([1] * rho.ndim))
    cu = c[:, 0].view(-1, *([1] * rho.ndim)) * ux
    cu = cu + c[:, 1].view(-1, *([1] * rho.ndim)) * uy
    cu = cu + c[:, 2].view(-1, *([1] * rho.ndim)) * uz
    usq = ux * ux + uy * uy + uz * uz
    return rho.unsqueeze(0) * w * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq.unsqueeze(0))


def macroscopics(f: torch.Tensor, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, ...]:
    """(rho, ux, uy, uz) from (19, nz, ny, nx)."""
    c = C.to(dtype)
    rho = f.sum(dim=0)
    ux = (f * c[:, 0].view(19, 1, 1, 1)).sum(dim=0) / rho
    uy = (f * c[:, 1].view(19, 1, 1, 1)).sum(dim=0) / rho
    uz = (f * c[:, 2].view(19, 1, 1, 1)).sum(dim=0) / rho
    return rho, ux, uy, uz


def collide_bgk(f: torch.Tensor, tau: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    rho, ux, uy, uz = macroscopics(f, dtype)
    feq = equilibrium3d(rho, ux, uy, uz, dtype)
    return f - (f - feq) / tau


@dataclass(frozen=True)
class TGV:
    n: int
    u0: float = 0.04
    tau: float = 0.65
    dtype: torch.dtype = torch.float32

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def k(self) -> float:
        return 2.0 * math.pi / self.n

    @property
    def mach(self) -> float:
        return self.u0 * math.sqrt(3.0)

    @property
    def re_k(self) -> float:
        """U0 / (nu k): how far the run is into the nonlinear (reference-invalid) regime."""
        return self.u0 / (self.nu * self.k)

    def laplacian_eigenvalue(self, discrete: bool = True) -> float:
        """|lap| eigenvalue of the single mode, in lattice units (positive).

        ``discrete`` gives ``sum_d 2 (1 - cos k_d) = 6 (1 - cos k)``, the exact
        eigenvalue of the second-order lattice Laplacian on a mode with wave
        number ``k`` in each of the three axes; its continuum limit is ``3 k^2``.
        """
        if discrete:
            return 6.0 * (1.0 - math.cos(self.k))
        return 3.0 * self.k**2

    @property
    def viscous_rate(self) -> float:
        """Exponent of the velocity-amplitude decay, ``nu * |lap eigenvalue|``."""
        return self.nu * self.laplacian_eigenvalue()

    def decay(self, t: float) -> float:
        """Velocity-amplitude factor of the Stokes solution (``e^{-3 nu k^2 t}``)."""
        return math.exp(-self.viscous_rate * t)

    def energy_decay(self, t: float) -> float:
        """Kinetic-energy factor of the Stokes solution (square of :meth:`decay`)."""
        return math.exp(-2.0 * self.viscous_rate * t)

    def _axes(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = torch.arange(self.n, dtype=self.dtype) * (2.0 * math.pi / self.n)
        z = base.view(-1, 1, 1)
        y = base.view(1, -1, 1)
        x = base.view(1, 1, -1)
        return x, y, z

    def initial_density(self) -> torch.Tensor:
        x, y, z = self._axes()
        return 1.0 + (3.0 / 16.0) * self.u0**2 * (torch.cos(2 * x) + torch.cos(2 * y)) * (
            torch.cos(2 * z) + 2.0
        )

    def initial_velocity(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, z = self._axes()
        ux = self.u0 * torch.sin(x) * torch.cos(y) * torch.cos(z)
        uy = -self.u0 * torch.cos(x) * torch.sin(y) * torch.cos(z)
        uz = torch.zeros_like(ux)
        return ux, uy, uz

    def initial_field(self) -> torch.Tensor:
        rho = self.initial_density()
        ux, uy, uz = self.initial_velocity()
        return equilibrium3d(rho, ux, uy, uz, self.dtype)

    def analytic_velocity(self, t: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ux, uy, uz = self.initial_velocity()
        amplitude = self.decay(t)
        return ux * amplitude, uy * amplitude, uz * amplitude

    def collide(self, f: torch.Tensor) -> torch.Tensor:
        return collide_bgk(f, self.tau, self.dtype)


def tgv_error(
    field: torch.Tensor, case: TGV, t: float
) -> dict[str, float]:
    """Velocity error of a full-domain field against the analytic TGV mode."""
    rho, ux, uy, uz = macroscopics(field, case.dtype)
    ax, ay, az = case.analytic_velocity(t)
    num = torch.cat([ux.flatten(), uy.flatten(), uz.flatten()])
    ref = torch.cat([ax.flatten(), ay.flatten(), az.flatten()])
    diff = num - ref
    rel_l2 = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref))
    amplitude_ratio = float(torch.linalg.vector_norm(num) / torch.linalg.vector_norm(ref))
    u_norm = float(torch.linalg.vector_norm(num))
    u0_norm = float(torch.linalg.vector_norm(torch.stack(case.initial_velocity())))
    fitted_rate = -math.log(u_norm / u0_norm) / t if t > 0 and u_norm > 0 else float("nan")
    return {
        "tgv_rel_l2": rel_l2,
        "tgv_amplitude_ratio": amplitude_ratio,
        "u_norm": u_norm,
        "u_norm_initial": u0_norm,
        "fitted_decay_rate": fitted_rate,
        "analytic_decay_rate": case.viscous_rate,
        "decay_rate_rel_error": (
            (fitted_rate - case.viscous_rate) / case.viscous_rate
            if case.viscous_rate > 0
            else float("nan")
        ),
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "u_max": float(torch.linalg.vector_norm(torch.stack([ux, uy, uz]), dim=0).max()),
    }