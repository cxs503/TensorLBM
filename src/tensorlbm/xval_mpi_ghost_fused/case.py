"""D3Q19 Taylor-Green vortex case used to cross-validate the ghost exchange.

The physical reference is the leading-order (linear-decay) Taylor-Green mode of
the incompressible Navier-Stokes equations on a 2*pi periodic box discretised by
``n`` lattice cells::

    u =  U0 sin(kx) cos(ky) cos(kz) e^{-3 nu k^2 t}
    v = -U0 cos(kx) sin(ky) cos(kz) e^{-3 nu k^2 t}
    w =  0,                               k = 2 pi / n

with the classic Taylor-Green pressure field ``p = p0 + (rho0 U0^2/16)
(cos 2kx + cos 2ky)(cos 2kz + 2)``.  The O(U0^2) harmonics of the true TGV
solution are *not* in the reference: they are part of the reported error, which
is stated in the report rather than absorbed into a fudge factor.

Decay exponent (this is the *three*-dimensional TGV, and the exponent is 3, not
2): the mode is an eigenfunction of the Laplacian with
``d^2/dx^2 = d^2/dy^2 = d^2/dz^2 = -k^2``, i.e. ``lap(u) = -3 k^2 u``, so the
Stokes balance ``du/dt = nu lap(u)`` gives ``du/dt = -3 nu k^2 u``.  The
z-invariant (2D, "2D-in-3D") Taylor-Green mode ``w = 0, no kz dependence`` only
has ``lap = -2 k^2`` and decays with ``2 nu k^2`` -- that exponent (Taylor 1937;
``tensorlbm.verification.analytic_l2.tgv2d_analytic``; also
``benchmarks/d3q27_activation/run_tgv_convergence.py``, whose reference is
explicitly z-invariant) is *not* the one for the cos(kz)-modulated mode used
here.  Using it in a round-1 revision of this cross-validation inflated the
reported rel-L2 error to ~25% at n=64; see
``xval/XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED/REFERENCE_AUDIT.md`` and
``scripts/xval_tgv_reference_audit.py``, which measures the effective decay rate
of the solver alone (no reference involved) and finds it at 1.5% from
``3 nu k^2`` and 52% from ``2 nu k^2``.

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
    def laplacian_eigenvalue(self) -> float:
        """``lap(u) / u`` of the mode: ``-3 k^2`` (``-2 k^2`` for the z-invariant TGV)."""
        return -3.0 * self.k**2

    @property
    def decay_rate(self) -> float:
        """Viscous decay rate ``-lap_eigenvalue * nu = 3 nu k^2`` of the mode."""
        return -self.laplacian_eigenvalue * self.nu

    @property
    def decay_rate_2d(self) -> float:
        """The z-invariant/2D TGV rate ``2 nu k^2`` -- kept for the audit only."""
        return 2.0 * self.nu * self.k**2

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

    def decay(self, t: float) -> float:
        return math.exp(-self.decay_rate * t)

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
    return {
        "tgv_rel_l2": rel_l2,
        "tgv_amplitude_ratio": amplitude_ratio,
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "u_max": float(torch.linalg.vector_norm(torch.stack([ux, uy, uz]), dim=0).max()),
    }