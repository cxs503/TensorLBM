"""Swift-Osborn-Bartholomew (2009) free-energy two-phase LBM with a density ratio.

This is the PF5-SWFE "matrix-completion arm" of the TRACK_COMB_W1
combination-comparison matrix.  It implements the Swift free-energy model:

* free energy   ``F = int [ (a/2) psi^2 - (a/4) psi^4 + (kappa/2) |grad psi|^2 ] dV``
  with ``a = -2 kappa / xi^2`` (analytic interface width ``xi``);
* chemical potential  ``mu = a psi (1 - psi^2) - kappa lap(psi)``;
* density ratio   ``rho = rho_g + (rho_l - rho_g) (psi + 1) / 2``;
* free-energy pressure  ``p = rho cs^2 + psi mu cs^2``  (the "nabla(psi) tensor
  force" is the ``psi mu cs^2`` capillary-pressure contribution, derived from the
  tensor ``grad psi (x) grad mu``);
* momentum  ``d(rho u)/dt = -grad(p I + rho u u) + rho g``.

The 2-D (D2Q9) kernel reuses the validated sailfish anisotropic coefficients and
streaming from :mod:`tensorlbm.phasefield.free_energy_pressure` (the ``nabla(psi)
tensor force`` is the pressure tensor embedded in the equilibrium).  The only
change vs the equal-density sailfish transplant is the **density ratio**: the
density is a function of the order parameter, ``rho(psi) = rho_g + (rho_l - rho_g)
(psi + 1) / 2``, and the free-energy pressure is the Swift ``p = rho cs^2 + psi mu
cs^2``.

The 3-D (D3Q19) kernel uses the same algebra with the D3Q19 lattice.

Both are pure-``torch`` and periodic (the water-entry probe handles the no-flux
outer boundary by mirroring the populations, not here).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .d2q9 import C as _C2
from .d3q19 import C as _C3
from .phasefield.free_energy_pressure import (
    anisotropic_coefficients as _aniso2,
    grad_lap as _grad_lap2,
    stream as _stream2,
)

CS2 = 1.0 / 3.0


@dataclass(frozen=True)
class SwiftParams:
    """Swift free-energy parameters in lattice units.

    ``a = -2 kappa / xi^2`` is derived from ``kappa`` and ``xi`` so that the
    analytic interface width is ``xi`` and the surface tension is
    ``sigma = (2/3) sqrt(2 kappa |a|) = (4/3) kappa / xi``.
    """

    kappa: float = 0.02
    xi: float = 3.0
    tau_f: float = 1.0
    tau_g: float = 1.0
    rho_g: float = 1.0
    rho_l: float = 10.0

    @property
    def a(self) -> float:
        """Double-well coefficient ``a = -2 kappa / xi^2`` (negative)."""
        return -2.0 * self.kappa / (self.xi * self.xi)

    @property
    def A(self) -> float:
        """Positive double-well coefficient ``A = -a = 2 kappa / xi^2``."""
        return 2.0 * self.kappa / (self.xi * self.xi)

    @property
    def sigma(self) -> float:
        """Analytic surface tension ``(2/3) sqrt(2 kappa A)``."""
        return (2.0 / 3.0) * (2.0 * self.kappa * self.A) ** 0.5

    @property
    def viscosity(self) -> float:
        return CS2 * (self.tau_f - 0.5)

    @property
    def mobility(self) -> float:
        """Cahn-Hilliard mobility ``M = (tau_g - 1/2) cs^2``."""
        return (self.tau_g - 0.5) * CS2

    def density(self, psi: torch.Tensor) -> torch.Tensor:
        """``rho = rho_g + (rho_l - rho_g)(psi + 1)/2``."""
        return self.rho_g + (self.rho_l - self.rho_g) * 0.5 * (psi + 1.0)

    def chemical_potential(self, psi: torch.Tensor, lap: torch.Tensor) -> torch.Tensor:
        """``mu = a psi (1 - psi^2) - kappa lap(psi)``  (``a = -A``)."""
        return -self.A * psi * (1.0 - psi * psi) - self.kappa * lap

    def free_energy_pressure(self, rho: torch.Tensor, psi: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """``p = rho cs^2 + psi mu cs^2``."""
        return rho * CS2 + psi * mu * CS2


# --------------------------------------------------------------------------- #
# 2-D (D2Q9) -- reuses the validated sailfish anisotropic coefficients
# --------------------------------------------------------------------------- #
def _c2(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _C2.to(dtype=dtype, device=device)


def _w2(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Standard BGK weights (for the Guo source only)."""
    return torch.tensor(
        (4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0),
        dtype=dtype,
        device=device,
    )


def _swift_feq_2d(rho, ux, uy, psi, p, device, dtype):
    """Swift fluid equilibrium (D2Q9).

    ``f_i = wi [p + rho c_i.u + 1.5 (rho (c_i.u)^2 - rho u^2/3)]
              + kappa [wxx_i (d_x psi)^2 + wyy_i (d_y psi)^2 + wxy_i d_x psi d_y psi]``

    with the sailfish anisotropic coefficients; rest = rho - sum_{i>0}.
    """
    cx = _c2(device, dtype)[:, 0].view(9, 1, 1)
    cy = _c2(device, dtype)[:, 1].view(9, 1, 1)
    wi, wxx, wyy, wxy = _aniso2()
    wi_t = torch.tensor((0.0,) + wi, dtype=dtype, device=device).view(9, 1, 1)
    wxx_t = torch.tensor((0.0,) + wxx, dtype=dtype, device=device).view(9, 1, 1)
    wyy_t = torch.tensor((0.0,) + wyy, dtype=dtype, device=device).view(9, 1, 1)
    wxy_t = torch.tensor((0.0,) + wxy, dtype=dtype, device=device).view(9, 1, 1)
    cu = cx * ux + cy * uy
    usq = ux * ux + uy * uy
    gx, gy, lap = _grad_lap2(psi)
    mu = p.chemical_potential(psi, lap)
    pb = p.free_energy_pressure(rho, psi, mu)
    t = wi_t * (pb + rho * cu + 1.5 * (rho * cu * cu - rho * usq / 3.0))
    t = t + p.kappa * (wxx_t * gx * gx + wyy_t * gy * gy + wxy_t * gx * gy)
    rest = rho - t.sum(dim=0)
    return torch.cat([rest.unsqueeze(0), t[1:]], dim=0)


def _swift_geq_2d(psi, ux, uy, mu, p, device, dtype):
    """Order-parameter equilibrium (Cahn-Hilliard moment closure, D2Q9).

    ``g_i = wi [M mu + c_i.u psi + 1.5 psi ((c_i.u)^2 - u^2/3)]``, rest = psi - sum.
    """
    cx = _c2(device, dtype)[:, 0].view(9, 1, 1)
    cy = _c2(device, dtype)[:, 1].view(9, 1, 1)
    wi, _, _, _ = _aniso2()
    wi_t = torch.tensor((0.0,) + wi, dtype=dtype, device=device).view(9, 1, 1)
    cu = cx * ux + cy * uy
    usq = ux * ux + uy * uy
    t = wi_t * (p.mobility * mu + cu * psi + 1.5 * psi * (cu * cu - usq / 3.0))
    rest = psi - t.sum(dim=0)
    return torch.cat([rest.unsqueeze(0), t[1:]], dim=0)


def swift_init_2d(
    psi0: torch.Tensor,
    p: SwiftParams,
    *,
    ux0: torch.Tensor | None = None,
    uy0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialise ``(f, g)`` at local equilibrium of ``psi0`` (D2Q9)."""
    rho = p.density(psi0)
    ux = torch.zeros_like(psi0) if ux0 is None else ux0
    uy = torch.zeros_like(psi0) if uy0 is None else uy0
    f = _swift_feq_2d(rho, ux, uy, psi0, p, psi0.device, psi0.dtype)
    _, _, lap = _grad_lap2(psi0)
    mu = p.chemical_potential(psi0, lap)
    g = _swift_geq_2d(psi0, ux, uy, mu, p, psi0.device, psi0.dtype)
    return f, g


def swift_step_2d(
    f: torch.Tensor,
    g: torch.Tensor,
    p: SwiftParams,
    *,
    gx: float = 0.0,
    gy: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One collide+stream step of the Swift model on D2Q9 (periodic).

    The capillary force is the free-energy pressure ``p = rho cs^2 + psi mu cs^2``
    embedded in the equilibrium (the "nabla(psi) tensor force").  A body force
    ``rho * (gx, gy)`` (gravity / buoyancy) is added to the momentum via Guo.
    """
    psi = g.sum(0)
    rho = f.sum(0)  # macroscopic density (mass-conserving)
    rho_swift = p.density(psi)  # physical density (for the body force)
    c = _c2(f.device, f.dtype)
    cx, cy = c[:, 0].view(9, 1, 1), c[:, 1].view(9, 1, 1)
    w = _w2(f.device, f.dtype).view(9, 1, 1)
    ux = (f * cx).sum(0) / rho.clamp(min=1e-12)
    uy = (f * cy).sum(0) / rho.clamp(min=1e-12)
    feq = _swift_feq_2d(rho, ux, uy, psi, p, f.device, f.dtype)
    # Guo source for the body force (gravity / buoyancy), scaled by local density
    fx, fy = rho_swift * gx, rho_swift * gy
    cf = cx * fx + cy * fy
    uf = ux * fx + uy * fy
    cu = cx * ux + cy * uy
    guo = (1.0 - 0.5 / p.tau_f) * w * (3.0 * cf + 9.0 * cu * cf - 3.0 * uf)
    feq = feq + guo
    f_post = f - (f - feq) / p.tau_f
    _, _, lap = _grad_lap2(psi)
    mu = p.chemical_potential(psi, lap)
    geq = _swift_geq_2d(psi, ux, uy, mu, p, f.device, f.dtype)
    g_post = g - (g - geq) / p.tau_g
    return _stream2(f_post), _stream2(g_post)


# --------------------------------------------------------------------------- #
# 3-D (D3Q19)
# --------------------------------------------------------------------------- #
def _shift3(field: torch.Tensor, dx: int, dy: int, dz: int) -> torch.Tensor:
    return torch.roll(field, shifts=(-dz, -dy, -dx), dims=(-3, -2, -1))


def grad_lap_3d(psi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Seven-point ``(d_x, d_y, d_z, lap)`` on the periodic grid."""
    fxp = _shift3(psi, 1, 0, 0)
    fxm = _shift3(psi, -1, 0, 0)
    fyp = _shift3(psi, 0, 1, 0)
    fym = _shift3(psi, 0, -1, 0)
    fzp = _shift3(psi, 0, 0, 1)
    fzm = _shift3(psi, 0, 0, -1)
    gx = 0.5 * (fxp - fxm)
    gy = 0.5 * (fyp - fym)
    gz = 0.5 * (fzp - fzm)
    lap = fxp + fxm + fyp + fym + fzp + fzm - 6.0 * psi
    return gx, gy, gz, lap


def _c3(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _C3.to(dtype=dtype, device=device)


def _w3(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    from .d3q19 import W_EXACT64
    return W_EXACT64.to(dtype=dtype, device=device)


def _stream3(f: torch.Tensor) -> torch.Tensor:
    """Pull streaming on the periodic grid: ``f_i(x) = f_i^post(x - c_i)``."""
    c = _C3
    out = torch.empty_like(f)
    for i in range(19):
        out[i] = torch.roll(
            f[i],
            shifts=(int(c[i, 2]), int(c[i, 1]), int(c[i, 0])),
            dims=(-3, -2, -1),
        )
    return out


def _swift_feq_3d(rho, ux, uy, uz, psi, p, device, dtype):
    """Swift fluid equilibrium (D3Q19).

    ``f_i = w_i [p + rho c_i.u + 1.5 (rho (c_i.u)^2 - rho u^2/3)]`` with the
    standard BGK weights; rest = rho - sum_{i>0}.  The capillary force is the
    free-energy pressure ``p = rho cs^2 + psi mu cs^2`` (the "nabla(psi) tensor
    force").  The anisotropic Korteweg stress is neglected (the Swift 2009 model
    uses the isotropic free-energy pressure, not the full pressure tensor).
    """
    cx = _c3(device, dtype)[:, 0].view(19, 1, 1, 1)
    cy = _c3(device, dtype)[:, 1].view(19, 1, 1, 1)
    cz = _c3(device, dtype)[:, 2].view(19, 1, 1, 1)
    wv = _w3(device, dtype).view(19, 1, 1, 1)
    cu = cx * ux + cy * uy + cz * uz
    usq = ux * ux + uy * uy + uz * uz
    _, _, _, lap = grad_lap_3d(psi)
    mu = p.chemical_potential(psi, lap)
    pb = p.free_energy_pressure(rho, psi, mu)
    t = wv * (pb + rho * cu + 1.5 * (rho * cu * cu - rho * usq / 3.0))
    rest = rho - t.sum(dim=0)
    return torch.cat([rest.unsqueeze(0), t[1:]], dim=0)


def _swift_geq_3d(psi, ux, uy, uz, mu, p, device, dtype):
    """Order-parameter equilibrium (Cahn-Hilliard moment closure, D3Q19)."""
    cx = _c3(device, dtype)[:, 0].view(19, 1, 1, 1)
    cy = _c3(device, dtype)[:, 1].view(19, 1, 1, 1)
    cz = _c3(device, dtype)[:, 2].view(19, 1, 1, 1)
    wv = _w3(device, dtype).view(19, 1, 1, 1)
    cu = cx * ux + cy * uy + cz * uz
    usq = ux * ux + uy * uy + uz * uz
    t = wv * (p.mobility * mu + cu * psi + 1.5 * psi * (cu * cu - usq / 3.0))
    rest = psi - t.sum(dim=0)
    return torch.cat([rest.unsqueeze(0), t[1:]], dim=0)


def swift_init_3d(
    psi0: torch.Tensor,
    p: SwiftParams,
    *,
    ux0: torch.Tensor | None = None,
    uy0: torch.Tensor | None = None,
    uz0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialise ``(f, g)`` at local equilibrium of ``psi0`` (D3Q19)."""
    rho = p.density(psi0)
    ux = torch.zeros_like(psi0) if ux0 is None else ux0
    uy = torch.zeros_like(psi0) if uy0 is None else uy0
    uz = torch.zeros_like(psi0) if uz0 is None else uz0
    f = _swift_feq_3d(rho, ux, uy, uz, psi0, p, psi0.device, psi0.dtype)
    _, _, _, lap = grad_lap_3d(psi0)
    mu = p.chemical_potential(psi0, lap)
    g = _swift_geq_3d(psi0, ux, uy, uz, mu, p, psi0.device, psi0.dtype)
    return f, g


def swift_step_3d(
    f: torch.Tensor,
    g: torch.Tensor,
    p: SwiftParams,
    *,
    gx: float = 0.0,
    gy: float = 0.0,
    gz: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One collide+stream step of the Swift model on D3Q19 (periodic).

    The capillary force is the free-energy pressure ``p = rho cs^2 + psi mu cs^2``
    embedded in the equilibrium (the "nabla(psi) tensor force").  A body force
    ``rho * (gx, gy, gz)`` (gravity / buoyancy) is added to the momentum via Guo.
    """
    psi = g.sum(0)
    rho = f.sum(0)  # macroscopic density (mass-conserving)
    rho_swift = p.density(psi)  # physical density (for the body force)
    c = _c3(f.device, f.dtype)
    cx, cy, cz = c[:, 0].view(19, 1, 1, 1), c[:, 1].view(19, 1, 1, 1), c[:, 2].view(19, 1, 1, 1)
    w = _w3(f.device, f.dtype).view(19, 1, 1, 1)
    ux = (f * cx).sum(0) / rho.clamp(min=1e-12)
    uy = (f * cy).sum(0) / rho.clamp(min=1e-12)
    uz = (f * cz).sum(0) / rho.clamp(min=1e-12)
    feq = _swift_feq_3d(rho, ux, uy, uz, psi, p, f.device, f.dtype)
    # Guo source for the body force (gravity / buoyancy), scaled by local density
    fx, fy, fz = rho_swift * gx, rho_swift * gy, rho_swift * gz
    cf = cx * fx + cy * fy + cz * fz
    uf = ux * fx + uy * fy + uz * fz
    cu = cx * ux + cy * uy + cz * uz
    guo = (1.0 - 0.5 / p.tau_f) * w * (3.0 * cf + 9.0 * cu * cf - 3.0 * uf)
    feq = feq + guo
    f_post = f - (f - feq) / p.tau_f
    _, _, _, lap = grad_lap_3d(psi)
    mu = p.chemical_potential(psi, lap)
    geq = _swift_geq_3d(psi, ux, uy, uz, mu, p, f.device, f.dtype)
    g_post = g - (g - geq) / p.tau_g
    return _stream3(f_post), _stream3(g_post)
