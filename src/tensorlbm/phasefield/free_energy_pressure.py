"""Free-energy binary-fluid LBM with the *pressure tensor embedded in the equilibrium*.

This module is a semantics transplant of the sailfish implementation
(``sailfish-team/sailfish``, LGPL3), file by file and line by line:

``sym_equilibrium.py:26-48``  ``free_energy_equilibrium_fluid``
    ``pb   = rho/3 + A(-phi^2/2 + 3 phi^4/4)``
    ``f_i  = w_i [ pb - kappa phi lap(phi) + rho (e_i.u)
                   + 3/2 ( rho (e_i.u)^2 + lambda (2 (e_i.u)(e_i.grad0)
                                           + e_i.e_i u.grad0)
                           - 1/3 (rho u.u + 3 lambda u.grad0) ) ]
             + kappa [ wxx_i (d_x phi)^2 + wyy_i (d_y phi)^2
                       + wxy_i (d_x phi)(d_y phi) ]``
    ``lambda = visc (1 - 3 cs^2) = 0`` on D2Q9/D3Q19 (``cssq == 1/3``), so the
    ``grad0`` terms vanish identically; they are kept in the docstring because
    they are the reason this model needs no Guo forcing.
    The rest population is ``f_0 = rho - sum_{i>0} f_i`` (mass conservation).

``sym_equilibrium.py:50-70``  ``free_energy_equilibrium_order_param``
    ``mu   = A(-phi + phi^3) - kappa lap(phi)``
    ``g_i  = w_i [ Gamma mu + (e_i.u) phi
                   + 3/2 phi ( (e_i.u)^2 - 1/3 u.u ) ]``,  ``g_0 = phi - sum``.

``lb_binary.py:185-268``  ``LBBinaryFluidFreeEnergy._prepare_symbols``
    the anisotropic second-moment coefficients ``wxx/wyy/wzz/wxy/wyz/wxz``
    (D2Q9 branch reproduced in :func:`anisotropic_coefficients`), and the
    ``wi`` used by the model: axis ``1/3``, diagonal ``1/12`` -- i.e. three
    times the standard BGK weights, which is what turns ``w_i pb`` into
    ``rho/3 + A(...)`` after the 2nd moment.

``templates/finite_difference_optimized.mako:46-48``  (D2Q9 branch)
    the nine-point isotropic operators used for ``grad phi`` and ``lap phi``
    (reproduced in :func:`grad_lap`).

Why the pressure tensor matters: unlike the ``-phi grad(mu)`` Korteweg-force
route, the second moment of ``f`` **is** the free-energy pressure tensor

    ``P_ab = [pb - kappa phi lap(phi)] delta_ab
             + kappa ( d_a phi d_b phi - 1/2 |grad phi|^2 delta_ab )``

so a pressure field can be measured from the populations (no reconstruction
postulate) and the flat-interface mechanical-equilibrium condition
``d P_NN / dx = 0`` and the Laplace law ``p_in - p_out = sigma / R`` become
direct diagnostics.  The equilibrium identity
``sum_i f_i^eq c_ia c_ib == P_ab + rho u_a u_b`` is checked to machine
precision in ``tests/test_free_energy_pressure_xval.py``.

Analytic anchors (``f = A(-phi^2/2 + phi^4/4) + kappa |grad phi|^2/2``):

* interface width    ``xi    = sqrt(2 kappa / A)``
* surface tension     ``sigma = (2/3) sqrt(2 kappa A) = (2/3) A xi``
* profile             ``phi(x) = tanh((x - x0) / xi)``
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

#: D2Q9 velocities, same order as ``sailfish.sym.D2Q9.basis``.
CX = (0, 1, 0, -1, 0, 1, -1, -1, 1)
CY = (0, 0, 1, 0, -1, 1, 1, -1, -1)

#: Standard BGK weights, ``sailfish.sym.D2Q9.weights``.
W_STD = (4.0 / 9.0,) + (1.0 / 9.0,) * 4 + (1.0 / 36.0,) * 4

CS2 = 1.0 / 3.0


def anisotropic_coefficients() -> tuple[tuple[float, ...], ...]:
    """Return ``(wi, wxx, wyy, wxy)`` with the D2Q9 values of ``lb_binary.py:249-268``.

    The ``wi`` are three times the BGK weights (axis ``1/3``, diagonal ``1/12``).
    """
    wi, wxx, wyy, wxy = [], [], [], []
    for k in range(1, 9):
        x, y = CX[k], CY[k]
        if x * x + y * y == 1:
            wi.append(1.0 / 3.0)
            wxx.append(1.0 / 3.0 if abs(x) == 1 else -1.0 / 6.0)
            wyy.append(1.0 / 3.0 if abs(y) == 1 else -1.0 / 6.0)
        else:
            wi.append(1.0 / 12.0)
            wxx.append(-1.0 / 24.0)
            wyy.append(-1.0 / 24.0)
        wxy.append(x * y / 4.0)
    return tuple(wi), tuple(wxx), tuple(wyy), tuple(wxy)


@dataclass(frozen=True)
class FreeEnergyBinaryParams:
    """Free-energy binary-fluid parameters in lattice units (``A = B`` wells)."""

    A: float = 0.1
    kappa: float = 0.2
    Gamma: float = 1.0
    tau_f: float = 1.0
    tau_phi: float = 1.0

    @property
    def xi(self) -> float:
        """Analytic interface width ``sqrt(2 kappa / A)``."""
        return (2.0 * self.kappa / self.A) ** 0.5

    @property
    def sigma(self) -> float:
        """Analytic surface tension ``(2/3) sqrt(2 kappa A)``."""
        return (2.0 / 3.0) * (2.0 * self.kappa * self.A) ** 0.5

    @property
    def viscosity(self) -> float:
        return CS2 * (self.tau_f - 0.5)

    @property
    def mobility(self) -> float:
        """Cahn-Hilliard mobility ``M = Gamma (tau_phi - 1/2) cs^2``."""
        return self.Gamma * (self.tau_phi - 0.5) * CS2


# --------------------------------------------------------------------------- #
# nine-point isotropic operators (sailfish finite_difference_optimized.mako)
# --------------------------------------------------------------------------- #
def _shift(field: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Return ``field(x + dx, y + dy)`` on the periodic (y, x) grid."""
    return torch.roll(field, shifts=(-dy, -dx), dims=(-2, -1))


def grad_lap(phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(d_x phi, d_y phi, lap phi)`` with the sailfish nine-point stencil."""
    fe = _shift(phi, 1, 0)
    fw = _shift(phi, -1, 0)
    fn = _shift(phi, 0, 1)
    fs = _shift(phi, 0, -1)
    fne = _shift(phi, 1, 1)
    fnw = _shift(phi, -1, 1)
    fse = _shift(phi, 1, -1)
    fsw = _shift(phi, -1, -1)

    gx = (-fnw - fsw + fse + fne) / 12.0 + (fe - fw) / 3.0
    gy = (-fse - fsw + fne + fnw) / 12.0 + (fn - fs) / 3.0
    lap = (fnw + fne + fsw + fse + 4.0 * (fe + fw + fn + fs) - 20.0 * phi) / 6.0
    return gx, gy, lap


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def bulk_pressure(rho: torch.Tensor, phi: torch.Tensor, A: float) -> torch.Tensor:
    """``pb = rho/3 + A(-phi^2/2 + 3 phi^4/4)`` (``sym_equilibrium.py:29``)."""
    return rho * CS2 + A * (-0.5 * phi**2 + 0.75 * phi**4)


def chemical_potential(
    phi: torch.Tensor, lap: torch.Tensor, A: float, kappa: float
) -> torch.Tensor:
    """``mu = A(-phi + phi^3) - kappa lap(phi)`` (``sym_equilibrium.py:55``)."""
    return A * (-phi + phi**3) - kappa * lap


def pressure_tensor_analytic(
    rho: torch.Tensor,
    phi: torch.Tensor,
    gx: torch.Tensor,
    gy: torch.Tensor,
    lap: torch.Tensor,
    A: float,
    kappa: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(Pxx, Pyy, Pxy)`` of the free-energy pressure tensor.

    ``P_ab = [pb - kappa phi lap(phi)] delta_ab
             + kappa (d_a phi d_b phi - |grad phi|^2 delta_ab / 2)``
    """
    pb = bulk_pressure(rho, phi, A) - kappa * phi * lap
    g2 = gx * gx + gy * gy
    Pxx = pb + kappa * (gx * gx - 0.5 * g2)
    Pyy = pb + kappa * (gy * gy - 0.5 * g2)
    Pxy = kappa * gx * gy
    return Pxx, Pyy, Pxy


def _coeff_tensors(
    dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, ...]:
    """Return ``(cx, cy, wi, wxx, wyy, wxy)`` as ``(9, 1, 1)`` tensors (k=0 -> 0)."""
    wi, wxx, wyy, wxy = anisotropic_coefficients()
    cx = torch.tensor(CX, dtype=dtype, device=device).view(9, 1, 1)
    cy = torch.tensor(CY, dtype=dtype, device=device).view(9, 1, 1)
    z = (0.0,)
    wi_t = torch.tensor(z + wi, dtype=dtype, device=device).view(9, 1, 1)
    wxx_t = torch.tensor(z + wxx, dtype=dtype, device=device).view(9, 1, 1)
    wyy_t = torch.tensor(z + wyy, dtype=dtype, device=device).view(9, 1, 1)
    wxy_t = torch.tensor(z + wxy, dtype=dtype, device=device).view(9, 1, 1)
    return cx, cy, wi_t, wxx_t, wyy_t, wxy_t


def equilibrium_f(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    phi: torch.Tensor,
    gx: torch.Tensor,
    gy: torch.Tensor,
    lap: torch.Tensor,
    A: float,
    kappa: float,
) -> torch.Tensor:
    """Fluid equilibrium ``f^eq`` of ``sym_equilibrium.py:26-48`` (D2Q9, lambda=0)."""
    cx, cy, wi, wxx, wyy, wxy = _coeff_tensors(phi.dtype, phi.device)
    body = bulk_pressure(rho, phi, A) - kappa * phi * lap
    cu = cx * ux + cy * uy
    usq = ux * ux + uy * uy
    t = wi * (body + rho * cu + 1.5 * (rho * cu * cu - rho * usq / 3.0))
    t = t + kappa * (wxx * gx * gx + wyy * gy * gy + wxy * gx * gy)
    rest = rho - t.sum(dim=0)
    return torch.cat([rest.unsqueeze(0), t[1:]], dim=0)


def equilibrium_g(
    phi: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    mu: torch.Tensor,
    Gamma: float,
) -> torch.Tensor:
    """Order-parameter equilibrium ``g^eq`` of ``sym_equilibrium.py:50-70``."""
    cx, cy, wi, _, _, _ = _coeff_tensors(phi.dtype, phi.device)
    cu = cx * ux + cy * uy
    usq = ux * ux + uy * uy
    t = wi * (Gamma * mu + cu * phi + 1.5 * phi * (cu * cu - usq / 3.0))
    rest = phi - t.sum(dim=0)
    return torch.cat([rest.unsqueeze(0), t[1:]], dim=0)


def macroscopic(
    f: torch.Tensor, g: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(rho, ux, uy, phi)`` from the populations (no forcing term)."""
    rho = f.sum(dim=0)
    phi = g.sum(dim=0)
    cx = torch.tensor(CX, dtype=f.dtype, device=f.device).view(9, 1, 1)
    cy = torch.tensor(CY, dtype=f.dtype, device=f.device).view(9, 1, 1)
    ux = (f * cx).sum(dim=0) / rho
    uy = (f * cy).sum(dim=0) / rho
    return rho, ux, uy, phi


@functools.lru_cache(maxsize=8)
def _stream_index(ny: int, nx: int) -> torch.Tensor:
    """Flat gather index realising ``f(x) = f(x - c)`` on the periodic grid."""
    yy = torch.arange(ny).view(-1, 1)
    xx = torch.arange(nx).view(1, -1)
    idx = [((yy - CY[k]) % ny) * nx + (xx - CX[k]) % nx for k in range(9)]
    return torch.stack([i.reshape(-1) for i in idx])


def stream(f: torch.Tensor) -> torch.Tensor:
    """Pull streaming on the periodic grid: ``f_i(x) = f_i^post(x - c_i)``."""
    ny, nx = f.shape[-2], f.shape[-1]
    idx = _stream_index(ny, nx).to(f.device)
    return torch.gather(f.reshape(9, -1), 1, idx).reshape(9, ny, nx)



def collide_and_stream(
    f: torch.Tensor, g: torch.Tensor, p: FreeEnergyBinaryParams
) -> tuple[torch.Tensor, torch.Tensor]:
    """One BGK collide + stream step of the free-energy binary model."""
    rho, ux, uy, phi = macroscopic(f, g)
    gx, gy, lap = grad_lap(phi)
    mu = chemical_potential(phi, lap, p.A, p.kappa)
    feq = equilibrium_f(rho, ux, uy, phi, gx, gy, lap, p.A, p.kappa)
    geq = equilibrium_g(phi, ux, uy, mu, p.Gamma)
    f_post = f - (f - feq) / p.tau_f
    g_post = g - (g - geq) / p.tau_phi
    return stream(f_post), stream(g_post)


def initial_condition(
    phi0: torch.Tensor,
    p: FreeEnergyBinaryParams,
    *,
    rho0: float = 1.0,
    ux0: torch.Tensor | None = None,
    uy0: torch.Tensor | None = None,
    phi_eq_g: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(f, g)`` initialised at local equilibrium of the given ``phi0``."""
    shape = phi0.shape
    rho = torch.full(shape, rho0, dtype=phi0.dtype, device=phi0.device)
    ux = torch.zeros_like(phi0) if ux0 is None else ux0
    uy = torch.zeros_like(phi0) if uy0 is None else uy0
    gx, gy, lap = grad_lap(phi0)
    f = equilibrium_f(rho, ux, uy, phi0, gx, gy, lap, p.A, p.kappa)
    if phi_eq_g:
        mu = chemical_potential(phi0, lap, p.A, p.kappa)
    else:
        mu = torch.zeros_like(phi0)
    g = equilibrium_g(phi0, ux, uy, mu, p.Gamma)
    return f, g


# --------------------------------------------------------------------------- #
# measurement (from the populations, never from the analytic formula)
# --------------------------------------------------------------------------- #
def pressure_tensor_measured(
    f: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(Pxx, Pyy, Pxy) = sum_i f_i c_ia c_ib`` (non-equilibrium included)."""
    cx = torch.tensor(CX, dtype=f.dtype, device=f.device).view(9, 1, 1)
    cy = torch.tensor(CY, dtype=f.dtype, device=f.device).view(9, 1, 1)
    Pxx = (f * cx * cx).sum(dim=0)
    Pyy = (f * cy * cy).sum(dim=0)
    Pxy = (f * cx * cy).sum(dim=0)
    return Pxx, Pyy, Pxy


def mean_pressure(f: torch.Tensor) -> torch.Tensor:
    """Isotropic pressure ``p = (P_xx + P_yy) / 2`` from the second moment."""
    Pxx, Pyy, _ = pressure_tensor_measured(f)
    return 0.5 * (Pxx + Pyy)


# --------------------------------------------------------------------------- #
# analytic anchors
# --------------------------------------------------------------------------- #
def periodic_distance(
    coordinate: torch.Tensor, position: float, period: float
) -> torch.Tensor:
    """Signed shortest distance on a periodic domain of length ``period``."""
    return torch.remainder(coordinate - position + period / 2.0, period) - period / 2.0


def tanh_profile(
    coordinate: torch.Tensor, position: float, xi: float, *, amplitude: float = 1.0
) -> torch.Tensor:
    """Equilibrium order-parameter profile ``amplitude * tanh((x - x0)/xi)``."""
    return amplitude * torch.tanh((coordinate - position) / xi)


def sigma_from_pressure_tensor(
    Pxx: torch.Tensor, Pyy: torch.Tensor, *, axis: int = -1
) -> torch.Tensor:
    """Surface tension ``integral (P_NN - P_TT)`` along ``axis`` (flat interface)."""
    return (Pxx - Pyy).sum(dim=axis)


def spurious_velocity(ux: torch.Tensor, uy: torch.Tensor) -> dict[str, float]:
    """Return max/RMS of ``|u|`` -- the parasitic current of a static interface."""
    speed = torch.sqrt(ux * ux + uy * uy)
    return {
        "max": float(speed.max().item()),
        "rms": float(torch.sqrt((speed * speed).mean()).item()),
    }