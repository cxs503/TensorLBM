"""Free-energy binary-fluid D2Q9 kernels with the **pressure tensor inside the
equilibrium** -- a semantics transplant of the OSS sailfish implementation.

Why this module exists
----------------------
TensorLBM's free-energy/phase-field chain (:mod:`tensorlbm.phasefield`) couples
the capillary stress to the momentum equation as an **explicit force**
(``mu grad(phi)`` / ``-phi grad(mu)``, Guo forcing).  In that formulation the
second moment of the populations is the *ideal-gas* moment ``cs^2 rho delta_ab
+ rho u_a u_b``; the non-ideal (Korteweg) part lives in the force term, so a
mechanical pressure cannot be read off the populations, and the Young-Laplace
law cannot be tested without postulating the pressure field (this is exactly the
``laplace_physical_validation: NOT_VALIDATED`` caveat recorded for the
pseudopotential route in :mod:`tensorlbm.scmc_pressure_tensor`).

The OSS reference implementation ``sailfish-team/sailfish`` (LGPL3) solves the
*same* free-energy binary-fluid model with a different discrete recipe: the
Korteweg stress is folded into the equilibrium distribution, so

    sum_i f_i^eq c_ia c_ib = P_ab + rho u_a u_b,          (exact, see below)
    P_ab = [pb - kappa phi lap(phi)] delta_ab
           + kappa (d_a phi d_b phi - 1/2 |grad phi|^2 delta_ab),
    pb   = rho/3 + A(-phi^2/2 + 3 phi^4/4).

This module is a clean-room transplant of that recipe.  Nothing is transcribed
from the LGPL3 sources: the algebraic form is *re-derived* here from the OSS
source **text** at run time by
``xval/XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT/derive_oss_pressure_tensor.py``,
which prints the OSS function with ``file:line`` and then proves symbolically
that the discrete second moment equals the continuum tensor above (residual
``0`` for the D2Q9 branch, ``ref/oss_discrete_pressure_tensor.json``).  The
coefficient tables and the nine-point operators below are the numbers that
derivation returns.

Provenance (file:line of the vendored OSS copy under ``xval/.../ref/sailfish_src``)
----------------------------------------------------------------------------------
``sailfish/sym_equilibrium.py:16-49``   ``free_energy_equilibrium_fluid``
``sailfish/sym_equilibrium.py:51-83``   ``free_energy_equilibrium_order_param``
``sailfish/lb_binary.py:224-259``       D2Q9 ``wi / wxx / wyy / wxy`` tables
``sailfish/templates/finite_difference_optimized.mako:46-48``  nine-point
    isotropic ``grad`` / ``lap`` (identical to the standard D2Q9 stencil:
    axis weights ``1/3``, diagonal ``1/12``)

Conventions
-----------
* ``A``      double-well amplitude, ``f = -A phi^2/2 + A phi^4/4 + kappa |grad phi|^2/2``
             (sailfish uses ``B = A``; the repo's :class:`DoubleWellFreeEnergy` keeps
             ``A`` and ``B`` separate, and this module accepts both).
* ``xi = sqrt(2 kappa / A)`` interface width, ``sigma = (2/3) A xi = (2/3) sqrt(2 kappa A)``
             surface tension (analytic, see the module-level tests).
* relaxation: ``omega = 1/tau`` for the fluid populations; ``Gamma`` is the order
  parameter relaxation coefficient (``Gamma = 1/omega_g`` in sailfish language).
* lattice: D2Q9, periodic wrap; tensor layout ``(9, ny, nx)``.

Everything is pure ``torch`` in ``float64``; no solver in the repository is
modified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

#: D2Q9 velocities in sailfish ``sym.D2Q9.basis`` order.
C = (
    (0, 0),
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
    (1, 1),
    (-1, 1),
    (-1, -1),
    (1, -1),
)
CS2 = 1.0 / 3.0


def oss_d2q9_coefficients() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """D2Q9 ``wi/wxx/wyy/wxy`` tables of ``sailfish/lb_binary.py:224-259``.

    ``wi`` are *three times* the BGK weights on the axes (``1/3`` vs ``1/9``) and
    *three times* on the diagonals (``1/12`` vs ``1/36``); that factor 3 is what
    turns ``sum_i wi_i pb c_ia c_ib`` into ``pb delta_ab``.
    """
    wi, wxx, wyy = [], [], []
    for ex, ey in C[1:]:
        if ex * ex + ey * ey == 1:
            wi.append(1.0 / 3.0)
            wxx.append(1.0 / 3.0 if abs(ex) == 1 else -1.0 / 6.0)
            wyy.append(1.0 / 3.0 if abs(ey) == 1 else -1.0 / 6.0)
        else:
            wi.append(1.0 / 12.0)
            wxx.append(-1.0 / 24.0)
            wyy.append(-1.0 / 24.0)
    wxy = [0.25 * ex * ey for ex, ey in C[1:]]
    return (
        torch.tensor(wi, dtype=torch.float64),
        torch.tensor(wxx, dtype=torch.float64),
        torch.tensor(wyy, dtype=torch.float64),
        torch.tensor(wxy, dtype=torch.float64),
    )


WI_OSS, WXX_OSS, WYY_OSS, WXY_OSS = oss_d2q9_coefficients()
#: 9-length versions with the rest direction zero-filled (``f_0`` is set separately).
WI = torch.cat([torch.zeros(1, dtype=torch.float64), WI_OSS])
WXX = torch.cat([torch.zeros(1, dtype=torch.float64), WXX_OSS])
WYY = torch.cat([torch.zeros(1, dtype=torch.float64), WYY_OSS])
WXY = torch.cat([torch.zeros(1, dtype=torch.float64), WXY_OSS])
CX = torch.tensor([c[0] for c in C], dtype=torch.float64)
CY = torch.tensor([c[1] for c in C], dtype=torch.float64)
W_STD = torch.tensor([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=torch.float64)


def _c(name: str) -> torch.Tensor:
    return {"x": CX, "y": CY, "wi": WI, "wxx": WXX, "wyy": WYY, "wxy": WXY, "w": W_STD}[name]


def _w(t: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    return t.view(-1, *([1] * len(shape))).to(torch.float64)


def shift(f: torch.Tensor, ex: int, ey: int) -> torch.Tensor:
    """Periodic ``+e`` shift: result[..., y, x] = f[..., y+ey, x+ex]."""
    return torch.roll(f, shifts=(-ey, -ex), dims=(-2, -1))


def grad_lap(phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nine-point isotropic gradient and Laplacian (OSS ``finite_difference_optimized.mako:46-48``).

    ``grad_x = (f_e - f_w)/3 + (f_ne + f_se - f_nw - f_sw)/12``
    ``grad_y = (f_n - f_s)/3 + (f_ne + f_nw - f_se - f_sw)/12``
    ``lap    = [4(f_e+f_w+f_n+f_s) + (f_ne+f_nw+f_se+f_sw) - 20 f]/6``
    """
    e, w = shift(phi, 1, 0), shift(phi, -1, 0)
    n, s = shift(phi, 0, 1), shift(phi, 0, -1)
    ne, nw = shift(phi, 1, 1), shift(phi, -1, 1)
    se, sw = shift(phi, 1, -1), shift(phi, -1, -1)
    gx = (e - w) / 3.0 + (ne + se - nw - sw) / 12.0
    gy = (n - s) / 3.0 + (ne + nw - se - sw) / 12.0
    lap = (4.0 * (e + w + n + s) + (ne + nw + se + sw) - 20.0 * phi) / 6.0
    return gx, gy, lap


def bulk_pressure(phi: torch.Tensor, rho: torch.Tensor, A: float, B: float | None = None) -> torch.Tensor:
    """OSS ``pb = rho/3 + A(-phi^2/2 + 3 phi^4/4)`` (``sym_equilibrium.py:26``)."""
    b = A if B is None else B
    return rho / 3.0 + A * (-(phi**2) / 2.0) + b * 3.0 * phi**4 / 4.0


def pressure_tensor(
    phi: torch.Tensor,
    *,
    A: float,
    kappa: float,
    rho: torch.Tensor | None = None,
    B: float | None = None,
    grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Continuum free-energy pressure tensor realised by the OSS equilibrium.

    ``P_ab = [pb - kappa phi lap(phi)] delta_ab + kappa (d_a phi d_b phi - |grad|^2/2 delta_ab)``
    """
    if rho is None:
        rho = torch.ones_like(phi)
    gx, gy, lap = grads if grads is not None else grad_lap(phi)
    pb = bulk_pressure(phi, rho, A, B)
    iso = pb - kappa * phi * lap
    g2 = gx * gx + gy * gy
    pxx = iso + kappa * (gx * gx - 0.5 * g2)
    pyy = iso + kappa * (gy * gy - 0.5 * g2)
    pxy = kappa * gx * gy
    return pxx, pyy, pxy


def fluid_equilibrium(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    phi: torch.Tensor,
    *,
    A: float,
    kappa: float,
    B: float | None = None,
    grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """OSS ``free_energy_equilibrium_fluid`` for D2Q9 with ``lambda = visc(1-3cs^2) = 0``.

    ``f_i = wi_i[pb - kappa phi lap phi + rho (e_i.u)] + (3/2) wi_i rho [(e_i.u)^2 - u^2/3]``
    ``      + kappa [wxx_i gx^2 + wyy_i gy^2 + wxy_i gx gy]``, ``f_0 = rho - sum_{i>0} f_i``
    """
    gx, gy, lap = grads if grads is not None else grad_lap(phi)
    shape = phi.shape
    pb = bulk_pressure(phi, rho, A, B)
    eu = _w(CX, shape) * ux + _w(CY, shape) * uy
    u2 = ux * ux + uy * uy
    w_o = _w(WI, shape)
    body = w_o * (pb.unsqueeze(0) - kappa * (phi * lap).unsqueeze(0) + rho.unsqueeze(0) * eu) \
        + 1.5 * w_o * rho.unsqueeze(0) * (eu * eu - u2.unsqueeze(0) / 3.0)
    aniso = kappa * (
        _w(WXX, shape) * (gx * gx).unsqueeze(0)
        + _w(WYY, shape) * (gy * gy).unsqueeze(0)
        + _w(WXY, shape) * (gx * gy).unsqueeze(0)
    )
    f = body + aniso  # row 0 is structurally zero (rest direction)
    f0 = rho - f.sum(dim=0)
    return torch.cat([f0.unsqueeze(0), f[1:]], dim=0)


def order_param_equilibrium(
    phi: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    mu: torch.Tensor,
    *,
    Gamma: float,
) -> torch.Tensor:
    """OSS ``free_energy_equilibrium_order_param`` (``sym_equilibrium.py:51-83``).

    ``g_i = wi_i[Gamma mu + (e_i.u) phi + (3/2) phi((e_i.u)^2 - u^2/3)]``, ``g_0 = phi - sum``
    """
    shape = phi.shape
    eu = _w(CX, shape) * ux + _w(CY, shape) * uy
    u2 = ux * ux + uy * uy
    w_o = _w(WI, shape)
    g = w_o * (Gamma * mu.unsqueeze(0) + eu * phi.unsqueeze(0)) \
        + 1.5 * w_o * phi.unsqueeze(0) * (eu * eu - u2.unsqueeze(0) / 3.0)
    g0 = phi - g.sum(dim=0)
    return torch.cat([g0.unsqueeze(0), g[1:]], dim=0)


def chemical_potential(
    phi: torch.Tensor, *, A: float, kappa: float, B: float | None = None
) -> torch.Tensor:
    """OSS ``mu = A(-phi + phi^3) - kappa lap(phi)`` (``sym_equilibrium.py:53``)."""
    b = A if B is None else B
    _, _, lap = grad_lap(phi)
    return -A * phi + b * phi**3 - kappa * lap


def bgk_equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Standard (ideal-gas) D2Q9 BGK equilibrium -- the TensorLBM force-based route."""
    shape = rho.shape
    eu = _w(CX, shape) * ux + _w(CY, shape) * uy
    u2 = ux * ux + uy * uy
    return _w(W_STD, shape) * rho.unsqueeze(0) * (
        1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2.unsqueeze(0)
    )


def moments(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``rho, ux, uy`` from populations (no force correction)."""
    shape = f.shape[1:]
    rho = f.sum(dim=0)
    ux = (f * _w(CX, shape)).sum(dim=0) / rho
    uy = (f * _w(CY, shape)).sum(dim=0) / rho
    return rho, ux, uy


def pressure_from_populations(
    f: torch.Tensor, *, with_velocity: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mechanical pressure tensor read **directly from the populations**.

    ``P_ab = sum_i f_i c_ia c_ib  [- rho u_a u_b]``.  With the OSS equilibrium the
    bracket-free expression equals ``P_ab + rho u_a u_b`` exactly, so subtracting
    the momentum flux returns the free-energy pressure tensor.
    """
    shape = f.shape[1:]
    pxx = (f * _w(CX * CX, shape)).sum(dim=0)
    pyy = (f * _w(CY * CY, shape)).sum(dim=0)
    pxy = (f * _w(CX * CY, shape)).sum(dim=0)
    if with_velocity:
        rho, ux, uy = moments(f)
        pxx = pxx - rho * ux * ux
        pyy = pyy - rho * uy * uy
        pxy = pxy - rho * ux * uy
    return pxx, pyy, pxy


def stream(f: torch.Tensor) -> torch.Tensor:
    """Periodic pull-streaming ``f_i(x) <- f_i(x - c_i)``."""
    out = []
    for i, (ex, ey) in enumerate(C):
        out.append(shift(f[i], ex, ey))
    return torch.stack(out, dim=0)


def collide(f: torch.Tensor, feq: torch.Tensor, tau: float) -> torch.Tensor:
    """BGK collision ``f + (feq - f)/tau``."""
    return f + (feq - f) / tau


# ---------------------------------------------------------------------------
# analytic anchors and diagnostics
# ---------------------------------------------------------------------------
def interface_width(A: float, kappa: float) -> float:
    """``xi = sqrt(2 kappa / A)``."""
    return math.sqrt(2.0 * kappa / A)


def analytic_sigma(A: float, kappa: float) -> float:
    """``sigma = (2/3) sqrt(2 kappa A) = (2/3) A xi``."""
    return (2.0 / 3.0) * math.sqrt(2.0 * kappa * A)


def sigma_from_flat_dip(
    pxx: torch.Tensor,
    pyy: torch.Tensor,
    *,
    axis: str = "y",
    center: int | None = None,
    half_width: int | None = None,
) -> float:
    """``sigma = int (P_NN - P_TT) dn`` across **one** flat interface.

    ``axis="y"``: normal along ``y`` (``P_NN = P_yy``); the dip is azimuthally
    averaged over the periodic ``x`` direction, then summed over a window
    ``[center - half_width, center + half_width]`` so that a periodic band with
    two interfaces is not counted twice (the mistake that inflates ``sigma`` by
    a factor 2 if the whole domain is summed).
    """
    dip = (pyy - pxx) if axis == "y" else (pxx - pyy)
    prof = dip.mean(dim=-1) if axis == "y" else dip.mean(dim=-2)
    n = prof.numel()
    if center is None:
        center = n // 4
    if half_width is None:
        half_width = n // 4
    idx = torch.arange(n)
    win = (idx >= center - half_width) & (idx <= center + half_width)
    return float(prof[win].sum())


def radial_profile(field: torch.Tensor, center: tuple[float, float] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Azimuthal mean of ``field`` on integer radius shells around ``center``.

    Returns ``(r, value)`` with ``value`` averaged over the 8-fold symmetric
    shells (so a circular droplet pays no O(1) angular discretisation error in
    the *profile*, only the tanh fit does).
    """
    ny, nx = field.shape
    cy, cx = center if center is not None else ((ny - 1) / 2.0, (nx - 1) / 2.0)
    yy = torch.arange(ny, dtype=torch.float64).view(-1, 1)
    xx = torch.arange(nx, dtype=torch.float64).view(1, -1)
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    nbins = int(r.max().item()) + 1
    acc = torch.zeros(nbins, dtype=torch.float64)
    cnt = torch.zeros(nbins, dtype=torch.float64)
    idx = torch.clamp(r.round().long(), max=nbins - 1)
    acc.index_add_(0, idx.flatten(), field.flatten())
    cnt.index_add_(0, idx.flatten(), torch.ones_like(field).flatten())
    rr = torch.arange(nbins, dtype=torch.float64)
    return rr[cnt > 0], (acc / cnt.clamp(min=1))[cnt > 0]


def sigma_from_radial_dip(phi: torch.Tensor, *, kappa: float, center: tuple[float, float] | None = None) -> float:
    """``sigma = int kappa (d phi/dr)^2 dr`` along the interface normal.

    ``P_NN - P_TT = kappa (d phi/dn)^2`` holds locally (it is the
    ``kappa (d_a phi d_b phi - 1/2 |grad phi|^2 delta_ab)`` tensor projected on
    the interface frame), so this integral measures the surface tension of a
    *curved* interface directly from the phase field and ``kappa`` -- it is
    independent of the analytic ``(2/3) sqrt(2 kappa A)`` value it is compared
    against.
    """
    r, prof = radial_profile(phi, center)
    dphi = torch.gradient(prof, spacing=1.0)[0]
    return float((kappa * dphi * dphi).sum())


def flat_interface_profile(
    ny: int,
    nx: int,
    *,
    A: float,
    kappa: float,
    y0: float | None = None,
) -> torch.Tensor:
    """Periodic flat-interface band: ``phi = -tanh((y - y0)/xi) tanh((y - (ny-y0))/xi)``.

    Two interfaces at ``y0`` and ``ny - y0``; each is measured in its own window.
    """
    xi = interface_width(A, kappa)
    y = torch.arange(ny, dtype=torch.float64).view(-1, 1).expand(ny, nx)
    if y0 is None:
        y0 = ny / 4.0
    return -torch.tanh((y - y0) / xi) * torch.tanh((y - (ny - y0)) / xi)


def droplet_profile(
    ny: int, nx: int, *, A: float, kappa: float, radius: float
) -> torch.Tensor:
    """``phi = -tanh((r - R)/xi)``: ``phi = -1`` inside (dense), ``+1`` outside."""
    xi = interface_width(A, kappa)
    yy = torch.arange(ny, dtype=torch.float64).view(-1, 1)
    xx = torch.arange(nx, dtype=torch.float64).view(1, -1)
    r = torch.sqrt((yy - (ny - 1) / 2.0) ** 2 + (xx - (nx - 1) / 2.0) ** 2)
    return -torch.tanh((r - radius) / xi)


@dataclass(frozen=True)
class FlatInterfaceResult:
    """Diagnostics of one quiescent flat-interface state (``u = 0`` everywhere)."""

    xi: float
    sigma_analytic: float
    sigma_measured: float
    rel_error: float
    dpnn_residual: float
    closure_residual: float
    max_speed: float


def quiescent_flat_interface(
    *,
    A: float,
    kappa: float,
    ny: int = 96,
    nx: int = 8,
    B: float | None = None,
    rho0: float = 1.0,
) -> FlatInterfaceResult:
    """Build the OSS equilibrium of a flat-interface band and measure ``sigma``.

    No time stepping: the equilibrium distribution is evaluated at ``u = 0``, so
    ``sum_i f_i c_ia c_ib`` *is* the pressure tensor and the measurement is a
    property of the equilibrium alone (``closure_residual`` checks that identity
    to machine precision).
    """
    phi = flat_interface_profile(ny, nx, A=A, kappa=kappa)
    rho = torch.full_like(phi, rho0)
    zero = torch.zeros_like(phi)
    grads = grad_lap(phi)
    f = fluid_equilibrium(rho, zero, zero, phi, A=A, kappa=kappa, B=B, grads=grads)
    pxx, pyy, _ = pressure_from_populations(f)
    pxx_c, pyy_c, _ = pressure_tensor(phi, A=A, kappa=kappa, rho=rho, B=B, grads=grads)
    half = max(2, int(round(4 * interface_width(A, kappa))))
    sigma_meas = sigma_from_flat_dip(pxx, pyy, axis="y", center=ny // 4, half_width=half)
    prof = pyy.mean(dim=-1)
    dpnn = torch.gradient(prof, spacing=1.0)[0]
    core = torch.zeros_like(prof, dtype=torch.bool)
    core[half : ny // 4] = True
    closure = float(max((pxx - pxx_c).abs().max(), (pyy - pyy_c).abs().max()))
    _, ux_m, uy_m = moments(f)
    sig_a = analytic_sigma(A, kappa)
    return FlatInterfaceResult(
        xi=interface_width(A, kappa),
        sigma_analytic=sig_a,
        sigma_measured=sigma_meas,
        rel_error=(sigma_meas - sig_a) / sig_a,
        dpnn_residual=float(dpnn[core].abs().max()) if core.any() else float(dpnn.abs().max()),
        closure_residual=closure,
        max_speed=float(torch.sqrt(ux_m**2 + uy_m**2).max()),
    )


@dataclass(frozen=True)
class DropletResult:
    """Diagnostics of one quiescent droplet (``u = 0``, ``rho = rho0``)."""

    radius: float
    xi: float
    sigma_analytic: float
    sigma_measured: float
    rel_error: float
    dp_bulk: float
    dp_curvature_bound: float
    max_speed: float


def quiescent_droplet(
    *,
    A: float,
    kappa: float,
    radius: float,
    ny: int | None = None,
    B: float | None = None,
    rho0: float = 1.0,
    pad: float = 5.0,
) -> DropletResult:
    """Surface tension of a quiescent droplet from the radial pressure-tensor dip.

    ``sigma`` is measured as ``int kappa (d phi/dr)^2 dr`` (:func:`sigma_from_radial_dip`)
    and compared with ``(2/3) sqrt(2 kappa A)``.

    ``dp_bulk`` is the *bulk* mechanical-pressure jump ``p_in - p_out`` with
    ``p = (P_xx + P_yy)/2``.  For the symmetric double well used by the OSS model
    ``pb(phi=+1) = pb(phi=-1) = rho/3 + A/4``, so ``dp_bulk`` is *identically*
    zero -- the curvature signal lives entirely inside the interface.  That is a
    property of this model variant, not a defect of the transplant, and the
    report states it explicitly instead of dressing an integral identity up as a
    Laplace-law verification.  ``dp_curvature_bound`` is the mechanical balance
    value ``-sigma/R`` for orientation.
    """
    xi = interface_width(A, kappa)
    n = ny or int(math.ceil(2 * (radius + pad * xi)))
    phi = droplet_profile(n, n, A=A, kappa=kappa, radius=radius)
    rho = torch.full_like(phi, rho0)
    zero = torch.zeros_like(phi)
    grads = grad_lap(phi)
    f = fluid_equilibrium(rho, zero, zero, phi, A=A, kappa=kappa, B=B, grads=grads)
    pxx, pyy, _ = pressure_from_populations(f)
    p = 0.5 * (pxx + pyy)
    yy = torch.arange(n, dtype=torch.float64).view(-1, 1)
    xx = torch.arange(n, dtype=torch.float64).view(1, -1)
    r = torch.sqrt((yy - (n - 1) / 2.0) ** 2 + (xx - (n - 1) / 2.0) ** 2)
    inside = r < (radius - 3.0 * xi)
    outside = r > (radius + 3.0 * xi)
    dp = float(p[inside].mean() - p[outside].mean())
    sig = analytic_sigma(A, kappa)
    sigma_meas = sigma_from_radial_dip(phi, kappa=kappa)
    _, ux_m, uy_m = moments(f)
    return DropletResult(
        radius=radius,
        xi=xi,
        sigma_analytic=sig,
        sigma_measured=sigma_meas,
        rel_error=(sigma_meas - sig) / sig,
        dp_bulk=dp,
        dp_curvature_bound=-sig / radius,
        max_speed=float(torch.sqrt(ux_m**2 + uy_m**2).max()),
    )


def run_quiescent_steps(
    *,
    A: float,
    kappa: float,
    radius: float,
    steps: int,
    tau: float = 1.0,
    B: float | None = None,
    rho0: float = 1.0,
    pad: float = 5.0,
) -> dict[str, float]:
    """Advance the OSS equilibrium state ``steps`` BGK steps with frozen ``phi``.

    A true equilibrium must stay put: this is the spurious-current gate
    (``max|u|``) plus the density-adjustment gate.  ``dp_bulk`` is the bulk
    pressure jump after the run -- it can only depart from zero by the amount the
    compressible branch permits (``sigma/(R cs^2)`` order), which is exactly the
    quantity to watch for the Laplace-law discussion.
    """
    xi = interface_width(A, kappa)
    n = int(math.ceil(2 * (radius + pad * xi)))
    phi = droplet_profile(n, n, A=A, kappa=kappa, radius=radius)
    rho = torch.full_like(phi, rho0)
    zero = torch.zeros_like(phi)
    grads = grad_lap(phi)
    feq = fluid_equilibrium(rho, zero, zero, phi, A=A, kappa=kappa, B=B, grads=grads)
    f = feq.clone()
    for _ in range(steps):
        f = stream(collide(f, feq, tau))
    pxx, pyy, _ = pressure_from_populations(f)
    p = 0.5 * (pxx + pyy)
    yy = torch.arange(n, dtype=torch.float64).view(-1, 1)
    xx = torch.arange(n, dtype=torch.float64).view(1, -1)
    r = torch.sqrt((yy - (n - 1) / 2.0) ** 2 + (xx - (n - 1) / 2.0) ** 2)
    inside = r < (radius - 3.0 * xi)
    outside = r > (radius + 3.0 * xi)
    dp = float(p[inside].mean() - p[outside].mean())
    sig = analytic_sigma(A, kappa)
    rho_m, ux_m, uy_m = moments(f)
    return {
        "steps": float(steps),
        "tau": tau,
        "dp_bulk": dp,
        "dp_over_sigma_R": dp / (sig / radius),
        "max_speed": float(torch.sqrt(ux_m**2 + uy_m**2).max()),
        "rho_drift_max": float((rho_m - rho0).abs().max()),
    }



# ---------------------------------------------------------------------------
# coupled free-energy binary solver: the Laplace-law arm
# ---------------------------------------------------------------------------
def sigma_from_measured_pressure(
    pxx: torch.Tensor,
    pyy: torch.Tensor,
    pxy: torch.Tensor | None = None,
    *,
    center: tuple[float, float] | None = None,
) -> float:
    """``sigma = int (P_rr - P_tt) dr`` from the pressure tensor **read off the populations**.

    ``P_rr - P_tt`` is the shell-projected anisotropy of the measured tensor
    (``P_ab r_a r_b`` vs ``P_ab t_a t_b``), so this is a *measurement* on the
    simulated state -- unlike :func:`sigma_from_radial_dip`, which differentiates
    the prescribed ``phi`` and closes the dip with ``kappa``.
    """
    ny, nx = pxx.shape
    cy, cx = center if center is not None else ((ny - 1) / 2.0, (nx - 1) / 2.0)
    yy = torch.arange(ny, dtype=torch.float64).view(-1, 1) - cy
    xx = torch.arange(nx, dtype=torch.float64).view(1, -1) - cx
    r = torch.sqrt(yy * yy + xx * xx)
    safe = r.clamp(min=1e-30)
    rx, ry = xx / safe, yy / safe
    prr = pxx * rx * rx + pyy * ry * ry
    if pxy is not None:
        prr = prr + 2.0 * pxy * rx * ry
    ptt = pxx + pyy - prr
    dip = prr - ptt
    dip = torch.where(r > 0.5, dip, torch.zeros_like(dip))
    _, prof = radial_profile(dip, center)
    return float(prof.sum())


@dataclass(frozen=True)
class CoupledDropletResult:
    """Diagnostics of one *coupled* (both distributions advanced) droplet arm."""

    radius: float
    xi: float
    steps: int
    sigma_analytic: float
    sigma_measured: float
    sigma_rel_error: float
    dp_bulk: float
    dp_tail_mean: float
    dp_laplace_ref: float
    laplace_rel_error: float
    laplace_tail_rel_error: float
    max_speed: float
    rho_min: float
    rho_max: float
    mass_err_rho: float
    mass_err_phi: float
    series: list


def coupled_step(
    f: torch.Tensor,
    g: torch.Tensor,
    *,
    A: float,
    kappa: float,
    Gamma: float,
    tau_f: float,
    tau_phi: float,
    B: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One BGK step of the coupled free-energy system (``f`` and ``g`` both advance).

    The OSS equilibrium is a functional of ``(rho, u, phi)`` for the fluid and of
    ``(phi, u, mu)`` for the order parameter, so the two distributions must be
    stepped together for the interface to relax to its *own* mechanical
    equilibrium -- freezing ``phi`` (as :func:`run_quiescent_steps` does) only
    proves that one equilibrium is a fixed point of the collision operator.
    """
    rho, ux, uy = moments(f)
    phi = g.sum(dim=0)
    mu = chemical_potential(phi, A=A, kappa=kappa, B=B)
    feq = fluid_equilibrium(rho, ux, uy, phi, A=A, kappa=kappa, B=B)
    geq = order_param_equilibrium(phi, ux, uy, mu, Gamma=Gamma)
    return stream(collide(f, feq, tau_f)), stream(collide(g, geq, tau_phi))


def run_coupled_droplet(
    *,
    A: float,
    kappa: float,
    radius: float,
    steps: int,
    tau_f: float = 1.0,
    tau_phi: float = 1.0,
    Gamma: float = 0.2,
    B: float | None = None,
    rho0: float = 1.0,
    pad: float = 5.0,
    record_every: int | None = None,
) -> CoupledDropletResult:
    """Advance the coupled free-energy droplet and measure ``sigma`` and the Laplace jump.

    ``dp_bulk = p_in - p_out`` with ``p = (P_xx + P_yy)/2`` from the populations;
    reference ``sigma_analytic / R``.  A non-finite state raises
    ``RuntimeError`` instead of being silently reported as a small error.
    """
    xi = interface_width(A, kappa)
    n = int(math.ceil(2 * (radius + pad * xi)))
    phi0 = droplet_profile(n, n, A=A, kappa=kappa, radius=radius)
    rho = torch.full_like(phi0, rho0)
    zero = torch.zeros_like(phi0)
    grads = grad_lap(phi0)
    f = fluid_equilibrium(rho, zero, zero, phi0, A=A, kappa=kappa, B=B, grads=grads)
    mu = chemical_potential(phi0, A=A, kappa=kappa, B=B)
    g = order_param_equilibrium(phi0, zero, zero, mu, Gamma=Gamma)
    yy = torch.arange(n, dtype=torch.float64).view(-1, 1)
    xx = torch.arange(n, dtype=torch.float64).view(1, -1)
    r = torch.sqrt((yy - (n - 1) / 2.0) ** 2 + (xx - (n - 1) / 2.0) ** 2)
    inside = r < (radius - 3.0 * xi)
    outside = r > (radius + 3.0 * xi)
    sig_a = analytic_sigma(A, kappa)
    every = record_every or max(1, steps // 10)
    series: list[dict] = []

    def _sample(step: int) -> dict:
        pxx, pyy, pxy = pressure_from_populations(f)
        p = 0.5 * (pxx + pyy)
        rho_m, ux_m, uy_m = moments(f)
        return {
            "step": int(step),
            "dp_bulk": float(p[inside].mean() - p[outside].mean()),
            "dp_over_sigma_R": float(p[inside].mean() - p[outside].mean()) / (sig_a / radius),
            "max_speed": float(torch.sqrt(ux_m**2 + uy_m**2).max()),
            "rho_min": float(rho_m.min()),
            "rho_max": float(rho_m.max()),
        }

    series.append(_sample(0))
    for s in range(1, steps + 1):
        f, g = coupled_step(f, g, A=A, kappa=kappa, Gamma=Gamma, tau_f=tau_f, tau_phi=tau_phi, B=B)
        if not torch.isfinite(f).all():
            raise RuntimeError(f"coupled droplet went non-finite at step {s}")
        if s % every == 0:
            series.append(_sample(s))

    pxx, pyy, pxy = pressure_from_populations(f)
    p = 0.5 * (pxx + pyy)
    dp = float(p[inside].mean() - p[outside].mean())
    # acoustic ringing makes the instantaneous jump oscillate by a few per cent;
    # the judged quantity is the mean over the last 40 % of the samples (declared
    # in predeclared.json *before* the sweep, not chosen after looking at it).
    tail = [r["dp_bulk"] for r in series if r["step"] >= 0.6 * steps and r["step"] > 0]
    dp_tail = sum(tail) / len(tail) if tail else dp
    sigma_m = sigma_from_measured_pressure(pxx, pyy, pxy)
    rho_m, ux_m, uy_m = moments(f)
    phi_m = g.sum(dim=0)
    return CoupledDropletResult(
        radius=radius,
        xi=xi,
        steps=steps,
        sigma_analytic=sig_a,
        sigma_measured=sigma_m,
        sigma_rel_error=(sigma_m - sig_a) / sig_a,
        dp_bulk=dp,
        dp_tail_mean=float(dp_tail),
        dp_laplace_ref=sig_a / radius,
        laplace_rel_error=(dp - sig_a / radius) / (sig_a / radius),
        laplace_tail_rel_error=(dp_tail - sig_a / radius) / (sig_a / radius),
        max_speed=float(torch.sqrt(ux_m**2 + uy_m**2).max()),
        rho_min=float(rho_m.min()),
        rho_max=float(rho_m.max()),
        mass_err_rho=float(abs(rho_m.sum() - rho0 * n * n) / (rho0 * n * n)),
        mass_err_phi=float(abs(phi_m.sum() - phi0.sum()) / max(1e-30, float(phi0.abs().sum()))),
        series=series,
    )


__all__ = [
    "C",
    "CS2",
    "DropletResult",
    "FlatInterfaceResult",
    "WI",
    "WI_OSS",
    "WXX",
    "WXX_OSS",
    "WXY",
    "WXY_OSS",
    "WYY",
    "WYY_OSS",
    "analytic_sigma",
    "bgk_equilibrium",
    "bulk_pressure",
    "chemical_potential",
    "coupled_step",
    "CoupledDropletResult",
    "collide",
    "droplet_profile",
    "flat_interface_profile",
    "fluid_equilibrium",
    "grad_lap",
    "interface_width",
    "moments",
    "order_param_equilibrium",
    "oss_d2q9_coefficients",
    "pressure_from_populations",
    "pressure_tensor",
    "quiescent_droplet",
    "quiescent_flat_interface",
    "radial_profile",
    "run_coupled_droplet",
    "run_quiescent_steps",
    "shift",
    "sigma_from_flat_dip",
    "sigma_from_measured_pressure",
    "sigma_from_radial_dip",
    "stream",
]
