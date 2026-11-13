"""Diffuse-interface free-energy LBM with wetting imposed by a *wall order parameter*.

This module is the TensorLBM port of the wetting treatment used by the open-source
code ``ebinan92/Fingering_dynamics`` (``lattice_boltzmann/validation.py``,
"wettability validation").  In that code the wall is **not** given a
wetting angle / wetting potential; instead a *ghost layer of the order
parameter* ``psi_wall_list`` is stacked above and below the fluid domain
(``np.vstack((psi_wall_list, np.vstack((psi, psi_wall_list))))``) and every
finite-difference stencil of the order parameter -- the isotropic 9-point
gradient and the isotropic 9-point Laplacian that build the chemical
potential -- reads that ghost value.  The wettability is therefore controlled
by a single scalar, the wall order parameter ``psi_w``, and the equilibrium
contact angle is an *emergent* function of ``psi_w``.

What is reproduced here
-----------------------
* free energy ``f(psi) = (a/4) (psi^2 - 1)^2`` with chemical potential
  ``mu = a psi (psi^2 - 1) - kappa lap(psi)``  (identical to the reference,
  ``getMu``), and the parameter pair

      ``kappa = (3/4) sigma xi``,  ``a = 2 kappa / xi^2``

  which gives the equilibrium profile ``psi = tanh(s/xi)``, an interface
  thickness ``xi``, and a model surface tension ``4 kappa / (3 xi) = sigma``
  (self-consistent -- checked by :func:`model_surface_tension`).
* the isotropic 9-point gradient / Laplacian of the reference
  (``getNabla_psix``, ``getNabla_psiy``, ``getNabla_psi2``; axis weight 4,
  diagonal weight 1, ``1/(12 dx)`` and ``1/(6 dx^2)`` normalisation).
* the order-parameter distribution ``g`` with the reference equilibrium
  (``getgeq``): advection of ``psi`` plus a mobility term
  ``Gamma mu / cs^2`` whose rest-direction part carries the ``-(1 - w_0)``
  correction so that ``sum_i geq_i = psi``.
* the momentum distribution ``f`` with the Korteweg force ``F = mu grad(psi)``
  inserted through the Guo forcing term ``(1 - 1/(2 tau_f)) w_i
  [(c_i - u)/cs^2 + c_i (c_i . u)/cs^4] . F`` (identical structure to
  ``getLarge_F``).
* half-way bounce-back for **both** distributions on the two walls
  (``halfway_bounceback``), i.e. a no-flux condition for the order parameter
  and a no-slip condition for the velocity; periodicity in ``x``.

Differences w.r.t. the reference (documented in
``docs/XVAL-LBM-DIFFUSE-INTERFACE-WETTING-VIA-WALL-ORDER-PARAMETER.md``):
the reference momentum equilibrium is the He/Liu incompressible pressure form
(``A0/B0`` split) with a non-Newtonian power-law viscosity that is disabled in
its own source; here the standard weakly compressible D2Q9 equilibrium is used
and the viscosity is Newtonian.  For static wetting -- the quantity that is
cross-validated -- neither difference affects the equilibrium contact angle
(the momentum field is only a diagnostic; spurious currents are reported).

Conventions
-----------
``psi`` is the order parameter in ``[-1, 1]``, ``psi = +1`` is the *wetting*
("liquid") phase.  Fluid rows are ``y = 0 .. ny-1``; the two ghost rows sit at
``y = -1`` and ``y = ny`` and hold ``psi_w``.  The halfway bounce-back wall
plane is at ``y = -1/2`` (bottom) and ``y = ny - 1/2`` (top); contact angles
below are measured against that wall plane.

Reference
---------
ebinan92/Fingering_dynamics, ``lattice_boltzmann/validation.py`` (MIT-style
research code, inventoried as OSS candidate), and the two-phase free-energy
LBM lineage it belongs to (Swift et al. 1996; Briant, Wagner & Yeomans 2004;
Pooley & Furtado 2008).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch

__all__ = [
    "EX",
    "EY",
    "W",
    "CS2",
    "WallOrderWettingConfig",
    "young_cos_theta",
    "analytic_contact_angle",
    "contact_angle_from_field",
    "free_energy_parameters",
    "grad_laplacian",
    "initialise_drop_on_wall",
    "macroscopics",
    "model_surface_tension",
    "relax_to_equilibrium",
    "run_to_equilibrium",
    "step",
]

#: D2Q9 velocity set, reference ordering: 0 rest, 1 E, 2 N, 3 W, 4 S,
#: 5 NE, 6 NW, 7 SW, 8 SE (identical to ``validation.py``'s ``self.e``).
EX: Tuple[float, ...] = (0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0)
EY: Tuple[float, ...] = (0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0)
W: Tuple[float, ...] = (4.0 / 9.0,) + (1.0 / 9.0,) * 4 + (1.0 / 36.0,) * 4
CS2: float = 1.0 / 3.0


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
def free_energy_parameters(surface_tension: float, interface_width: float) -> Tuple[float, float]:
    """``(a, kappa)`` of the double-well model for a target ``sigma`` and ``xi``.

    ``kappa = (3/4) sigma xi`` and ``a = 2 kappa / xi^2`` -- exactly the
    reference's choice -- give ``psi = tanh(s/xi)`` and ``4 kappa / (3 xi) = sigma``.
    """
    if surface_tension <= 0.0 or interface_width <= 0.0:
        raise ValueError("surface_tension and interface_width must be positive")
    kappa = 0.75 * surface_tension * interface_width
    a = 2.0 * kappa / interface_width**2
    return a, kappa


def model_surface_tension(kappa: float, interface_width: float) -> float:
    """Surface tension implied by ``kappa`` for a ``tanh(s/xi)`` profile."""
    return 4.0 * kappa / (3.0 * interface_width)


def young_cos_theta(psi_wall: float) -> float:
    """``cos(theta) = (3 psi_w - psi_w^3)/2`` for a wall order parameter ``psi_w``.

    Derived independently (docs/XVAL-LBM-DIFFUSE-INTERFACE-WETTING-VIA-WALL-ORDER-PARAMETER.md
    section 2) by truncating the equilibrium ``tanh`` profile at the wall value and
    differencing the wall excess energy of the two bulk phases:

        ``G(psi) = (kappa/xi) [2/3 -+ psi -+ psi^3/3]``,
        ``gamma_sg - gamma_sl = G_gas - G_liq = (2 kappa/xi)(psi_w - psi_w^3/3)``,
        ``sigma = 4 kappa/(3 xi)``  ->  Young's law above.

    This is the single source of truth shared by the diffuse-interface relaxation
    (:func:`analytic_contact_angle`), the wall-flux condition of the CAC solver
    (``tensorlbm.cac_lbm2d``, ``wall_phi_policy="wall_order"``) and the xval harness.
    """
    p = float(psi_wall)
    if p < -1.0 or p > 1.0:
        raise ValueError("psi_wall must lie in [-1, 1]")
    return 0.5 * (3.0 * p - p**3)


def analytic_contact_angle(psi_wall: float) -> float:
    """Contact angle (rad) of the wall-order-parameter model in the sharp limit.

    The Dirichlet wall value ``psi_w`` acts as a wall free energy whose
    difference between the two phases gives Young's law

        ``cos(theta) = (3 psi_w - psi_w^3) / 2``

    (``psi_w = 0  -> 90 deg``, ``psi_w = +1 -> 0 deg`` complete wetting,
    ``psi_w = -1 -> 180 deg``).  The relation is *tested* in the xval harness;
    see ``xval/.../runs/**/result.json`` for the measured mapping.
    """
    return float(math.acos(max(-1.0, min(1.0, young_cos_theta(psi_wall)))))


# ---------------------------------------------------------------------------
# isotropic stencils (reference getNabla_psi*)
# ---------------------------------------------------------------------------
def _pad_y(psi: torch.Tensor, psi_wall: float, psi_wall_top: float | None = None) -> torch.Tensor:
    """Rows ``[-1, 0 .. ny-1, ny]``; the two outer rows hold the wall values."""
    top = float(psi_wall) if psi_wall_top is None else float(psi_wall_top)
    lo = torch.full((1, psi.shape[1]), float(psi_wall), dtype=psi.dtype, device=psi.device)
    hi = torch.full((1, psi.shape[1]), top, dtype=psi.dtype, device=psi.device)
    return torch.cat((lo, psi, hi), dim=0)


def grad_laplacian(
    psi: torch.Tensor, psi_wall: float, psi_wall_top: float | None = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Isotropic 9-point ``d/dx``, ``d/dy`` and ``lap`` in lattice units.

    ``psi`` has shape ``(ny, nx)``; ``x`` is periodic and the two walls are
    supplied through the ghost rows carrying ``psi_wall``.
    """
    padded = _pad_y(psi, psi_wall, psi_wall_top)
    north = padded[2:]
    south = padded[:-2]
    east = torch.roll(psi, -1, dims=1)
    west = torch.roll(psi, 1, dims=1)
    ne = torch.roll(north, -1, dims=1)
    nw = torch.roll(north, 1, dims=1)
    se = torch.roll(south, -1, dims=1)
    sw = torch.roll(south, 1, dims=1)
    gx = (4.0 * (east - west) + (ne + se - nw - sw)) / 12.0
    gy = (4.0 * (north - south) + (ne + nw - se - sw)) / 12.0
    lap = (4.0 * (east + west + north + south) + (ne + nw + sw + se) - 20.0 * psi) / 6.0
    return gx, gy, lap


# ---------------------------------------------------------------------------
# configuration / state
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WallOrderWettingConfig:
    """Physical + numerical parameters of one run (all in lattice units)."""

    ny: int = 64
    nx: int = 96
    psi_wall: float = 0.0
    psi_wall_top: "float | None" = None
    interface_width: float = 2.0
    surface_tension: float = 0.01
    tau_f: float = 0.8
    tau_g: float = 0.7
    mobility: float = 0.1
    closure: str = "isotropic"
    dtype: torch.dtype = torch.float64

    @property
    def psi_wall_top_value(self) -> float:
        """Wall order parameter of the upper wall (defaults to ``psi_wall``)."""
        return self.psi_wall if self.psi_wall_top is None else float(self.psi_wall_top)

    @property
    def params(self) -> Tuple[float, float]:
        return free_energy_parameters(self.surface_tension, self.interface_width)

    @property
    def a(self) -> float:
        return self.params[0]

    @property
    def kappa(self) -> float:
        return self.params[1]

    @property
    def capillary_number_note(self) -> str:
        return (
            "lattice units: xi={xi}, sigma={s}, a={a:.6g}, kappa={k:.6g}, "
            "nu={nu:.4g}, mobility={m}".format(
                xi=self.interface_width,
                s=self.surface_tension,
                a=self.a,
                k=self.kappa,
                nu=(self.tau_f - 0.5) / 3.0,
                m=self.mobility,
            )
        )


# ---------------------------------------------------------------------------
# initialisation
# ---------------------------------------------------------------------------
def initialise_drop_on_wall(
    cfg: WallOrderWettingConfig,
    radius: float,
    centre_y: float,
    centre_x: float | None = None,
) -> Dict[str, torch.Tensor]:
    """Initialise a circular cap of ``psi = +1`` on the bottom wall.

    The distribution functions are set to the local equilibrium of the
    (zero-velocity) initial state, exactly as the reference does.
    """
    ny, nx = cfg.ny, cfg.nx
    yy = torch.arange(ny, dtype=cfg.dtype).view(ny, 1)
    xx = torch.arange(nx, dtype=cfg.dtype).view(1, nx)
    x0 = 0.5 * nx if centre_x is None else float(centre_x)
    r = torch.sqrt((xx - x0) ** 2 + (yy - centre_y) ** 2)
    # equilibrium diffuse profile psi = tanh(s/xi), s = signed distance *inside* the cap
    psi = torch.tanh((radius - r) / cfg.interface_width)
    rho = torch.ones((ny, nx), dtype=cfg.dtype)
    ux = torch.zeros((ny, nx), dtype=cfg.dtype)
    uy = torch.zeros((ny, nx), dtype=cfg.dtype)
    f = _equilibrium(rho, ux, uy)
    g = _order_parameter_equilibrium(psi, None, cfg)
    return {"f": f, "g": g}


def _equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    ex = torch.tensor(EX, dtype=rho.dtype).view(9, 1, 1)
    ey = torch.tensor(EY, dtype=rho.dtype).view(9, 1, 1)
    w = torch.tensor(W, dtype=rho.dtype).view(9, 1, 1)
    cu = ex * ux.unsqueeze(0) + ey * uy.unsqueeze(0)
    usq = (ux**2 + uy**2).unsqueeze(0)
    return w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usq)


def _order_parameter_equilibrium(
    psi: torch.Tensor,
    mu: torch.Tensor | None,
    cfg: WallOrderWettingConfig,
    ux: torch.Tensor | None = None,
    uy: torch.Tensor | None = None,
) -> torch.Tensor:
    """Equilibrium of the order-parameter distribution ``g``.

    ``cfg.closure = "reference"`` reproduces the OSS ``getgeq``: the mobility
    term carries the ``-(1 - w_0)`` correction on the rest direction and
    ``w_i Gamma mu / cs^2`` on all moving directions (its second moment is the
    same as the isotropic one, so the diffusivity is nominally identical).

    ``cfg.closure = "isotropic"`` uses the standard free-energy closure
    ``w_i (Gamma mu / cs^2) (c_i^2 / cs^2 - D)`` (Pooley & Furtado 2008), which
    additionally cancels the spurious ``psi``-diffusion carried by the
    advection part of the equilibrium.  The flat-interface steadiness test in
    the xval harness measures the difference.
    """
    w = torch.tensor(W, dtype=psi.dtype).view(9, 1, 1)
    if cfg.closure == "zero_second_moment":
        c_sq = torch.tensor([EX[i] ** 2 + EY[i] ** 2 for i in range(9)], dtype=psi.dtype).view(9, 1, 1)
        shape_i = w * (2.0 - 0.5 * c_sq / CS2)  # sums to 1, second moment zero
    else:
        shape_i = w
    if ux is None or uy is None:
        geq = shape_i * psi.unsqueeze(0)
    else:
        ex = torch.tensor(EX, dtype=psi.dtype).view(9, 1, 1)
        ey = torch.tensor(EY, dtype=psi.dtype).view(9, 1, 1)
        cu = ex * ux.unsqueeze(0) + ey * uy.unsqueeze(0)
        usq = (ux**2 + uy**2).unsqueeze(0)
        geq = shape_i * psi.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usq)
    if mu is not None:
        mu_term = cfg.mobility * mu / CS2
        if cfg.closure == "reference":
            extra = w * mu_term.unsqueeze(0)
            extra[0] = -(1.0 - W[0]) * mu_term
        elif cfg.closure in ("isotropic", "zero_second_moment"):
            c_sq = torch.tensor([EX[i] ** 2 + EY[i] ** 2 for i in range(9)], dtype=psi.dtype).view(9, 1, 1)
            extra = w * mu_term.unsqueeze(0) * (c_sq / CS2 - 2.0)
        else:  # pragma: no cover - guarded by the config dataclass
            raise ValueError("unknown closure %r" % (cfg.closure,))
        geq = geq + extra
    return geq


# ---------------------------------------------------------------------------
# one LBM step
# ---------------------------------------------------------------------------
def macroscopics(state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(rho, ux, uy)`` from the momentum distribution (no force correction)."""
    f = state["f"]
    ex = torch.tensor(EX, dtype=f.dtype).view(9, 1, 1)
    ey = torch.tensor(EY, dtype=f.dtype).view(9, 1, 1)
    rho = f.sum(dim=0)
    return rho, (f * ex).sum(dim=0), (f * ey).sum(dim=0)


def _stream_bounce(pop: torch.Tensor) -> torch.Tensor:
    """Stream in the reference order with half-way bounce-back on both walls."""
    out = torch.empty_like(pop)
    for i in range(9):
        out[i] = torch.roll(pop[i], shifts=(int(EX[i]), int(EY[i])), dims=(1, 0))
    # bottom wall (y = -1/2): populations with e_y > 0 are reflected
    out[2, 0, :] = pop[4, 0, :]
    out[5, 0, :] = pop[7, 0, :]
    out[6, 0, :] = pop[8, 0, :]
    # top wall (y = ny - 1/2)
    out[4, -1, :] = pop[2, -1, :]
    out[7, -1, :] = pop[5, -1, :]
    out[8, -1, :] = pop[6, -1, :]
    return out


def step(state: Dict[str, torch.Tensor], cfg: WallOrderWettingConfig) -> Dict[str, torch.Tensor]:
    """Advance one lattice step (collide + force + stream + wall bounce-back)."""
    f, g = state["f"], state["g"]
    dtype = f.dtype
    rho = f.sum(dim=0)
    psi = g.sum(dim=0)

    gx, gy, lap = grad_laplacian(psi, cfg.psi_wall, cfg.psi_wall_top_value)
    mu = cfg.a * psi * (psi**2 - 1.0) - cfg.kappa * lap
    fx, fy = mu * gx, mu * gy

    ex = torch.tensor(EX, dtype=dtype).view(9, 1, 1)
    ey = torch.tensor(EY, dtype=dtype).view(9, 1, 1)
    w = torch.tensor(W, dtype=dtype).view(9, 1, 1)

    # velocity with the Guo half-force correction
    rho_safe = torch.where(rho.abs() < 1e-30, torch.ones_like(rho), rho)
    ux = ((f * ex).sum(dim=0) + 0.5 * fx) / rho_safe
    uy = ((f * ey).sum(dim=0) + 0.5 * fy) / rho_safe

    cu = ex * ux.unsqueeze(0) + ey * uy.unsqueeze(0)
    usq = (ux**2 + uy**2).unsqueeze(0)
    feq = w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usq)
    force_i = (
        (1.0 - 0.5 / cfg.tau_f)
        * w
        * (3.0 * ((ex - ux.unsqueeze(0)) * fx.unsqueeze(0) + (ey - uy.unsqueeze(0)) * fy.unsqueeze(0))
           + 9.0 * cu * (ex * fx.unsqueeze(0) + ey * fy.unsqueeze(0)))
    )
    f_post = f - (f - feq) / cfg.tau_f + force_i

    geq = _order_parameter_equilibrium(psi, mu, cfg, ux, uy)
    g_post = g - (g - geq) / cfg.tau_g

    return {"f": _stream_bounce(f_post), "g": _stream_bounce(g_post)}


# ---------------------------------------------------------------------------
# static equilibration (mass-conserving Allen-Cahn descent)
# ---------------------------------------------------------------------------
def relax_to_equilibrium(
    cfg: WallOrderWettingConfig,
    radius: float,
    centre_y: float,
    max_iter: int = 60000,
    check_every: int = 200,
    tol: float = 1e-10,
    omega: float = 0.25,
    centre_x: float | None = None,
) -> Dict[str, object]:
    """Relax the free energy to its static equilibrium with the wall order parameter.

    Descent ``psi <- psi - omega (mu - <mu>)``: the mean subtraction makes the
    update mass conserving (``sum psi`` is an invariant), and the fixed point is
    ``mu = const`` -- *the same* equilibrium condition as the Cahn-Hilliard
    equation solved by the LBM (``div(M grad mu) = 0`` with no-flux walls).
    It is used for the static contact-angle law because this descent is
    unconditionally robust at the interface widths of interest, whereas the
    reference order-parameter closure of the LBM is not (see the spurious
    ``psi``-diffusion test in the xval harness).

    The wall condition is *identical* to the LBM's: the ghost rows hold
    ``psi_wall`` (lower) / ``psi_wall_top`` (upper) and enter both the isotropic
    gradient and the isotropic Laplacian of ``psi``.
    """
    psi = initialise_drop_on_wall(cfg, radius=radius, centre_y=centre_y, centre_x=centre_x)["g"].sum(dim=0)
    psi_prev_check = psi.clone()
    mass0 = float(psi.sum())
    converged = False
    it = 0
    drift = float("inf")
    while it < max_iter:
        for _ in range(check_every):
            _gx, _gy, lap = grad_laplacian(psi, cfg.psi_wall, cfg.psi_wall_top_value)
            mu = cfg.a * psi * (psi**2 - 1.0) - cfg.kappa * lap
            psi = psi - omega * (mu - mu.mean())
        it += check_every
        drift = float((psi - psi_prev_check).abs().max()) if it > check_every else float("inf")
        psi_prev_check = psi.clone()
        if drift < tol:
            converged = True
            break
    return {
        "psi": psi,
        "iterations": it,
        "converged": converged,
        "drift": drift,
        "mass": float(psi.sum()),
        "mass_error": abs(float(psi.sum()) - mass0) / max(abs(mass0), 1e-30),
    }


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def interface_points(psi: torch.Tensor, level: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sub-cell ``psi = level`` crossing points of the field, shape ``(n, 2)``.

    Both x- and y-directed crossings are returned as ``(y, x)`` pairs.
    """
    p = psi.detach().to(torch.float64).cpu()
    dev = p - level
    pts: List[Tuple[float, float]] = []
    # x-direction crossings
    s = dev[:, :-1] * dev[:, 1:]
    ij = (s < 0).nonzero()
    for i, j in ij.tolist():
        t = dev[i, j] / (dev[i, j] - dev[i, j + 1])
        pts.append((float(i), float(j + t)))
    # y-direction crossings
    s = dev[:-1, :] * dev[1:, :]
    ij = (s < 0).nonzero()
    for i, j in ij.tolist():
        t = dev[i, j] / (dev[i, j] - dev[i + 1, j])
        pts.append((float(i + t), float(j)))
    out = torch.tensor(pts, dtype=torch.float64) if pts else torch.zeros((0, 2), dtype=torch.float64)
    return out


def _circle_fit(pts: torch.Tensor) -> Tuple[float, float, float]:
    """Algebraic (Kasa) least-squares circle fit: returns ``(yc, xc, R)``."""
    x = pts[:, 1]
    y = pts[:, 0]
    A = torch.stack((x, y, torch.ones_like(x)), dim=1)
    b = x**2 + y**2
    sol, *_ = torch.linalg.lstsq(A, b.unsqueeze(1))
    c = sol.squeeze(1)
    xc = float(c[0] / 2.0)
    yc = float(c[1] / 2.0)
    R = float(math.sqrt(max(1e-30, float(c[2]) + xc**2 + yc**2)))
    return yc, xc, R


def _zero_crossing_1d(values: torch.Tensor, last: bool = False) -> float:
    """A sign change of a 1D array, returned as a fractional index.

    ``last = False`` gives the first crossing (scanning from index 0), ``True``
    the last one.  The apex of a cap is the *last* crossing of the centre
    column, which is not the same as the first once the cap dewets (the
    contour then crosses the column twice).
    """
    v = values.detach().to(torch.float64).cpu()
    s = v[:-1] * v[1:]
    idx = (s < 0).nonzero()
    if idx.numel() == 0:
        return float("nan")
    i = int(idx[-1, 0]) if last else int(idx[0, 0])
    return float(i) + float(v[i] / (v[i] - v[i + 1]))


def theta_from_cap_area(area: float, height: float) -> float:
    """Solve the Laplace-cap relation ``A/h^2 = (th - sin th cos th)/(1-cos th)^2``.

    Both the excess area ``A`` (the conserved "volume" of the cap) and the apex
    height ``h`` are convex in the cap geometry, so this inversion is well
    conditioned for shallow (wetting) as well as deep (non-wetting) caps -- it
    does not require the ``psi = 0`` contour to reach the wall plane, which it
    does not once a precursor film forms.
    """
    if not (area > 0.0 and height > 0.0):
        return float("nan")
    target = area / height**2

    def f(th: float) -> float:
        denom = (1.0 - math.cos(th)) ** 2
        return (th - math.sin(th) * math.cos(th)) / denom - target

    lo, hi = 1e-4, math.pi - 1e-4
    if f(lo) < 0.0:
        return float("nan")
    if f(hi) > 0.0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def contact_angle_from_field(
    psi: torch.Tensor,
    wall_y: float = -0.5,
    psi_wall: float = 0.0,
    centre_x: float | None = None,
    interface_width: float = 3.0,
) -> Dict[str, float]:
    """Measure the static contact angle of a cap resting on the lower wall.

    Primary estimator ``theta_cap_area``: inversion of the Laplace-cap relation
    from the *conserved* excess area ``A = sum (psi + 1)/2`` and the apex height
    ``h`` above the wall plane (see :func:`theta_from_cap_area`).  This is the
    estimator used for the wetting law: it needs neither the contour to reach
    the wall (a precursor film can form for wetting walls) nor a circle fit of
    a shallow arc.

    Secondary estimator ``theta_circle``: algebraic circle fit of the level-set
    points more than 1.5 interface widths above the wall, then
    ``cos(theta) = (y_c - y_wall)/R``.  Reported as an independent geometric
    cross-check for the non-wetting branch (where the cap is nearly spherical).

    ``R_curv`` is the fitted radius of curvature at the apex, used for the
    Young-Laplace cross-check of the numerical surface tension.

    Validity diagnostics (both estimators above assume a *sharp* cap):

    ``layer_area_per_column``  liquid area the equilibrium wall adsorption layer
                               consumes per column (from the 1-D profile far from
                               the drop);
    ``layer_fraction``         ``layer_area_per_column * nx / area``.  When this
                               approaches 1 the drop has been absorbed by the
                               layer and the cap measure is meaningless, which is
                               what happens for shallow caps on strongly wetting
                               walls (``psi_w >= 0.7`` here).  ``cap_measurable``
                               is the resulting gate flag.
    """
    ny, nx = psi.shape
    xc = (nx - 1) / 2.0 if centre_x is None else float(centre_x)
    j = max(1, min(nx - 2, int(round(xc))))

    apex = _zero_crossing_1d(psi[:, j], last=True)
    height = apex - wall_y
    area = float(((psi.detach() + 1.0) * 0.5).sum())
    theta_cap = theta_from_cap_area(area, height)

    # Adsorption-layer (precursor film) capacity of the wall: the equilibrium
    # 1-D profile far from the drop, integrated over y, is the liquid area the
    # wall layer consumes *per column*.  For a shallow cap the footprint
    # (>= 2 R sin(theta)) is entirely covered by that layer, so the layer area
    # can reach or exceed the drop area and the area/height inversion becomes
    # ill-posed.  The ratio is reported so such a run is *flagged* rather than
    # silently trusted (mechanism documented in the xval report, section 5).
    edge = max(2, min(6, nx // 10))
    film_cols = torch.cat((psi[:, :edge], psi[:, -edge:]), dim=1)
    film_profile = film_cols.mean(dim=1)
    layer_area_per_column = float(((film_profile.detach() + 1.0) * 0.5).sum())
    cap_excess = float(area - layer_area_per_column * nx)
    layer_fraction = float(layer_area_per_column * nx / area) if area > 0 else float("nan")

    pts = interface_points(psi)
    theta_circle = float("nan")
    radius = float("nan")
    centre_y = float("nan")
    if pts.shape[0] >= 20:
        mask = pts[:, 0] > 1.5 * interface_width
        pts_used = pts[mask] if int(mask.sum()) >= 20 else pts
        try:
            yc, xcc, R = _circle_fit(pts_used)
            radius, centre_y = R, yc
            theta_circle = float(math.acos(max(-1.0, min(1.0, (yc - wall_y) / R))))
        except Exception:  # pragma: no cover - degenerate fits
            pass

    return {
        "theta_cap": theta_cap,
        "theta_cap_deg": math.degrees(theta_cap),
        "theta_circle": theta_circle,
        "theta_circle_deg": math.degrees(theta_circle),
        "apex_height": float(height),
        "excess_area": area,
        "R_curv": float(radius),
        "centre_y": float(centre_y),
        "n_points": float(pts.shape[0]),
        "layer_area_per_column": layer_area_per_column,
        "cap_excess_area": cap_excess,
        "layer_fraction": layer_fraction,
        "cap_measurable": bool(layer_fraction < 0.5 and cap_excess > 0.0),
    }


def _solve_cap_area(area: float, radius: float) -> float:
    """Solve ``R^2 (theta - sin theta cos theta) = area`` for ``theta`` in (0, pi)."""

    def f(th: float) -> float:
        return radius**2 * (th - math.sin(th) * math.cos(th)) - area

    lo, hi = 1e-6, math.pi - 1e-6
    if f(lo) > 0.0 or f(hi) < 0.0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_to_equilibrium(
    cfg: WallOrderWettingConfig,
    radius: float,
    centre_y: float,
    max_steps: int = 60000,
    check_every: int = 500,
    tol: float = 1e-9,
    min_steps: int = 2000,
) -> Dict[str, object]:
    """Integrate until the order-parameter field stops changing.

    Stopping rule (predeclared in the harness): the run stops when
    ``max|psi(t) - psi(t - check_every)| < tol`` for two consecutive checks,
    or when ``max_steps`` is reached (``converged = False``).
    """
    state = initialise_drop_on_wall(cfg, radius=radius, centre_y=centre_y)
    psi_prev = state["g"].sum(dim=0).clone()
    hits = 0
    converged = False
    steps = 0
    while steps < max_steps:
        for _ in range(check_every):
            state = step(state, cfg)
        steps += check_every
        psi = state["g"].sum(dim=0)
        drift = float((psi - psi_prev).abs().max())
        psi_prev = psi.clone()
        if steps >= min_steps and drift < tol:
            hits += 1
            if hits >= 2:
                converged = True
                break
        else:
            hits = 0
    rho, ux, uy = macroscopics(state)
    psi = state["g"].sum(dim=0)
    span = float(psi.max() - psi.min())
    return {
        "psi": psi,
        "steps": steps,
        "converged": converged,
        "max_u": float(torch.sqrt(ux**2 + uy**2).max()),
        "psi_span": span,
        "rho_span": float(rho.max() - rho.min()),
        "psi_mean": float(psi.mean()),
    }