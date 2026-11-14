"""Two independent implementations of one static contact-angle wall condition.

Physics
-------
The static contact angle of a drop on a wall is fixed by Young's law
``sigma_lg cos(theta) = sigma_sg - sigma_sl``.  In a diffuse-interface
(Cahn-Hilliard / Allen-Cahn) model the same law can be imposed at the wall in
two ways that share *no* code and *no* intermediate quantity:

``dirichlet``  a wall *value* of the order parameter.  The wall is a Dirichlet
    boundary ``psi = psi_w``; the equilibrium profile truncated at ``psi_w``
    carries a wall excess energy whose two-phase difference is Young's law,

        ``cos(theta) = (3 psi_w - psi_w^3)/2``

    (inverted by :func:`psi_wall_from_angle`).  This is the "phase-field /
    Neumann condition derived from interface thickness and angle" family used by
    the walberla ``PhaseFieldAllenCahn`` codegen contact-angle boundary
    (``oss_solutions`` row 281) and by ``sailfish`` ``bc_wall_grad_phase``
    (row 91).

``neumann_flux``  a wall *flux*.  Same wall free energy
    ``gamma_w(psi) = -(sigma/2) cos(theta) P(psi)``, ``P(psi)=(3 psi - psi^3)/2``,
    but the natural boundary condition ``kappa d_n psi = -gamma_w'(psi)`` is
    imposed instead of the value:

        ``kappa d_n psi = (3/4) sigma cos(theta) (1 - psi^2)``,

    with ``d_n`` the wall normal pointing *out of the fluid*.  Nothing about
    ``psi_w`` enters; the wall value is an outcome.

``geometric_ghost``  a purely *geometric* ghost value.  No wall free energy at
    all: the ghost row is constructed from the level-set geometry of the
    adjacent fluid row so that the interface normal makes the prescribed angle
    with the wall,

        ``psi_ghost = psi_0 + cot(theta) |d_t psi|``

    (``d_t`` tangential, ``|.|`` keeps a symmetric cap symmetric).  This is the
    free-surface / geometric family of walberla's ``ContactAngle.h`` and of
    ``openLBMPM`` (``oss_solutions`` row 5), applied here to the order parameter
    instead of a colour gradient.

All three are realisations of the same continuum condition: for a local
``tanh(s/xi)`` interface at angle ``theta`` one has
``d_n psi = -cos(theta)(1-psi^2)/xi`` and ``|d_t psi| = sin(theta)(1-psi^2)/xi``,
so the flux form (using ``(3/4)sigma/kappa = 1/xi``) and the geometric form are
algebraically identical; the *discretisations* are not.  Cross-validating them
is the point of the anchor.

Measurement is shared by all arms (``contact_angle_from_field`` in
:mod:`tensorlbm.wetting.wall_order_parameter_2d`): a conserved-area Laplace-cap
inversion and a level-set circle fit.  The implementation-internal read-back
:func:`wall_angle_readback` reads the angle the arm actually applied and is
reported as a diagnostic only, never as the acceptance quantity.

References
----------
* Young's law / wall free energy: de Gennes, *Rev. Mod. Phys.* 57 (1985) 827;
  Jacqmin, *J. Comput. Phys.* 155 (1999) 96 (wetting BC for diffuse interfaces).
* Geometric wetting ghost: Akai et al., *Phys. Rev. E* 98, 043302 (2018) for the
  colour-gradient form; walberla free-surface ``ContactAngle.h`` for the
  free-surface form (OSS inventory row 281).
* Order-parameter wetting in LBM: Briant, Wagner & Yeomans, *Phys. Rev. E* 69
  (2004) 031602; Pooley & Furtado, *Phys. Rev. E* 77 (2008) 046702.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from .wall_order_parameter_2d import (
    WallOrderWettingConfig,
    contact_angle_from_field,
    free_energy_parameters,
    initialise_drop_on_wall,
    interface_points,
    theta_from_cap_area,
    young_cos_theta,
)

__all__ = [
    "ARMS",
    "TwoImplConfig",
    "cap_target_area",
    "grad_laplacian_ghost",
    "initialise_matched_cap",
    "measure_contact_angle",
    "psi_wall_from_angle",
    "relax_with_wall_condition",
    "run_arm",
    "wall_angle_readback",
    "wall_ghost_rows",
]

#: The two independent implementations (plus the shared reference arm).
ARMS: Tuple[str, ...] = ("dirichlet", "neumann_flux", "geometric_ghost")


# ---------------------------------------------------------------------------
# angle <-> wall value (Young's law)
# ---------------------------------------------------------------------------
def psi_wall_from_angle(theta: float) -> float:
    """Invert ``cos(theta) = (3 p - p^3)/2`` for the wall value ``p`` in [-1, 1].

    Monotone in ``p`` (the physical branch), solved by bisection so that it can
    never drift from :func:`young_cos_theta`, which stays the single source of
    truth for the law.
    """
    target = math.cos(float(theta))
    lo, hi = -1.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if young_cos_theta(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# wall condition -> ghost rows
# ---------------------------------------------------------------------------
def _tangential(f: torch.Tensor) -> torch.Tensor:
    """Central-difference ``d/dx`` of a row of the field (x periodic)."""
    return 0.5 * (torch.roll(f, -1, dims=-1) - torch.roll(f, 1, dims=-1))


def wall_ghost_rows(
    arm: str,
    psi: torch.Tensor,
    theta: float,
    kappa: float,
    surface_tension: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ghost rows (``y = -1`` and ``y = ny``) implied by ``arm`` for ``theta``.

    ``psi`` is the ``(ny, nx)`` fluid field; row 0 touches the lower wall at
    ``y = -1/2`` and row ``ny-1`` the upper wall at ``y = ny - 1/2``.  The upper
    wall is *neutral* in every arm (mirror: a zero normal gradient), so the two
    walls are identical across arms and only the lower wall carries the angle
    under test.
    """
    ny, _nx = psi.shape
    lower, upper = psi[0], psi[ny - 1]
    upper_ghost = upper.clone()  # neutral (mirror) in every arm

    if arm == "dirichlet":
        p = psi_wall_from_angle(theta)
        return torch.full_like(lower, p), upper_ghost

    if arm == "neumann_flux":
        # kappa d_n psi = (3/4) sigma cos(theta) (1 - psi^2), d_n out of the fluid
        amp = 0.75 * surface_tension * math.cos(theta) / kappa
        ghost = lower.clone()
        for _ in range(3):  # 3 fixed-point sweeps for psi at the wall plane
            psi_wall_plane = 0.5 * (lower + ghost)
            ghost = lower + amp * (1.0 - psi_wall_plane**2)
        return ghost, upper_ghost

    if arm == "geometric_ghost":
        if abs(math.sin(theta)) < 1e-12:
            raise ValueError("geometric ghost is singular at theta = 0 / 180 deg")
        cot = math.cos(theta) / math.sin(theta)
        return lower + cot * _tangential(lower).abs(), upper_ghost

    raise ValueError("unknown arm %r (expected one of %s)" % (arm, ARMS))


def grad_laplacian_ghost(
    psi: torch.Tensor, ghost_lo: torch.Tensor, ghost_hi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Isotropic 9-point ``d/dx``, ``d/dy``, ``lap`` with *per-column* ghosts.

    Identical stencils to :func:`tensorlbm.wetting.wall_order_parameter_2d.grad_laplacian`;
    with constant ghost rows the two agree bit-for-bit (tested).
    """
    padded = torch.cat((ghost_lo.unsqueeze(0), psi, ghost_hi.unsqueeze(0)), dim=0)
    north, south = padded[2:], padded[:-2]
    east, west = torch.roll(psi, -1, dims=1), torch.roll(psi, 1, dims=1)
    ne, nw = torch.roll(north, -1, dims=1), torch.roll(north, 1, dims=1)
    se, sw = torch.roll(south, -1, dims=1), torch.roll(south, 1, dims=1)
    gx = (4.0 * (east - west) + (ne + se - nw - sw)) / 12.0
    gy = (4.0 * (north - south) + (ne + nw - se - sw)) / 12.0
    lap = (4.0 * (east + west + north + south) + (ne + nw + sw + se) - 20.0 * psi) / 6.0
    return gx, gy, lap


def wall_angle_readback(
    arm: str, psi: torch.Tensor, theta: float, kappa: float, surface_tension: float
) -> Dict[str, float]:
    """Angle the arm *actually applied*, read back from the wall gradient.

    ``tan(theta) = |d_t psi| / (-d_n psi)`` at the wall-adjacent row
    (``d_n = psi[0] - psi_ghost`` is positive into the fluid, while the continuum
    condition is ``d_n psi = -cos(theta)(1-psi^2)/xi``, i.e.
    ``cos(theta) = -d_n xi/(1-psi^2)`` and ``sin(theta) = |d_t psi| xi/(1-psi^2)``;
    the earlier form used ``+d_n`` and therefore returned ``180 deg - theta``
    above 90 deg -- measured 70.0 deg for a prescribed 110 deg, fixed here).
    For the two *gradient* arms (``neumann_flux``,
    ``geometric_ghost``) this is exactly the continuum condition the ghost
    encodes, and it returns the prescribed angle.  For the ``dirichlet`` arm it
    does **not**: the ghost carries a wall *value* ``psi_w``, so
    ``psi[0] - psi_w`` is the profile difference across one half spacing, not
    ``d_n psi`` of the imposed condition -- measured 53.4 deg at a prescribed
    90 deg and 36.3 deg at 110 deg (benchmark
    ``benchmarks/case_pool/two_impl_contact_angle``), i.e. the read-back of the
    Dirichlet arm is NOT an angle and must not be used as evidence.
    Averaged over the columns where the interface meets the wall
    (``|d_t psi| > 10 % of its maximum``).  Diagnostic only: it is built from the
    same ghost the arm imposed, so it validates the *implementation*, not the
    physics (the shape estimators do that).
    """
    ghost_lo, _ = wall_ghost_rows(arm, psi, theta, kappa, surface_tension)
    dt = _tangential(psi[0]).abs()
    dn = psi[0] - ghost_lo  # positive into the fluid
    mask = dt > 0.1 * float(dt.max())
    if int(mask.sum()) < 2:
        return {"theta_deg": float("nan"), "n_columns": float(mask.sum())}
    ang = torch.atan2(dt[mask], -dn[mask])
    return {
        "theta_deg": float(torch.rad2deg(torch.median(ang))),
        "theta_mean_deg": float(torch.rad2deg(ang.mean())),
        "n_columns": float(mask.sum()),
    }


# ---------------------------------------------------------------------------
# volume-matched initialisation and gas-offset-free measurement
# ---------------------------------------------------------------------------
def cap_target_area(theta: float, radius: float) -> float:
    """Area of the sharp circular cap ``R^2 (theta - sin theta cos theta)``."""
    return float(radius) ** 2 * (float(theta) - math.sin(theta) * math.cos(theta))


def initialise_matched_cap(
    theta: float,
    radius: float,
    cfg: "TwoImplConfig",
    wall_y: float = -0.5,
    sweeps: int = 6,
) -> torch.Tensor:
    """Smeared cap whose conserved liquid *content* equals the sharp cap volume.

    A ``tanh((R - r)/xi)`` field carries more liquid than ``pi R^2/2`` because the
    interface of a curved boundary is not content-neutral.  Relaxing such a field
    with a mass-conserving descent then *supersaturates the gas* (measured: the
    far field settles near ``psi = -0.96`` instead of ``-1``), which biases every
    area-based angle estimator.  A few Picard sweeps on the radius remove the
    offset at the source (residual < 1e-3 of the cap area).
    """
    wcfg = WallOrderWettingConfig(
        ny=cfg.ny,
        nx=cfg.nx,
        interface_width=cfg.interface_width,
        surface_tension=cfg.surface_tension,
        dtype=cfg.dtype,
    )
    target = cap_target_area(theta, radius)
    r = float(radius)
    psi = None
    for _ in range(sweeps):
        psi = initialise_drop_on_wall(
            wcfg, radius=r, centre_y=_cap_centre_y(theta, r, wall_y)
        )["g"].sum(dim=0)
        content = float(((psi - psi.min()) * 0.5).sum())
        if content <= 0.0:
            raise ValueError("degenerate initial cap")
        r *= math.sqrt(target / content)
    return psi


def _gas_reference(psi: torch.Tensor) -> float:
    """Far-field gas value, sampled in the two top corners (never in the drop)."""
    top_lo = float(psi[-4:, :4].mean())
    top_hi = float(psi[-4:, -4:].mean())
    return 0.5 * (top_lo + top_hi)


def _level_crossing(values: torch.Tensor, level: float) -> float:
    """Last sub-cell crossing of ``level`` along a 1-D profile."""
    dev = values - level
    s = dev[:-1] * dev[1:]
    idx = (s < 0).nonzero()
    if idx.numel() == 0:
        return float("nan")
    i = int(idx[-1, 0])
    return float(i) + float(dev[i] / (dev[i] - dev[i + 1]))


def measure_contact_angle(
    psi: torch.Tensor,
    interface_width: float,
    wall_y: float = -0.5,
    centre_x: float | None = None,
) -> Dict[str, float]:
    """Two shape estimators, both corrected for the measured gas offset.

    ``theta_cap``  conserved-content Laplace-cap inversion:
        ``A = sum (psi - psi_gas)/2`` (the offset removal is essential -- with an
        uncorrected ``sum (psi+1)/2`` a 3 % gas supersaturation moves the estimate
        by ~18 deg at 90 deg) and the apex height ``h`` of the mid-level contour.
    ``theta_circle``  algebraic circle fit of the mid-level contour points more
        than ``1.5 xi`` above the wall, with
        ``theta = acos((y_wall - y_c)/R)`` (the sign that puts a cap centred
        *below* the wall at ``theta < 90``; the opposite sign, still present in
        ``contact_angle_from_field``, returns the supplementary angle).

    Both estimators are implementation-blind: they use only the final field.
    """
    ny, nx = psi.shape
    p = psi.detach().to(torch.float64)
    gas = _gas_reference(p)
    liquid = float(p.max())
    level = 0.5 * (gas + liquid)
    content = float(((p - gas) * 0.5).sum())

    j = max(1, min(nx - 2, int(round((nx - 1) / 2.0 if centre_x is None else centre_x))))
    apex = _level_crossing(p[:, j], level)
    height = apex - wall_y
    theta_cap = math.degrees(theta_from_cap_area(content, height)) if (height > 0) else float("nan")

    theta_circle = float("nan")
    rms = float("nan")
    radius = float("nan")
    y_c = float("nan")
    pts = interface_points(p, level=level)
    n_used = 0
    if pts.shape[0] >= 20:
        mask = pts[:, 0] > 1.5 * interface_width
        use = pts[mask] if int(mask.sum()) >= 20 else pts
        n_used = int(use.shape[0])
        x = use[:, 1]
        y = use[:, 0]
        A = torch.stack((x, y, torch.ones_like(x)), dim=1)
        b = -(x**2 + y**2)
        sol = torch.linalg.lstsq(A, b).solution
        x_c = -0.5 * sol[0]
        y_c = -0.5 * sol[1]
        radius = math.sqrt(max(x_c**2 + y_c**2 - sol[2], 1e-30))
        rms = float(torch.sqrt(((torch.sqrt((x - x_c) ** 2 + (y - y_c) ** 2) - radius) ** 2).mean()))
        theta_circle = math.degrees(math.acos(max(-1.0, min(1.0, (wall_y - y_c) / radius))))

    return {
        "theta_cap_deg": theta_cap,
        "theta_circle_deg": theta_circle,
        "estimator_gap_deg": abs(theta_cap - theta_circle) if (
            math.isfinite(theta_cap) and math.isfinite(theta_circle)) else float("nan"),
        "apex_height": float(height),
        "radius": float(radius),
        "centre_y": float(y_c),
        "circle_rms": rms,
        "n_contour_points": float(n_used),
        "content": content,
        "gas_reference": gas,
        "gas_supersaturation": gas + 1.0,
        "layer_fraction": float("nan"),
        "measurable": bool(math.isfinite(theta_cap) and height > 0 and rms < 0.15 * interface_width),
    }


# ---------------------------------------------------------------------------
# configuration / relaxation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TwoImplConfig:
    """Grid + free-energy parameters of one relaxation (lattice units)."""

    ny: int = 56
    nx: int = 80
    interface_width: float = 3.0
    surface_tension: float = 0.05
    dtype: torch.dtype = torch.float64

    @property
    def params(self) -> Tuple[float, float]:
        return free_energy_parameters(self.surface_tension, self.interface_width)

    @property
    def a(self) -> float:
        return self.params[0]

    @property
    def kappa(self) -> float:
        return self.params[1]


def _cap_centre_y(theta: float, radius: float, wall_y: float = -0.5) -> float:
    """Circle centre that puts an initially circular cap at contact angle ``theta``."""
    return wall_y - radius * math.cos(theta)


def relax_with_wall_condition(
    arm: str,
    theta: float,
    cfg: TwoImplConfig | None = None,
    radius: float = 16.0,
    max_iter: int = 30000,
    check_every: int = 200,
    tol: float = 1e-9,
    omega: float = 0.4,
    psi0: torch.Tensor | None = None,
) -> Dict[str, object]:
    """Static equilibrium by mass-conserving Allen-Cahn descent under ``arm``.

    Same descent as
    :func:`tensorlbm.wetting.wall_order_parameter_2d.relax_to_equilibrium`
    (``psi <- psi - omega (mu - <mu>)``, ``mu = a psi(psi^2-1) - kappa lap psi``),
    but the wall enters through :func:`wall_ghost_rows` instead of a constant
    ghost, so the three arms differ *only* in the wall condition.

    The initial field is a circular cap whose centre is placed so that the
    initial contact angle equals the prescribed one
    (``y_c = wall - R cos(theta)``); this removes the long initial transient of a
    wrong-shaped drop (an honest source of the drift seen in earlier rounds).
    """
    cfg = cfg or TwoImplConfig()
    a, kappa = cfg.a, cfg.kappa
    if psi0 is None:
        psi = initialise_matched_cap(theta, radius, cfg)
    else:
        psi = psi0.clone()

    mass0 = float(psi.sum())
    prev = psi.clone()
    drift = float("inf")
    converged = False
    it = 0
    while it < max_iter:
        for _ in range(check_every):
            ghost_lo, ghost_hi = wall_ghost_rows(arm, psi, theta, kappa, cfg.surface_tension)
            _gx, _gy, lap = grad_laplacian_ghost(psi, ghost_lo, ghost_hi)
            mu = a * psi * (psi**2 - 1.0) - kappa * lap
            psi = psi - omega * (mu - mu.mean())
        it += check_every
        drift = float((psi - prev).abs().max())
        prev = psi.clone()
        if drift < tol:
            converged = True
            break

    ghost_lo, _ = wall_ghost_rows(arm, psi, theta, kappa, cfg.surface_tension)
    return {
        "psi": psi,
        "iterations": it,
        "converged": converged,
        "drift": drift,
        "mass_error": abs(float(psi.sum()) - mass0) / max(abs(mass0), 1e-30),
        "wall_value_mean": float(ghost_lo.mean()),
    }


def run_arm(
    arm: str,
    theta: float,
    cfg: TwoImplConfig | None = None,
    radius: float = 16.0,
    max_iter: int = 30000,
    omega: float = 0.4,
    tol: float = 1e-9,
) -> Dict[str, object]:
    """Relax one arm at one prescribed angle and measure the resulting angle."""
    cfg = cfg or TwoImplConfig()
    out = relax_with_wall_condition(
        arm, theta, cfg=cfg, radius=radius, max_iter=max_iter, omega=omega, tol=tol
    )
    psi = out["psi"]
    meas = measure_contact_angle(psi, cfg.interface_width, wall_y=-0.5)
    legacy = contact_angle_from_field(
        psi, wall_y=-0.5, psi_wall=0.0, interface_width=cfg.interface_width
    )
    rb = wall_angle_readback(arm, psi, theta, cfg.kappa, cfg.surface_tension)
    return {
        "arm": arm,
        "theta_prescribed_deg": math.degrees(theta),
        "theta_cap_deg": meas["theta_cap_deg"],
        "theta_circle_deg": meas["theta_circle_deg"],
        "estimator_gap_deg": meas["estimator_gap_deg"],
        "cap_measurable": meas["measurable"],
        "layer_fraction": legacy["layer_fraction"],
        "apex_height": meas["apex_height"],
        "R_curv": meas["radius"],
        "gas_supersaturation": meas["gas_supersaturation"],
        "circle_rms": meas["circle_rms"],
        "wall_angle_readback_deg": rb["theta_deg"],
        "wall_readback_columns": rb["n_columns"],
        "wall_value_mean": out["wall_value_mean"],
        "iterations": out["iterations"],
        "converged": out["converged"],
        "drift": out["drift"],
        "mass_error": out["mass_error"],
        "psi_min": float(psi.min()),
        "psi_max": float(psi.max()),
        "psi": psi,
        "cfg": cfg,
    }