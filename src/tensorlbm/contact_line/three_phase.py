"""Three-phase (ice / water / air) contact line on a Shan-Chen lattice.

Why this module exists
----------------------
:mod:`tensorlbm.contact_line.engine` provides the wall-bounded Shan-Chen stepper
and :mod:`tensorlbm.contact_line.scaling` the Cox-Voinov post-processing, but the
*phase identities* of a cryogenic contact line - solid ice, liquid water, air -
were never tied together, and the static angle was only ever obtained as an
empirical calibration ``theta(f_w)`` of the wall parameter, never predicted.
This module closes that gap with three measurement primitives that make Young's
equation a *prediction* instead of a fitting curve:

1. :func:`interaction_pressure_tensor` - the discrete Shan-Chen pressure tensor
   on the D2Q9 stencil,

       P_ab(x) = p_0(rho) delta_ab
                 + (G/2) psi(x) sum_i w_i [psi(x + c_i) - psi(x)] c_ia c_ib

   evaluated exactly on the lattice (no continuum gradient approximation), with
   the solid handled as a *frozen* pseudopotential ``psi_w`` - the same
   substitution the force kernel performs (:func:`tensorlbm.multiphase._apply_wall_psi`).
2. :func:`surface_tension` - the mechanical definition
   ``gamma = int (P_T - P_N) d(normal)`` applied to a one-dimensional
   equilibrium profile.  The same integral is used for the water/air interface
   (``gamma_lg``), the ice/water interface (``gamma_sl``) and the ice/air
   interface (``gamma_sg``); because the integral is linear in the pressure
   tensor, the *discretisation prefactor cancels in the Young ratio*
   ``cos theta_Y = (gamma_sg - gamma_sl) / gamma_lg``, so the contact angle is
   a prediction of the model, not of the post-processing.
3. :class:`~tensorlbm.contact_line.scaling.ScalingPoint`-compatible diagnostics
   (:func:`contact_line_position`, :func:`contact_line_velocity`,
   :func:`mass_total`) so a driven droplet reduces to the Cox-Voinov fit of
   :func:`tensorlbm.contact_line.scaling.cox_voinov_fit`.

Physical mapping (ice / water / air)
------------------------------------
The lattice is a two-phase (dense = water, light = air) single-component
Shan-Chen fluid; the ice is the **solid substrate** whose wettability is set by
the wall pseudopotential ``psi_w``.  Interface tensions are produced in lattice
units and mapped to SI by :func:`to_physical_units`, which only needs the
*ratios* from the lattice (SI values of gamma_water-air and the ice/water contact
angle anchor the scale), so no lattice-unit number is ever passed off as a
physical one.

Sign and normal conventions
---------------------------
For a profile varying along ``y`` the interface normal is ``e_y``; the tangential
stress is ``P_xx`` and the normal one ``P_yy``.  The lattice sum above yields
``P_yy - P_xx > 0`` inside a liquid film, hence ``gamma = int (P_xx - P_yy) dy``
is positive for a stable interface (verified numerically against the wall-free
liquid/air film, see ``tests/test_contact_line_three_phase.py``).  Since only
ratios of ``gamma`` enter the Young test, an overall sign flip of the convention
would be detected immediately by a negative ``gamma_lg``.

Limitations (documented, not hidden)
------------------------------------
* The pressure-tensor *prefactor* is fixed by the lattice stencil, not
  calibrated here; absolute ``gamma`` therefore carries the usual few-percent
  D2Q9 anisotropy error, which is why the Laplace-law cross-check is reported
  side by side with the planar value in the campaign output.
* The wall is a sharp (one-cell) boundary; the wall/fluid tension is the
  integral of a discrete stress over the first few fluid rows, so its
  ``psi_w``-dependence is only resolved to the accuracy of a ~one-cell
  resolution study (reported as ``resolution_spread``).
* Single-component Shan-Chen: "ice" is a rigid wall, not a melting solid, so
  phase change and melt-water films are outside the scope of this node.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch

from ..d2q9 import C, W, OPPOSITE, equilibrium, macroscopic
from ..multiphase import collide_sc_single_component, psi_exp
from ..solver import stream
from .engine import (
    Channel,
    RHO_L_DEFAULT,
    RHO_V_DEFAULT,
    SIGMA_DEFAULT,
    WallStepper,
    meniscus_state,
)
from .theta_fit import angle_estimate, circle_fit, interface_points, wall_contact_angle

__all__ = [
    "WallTension",
    "free_droplet",
    "sc_pressure",
    "sessile_contact_angle",
    "wall_force",
    "wedge_contact_angle",
    "WettingTension",
    "wall_coupling_moment",
    "wall_coupling_weights",
    "wetting_tension_ti",
    "contact_line_position",
    "contact_line_velocity",
    "flat_interface_tension",
    "interaction_pressure_tensor",
    "mass_total",
"planar_liquid_gas_tension",
    "planar_wall_tension",
    "to_physical_units",
    "wall_tension_pair",
    "young_angle_deg",
]


def _shift(field: torch.Tensor, cx: int, cy: int) -> torch.Tensor:
    """``field(x + c)`` for the D2Q9 direction ``c = (cx, cy)`` (periodic)."""
    return torch.roll(field, shifts=(-int(cy), -int(cx)), dims=(-2, -1))


def interaction_pressure_tensor(
    rho: torch.Tensor,
    *,
    G: float = 5.0,
    psi_fn: Callable[[torch.Tensor], torch.Tensor] = psi_exp,
    solid_mask: torch.Tensor | None = None,
    wall_psi: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discrete Shan-Chen interaction pressure tensor ``(P_xx, P_yy)``.

    Computes exactly

    .. math::

        P^{int}_{ab} = \\frac{G}{2}\\psi(x)\\sum_i w_i
                       [\\psi(x+c_i)-\\psi(x)]\\,c_{ia}c_{ib}

    i.e. the anisotropic (interface-tension carrying) part of the pressure
    tensor implied by the standard Shan-Chen force.  Solid cells contribute the
    frozen pseudopotential ``wall_psi`` (identical substitution to the force
    kernel), so the same routine serves a fluid/fluid interface and a
    wall/fluid interface.

    Args:
        rho:        Density field ``(ny, nx)``.
        G:          Shan-Chen coupling (library convention, ``> 0`` attractive).
        psi_fn:     Pseudopotential.
        solid_mask: Boolean mask of solid cells (``True`` = solid).
        wall_psi:   Pseudopotential inside solid cells (``None`` = dry wall, 0).

    Returns:
        ``(P_xx, P_yy)``.  The isotropic part ``p_0(rho)`` is *not* included:
        it drops out of ``gamma = int (P_xx - P_yy) dy``.
    """
    if rho.ndim != 2:
        raise ValueError(f"rho must be 2-D, got shape {tuple(rho.shape)}")
    psi = psi_fn(rho)
    acc_xx = torch.zeros_like(rho)
    acc_yy = torch.zeros_like(rho)
    for i in range(9):
        cx, cy = int(C[i, 0]), int(C[i, 1])
        if cx == 0 and cy == 0:
            continue
        p_shift = _shift(psi, cx, cy)
        if solid_mask is not None:
            fill = 0.0 if wall_psi is None else wall_psi
            if isinstance(fill, torch.Tensor):
                fill = fill.to(dtype=psi.dtype, device=psi.device)
            p_shift = torch.where(solid_mask, torch.full_like(p_shift, 0.0) + fill, p_shift)
        delta = p_shift - psi
        acc_xx = acc_xx + float(W[i]) * delta * float(cx * cx)
        acc_yy = acc_yy + float(W[i]) * delta * float(cy * cy)
    scale = 0.5 * float(G) * psi
    return scale * acc_xx, scale * acc_yy


def flat_interface_tension(
    rho: torch.Tensor,
    *,
    G: float = 5.0,
    psi_fn: Callable[[torch.Tensor], torch.Tensor] = psi_exp,
    solid_mask: torch.Tensor | None = None,
    wall_psi: float | torch.Tensor | None = None,
    n_walls: int = 0,
) -> float:
    """Interface tension ``gamma = int (P_T - P_N) dy`` of a y-varying profile.

    The profile is assumed homogeneous along ``x`` (averaged over columns), so
    the interface normal is ``e_y``.

    Args:
        rho:        Density field ``(ny, nx)`` (equilibrium profile).
        G:          Shan-Chen coupling.
        psi_fn:     Pseudopotential.
        solid_mask: Solid mask; tension is integrated over *fluid* rows only.
        wall_psi:   Pseudopotential inside the solid.
        n_walls:    Number of equivalent wall interfaces in the domain; the
                    integral is divided by it (``2`` for a film between two
                    identical walls, ``0`` = no division).

    Returns:
        Signed tension in lattice units (positive for a stable interface when
        the normal is ``e_y``).
    """
    p_xx, p_yy = interaction_pressure_tensor(
        rho, G=G, psi_fn=psi_fn, solid_mask=solid_mask, wall_psi=wall_psi
    )
    aniso = (p_xx - p_yy)
    if solid_mask is not None:
        aniso = torch.where(solid_mask, torch.zeros_like(aniso), aniso)
    total = float(aniso.mean(dim=1).sum())
    if n_walls > 0:
        total /= float(n_walls)
    return total


@dataclass
class WallTension:
    """One planar wall/fluid interfacial-tension measurement."""

    bulk_density: float
    wall_fraction: float
    gamma: float
    n_steps: int
    #: relative mass drift of the *whole* domain (the physical conservation
    #: statement); the fluid-only drift is dominated by the mass the full-way
    #: bounce-back stores inside the wall rows and is not a conservation error
    mass_drift: float
    rho_wall: float
    resolution_spread: float = float("nan")
    gamma_by_resolution: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "bulk_density": self.bulk_density,
            "wall_fraction": self.wall_fraction,
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "mass_drift": self.mass_drift,
            "rho_wall": self.rho_wall,
            "resolution_spread": self.resolution_spread,
            "gamma_by_resolution": self.gamma_by_resolution,
        }


def _relax_wall_profile(
    *,
    bulk_density: float,
    wall_fraction: float,
    ny: int,
    nx: int,
    wall_rows: int,
    n_steps: int,
    G: float,
    tau: float,
    rho_l: float,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Relax a uniform fluid against two identical walls; return ``(f, rho, solid, psi_w)``."""
    if nx < 8:
        raise ValueError("nx must be >= 8 (Channel requirement)")
    channel = Channel(nx=nx, ny=ny, wall_rows=wall_rows)
    solid = channel.solid_mask()
    rho0 = torch.full(channel.shape, float(bulk_density), dtype=dtype)
    rho0 = torch.where(solid, torch.full_like(rho0, float(bulk_density)), rho0)
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero).to(dtype)
    stepper = WallStepper(
        channel,
        G=G,
        tau=tau,
        rho_l=rho_l,
        wall_fraction=wall_fraction,
        forcing="edm",
        dtype=dtype,
    )
    psi_l = float(psi_exp(torch.tensor(float(rho_l), dtype=dtype)))
    psi_w = float(wall_fraction) * psi_l
    m0 = float(rho0.sum())
    f = stepper.run(f, int(n_steps), every=0)
    rho = macroscopic(f)[0]
    m1 = float(rho.sum())
    drift = abs(m1 - m0) / abs(m0) if m0 != 0.0 else float("nan")
    return f, rho, solid, psi_w, drift


def planar_wall_tension(
    *,
    bulk_density: float,
    wall_fraction: float,
    ny: int = 64,
    nx: int = 8,
    wall_rows: int = 2,
    n_steps: int = 6000,
    G: float = 5.0,
    tau: float = 1.0,
    rho_l: float = RHO_L_DEFAULT,
    dtype: torch.dtype = torch.float64,
    n_resolutions: int = 3,
) -> WallTension:
    """Measure ``gamma_sf`` between a wall and a fluid at ``bulk_density``.

    The wall pseudopotential is ``psi_w = wall_fraction * psi(rho_l)`` — the
    convention of :class:`~tensorlbm.contact_line.engine.WallStepper` — so a
    ``wall_fraction`` means the same thing for the water-side and the air-side
    interface, which is what makes the pair ``(gamma_sl, gamma_sg)`` a
    *one-parameter* family.

    Args:
        bulk_density:  Density of the bulk fluid far from the wall
                       (``rho_l`` for ice/water, ``rho_v`` for ice/air).
        wall_fraction: ``psi_w / psi(rho_l)``.
        ny, nx:        Domain size; ``nx`` only needs a few columns since the
                       profile is x-homogeneous (keeps the 1-D problem cheap).
        wall_rows:     Solid rows at *each* wall; the tension is halved.
        n_steps:       Relaxation steps.
        n_resolutions: Row counts swept for the resolution spread
                       ``ny, 2*ny, 3*ny`` (relative spread reported).

    Returns:
        :class:`WallTension`.
    """
    if bulk_density <= 0.0:
        raise ValueError("bulk_density must be positive")
    if wall_fraction < 0.0:
        raise ValueError("wall_fraction must be >= 0")

    by_res: dict[str, float] = {}
    primary: WallTension | None = None
    for k in range(max(1, int(n_resolutions))):
        ny_k = int(ny) * (k + 1)
        f, rho, solid, psi_w, drift = _relax_wall_profile(
            bulk_density=bulk_density,
            wall_fraction=wall_fraction,
            ny=ny_k,
            nx=nx,
            wall_rows=wall_rows,
            n_steps=int(n_steps),
            G=G,
            tau=tau,
            rho_l=rho_l,
            dtype=dtype,
        )
        gamma = flat_interface_tension(
            rho,
            G=G,
            solid_mask=solid,
            wall_psi=psi_w,
            n_walls=2,
        )
        by_res[str(ny_k)] = gamma
        if k == 0:
            first_row = int(wall_rows)
            primary = WallTension(
                bulk_density=float(bulk_density),
                wall_fraction=float(wall_fraction),
                gamma=gamma,
                n_steps=int(n_steps),
                mass_drift=drift,
                rho_wall=float(rho[first_row].mean()),
            )
    assert primary is not None
    vals = torch.tensor(list(by_res.values()), dtype=torch.float64)
    mean = float(vals.mean())
    primary.gamma_by_resolution = {k: float(v) for k, v in by_res.items()}
    primary.resolution_spread = float((vals.max() - vals.min()) / abs(mean)) if mean != 0.0 else float("nan")
    return primary


def wall_tension_pair(
    wall_fraction: float, *, rho_l: float = RHO_L_DEFAULT, rho_v: float = RHO_V_DEFAULT, **kwargs
) -> tuple[WallTension, WallTension]:
    """``(gamma_sl, gamma_sg)`` for one wall wettability ``wall_fraction``.

    ``gamma_sl`` is measured with the wall in contact with liquid water,
    ``gamma_sg`` with the wall in contact with air.
    """
    gamma_sl = planar_wall_tension(bulk_density=rho_l, wall_fraction=wall_fraction, **kwargs)
    gamma_sg = planar_wall_tension(bulk_density=rho_v, wall_fraction=wall_fraction, **kwargs)
    return gamma_sl, gamma_sg


def planar_liquid_gas_tension(
    *,
    ny: int = 128,
    nx: int = 4,
    half_width: float = 24.0,
    n_steps: int = 6000,
    G: float = 5.0,
    tau: float = 1.0,
    rho_l: float = RHO_L_DEFAULT,
    rho_v: float = RHO_V_DEFAULT,
    interface_width: float = 1.5,
    dtype: torch.dtype = torch.float64,
) -> dict[str, float]:
    """Water/air tension ``gamma_lg`` from a wall-free liquid film (Laplace-free).

    A flat liquid slab (half thickness ``half_width``) is relaxed in a fully
    periodic box; the mechanical integral ``int (P_xx - P_yy) dy`` then gives
    ``gamma_lg`` with the *same* pressure tensor that measures the wall
    tensions, so the Young ratio contains no mixed-method error.  The result is
    reported next to the documented Laplace value ``SIGMA_DEFAULT`` as an
    absolute cross-check of the discrete prefactor.

    Returns:
        ``dict`` with ``gamma``, ``gamma_laplace_reference``,
        ``relative_difference``, relaxed ``rho_l``/``rho_v`` (measured plateau
        values), ``n_steps`` and ``mass_drift``.
    """
    yy = torch.arange(ny, dtype=dtype).view(ny, 1)
    dist = torch.abs(yy - 0.5 * ny) - float(half_width)
    rho0 = rho_v + (rho_l - rho_v) * 0.5 * (1.0 - torch.tanh(dist / float(interface_width)))
    rho0 = rho0.expand(ny, nx).contiguous()
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero).to(dtype)
    m0 = float(rho0.sum())
    for _ in range(int(n_steps)):
        f = collide_sc_single_component(f, G, tau=tau, psi_fn=psi_exp, forcing="edm")
        f = stream(f)
    rho = macroscopic(f)[0]
    m1 = float(rho.sum())
    gamma = flat_interface_tension(rho, G=G)
    prof = rho.mean(dim=1)
    return {
        "gamma": float(gamma),
        "gamma_laplace_reference": float(SIGMA_DEFAULT),
        "relative_difference": float(gamma / SIGMA_DEFAULT - 1.0),
        "rho_l": float(prof.max()),
        "rho_v": float(prof.min()),
        "n_steps": int(n_steps),
        "mass_drift": abs(m1 - m0) / abs(m0),
        "profile": [float(v) for v in prof.tolist()],
    }


#: Speed of sound squared of the D2Q9 lattice (``c_s^2 = 1/3``).
CS2 = 1.0 / 3.0


def sc_pressure(
    rho: torch.Tensor,
    *,
    G: float = 5.0,
    psi_fn: Callable[[torch.Tensor], torch.Tensor] = psi_exp,
) -> torch.Tensor:
    """Shan-Chen equation of state of this scheme, ``p = c_s^2 (rho + (G_eff/2) psi^2)``.

    With the library's sign convention (``G > 0`` attractive, i.e. the classic
    ``G_eff = -G``) the lattice EOS is

    .. math:: p = c_s^2\rho - \frac{G c_s^2}{2}\psi^2 .

    Validated in this node against the *coexistence* requirement (equal pressure
    in the liquid and the vapour plateau of a relaxed film, agreement 0.3 %, see
    ``docs/P262W1_N02_three_phase_contact_line.md``), which is what allows a
    Laplace pressure difference - and hence ``gamma_lg`` - to be measured without
    any hand-copied constant.
    """
    psi = psi_fn(rho)
    return CS2 * (rho - 0.5 * float(G) * psi**2)


def wall_force(
    rho: torch.Tensor,
    *,
    solid_mask: torch.Tensor,
    wall_psi: float | torch.Tensor,
    G: float = 5.0,
    psi_fn: Callable[[torch.Tensor], torch.Tensor] = psi_exp,
) -> dict[str, float]:
    """Net Shan-Chen force transmitted *to the solid* (momentum balance).

    The fluid-solid pairs are evaluated exactly as the force kernel does, but
    summed over the *solid* cells: by Newton's third law this is the momentum
    flux the fluid delivers to the wall.  For a flat wall that force has a
    tangential component only where the fluid state varies along the wall, i.e.
    at a contact line, and its integral is the **wetting tension**
    ``gamma_sg - gamma_sl`` per unit contact line (the Wilhelmy-plate
    measurement of Young's equation; a measurement that is independent of any
    interface-shape fit).

    Args:
        rho:        Density field ``(ny, nx)``.
        solid_mask: Boolean solid mask.
        wall_psi:   Pseudopotential inside the solid.
        G, psi_fn:  Shan-Chen parameters.

    Returns:
        ``dict`` with ``fx``/``fy`` (total, lattice units), ``fx_per_length`` /
        ``fy_per_length`` (divided by the wall length ``nx``) and
        ``fx_localised`` (fraction of ``|fx|`` carried by the 20 %% of columns
        with the largest contribution - a locality check: the signal must come
        from the contact line, not from a background drift).
    """
    psi = psi_fn(rho)
    fill = wall_psi
    if isinstance(fill, torch.Tensor):
        fill = fill.to(dtype=rho.dtype, device=rho.device)
    psi_full = torch.where(solid_mask, torch.full_like(psi, 0.0) + fill, psi)
    acc_x = torch.zeros_like(rho)
    acc_y = torch.zeros_like(rho)
    for i in range(9):
        cx, cy = int(C[i, 0]), int(C[i, 1])
        if cx == 0 and cy == 0:
            continue
        back = _shift(psi_full, -cx, -cy)
        acc_x = acc_x + float(W[i]) * back * float(cx)
        acc_y = acc_y + float(W[i]) * back * float(cy)
    fx_field = -float(G) * psi_full * acc_x
    fy_field = -float(G) * psi_full * acc_y
    fx_field = torch.where(solid_mask, fx_field, torch.zeros_like(fx_field))
    fy_field = torch.where(solid_mask, fy_field, torch.zeros_like(fy_field))
    fx = float(fx_field.sum())
    fy = float(fy_field.sum())
    nx = rho.shape[1]
    col = fx_field.sum(dim=0).abs()
    n_top = max(1, int(round(0.2 * nx)))
    top = float(torch.topk(col, min(n_top, nx)).values.sum())
    localised = top / float(col.sum()) if float(col.sum()) > 0 else float("nan")
    return {
        "fx": fx,
        "fy": fy,
        "fx_per_length": fx / nx,
        "fy_per_length": fy / nx,
        "fx_localised": localised,
    }


def free_droplet(
    *,
    n: int = 160,
    radius: float = 30.0,
    n_steps: int = 8000,
    G: float = 5.0,
    tau: float = 1.0,
    rho_l: float = RHO_L_DEFAULT,
    rho_v: float = RHO_V_DEFAULT,
    interface_width: float = 1.5,
    dtype: torch.dtype = torch.float64,
) -> dict[str, float]:
    """Relax a free (wall-free) droplet in a periodic box; return densities + tension.

    Voronoi-style diagnostics are avoided on purpose: the interface radius comes
    from the algebraic circle fit of the level set and the pressure difference
    from :func:`sc_pressure`, so ``gamma_lg = (p_in - p_out) R`` is the Laplace
    law applied to the *same* density field.

    Returns:
        ``dict`` with ``gamma`` (Laplace), ``radius``, ``dp``, ``rho_l``/``rho_v`` (plateau
        values at the centre and in the corner), the level-set sensitivity
        ``gamma_level_spread`` and ``mass_drift``.
    """
    yy, xx = torch.meshgrid(
        torch.arange(n, dtype=dtype), torch.arange(n, dtype=dtype), indexing="ij"
    )
    dist = torch.sqrt((xx - 0.5 * n) ** 2 + (yy - 0.5 * n) ** 2) - float(radius)
    rho0 = rho_v + (rho_l - rho_v) * 0.5 * (1.0 - torch.tanh(dist / float(interface_width)))
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero).to(dtype)
    m0 = float(rho0.sum())
    for _ in range(int(n_steps)):
        f = collide_sc_single_component(f, G, tau=tau, psi_fn=psi_exp, forcing="edm")
        f = stream(f)
    rho = macroscopic(f)[0]
    m1 = float(rho.sum())
    level = 0.5 * (rho_l + rho_v)
    pts = interface_points(rho, level)
    fit = circle_fit(pts)
    p = sc_pressure(rho, G=G)
    c = int(round(0.5 * n))
    r_in = max(2, int(0.25 * fit.r))
    p_in = float(p[c - r_in : c + r_in, c - r_in : c + r_in].mean())
    p_out = float(
        torch.cat(
            [
                p[: r_in, : r_in].flatten(),
                p[: r_in, -r_in:].flatten(),
                p[-r_in:, : r_in].flatten(),
                p[-r_in:, -r_in:].flatten(),
            ]
        ).mean()
    )
    dp = p_in - p_out
    gamma = dp * float(fit.r)
    # level-set sensitivity of the Laplace estimate
    gammas: list[float] = []
    for f_level in (0.9, 1.0, 1.1):
        pts_l = interface_points(rho, level * f_level)
        if pts_l.shape[0] < 8:
            continue
        try:
            fit_l = circle_fit(pts_l)
        except ValueError:
            continue
        gammas.append(dp * float(fit_l.r))
    spread = (max(gammas) - min(gammas)) / abs(gamma) if gammas and gamma != 0 else float("nan")
    return {
        "gamma": float(gamma),
        "radius": float(fit.r),
        "radius_rms": float(fit.rms),
        "dp": float(dp),
        "rho_l": float(rho[c - r_in : c + r_in, c - r_in : c + r_in].max()),
        "rho_v": float(rho[:4, :4].mean()),
        "gamma_level_spread": float(spread),
        "mass_drift": abs(m1 - m0) / abs(m0),
        "rho_field": rho,
    }


def wedge_contact_angle(
    rho: torch.Tensor,
    *,
    y_wall: float,
    solid: torch.Tensor,
    level: float,
    branch: str = "max",
    below: bool = True,
) -> dict[str, object]:
    """Contact angle of a channel meniscus (plug) at a horizontal wall.

    Uses the local ``x(y)`` tangent estimator of
    :func:`tensorlbm.contact_line.theta_fit.wall_contact_angle`, which is the
    estimator designed for a wall-spanning meniscus (no circle-shape
    assumption, hence independent of the droplet/Laplace model).

    Args:
        rho:    Density field.
        y_wall: Wall plane height.
        solid:  Solid mask.
        level:  Level-set value defining the interface.
        branch: ``"max"`` (right crossing) or ``"min"`` (left crossing) when two
                menisci share a row (periodic plug).
        below:  Wall orientation.

    Returns:
        ``dict`` with ``theta_deg``, ``unc_deg`` and the per-window values.
    """
    pts = interface_points(rho, level, solid)
    if pts.shape[0] < 8:
        raise ValueError("interface too short for a wedge angle fit")
    ang = wall_contact_angle(pts, y_wall, below=below, branch=branch)
    return ang.to_dict()


def sessile_contact_angle(
    rho: torch.Tensor,
    *,
    y_wall: float,
    solid: torch.Tensor,
    level: float,
    side: str = "cap",
    centre_x: float | None = None,
) -> dict[str, object]:
    """Contact angle of a sessile droplet by circle fit (see ``theta_fit``).

    Args:
        side: ``"cap"`` (whole droplet), ``"front"`` (advancing) or ``"rear"``.

    Returns:
        ``dict`` from :meth:`tensorlbm.contact_line.theta_fit.AngleEstimate.to_dict`.
    """
    pts = interface_points(rho, level, solid)
    if pts.shape[0] < 8:
        raise ValueError("interface too short for a droplet angle fit")
    est = angle_estimate(pts, y_wall, side=side, centre_x=centre_x)
    return est.to_dict()


def wall_coupling_weights(
    shape: tuple[int, int], *, solid_mask: torch.Tensor
) -> torch.Tensor:
    """Lattice weight ``S(x) = sum_{i: x+c_i in solid} w_i`` of the wall coupling.

    A fluid cell interacts with the wall through the directions whose neighbour
    lies inside the solid; ``S`` is the weight of that interaction.  It enters
    the wall/fluid interaction energy per unit wall length

    .. math:: E_{wall}/A = -G\,\psi_w\,\langle\psi\,S\rangle_x ,

    which is the quantity whose ``psi_w``-derivative gives the wall tension
    (:func:`wetting_tension_ti`).  The factor convention is the same double-sum
    convention as the Shan-Chen force (each fluid/solid pair counted once).
    """
    s = torch.zeros(shape, dtype=torch.float64, device=solid_mask.device)
    solid = solid_mask.to(torch.float64)
    for i in range(9):
        cx, cy = int(C[i, 0]), int(C[i, 1])
        if cx == 0 and cy == 0:
            continue
        s = s + float(W[i]) * _shift(solid, cx, cy)
    return s


def wall_coupling_moment(
    rho: torch.Tensor,
    *,
    solid_mask: torch.Tensor,
    weights: torch.Tensor,
    n_walls: int = 2,
) -> float:
    """``<psi S>_x`` — the wall-coupling moment whose derivative is the wall tension.

    Args:
        rho:        Density field (equilibrium profile against the wall).
        solid_mask: Solid mask.
        weights:    :func:`wall_coupling_weights` for the same mask.
        n_walls:    Number of identical walls in the domain (2 for a channel).

    Returns:
        ``(1/nx) * sum_{fluid} psi(x) S(x) / n_walls``.
    """
    psi = psi_exp(rho)
    contrib = torch.where(solid_mask, torch.zeros_like(psi), psi * weights)
    nx = rho.shape[1]
    return float(contrib.sum()) / float(nx) / float(max(1, n_walls))


@dataclass
class WettingTension:
    """Thermodynamic-integration measurement of the wetting tension ``gamma_sg-gamma_sl``."""

    wall_fraction: float
    psi_w: float
    delta_gamma: float
    gamma_sl: float
    gamma_sg: float
    n_ladder: int
    n_steps: int
    ny: int
    mass_drift_l: float
    mass_drift_g: float
    ladder: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "wall_fraction": self.wall_fraction,
            "psi_w": self.psi_w,
            "delta_gamma": self.delta_gamma,
            "gamma_sl": self.gamma_sl,
            "gamma_sg": self.gamma_sg,
            "n_ladder": self.n_ladder,
            "n_steps": self.n_steps,
            "ny": self.ny,
            "mass_drift_l": self.mass_drift_l,
            "mass_drift_g": self.mass_drift_g,
            "ladder": self.ladder,
        }


def wetting_tension_ti(
    wall_fraction: float,
    *,
    rho_l: float = RHO_L_DEFAULT,
    rho_v: float = RHO_V_DEFAULT,
    ny: int = 64,
    nx: int = 8,
    wall_rows: int = 2,
    n_ladder: int = 13,
    n_steps: int = 5000,
    G: float = 5.0,
    tau: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> WettingTension:
    """Wetting tension ``gamma_sg - gamma_sl`` by thermodynamic integration in ``psi_w``.

    Why this and not a pressure-tensor integral
    -------------------------------------------
    At a *rigid* wall the Shan-Chen interaction force is purely normal, so a
    pressure-tensor integral across a wall measures the disjoining/wall pressure,
    not the wall tension (measured bias ``+0.024`` lattice units, ~0.43
    ``gamma_lg``; see the falsified-hypothesis section of the node report).  The
    rigorous route is thermodynamic: the grand potential is stationary with
    respect to the density profile, so the derivative of the wall tension with
    respect to the wall coupling equals the *partial* derivative of the wall
    interaction energy at fixed profile,

    .. math::

        \frac{\partial \gamma_{sf}}{\partial \psi_w}
            = -G\,\langle \psi\, S\rangle_x ,

    and a trapezoidal integral over a ladder of ``psi_w`` values from 0 (dry
    wall) up to the target ``psi_w`` gives ``gamma_sf``.  The unknown
    ``gamma_sf(psi_w=0)`` is common to the liquid and the gas measurement and
    therefore cancels in the wetting tension - which is exactly the combination
    Young's law needs.  Each ladder point is warm-started from the previous one.

    Args:
        wall_fraction: ``psi_w / psi(rho_l)`` target of the ladder.
        rho_l, rho_v:  Bulk densities of the liquid and gas measurements.
        ny, nx:        Domain (``nx`` only keeps the profile x-homogeneous).
        n_ladder:      Number of ladder points (trapezoid resolution).
        n_steps:       Relaxation steps per ladder point.
        n_resolutions: not used here (resolution study is done by the caller).

    Returns:
        :class:`WettingTension`.
    """
    if wall_fraction < 0.0:
        raise ValueError("wall_fraction must be >= 0")
    psi_l_val = float(psi_exp(torch.tensor(float(rho_l), dtype=dtype)))
    psi_w_target = float(wall_fraction) * psi_l_val
    channel = Channel(nx=nx, ny=ny, wall_rows=wall_rows)
    solid = channel.solid_mask()
    weights = wall_coupling_weights(channel.shape, solid_mask=solid)
    ps = torch.linspace(0.0, psi_w_target, int(max(2, n_ladder)), dtype=dtype)
    dpsi = float(ps[1] - ps[0])

    moments: dict[str, list[float]] = {"l": [], "g": []}
    drift: dict[str, float] = {"l": 0.0, "g": 0.0}
    states: dict[str, torch.Tensor] = {}
    for key, rho_d in (("l", float(rho_l)), ("g", float(rho_v))):
        r0 = torch.full(channel.shape, rho_d, dtype=dtype)
        r0 = torch.where(solid, torch.full_like(r0, rho_d), r0)
        zero = torch.zeros_like(r0)
        states[key] = equilibrium(r0, zero, zero).to(dtype)

    ladder: list[dict[str, float]] = []
    for k, psi_w in enumerate(ps.tolist()):
        step_moments: dict[str, float] = {}
        for key, rho_d in (("l", float(rho_l)), ("g", float(rho_v))):
            stepper = WallStepper(
                channel,
                G=G,
                tau=tau,
                rho_l=rho_l,
                wall_fraction=psi_w / psi_l_val if psi_l_val > 0 else 0.0,
                forcing="edm",
                dtype=dtype,
            )
            f = states[key]
            m0 = float(macroscopic(f)[0][~solid].sum())
            f = stepper.run(f, int(n_steps), every=0)
            rho = macroscopic(f)[0]
            m1 = float(rho[~solid].sum())
            states[key] = f
            drift[key] = max(drift[key], abs(m1 - m0) / abs(m0) if m0 else float("nan"))
            step_moments[key] = wall_coupling_moment(
                rho, solid_mask=solid, weights=weights, n_walls=2
            )
        moments["l"].append(step_moments["l"])
        moments["g"].append(step_moments["g"])
        ladder.append(
            {"psi_w": float(psi_w), "moment_l": step_moments["l"], "moment_g": step_moments["g"]}
        )

    def _integrate(vals: list[float]) -> float:
        acc = 0.0
        for i in range(len(vals) - 1):
            acc += 0.5 * (vals[i] + vals[i + 1]) * dpsi
        return -G * acc

    gamma_sl = _integrate(moments["l"])
    gamma_sg = _integrate(moments["g"])
    return WettingTension(
        wall_fraction=float(wall_fraction),
        psi_w=psi_w_target,
        delta_gamma=float(gamma_sg - gamma_sl),
        gamma_sl=float(gamma_sl),
        gamma_sg=float(gamma_sg),
        n_ladder=int(len(ps)),
        n_steps=int(n_steps),
        ny=int(ny),
        mass_drift_l=float(drift["l"]),
        mass_drift_g=float(drift["g"]),
        ladder=ladder,
    )


def young_angle_deg(gamma_sg: float, gamma_sl: float, gamma_lg: float) -> float:
    """Young contact angle ``theta_Y = arccos((gamma_sg - gamma_sl)/gamma_lg)``.

    Args:
        gamma_sg: ice/air interfacial tension (lattice or SI — only ratios matter).
        gamma_sl: ice/water interfacial tension.
        gamma_lg: water/air interfacial tension.

    Returns:
        ``theta_Y`` in degrees.  ``nan`` when ``|cos| > 1`` (no partial wetting
        solution exists, i.e. the wall parameter drives complete wetting or
        complete drying).

    Raises:
        ValueError: if ``gamma_lg`` is not positive.
    """
    if not (gamma_lg > 0.0):
        raise ValueError(f"gamma_lg must be positive, got {gamma_lg}")
    cos = (float(gamma_sg) - float(gamma_sl)) / float(gamma_lg)
    if not math.isfinite(cos) or abs(cos) > 1.0:
        return float("nan")
    return math.degrees(math.acos(cos))


def mass_total(rho: torch.Tensor, solid_mask: torch.Tensor | None = None) -> float:
    """Total fluid mass (sum of the density field over fluid cells)."""
    if solid_mask is not None:
        rho = torch.where(solid_mask, torch.zeros_like(rho), rho)
    return float(rho.sum())


def contact_line_position(
    rho: torch.Tensor,
    *,
    level: float,
    y_wall: float,
    side: str,
    centre_x: float,
    band: tuple[float, float] = (1.5, 6.0),
) -> float:
    """Streamwise position where the interface crosses ``level`` near the wall.

    The interface is located in a band ``y_wall + band[0] <= y <= y_wall + band[1]``
    (*outside* the wall adsorption layer, whose density peak is one cell thick)
    by linear interpolation between the two rows that bracket ``level``, then
    the crossing nearest to the drop edge on the requested side is returned.

    Args:
        rho:       Density field ``(ny, nx)``.
        level:     Level set value, normally ``0.5*(rho_l + rho_v)``.
        y_wall:    Height of the wall plane (lattice units).
        side:      ``"front"`` (larger ``x`` than ``centre_x``) or ``"rear"``.
        centre_x:  Streamwise droplet centre.
        band:      Height band above the wall in which the crossing is searched.

    Returns:
        Streamwise coordinate (lattice units).  ``nan`` if no crossing exists in
        the band (interface detached from the wall).

    Raises:
        ValueError: on an unknown ``side``.
    """
    if side not in ("front", "rear"):
        raise ValueError(f"side must be 'front' or 'rear', got {side!r}")
    ny, nx = rho.shape
    y0 = int(round(y_wall + float(band[0])))
    y1 = int(round(y_wall + float(band[1])))
    y0 = max(y0, 0)
    y1 = min(y1, ny - 1)
    xs = torch.arange(nx, dtype=rho.dtype)
    best: float = float("nan")
    best_dist: float = float("inf")
    for y in range(y0, y1 + 1):
        row = rho[y]
        above = row >= level
        # crossings: above[i] != above[i+1] (periodic)
        idx = torch.nonzero(above != torch.roll(above, -1), as_tuple=False).flatten()
        for i in idx.tolist():
            j = (i + 1) % nx
            r0, r1 = float(row[i]), float(row[j])
            if r0 == r1:
                continue
            frac = (level - r0) / (r1 - r0)
            x = float(xs[i]) + frac * ((float(xs[j]) - float(xs[i])) % nx)
            x = x % nx
            # unwrap relative to centre
            dx = (x - centre_x + 0.5 * nx) % nx - 0.5 * nx
            if (side == "front" and dx <= 0.0) or (side == "rear" and dx >= 0.0):
                continue
            if abs(dx) < best_dist:
                best_dist = abs(dx)
                best = centre_x + dx
    return best


def contact_line_velocity(samples: list[tuple[float, float]]) -> dict[str, float]:
    """Least-squares drift speed of a contact-line trajectory.

    Args:
        samples: ``[(step, x), ...]`` pairs (``x`` already unwrapped).

    Returns:
        ``{"velocity", "stderr", "r2", "n_points"}``; ``velocity`` is in lattice
        units per step.
    """
    pts = [(float(s), float(x)) for s, x in samples if math.isfinite(float(x))]
    n = len(pts)
    if n < 3:
        return {"velocity": float("nan"), "stderr": float("nan"), "r2": float("nan"), "n_points": n}
    t = torch.tensor([p[0] for p in pts], dtype=torch.float64)
    x = torch.tensor([p[1] for p in pts], dtype=torch.float64)
    A = torch.stack([t, torch.ones_like(t)], dim=1)
    sol = torch.linalg.lstsq(A, x.unsqueeze(1)).solution.squeeze(1)
    resid = x - A @ sol
    ss_res = float((resid**2).sum())
    ss_tot = float(((x - x.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dof = max(n - 2, 1)
    stderr = math.sqrt(max(ss_res / dof / float(((t - t.mean()) ** 2).sum()), 0.0))
    return {"velocity": float(sol[0]), "stderr": stderr, "r2": r2, "n_points": n}


#: SI anchor values for the ice / water / air system (see module docstring).
SI_GAMMA_LG_WATER_AIR = 0.0728  # N/m, 20 C
SI_GAMMA_SL_ICE_WATER = 0.0291  # N/m, ice/water interfacial free energy (0 C)
SI_GAMMA_SG_ICE_AIR = 0.109  # N/m, ice/air surface free energy (0 C)


def to_physical_units(
    *,
    gamma_lg: float,
    gamma_sl: float,
    gamma_sg: float,
    gamma_lg_si: float = SI_GAMMA_LG_WATER_AIR,
    gamma_sl_si: float = SI_GAMMA_SL_ICE_WATER,
    gamma_sg_si: float = SI_GAMMA_SG_ICE_AIR,
) -> dict[str, float]:
    """Map lattice tensions onto the ice/water/air system.

    The lattice Shan-Chen model fixes only the *ratios* of the three tensions;
    the mapping takes the water/air tension as the scale anchor and reports the
    implied SI values plus the contact angle they predict.  The reference SI
    triple itself gives ``theta = arccos((0.109-0.0291)/0.0728)`` which is
    unphysical (|cos| > 1) because ice is fully wetted by water below the
    melting point - the reference is quoted for transparency, the lattice ratios
    are what the campaigns test.

    Args:
        gamma_lg, gamma_sl, gamma_sg: lattice-unit tensions.
        gamma_lg_si, gamma_sl_si, gamma_sg_si: SI anchors (N/m).

    Returns:
        ``dict`` with the lattice ratios, the SI-scaled tensions, the Young
        angle from the lattice ratios and from the SI anchors (degrees).
    """
    if not (gamma_lg > 0.0):
        raise ValueError("gamma_lg must be positive")
    scale = float(gamma_lg_si) / float(gamma_lg)
    return {
        "scale_N_per_m_per_lattice": scale,
        "ratio_sg_over_lg": float(gamma_sg) / float(gamma_lg),
        "ratio_sl_over_lg": float(gamma_sl) / float(gamma_lg),
        "gamma_lg_si": float(gamma_lg) * scale,
        "gamma_sl_si": float(gamma_sl) * scale,
        "gamma_sg_si": float(gamma_sg) * scale,
        "theta_young_lattice_deg": young_angle_deg(gamma_sg, gamma_sl, gamma_lg),
        "theta_young_si_anchor_deg": young_angle_deg(gamma_sg_si, gamma_sl_si, gamma_lg_si),
    }