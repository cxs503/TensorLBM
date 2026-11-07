"""Herschel-Bulkley lattice-Boltzmann kernels, runners and yield-surface probes.

This module carries the *numerical* half of the yield-stress work: a
variable-viscosity BGK collision with the Papanastasiou-regularised H-B
viscosity (:mod:`tensorlbm.herschel_bulkley`), Guo body forcing generalised to
a per-cell relaxation time, half-way bounce-back walls (stationary and moving),
and three end-to-end runners used by the verification benchmark:

* :func:`run_hb_pipe`     -- 3-D D3Q19 circular pipe (plug core), periodic in the
  axial direction, body-force driven, staircase half-way bounce-back wall.
* :func:`run_hb_couette`  -- 2-D D2Q9 plane Couette-Poiseuille with an adverse
  gradient and a *moving* upper plate: static yield layer glued to the lower
  plate, yield surface at ``y = h*tau_y/tau_w``.
* :func:`run_hb_channel`  -- 2-D D2Q9 plane Poiseuille (both plates at rest),
  used for the sub-yield negative control.

Consistency notes
-----------------
* The stress field of a fully developed duct flow is fixed by the momentum
  balance, ``sigma = a*r/2`` (pipe) or ``sigma = a*y`` (slit), *independently of
  the rheology*.  The plug boundary is therefore the locus where the regularised
  constitutive stress equals ``tau_y``; that is exactly what
  :func:`measure_pipe_plug_radius` / :func:`measure_yield_surface` compute.
* Velocity gradients use second-order central differences in the interior with
  one-sided (clamped) stencils at the domain edges; solid cells hold the
  bounce-back state (zero velocity for a resting wall), so the gradient next to
  a wall is evaluated against the wall node itself, not the half-way point.
  This is the same convention as :mod:`tensorlbm.powerlaw`; the benchmark
  cross-checks *integral* observables (flow rate, plug radius) instead of
  relying on a pointwise wall gradient.
* Body force: Guo (2002) in its velocity-shift form, ``u* = u + a/2``, with
  ``F_i = w_i (1 - 1/(2 tau)) rho [ 3 (c_i - u*)·a + 9 (c_i·u*)(c_i·a) ]``, which
  injects exactly ``rho*a`` momentum per step for any local ``tau``.

References
----------
* Guo, Z., Zheng, C. & Shi, B. (2002) *Phys. Rev. E* 65, 046308.
* Ladd, A.J.C. (1994) *J. Fluid Mech.* 271, 285 (moving-wall bounce-back).
* Papanastasiou, T.C. (1987) *J. Rheol.* 31, 385.
"""

from __future__ import annotations

import math

import torch

from .d2q9 import C as C2
from .d2q9 import W as W2
from .d2q9 import equilibrium, macroscopic
from .d3q19 import C as C3
from .d3q19 import W as W3
from .d3q19 import equilibrium3d, macroscopic3d
from .herschel_bulkley import HBRheology, hb_regularized_viscosity
from .powerlaw import guo_force_term, strain_rate_shear_rate_2d
from .solver import stream
from .solver3d import stream3d

__all__ = [
    "central_difference_axis",
    "shear_rate_3d",
    "shear_rate_3d_xinvariant",
    "wall_aware_gradient",
    "box_smooth",
    "shear_rate_pipe",
    "shear_rate_channel",
    "hb_relaxation_time",
    "collide_hb_bgk_forced_2d",
    "collide_hb_bgk_forced_3d",
    "bounce_back_solid_3d",
    "moving_wall_bounce_back_2d",
    "run_hb_pipe",
    "run_hb_couette",
    "run_hb_channel",
    "measure_pipe_plug_radius",
    "measure_yield_surface",
]


# ---------------------------------------------------------------------------
# Gradients / shear rate
# ---------------------------------------------------------------------------


def central_difference_axis(field: torch.Tensor, axis: int) -> torch.Tensor:
    """Second-order central difference along ``axis`` (unit grid spacing).

    Interior slices use ``(f[i+1] - f[i-1])/2``; the two boundary planes use
    one-sided differences.  Works for 2-D ``(ny, nx)`` and 3-D ``(nz, ny, nx)``
    fields (``axis`` counted from the end: ``-1`` = x).
    """
    ax = axis if axis >= 0 else field.dim() + axis
    if ax < 0 or ax >= field.dim():
        msg = f"axis {axis} out of range for a {field.dim()}-D field"
        raise ValueError(msg)
    n = field.shape[ax]
    if n < 3:
        msg = f"axis {axis} needs >= 3 samples, got {n}"
        raise ValueError(msg)
    grad = torch.zeros_like(field)

    def sl(start: int, stop: int) -> tuple[slice, ...]:
        idx = [slice(None)] * field.dim()
        idx[ax] = slice(start, stop)
        return tuple(idx)

    grad[sl(1, n - 1)] = 0.5 * (field[sl(2, n)] - field[sl(0, n - 2)])
    grad[sl(0, 1)] = field[sl(1, 2)] - field[sl(0, 1)]
    grad[sl(n - 1, n)] = field[sl(n - 1, n)] - field[sl(n - 2, n - 1)]
    return grad


def shear_rate_3d(ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor) -> torch.Tensor:
    """Shear rate ``gamma_dot = sqrt(2 S:S)`` for a 3-D field ``(nz, ny, nx)``.

    ``S_ab = (du_a/dx_b + du_b/dx_a)/2``, hence

        gamma_dot^2 = 2[(du/dx)^2 + (dv/dy)^2 + (dw/dz)^2]
                      + (du/dy + dv/dx)^2 + (du/dz + dw/dx)^2 + (dv/dz + dw/dy)^2.
    """
    du_dx = central_difference_axis(ux, -1)
    du_dy = central_difference_axis(ux, -2)
    du_dz = central_difference_axis(ux, -3)
    dv_dx = central_difference_axis(uy, -1)
    dv_dy = central_difference_axis(uy, -2)
    dv_dz = central_difference_axis(uy, -3)
    dw_dx = central_difference_axis(uz, -1)
    dw_dy = central_difference_axis(uz, -2)
    dw_dz = central_difference_axis(uz, -3)
    return torch.sqrt(
        torch.clamp(
            2.0 * (du_dx**2 + dv_dy**2 + dw_dz**2)
            + (du_dy + dv_dx) ** 2
            + (du_dz + dw_dx) ** 2
            + (dv_dz + dw_dy) ** 2,
            min=0.0,
        )
    )


def wall_aware_gradient(field: torch.Tensor, fluid: torch.Tensor, axis: int) -> torch.Tensor:
    """Distance-weighted central difference that respects half-way walls.

    Solid cells hold the bounce-back state (zero velocity for a resting wall)
    and the physical wall sits *half a lattice unit* from the adjacent fluid
    node.  A plain central difference across that interface divides by 2 while
    the data spans 1.5 lattice units, which underestimates the near-wall shear
    rate by up to a factor 2 -- harmless for a constant viscosity but a large
    bias for a yield-stress fluid, whose viscosity scales like ``tau_y/g``.

    This routine divides by the *actual* distance between the two samples:
    ``1 + 1`` in the interior (second-order central difference), ``1 + 0.5``
    when one side is a wall node.  One-sided edge stencils are used at domain
    boundaries, which are solid in all geometries used here.

    Args:
        field: Velocity component, 2-D ``(ny, nx)`` or 3-D ``(nz, ny, nx)``.
        fluid: Boolean fluid mask, same shape as ``field``.
        axis: Axis to differentiate (negative indices allowed).

    Returns:
        Gradient field, same shape.
    """
    ax = axis if axis >= 0 else field.dim() + axis
    if field.shape != fluid.shape:
        msg = f"field {tuple(field.shape)} and fluid mask {tuple(fluid.shape)} must match"
        raise ValueError(msg)
    # distance to a neighbour: 1.0 lattice if fluid, 0.5 if the wall is half-way
    dist = torch.where(fluid, 1.0, 0.5).to(field.dtype)

    def shift(t: torch.Tensor, step: int) -> torch.Tensor:
        """``out[i] = t[i+step]`` with edge replication."""
        idx_src = [slice(None)] * t.dim()
        idx_dst = [slice(None)] * t.dim()
        n = t.shape[ax]
        if step > 0:
            idx_src[ax] = slice(step, n)
            idx_dst[ax] = slice(0, n - step)
        else:
            idx_src[ax] = slice(0, n + step)
            idx_dst[ax] = slice(-step, n)
        out = t.clone()
        out[tuple(idx_dst)] = t[tuple(idx_src)]
        return out

    up, dn = shift(field, 1), shift(field, -1)
    d_up, d_dn = shift(dist, 1), shift(dist, -1)
    grad = (up - dn) / (d_up + d_dn)
    # the two edge planes of the axis have no up/down pair: use one-sided
    # differences with the same distance weighting
    edge_lo = [slice(None)] * field.dim()
    edge_lo[ax] = 0
    edge_hi = [slice(None)] * field.dim()
    edge_hi[ax] = -1
    nb_lo = [slice(None)] * field.dim()
    nb_lo[ax] = 1
    nb_hi = [slice(None)] * field.dim()
    nb_hi[ax] = -2
    grad[tuple(edge_lo)] = (field[tuple(nb_lo)] - field[tuple(edge_lo)]) / shift(dist, 1)[tuple(edge_lo)]
    grad[tuple(edge_hi)] = (field[tuple(edge_hi)] - field[tuple(nb_hi)]) / shift(dist, -1)[tuple(edge_hi)]
    return grad


def box_smooth(field: torch.Tensor, fluid: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Normalised ``k x k`` box average of a field over its last two axes.

    Generalised-Newtonian LBM evaluates the viscosity from *local* velocity
    gradients, and a yield-stress law amplifies gradient noise because
    ``nu ~ tau_y/gamma_dot``.  In the plug, where the true shear rate is orders of
    magnitude below the wall value, lattice-scale noise in the gradient is enough
    to collapse the plug viscosity and let the core creep.  Smoothing the shear
    rate over a small stencil (a standard device in non-Newtonian LBM, cf. the
    filtered strain-rate input of OpenLB's yield-stress models) keeps the plug
    stiff without touching the yielded region, where the gradient is large and
    smooth.

    The average is a *normalised* convolution, so solid cells do not dilute it.

    Args:
        field: 3-D ``(nz, ny, nx)`` field (the average acts on ``(nz, ny)``).
        fluid: Boolean fluid mask, same shape.
        k: Odd stencil width (``1`` disables the filter).

    Returns:
        Smoothed field, same shape.
    """
    if k <= 1:
        return field
    if k % 2 == 0:
        msg = f"stencil width must be odd, got {k}"
        raise ValueError(msg)
    nz, ny, nx = field.shape
    # the stencil acts on the cross-section (nz, ny); the axial direction is the
    # batch dimension because the flow is streamwise invariant
    fw = field.permute(2, 0, 1)
    mw = fluid.to(field.dtype).permute(2, 0, 1)
    kernel = torch.ones(1, 1, k, k, dtype=field.dtype, device=field.device)
    num = torch.nn.functional.conv2d((fw * mw).unsqueeze(1), kernel, padding=k // 2)
    den = torch.nn.functional.conv2d(mw.unsqueeze(1), kernel, padding=k // 2)
    out = num / torch.clamp(den, min=1.0)
    return out.squeeze(1).permute(1, 2, 0).clamp(min=0.0)


def shear_rate_pipe(ux: torch.Tensor, fluid: torch.Tensor) -> torch.Tensor:
    """Shear rate of a fully developed pipe flow ``u = (u(y,z), 0, 0)``.

    ``gamma_dot = sqrt((du/dy)^2 + (du/dz)^2)`` with wall-aware gradients
    (:func:`wall_aware_gradient`).  Using this instead of the full
    :func:`shear_rate_3d` is *exact* for a streamwise-invariant unidirectional
    flow (all x-derivatives vanish, transverse velocities are zero) and avoids
    feeding lattice noise from those components into a viscosity that scales
    like ``1/gamma_dot``.
    """
    if ux.dim() != 3:
        msg = f"ux must be 3-D (nz, ny, nx), got {tuple(ux.shape)}"
        raise ValueError(msg)
    du_dy = wall_aware_gradient(ux, fluid, -2)
    du_dz = wall_aware_gradient(ux, fluid, -3)
    return torch.sqrt(torch.clamp(du_dy**2 + du_dz**2, min=0.0))


def shear_rate_channel(ux: torch.Tensor, fluid: torch.Tensor) -> torch.Tensor:
    """Shear rate of a plane channel flow ``u = (u(y), 0)`` (2-D, wall-aware)."""
    if ux.dim() != 2:
        msg = f"ux must be 2-D (ny, nx), got {tuple(ux.shape)}"
        raise ValueError(msg)
    return torch.abs(wall_aware_gradient(ux, fluid, 0))


def hb_relaxation_time(
    gamma_dot: torch.Tensor,
    rheo: HBRheology,
    m: float = 1.0e4,
    nu_min: float = 1.0e-8,
    nu_max: float | None = None,
    tau_min: float = 0.5001,
    tau_max: float | None = None,
) -> torch.Tensor:
    """Per-cell relaxation time ``tau = 3 nu(g) + 1/2`` (clipped).

    ``tau_min`` must stay strictly above ``1/2``; ``tau_max`` guards the plug,
    where the regularised viscosity is largest.
    """
    if tau_min <= 0.5:
        msg = f"tau_min must be > 0.5, got {tau_min}"
        raise ValueError(msg)
    nu = hb_regularized_viscosity(gamma_dot, rheo, m=m, nu_min=nu_min, nu_max=nu_max)
    tau = torch.clamp(3.0 * nu + 0.5, min=tau_min)
    if tau_max is not None:
        tau = torch.clamp(tau, max=tau_max)
    return tau


# ---------------------------------------------------------------------------
# Collision operators
# ---------------------------------------------------------------------------


def collide_hb_bgk_forced_2d(
    f: torch.Tensor,
    ax: float | torch.Tensor,
    rheo: HBRheology,
    m: float = 1.0e4,
    nu_min: float = 1.0e-8,
    nu_max: float | None = None,
    tau_min: float = 0.5001,
    tau_max: float | None = None,
    tau_field: torch.Tensor | None = None,
) -> torch.Tensor:
    """H-B BGK collision + Guo body force for D2Q9 (per-cell ``tau``).

    Velocity -> shear rate -> regularised ``nu(g)`` -> ``tau = 3nu + 1/2`` -> BGK
    relaxation towards ``feq(rho, u*)`` plus the Guo momentum source, exactly as
    in :mod:`tensorlbm.powerlaw` but with the yield-stress viscosity.

    Args:
        f: Distribution tensor ``(9, ny, nx)``.
        ax: Body-force acceleration in +x (scalar or ``(ny, nx)`` field).
        rheo: H-B parameters (kinematic).
        tau_field: Optional precomputed relaxation-time field; when given the
            viscosity is not recomputed (keeps the constitutive stress and the
            collision exactly consistent).

    Returns:
        Post-collision distribution, same shape.
    """
    if f.dim() != 3 or f.shape[0] != 9:
        msg = f"f must have shape (9, ny, nx), got {tuple(f.shape)}"
        raise ValueError(msg)
    rho, ux, uy = macroscopic(f)
    tau = tau_field
    if tau is None:
        tau = hb_relaxation_time(
            strain_rate_shear_rate_2d(ux, uy),
            rheo,
            m=m,
            nu_min=nu_min,
            nu_max=nu_max,
            tau_min=tau_min,
            tau_max=tau_max,
        )
    ux_star = ux + 0.5 * ax
    feq = equilibrium(rho, ux_star, uy)
    fi = guo_force_term(f, ax, tau, ux_star=ux_star)
    return f - (f - feq) / tau.unsqueeze(0) + fi


def _opposite_table(c: torch.Tensor) -> torch.Tensor:
    """Index of the opposite direction for an ``(q, 3)`` velocity set."""
    dist = torch.cdist((-c).float(), c.float())
    idx = torch.argmin(dist, dim=1)
    if not torch.allclose(c[idx], -c):
        msg = "velocity set is not centrally symmetric; bounce-back impossible"
        raise ValueError(msg)
    return idx


# D3Q19 opposite-direction table (derived from the velocity set, not hard-coded).
_OPP3 = _opposite_table(C3)


def collide_hb_bgk_forced_3d(
    f: torch.Tensor,
    ax: float | torch.Tensor,
    rheo: HBRheology,
    m: float = 1.0e4,
    nu_min: float = 1.0e-8,
    nu_max: float | None = None,
    tau_min: float = 0.5001,
    tau_max: float | None = None,
    tau_field: torch.Tensor | None = None,
) -> torch.Tensor:
    """H-B BGK collision + Guo body force for D3Q19 (per-cell ``tau``).

    Grid is ``(nz, ny, nx)``; the body force acts in +x (the pipe axis in
    :func:`run_hb_pipe`).  See :func:`collide_hb_bgk_forced_2d` for the scheme.
    """
    if f.dim() != 4 or f.shape[0] != 19:
        msg = f"f must have shape (19, nz, ny, nx), got {tuple(f.shape)}"
        raise ValueError(msg)
    rho, ux, uy, uz = macroscopic3d(f)
    tau = tau_field
    if tau is None:
        tau = hb_relaxation_time(
            shear_rate_3d(ux, uy, uz),
            rheo,
            m=m,
            nu_min=nu_min,
            nu_max=nu_max,
            tau_min=tau_min,
            tau_max=tau_max,
        )
    a = torch.as_tensor(ax, dtype=f.dtype, device=f.device)
    ux_star = ux + 0.5 * a
    feq = equilibrium3d(rho, ux_star, uy, uz)
    c = C3.to(device=f.device, dtype=f.dtype)
    w = W3.to(device=f.device, dtype=f.dtype)
    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)
    cu = 3.0 * (cx * ux_star.unsqueeze(0) + cy * uy.unsqueeze(0) + cz * uz.unsqueeze(0))
    pref = (1.0 - 0.5 / tau).unsqueeze(0)
    fi = (
        w.view(19, 1, 1, 1)
        * rho.unsqueeze(0)
        * a.unsqueeze(0)
        * pref
        * (3.0 * (cx - ux_star.unsqueeze(0)) + 9.0 * cx * cu)
    )
    return f - (f - feq) / tau.unsqueeze(0) + fi


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


def bounce_back_solid_3d(f: torch.Tensor, solid: torch.Tensor) -> torch.Tensor:
    """Post-streaming half-way bounce-back in solid cells (D3Q19, resting wall).

    Args:
        f: Post-streaming distribution ``(19, nz, ny, nx)``.
        solid: Boolean mask ``(nz, ny, nx)``, True = solid.

    Returns:
        Distribution with the solid-cell entries replaced by the opposite
        direction of the same cell (wall half-way between the two nodes).
    """
    if f.shape[0] != 19:
        msg = f"f must have 19 directions, got {f.shape[0]}"
        raise ValueError(msg)
    if solid.shape != f.shape[1:]:
        msg = f"solid mask {tuple(solid.shape)} != field {tuple(f.shape[1:])}"
        raise ValueError(msg)
    f_opp = f[_OPP3]
    mask = solid.unsqueeze(0).expand_as(f)
    return torch.where(mask, f_opp, f)


def moving_wall_bounce_back_2d(
    f: torch.Tensor,
    wall: torch.Tensor,
    u_wall_x: float,
    u_wall_y: float = 0.0,
) -> torch.Tensor:
    """Post-streaming bounce-back on walls moving with ``(u_wall_x, u_wall_y)``.

    ``f_i = f_i^post,opp + 6 w_i rho_w (c_i · u_w)`` with ``c_s^2 = 1/3``
    (Ladd's second-order moving-wall bounce-back); ``rho_w`` is the local
    density in the wall cell taken from the post-streaming state.
    """
    if f.shape[0] != 9:
        msg = f"f must have 9 directions, got {f.shape[0]}"
        raise ValueError(msg)
    if wall.shape != f.shape[1:]:
        msg = f"wall mask {tuple(wall.shape)} != field {tuple(f.shape[1:])}"
        raise ValueError(msg)
    rho, _, _ = macroscopic(f)
    opp = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6])
    c = C2.to(dtype=f.dtype, device=f.device)
    w = W2.to(dtype=f.dtype, device=f.device)
    cu = (c[:, 0] * u_wall_x + c[:, 1] * u_wall_y).view(9, 1, 1)
    f_bb = f[opp] + 6.0 * w.view(9, 1, 1) * rho.unsqueeze(0) * cu
    mask = wall.unsqueeze(0).expand_as(f)
    return torch.where(mask, f_bb, f)# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def _pipe_solid_mask(n: int, R: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Staircase circular pipe mask on an ``n x n`` cross-section.

    The axis sits at lattice coordinate ``(n-1)/2`` (a cell centre for odd
    ``n``, a cell corner for even ``n``); a cell is solid when its centre lies
    farther than ``R`` from the axis.
    """
    if n < 5:
        msg = f"cross-section must be at least 5 cells, got {n}"
        raise ValueError(msg)
    coord = torch.arange(n, dtype=torch.float64) - 0.5 * (n - 1)
    zz, yy = torch.meshgrid(coord, coord, indexing="ij")
    rr = torch.sqrt(yy**2 + zz**2)
    return rr > R, rr


def run_hb_pipe(
    R: float,
    rheo: HBRheology,
    a: float,
    n_steps: int = 6000,
    n_cross: int | None = None,
    n_axial: int = 4,
    m: float = 1.0e6,
    nu_max: float | None = None,
    tau_max: float | None = None,
    check_every: int = 200,
    tol: float = 1.0e-9,
    u_init: torch.Tensor | None = None,
    viscosity_smoothing: int = 1,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> dict:
    """Steady H-B flow in a 3-D circular pipe (D3Q19), body-force driven.

    Geometry: axis along x, ``n_cross x n_cross`` cross-section (default
    ``2*ceil(R) + 4``), staircase wall, ``n_axial`` cells in the periodic axial
    direction (fully developed flow is x-invariant, so this is exact).  The
    body-force acceleration ``a`` drives the flow and the momentum balance fixes
    the wall stress ``tau_w = a*R_eff/2``.

    Args:
        R: Nominal pipe radius in lattice units.
        rheo: H-B parameters (kinematic).
        a: Body-force acceleration in +x.
        n_steps: Maximum number of steps.
        n_cross: Cross-section size (``None`` -> ``2*ceil(R) + 4``).
        n_axial: Axial (periodic) extent.
        m: Papanastasiou regularisation rate.
        nu_max: Upper viscosity clamp (``None`` -> unclamped).
        tau_max: Optional relaxation-time clamp.
        check_every: Flow-rate convergence check interval.
        tol: Relative flow-rate change considered converged.
        u_init: Optional initial axial velocity ``(n, n)`` (or ``(n,n,n_axial)``)
            used to warm-start the run, e.g. from the 1-D reference profile.
            A warm start changes only the transient, never the steady state; the
            benchmark also runs the primary case from rest to confirm this.
        dtype, device: Torch dtype / device.

    Returns:
        dict with ``ux``, ``solid``, ``rr`` (``(n,n)`` radius field),
        ``flow_rate``, ``steps``, ``converged``, ``history``, ``tau_field``.
    """
    n = n_cross if n_cross is not None else 2 * int(math.ceil(R)) + 4
    solid_cpu, rr = _pipe_solid_mask(n, R)
    solid = solid_cpu.to(device)
    # D3Q19 fields are (q, nz, ny, nx): the pipe axis is x, the cross-section
    # spans (z, y).  Fully developed flow is x-invariant, so n_axial can be small.
    solid3 = solid.unsqueeze(-1).expand(n, n, n_axial).contiguous()
    f = torch.zeros((19, n, n, n_axial), dtype=dtype, device=device)
    rho0 = torch.ones((n, n, n_axial), dtype=dtype, device=device)
    zero = torch.zeros_like(rho0)
    if u_init is None:
        f[:] = equilibrium3d(rho0, zero, zero, zero)
    else:
        u0 = u_init.to(device=device, dtype=dtype)
        if u0.dim() == 2:
            u0 = u0.unsqueeze(-1).expand(n, n, n_axial)
        f[:] = equilibrium3d(rho0, u0, zero, zero)
    fluid = (~solid3).to(dtype)
    fluid_b = ~solid3
    history: list[tuple[int, float]] = []
    q_prev = 0.0
    converged = False
    steps = 0
    tau_field = None
    for step in range(1, n_steps + 1):
        tau_field = hb_relaxation_time(
            box_smooth(shear_rate_pipe(macroscopic3d(f)[1], fluid_b), fluid_b,
                       k=viscosity_smoothing),
            rheo,
            m=m,
            nu_max=nu_max,
            tau_max=tau_max,
        )
        f = collide_hb_bgk_forced_3d(f, a, rheo, tau_field=tau_field)
        f = stream3d(f)
        f = bounce_back_solid_3d(f, solid3)
        steps = step
        if step % check_every == 0 or step == n_steps:
            _, ux, _, _ = macroscopic3d(f)
            q = float((ux * fluid).sum())
            history.append((step, q))
            if q_prev != 0.0 and abs(q - q_prev) <= tol * abs(q_prev):
                converged = True
                break
            q_prev = q
    _, ux, _, _ = macroscopic3d(f)
    return {
        "ux": ux,
        "solid": solid,
        "rr": rr,
        "flow_rate": float((ux * fluid).sum()),
        "steps": steps,
        "converged": converged,
        "history": history,
        "tau_field": tau_field,
    }


def run_hb_channel(
    h: int,
    rheo: HBRheology,
    a: float,
    u_wall: float = 0.0,
    nx: int = 4,
    n_steps: int = 4000,
    m: float = 1.0e6,
    nu_max: float | None = None,
    tau_max: float | None = None,
    check_every: int = 100,
    tol: float = 1.0e-10,
    u_init: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> dict:
    """H-B plane channel (D2Q9): rest lower plate, upper plate at ``u_wall``.

    The grid is ``h + 2`` rows tall, so bounce-back places the walls at
    ``y = 0.5`` and ``y = h - 0.5`` -- a gap of exactly ``h`` -- and a uniform
    body force ``a`` acts in +x.  Two configurations are used here:

    * **plane Poiseuille** (``u_wall = 0``, ``a > 0``): symmetric plug centred on
      the mid-plane, plug half-height ``(h/2)*tau_y/tau_w``;
    * **Couette-Poiseuille with adverse gradient** (``a = -beta < 0``,
      ``u_wall = U``): stress ``sigma(y) = beta*y``, static plug glued to the
      lower plate, yield surface at ``y = h*tau_y/tau_w``.

    Args:
        h: Gap height in lattice units.
        rheo: H-B parameters.
        a: Body-force acceleration in +x.
        u_wall: Upper-plate speed in +x (0 -> both plates at rest).
        nx: Periodic streamwise extent.
        n_steps, check_every, tol: Steady-state driver; convergence is checked
            on the x-averaged streamwise velocity at the plate.
        m, nu_max, tau_max: Regularisation / clamping parameters.
        dtype, device: Torch dtype / device.

    Returns:
        dict with ``ux`` (``(ny, nx)``), ``y`` (cell-centre distance above the
        lower wall, ``(ny,)``), ``u_profile`` (x-average over the fluid rows),
        ``gamma_dot`` (``(ny, nx)``), ``steps``, ``converged``, ``history``.
    """
    ny = h + 2
    f = torch.zeros((9, ny, nx), dtype=dtype, device=device)
    rho0 = torch.ones((ny, nx), dtype=dtype, device=device)
    zero = torch.zeros_like(rho0)
    if u_init is None:
        f[:] = equilibrium(rho0, zero, zero)
    else:
        f[:] = equilibrium(rho0, u_init.to(device=device, dtype=dtype).unsqueeze(1).expand(ny, nx), zero)
    lower = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    lower[0, :] = True
    upper = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    upper[-1, :] = True
    fluid_b = torch.ones((ny, nx), dtype=torch.bool, device=device)
    fluid_b[0, :] = False
    fluid_b[-1, :] = False
    rho_idx = torch.arange(ny, dtype=torch.float64, device=device) - 0.5
    history: list[tuple[int, float]] = []
    converged = False
    steps = 0
    tau_field = None
    u_prev = 0.0
    ux = torch.zeros_like(rho0)
    uy = torch.zeros_like(rho0)
    for step in range(1, n_steps + 1):
        _, ux, uy = macroscopic(f)
        tau_field = hb_relaxation_time(
            shear_rate_channel(ux, fluid_b),
            rheo,
            m=m,
            nu_max=nu_max,
            tau_max=tau_max,
        )
        f = collide_hb_bgk_forced_2d(f, a, rheo, tau_field=tau_field)
        f = stream(f)
        f = moving_wall_bounce_back_2d(f, lower, 0.0, 0.0)
        f = moving_wall_bounce_back_2d(f, upper, u_wall, 0.0)
        steps = step
        if step % check_every == 0 or step == n_steps:
            _, ux, _ = macroscopic(f)
            probe = float(ux[1:-1, :].abs().mean())
            history.append((step, probe))
            if step > check_every and abs(probe - u_prev) <= tol * max(1.0e-30, abs(u_prev)):
                converged = True
                break
            u_prev = probe
    _, ux, uy = macroscopic(f)
    gamma = shear_rate_channel(ux, fluid_b)
    return {
        "ux": ux,
        "y": rho_idx,
        "u_profile": ux.mean(dim=1),
        "gamma_dot": gamma,
        "steps": steps,
        "converged": converged,
        "history": history,
        "tau_field": tau_field,
    }


def run_hb_couette(
    h: int,
    rheo: HBRheology,
    beta: float,
    nx: int = 4,
    **kwargs,
) -> dict:
    """Couette-Poiseuille with an adverse gradient; plate speed set analytically.

    The upper-plate speed is taken from
    :func:`tensorlbm.herschel_bulkley.hb_couette_adverse_wall_velocity`, i.e. the
    value that makes the wall shear stress vanish at the *resting* lower plate
    and puts the yield surface at ``y = h*tau_y/tau_w``.  Remaining keyword
    arguments are forwarded to :func:`run_hb_channel`.
    """
    from .herschel_bulkley import hb_couette_adverse_wall_velocity

    meta = hb_couette_adverse_wall_velocity(beta, float(h), rheo)
    res = run_hb_channel(h, rheo, -beta, u_wall=meta["U"], nx=nx, **kwargs)
    res["analytic_meta"] = meta
    return res


# ---------------------------------------------------------------------------
# Yield-surface probes
# ---------------------------------------------------------------------------


def _azimuthal_profile(
    ux: torch.Tensor,
    rr: torch.Tensor,
    fluid: torch.Tensor,
    n_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Azimuthally averaged axial velocity in unit radial bins ``[k, k+1)``."""
    r = rr[fluid]
    u = ux[fluid].to(torch.float64)
    idx = torch.clamp(r.floor().long(), max=n_bins - 1)
    sums = torch.zeros(n_bins, dtype=torch.float64)
    counts = torch.zeros(n_bins, dtype=torch.float64)
    sums.index_add_(0, idx, u)
    counts.index_add_(0, idx, torch.ones_like(u))
    counts = torch.clamp(counts, min=1.0)
    centres = torch.arange(n_bins, dtype=torch.float64) + 0.5
    return centres, sums / counts


def _profile_derivative(coord: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """|dvalues/dcoord| on a uniformly spaced profile (one-sided at the ends)."""
    g = torch.zeros_like(values)
    if len(values) >= 3:
        g[1:-1] = torch.abs(0.5 * (values[2:] - values[:-2]))
        g[0] = abs(float(values[1] - values[0]))
        g[-1] = abs(float(values[-1] - values[-2]))
    return g / max(1.0e-300, float(coord[1] - coord[0]) if len(coord) > 1 else 1.0)


def probe_plug_radius(
    coord: torch.Tensor,
    profile: torch.Tensor,
    eps: float = 0.05,
    growing: bool = False,
) -> float | None:
    """Plug radius from a velocity profile: ``|du/dr| < eps*|du/dr|_max``.

    The rigid core is where the profile is flat.  Regularisation makes the plug
    *creep* slightly, so the core boundary has to be defined relative to the
    wall shear rate: this probe returns the first radius (ascending) at which
    ``|du/dr|`` reaches ``eps`` times its maximum value (``eps = 0.05`` by
    default, i.e. a 5 % shear-rate threshold).

    The probe is a *definition*, not an estimator of truth: the benchmark
    applies the very same probe to the analytic profile so that a definitional
    bias cannot be mistaken for a solver error.

    Args:
        coord: Radial (or wall-normal) coordinates, ascending.
        profile: Velocity on ``coord``.
        eps: Threshold fraction of the maximum |du/dcoord|.
        growing: Unused placeholder for API symmetry.

    Returns:
        The crossing coordinate, or ``None`` when the profile never reaches the
        threshold.
    """
    del growing
    if len(coord) < 3:
        return None
    g = _profile_derivative(coord, profile)
    g_ref = float(g.max())
    if g_ref <= 0.0:
        return None
    target = eps * g_ref
    for k in range(1, len(g)):
        if float(g[k]) >= target and float(g[k - 1]) < target:
            v0, v1 = float(g[k - 1]), float(g[k])
            w = (target - v0) / (v1 - v0) if v1 > v0 else 0.0
            return float(coord[k - 1] + w * (coord[k] - coord[k - 1]))
    return float(coord[k]) if False else None


def measure_pipe_plug_radius(
    ux: torch.Tensor,
    rr: torch.Tensor,
    solid: torch.Tensor,
    rheo: HBRheology,
    m: float = 1.0e6,
    nu_max: float | None = None,
    eps: float = 0.05,
) -> dict:
    """Numerical plug radius of a pipe flow.

    The azimuthally averaged axial profile is built in unit radial bins, and the
    plug radius is the radius where ``|du/dr|`` first reaches ``eps`` times its
    wall value (:func:`probe_plug_radius`).  Two independent diagnostics are
    returned alongside: the constitutive crossing (where the regularised
    stress ``nu_reg(g) g`` equals ``tau_y``, meaningful in the clamped core) and
    the momentum-balance value ``R*tau_y/tau_w`` supplied by the caller.

    Args:
        ux: Axial velocity field ``(nz, ny, nx)`` (averaged over the streamwise
            direction, where a fully developed flow is invariant).
        rr: Radius field ``(n, n)`` from :func:`run_hb_pipe`.
        solid: Solid mask ``(n, n)``.
        rheo: H-B parameters.
        m, nu_max: Regularisation / clamping used by the solver.
        eps: Shear-rate threshold fraction for the plug probe.

    Returns:
        dict with ``r_c`` (threshold probe), ``r_c_const`` (constitutive
        crossing), ``gamma_yield``, the profile payload and ``r_max``.
    """
    from .herschel_bulkley import hb_constitutive_stress, yield_surface_gamma_dot

    fluid = ~solid
    ux_plane = ux.mean(dim=-1) if ux.dim() == 3 else ux
    n_bins = int(float(rr[fluid].max()).__floor__()) + 1
    centres, prof = _azimuthal_profile(ux_plane, rr, fluid, n_bins)
    keep = torch.zeros(n_bins, dtype=torch.bool)
    idx = torch.clamp(rr[fluid].floor().long(), max=n_bins - 1)
    keep.index_fill_(0, idx, True)
    centres, prof = centres[keep], prof[keep]
    gamma = _profile_derivative(centres, prof)
    stress = hb_constitutive_stress(gamma, rheo, m=m, nu_max=nu_max)
    r_c_const = None
    if rheo.tau_y > 0:
        for k in range(1, len(centres)):
            s0, s1 = float(stress[k - 1]), float(stress[k])
            if s0 < rheo.tau_y <= s1 and s1 > s0:
                w = (rheo.tau_y - s0) / (s1 - s0)
                r_c_const = float(centres[k - 1] + w * (centres[k] - centres[k - 1]))
                break
    return {
        "r_c": probe_plug_radius(centres, prof, eps=eps),
        "r_c_const": r_c_const,
        "r_max": float(rr[fluid].max()),
        "gamma_yield": yield_surface_gamma_dot(rheo, m=m, nu_max=nu_max),
        "profile_r": centres.tolist(),
        "profile_u": prof.tolist(),
        "profile_gamma": gamma.tolist(),
    }


def measure_yield_surface(
    u_profile: torch.Tensor,
    y: torch.Tensor,
    rheo: HBRheology,
    m: float = 1.0e6,
    nu_max: float | None = None,
    eps: float = 0.05,
) -> dict:
    """Yield-surface position of a 1-D (plate-normal) H-B profile.

    Same probe as :func:`measure_pipe_plug_radius` in Cartesian form: the shear
    rate is ``|du/dy|`` on the x-averaged profile (wall rows trimmed off) and the
    yield surface is the first wall-normal position, counting from the unyielded
    side, where it reaches ``eps`` times the maximum (at the moving plate).

    Args:
        u_profile: x-averaged streamwise velocity ``(ny,)``, including the two
            wall rows (trimmed internally).
        y: Cell-centre distance above the lower wall ``(ny,)``.
        rheo, m, nu_max, eps: Rheology, regularisation and probe threshold.

    Returns:
        dict with ``y_yield``, ``y_yield_const`` (constitutive crossing),
        ``gamma_yield``, the profile payload and ``u_max``.
    """
    from .herschel_bulkley import hb_constitutive_stress, yield_surface_gamma_dot

    u = u_profile[1:-1].to(torch.float64)
    yy = y[1:-1].to(torch.float64)
    g = _profile_derivative(yy, u)
    stress = hb_constitutive_stress(g, rheo, m=m, nu_max=nu_max)
    y_const = None
    if rheo.tau_y > 0:
        for k in range(1, len(yy)):
            s0, s1 = float(stress[k - 1]), float(stress[k])
            if s0 < rheo.tau_y <= s1 and s1 > s0:
                w = (rheo.tau_y - s0) / (s1 - s0)
                y_const = float(yy[k - 1] + w * (yy[k] - yy[k - 1]))
                break
    return {
        "y_yield": probe_plug_radius(yy, u, eps=eps),
        "y_yield_const": y_const,
        "gamma_yield": yield_surface_gamma_dot(rheo, m=m, nu_max=nu_max),
        "profile_y": yy.tolist(),
        "profile_u": u.tolist(),
        "profile_gamma": g.tolist(),
        "u_max": float(u.abs().max()),
    }
