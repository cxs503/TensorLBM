"""Momentum Exchange Method (MEM) variants for D3Q19 LBM drag computation.

Implements five force-measurement strategies for stationary walls:

1. **Standard MEM** (Ladd 1994) — link-wise momentum exchange using
   post-streaming distributions at the solid-fluid interface.

2. **Galilean-invariant MEM** (Lorenz 2014 / Caiazzo 2007) — corrects the
   Galilean-invariance violation of the standard MEM by weighting each
   link with ``(1 + 1/(2τ))``.  More accurate for moving walls and
   non-zero background velocity.

2b. **Background-subtracted MEM** — Ladd sum with the uniform free-stream
   equilibrium background removed.  On curved staircase surfaces the
   discrete surface is not closed (Σ n̂ dA ≠ 0) so the free-stream
   equilibrium background of the Ladd sum does NOT cancel and produces a
   spurious force; subtracting the closed-form f^eq background fixes it.

3. **BFL MEM** (Yu 2003 / Bouzidi-Firdaouss-Lallemand 2001) — momentum
   exchange weighted by the fractional wall distance *q* for interpolated
   bounce-back boundaries.

4. **Stress integration** — surface integral of the viscous shear stress
   ``F = ∮ τ · n dA``.  Delegates to :func:`drag_pressure.drag_friction_integration`.

5. **Pressure + friction integration** — the legacy control-surface method
   ``F = ∮ p·n dA + ∮ τ·n dA``.  Delegates to
   :func:`drag_pressure.drag_pressure_integration` +
   :func:`drag_pressure.drag_friction_integration`.

All MEM variants return the **force on the wall** (drag-positive convention:
a positive *x*-component means drag opposing the flow).

References
----------
Ladd, A. J. C. (1994). "Numerical simulations of particulate suspensions
    via a discretized Boltzmann equation. Part I." *J. Fluid Mech.* 271.
Lorenz, E. (2014). "Towards Galilean-invariant moving-boundary LBM."
    *Comput. Math. Appl.* 67(2).
Yu, D.; Mei, R.; Luo, L.-S.; Shyy, W. (2003). "Viscous flow computation
    with the method of lattice Boltzmann equation." *Prog. Aero. Sci.* 39.
Caiazzo, A. (2007). "Analysis of correction in Galilean-invariant LBM."
    *J. Comput. Phys.* 225(2).
"""
from __future__ import annotations

import torch

from .d3q19 import C, OPPOSITE, W


# ---------------------------------------------------------------------------
# Helper: build the fluid→solid crossing mask for a given direction
# ---------------------------------------------------------------------------
def _crossing_mask(
    near: torch.Tensor,
    solid: torch.Tensor,
    cqx: int,
    cqy: int,
    cqz: int,
) -> torch.Tensor:
    """Boolean mask of near-wall fluid cells whose c_q-neighbour is solid.

    ``solid_shifted[x] = solid[x + c_q]``  →  True where the c_q-neighbour
    of a near-wall fluid cell is solid (a fluid→solid link in direction q).
    """
    solid_shifted = torch.roll(solid, (-cqz, -cqy, -cqx), dims=(0, 1, 2))
    return near & solid_shifted


# ---------------------------------------------------------------------------
# 1. Standard MEM (Ladd 1994)
# ---------------------------------------------------------------------------
def momentum_exchange_standard(
    f: torch.Tensor,
    solid: torch.Tensor,
    near: torch.Tensor,
) -> tuple[float, float, float]:
    """Standard momentum-exchange force (Ladd 1994) for D3Q19.

    For each near-wall fluid cell *x_f* and each direction *q* where the
    neighbour ``x_f + c_q`` is solid:

        F += (f_q(x_f) + f_{opp_q}(x_s)) · c_q

    where ``x_s = x_f + c_q`` is the solid neighbour.  The sum of the
    incoming population (moving toward the wall) and the bounced-back
    population (moving away) gives the total momentum transferred to the
    wall per link.

    This is the **sum form** of Ladd's MEM, which includes the
    equilibrium background.  For flat walls the equilibrium contributions
    cancel across opposite-direction pairs, giving exact results.

    Args:
        f:     Distribution tensor ``(19, nz, ny, nx)`` — post-streaming.
        solid: Boolean solid mask ``(nz, ny, nx)``.
        near:  Near-wall fluid mask ``(nz, ny, nx)``.

    Returns:
        ``(fx, fy, fz)`` — force on the wall (drag-positive).
    """
    device = f.device
    c = C.to(device).float()
    opp = OPPOSITE.to(device)

    fx = torch.tensor(0.0, device=device, dtype=f.dtype)
    fy = torch.tensor(0.0, device=device, dtype=f.dtype)
    fz = torch.tensor(0.0, device=device, dtype=f.dtype)

    for i in range(1, 19):
        opp_i = int(opp[i].item())
        ci = c[i]
        di = int(ci[0].item())
        dj = int(ci[1].item())
        dk = int(ci[2].item())

        crossing = _crossing_mask(near, solid, di, dj, dk)
        if not crossing.any():
            continue

        # f_opp at the solid neighbour (shifted from fluid cell)
        f_opp_solid = torch.roll(f[opp_i], (-dk, -dj, -di), dims=(0, 1, 2))

        # Ladd sum: (f_i[fluid] + f_opp_i[solid]) * c_i
        contrib = ((f[i] + f_opp_solid) * crossing.float()).sum()
        fx = fx + float(ci[0].item()) * contrib
        fy = fy + float(ci[1].item()) * contrib
        fz = fz + float(ci[2].item()) * contrib

    return float(fx.item()), float(fy.item()), float(fz.item())


# ---------------------------------------------------------------------------
# 2. Galilean-invariant MEM (Lorenz 2014)
# ---------------------------------------------------------------------------
def momentum_exchange_galilean(
    f: torch.Tensor,
    solid: torch.Tensor,
    near: torch.Tensor,
    tau: float,
) -> tuple[float, float, float]:
    """Galilean-invariant momentum-exchange force (Lorenz 2014).

    Corrects the Galilean-invariance violation of the standard MEM by
    weighting each link with the factor ``(1 + 1/(2τ))``:

        F += (f_q + f_{opp_q}) · c_q · (1 + 1/(2τ))

    The standard MEM violates Galilean invariance because it uses
    post-streaming populations that contain both pre- and post-collision
    contributions.  The correction factor accounts for the missing
    pre-collision contribution, making the result invariant under
    Galilean boosts (constant velocity shifts).

    More accurate for:
      - Moving walls (Couette with non-zero wall velocity)
      - Flows with strong background velocity (channel, external flow)
      - High Reynolds numbers where the equilibrium background is large

    Args:
        f:     Distribution tensor ``(19, nz, ny, nx)`` — post-streaming.
        solid: Boolean solid mask ``(nz, ny, nx)``.
        near:  Near-wall fluid mask ``(nz, ny, nx)``.
        tau:   Relaxation time (must match the collision operator).

    Returns:
        ``(fx, fy, fz)`` — Galilean-corrected force on the wall.
    """
    device = f.device
    c = C.to(device).float()
    opp = OPPOSITE.to(device)

    # Galilean correction factor
    gi_factor = 1.0 + 1.0 / (2.0 * tau)

    fx = torch.tensor(0.0, device=device, dtype=f.dtype)
    fy = torch.tensor(0.0, device=device, dtype=f.dtype)
    fz = torch.tensor(0.0, device=device, dtype=f.dtype)

    for i in range(1, 19):
        opp_i = int(opp[i].item())
        ci = c[i]
        di = int(ci[0].item())
        dj = int(ci[1].item())
        dk = int(ci[2].item())

        crossing = _crossing_mask(near, solid, di, dj, dk)
        if not crossing.any():
            continue

        f_opp_solid = torch.roll(f[opp_i], (-dk, -dj, -di), dims=(0, 1, 2))

        contrib = ((f[i] + f_opp_solid) * crossing.float()).sum() * gi_factor
        fx = fx + float(ci[0].item()) * contrib
        fy = fy + float(ci[1].item()) * contrib
        fz = fz + float(ci[2].item()) * contrib

    return float(fx.item()), float(fy.item()), float(fz.item())


# ---------------------------------------------------------------------------
# 2b. Uniform-background-subtracted MEM (free-stream equilibrium removal)
# ---------------------------------------------------------------------------
def momentum_exchange_background_subtracted(
    f: torch.Tensor,
    solid: torch.Tensor,
    near: torch.Tensor,
    rho0: float = 1.0,
    u0: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    """Ladd-sum MEM with the uniform free-stream equilibrium background removed.

    The standard Ladd sum form ``F = Σ_q (f_q + f_opp) · c_q`` contains the
    equilibrium background of the free stream.  On a flat wall the background
    cancels across opposite link pairs, but on a curved staircase surface the
    discrete surface is not closed (``Σ n̂ dA ≠ 0``), so the background does
    NOT cancel and produces a spurious force (G15: +268% on sphere Re=100).

    For a uniform flow ``f = f^eq(ρ₀, U₀)`` the per-link background is exact:

        f^eq_q(ρ₀,U₀) + f^eq_opp(ρ₀,U₀)
            = 2·ρ₀·w_q·[1 + 4.5·(c_q·U₀)² − 1.5·|U₀|²]

    (the linear term cancels because ``c_opp = −c_q``).  Subtracting this
    closed-form background from the Ladd sum removes the spurious free-stream
    force:

        F_corrected = F_std − Σ_q 2·ρ₀·w_q·[1 + 4.5·(c_q·U₀)² − 1.5·|U₀|²]·c_q

    Args:
        f:     Distribution tensor ``(19, nz, ny, nx)`` — post-streaming.
        solid: Boolean solid mask ``(nz, ny, nx)``.
        near:  Near-wall fluid mask ``(nz, ny, nx)``.
        rho0:  Free-stream density (lattice units, ≈ 1.0).
        u0:    Free-stream velocity ``(u0x, u0y, u0z)`` in lattice units.

    Returns:
        ``(fx, fy, fz)`` — background-corrected force on the wall.
    """
    device = f.device
    c = C.to(device).float()
    w = W.to(device).float()
    opp = OPPOSITE.to(device)

    u0x, u0y, u0z = u0
    u0sq = u0x * u0x + u0y * u0y + u0z * u0z

    fx = torch.tensor(0.0, device=device, dtype=f.dtype)
    fy = torch.tensor(0.0, device=device, dtype=f.dtype)
    fz = torch.tensor(0.0, device=device, dtype=f.dtype)

    for i in range(1, 19):
        opp_i = int(opp[i].item())
        ci = c[i]
        di = int(ci[0].item())
        dj = int(ci[1].item())
        dk = int(ci[2].item())

        crossing = _crossing_mask(near, solid, di, dj, dk)
        if not crossing.any():
            continue

        # Ladd sum contribution
        f_opp_solid = torch.roll(f[opp_i], (-dk, -dj, -di), dims=(0, 1, 2))
        contrib = ((f[i] + f_opp_solid) * crossing.float()).sum()

        # Uniform free-stream equilibrium background on the same crossing set
        cdotu = float(ci[0].item()) * u0x + float(ci[1].item()) * u0y + float(ci[2].item()) * u0z
        bg = 2.0 * rho0 * float(w[i].item()) * (1.0 + 4.5 * cdotu * cdotu - 1.5 * u0sq)
        bg_contrib = bg * float(crossing.sum().item())

        net = contrib - bg_contrib
        fx = fx + float(ci[0].item()) * net
        fy = fy + float(ci[1].item()) * net
        fz = fz + float(ci[2].item()) * net

    return float(fx.item()), float(fy.item()), float(fz.item())


# ---------------------------------------------------------------------------
# 3. BFL MEM (Yu 2003)
# ---------------------------------------------------------------------------
def momentum_exchange_bfl(
    f: torch.Tensor,
    solid: torch.Tensor,
    near: torch.Tensor,
    q_wall: torch.Tensor,
) -> tuple[float, float, float]:
    """BFL-interpolated momentum-exchange force (Yu 2003).

    For interpolated (Bouzidi-Firdaouss-Lallemand) bounce-back, the wall
    lies at a fractional distance *q* between the fluid and solid cells.
    The momentum exchange is weighted by *q* to account for the
    non-integer wall position:

        F += (f_q + f_{opp_q}) · c_q · (1 / q)

    For ``q = 0.5`` (half-way BB, flat wall) this reduces to the standard
    MEM with a factor of 2, consistent with the half-way bounce-back
    interpretation.

    Args:
        f:      Distribution tensor ``(19, nz, ny, nx)`` — post-streaming.
        solid:  Boolean solid mask ``(nz, ny, nx)``.
        near:   Near-wall fluid mask ``(nz, ny, nx)``.
        q_wall: Fractional wall distance.  Either a scalar field
                ``(nz, ny, nx)`` in [0, 1] used for all directions, or a
                per-direction field ``(19, nz, ny, nx)`` matching the
                lattice stencil (the physically correct BFL form — each
                link has its own fractional intersection distance).
                Typically 0.5 for flat walls; varies for curved surfaces.

    Returns:
        ``(fx, fy, fz)`` — BFL-weighted force on the wall.
    """
    device = f.device
    c = C.to(device).float()
    opp = OPPOSITE.to(device)

    # Support both scalar (nz,ny,nx) and per-direction (19,nz,ny,nx) q fields.
    per_direction = q_wall.dim() == 4 and q_wall.shape[0] == 19
    if per_direction:
        # Per-direction inverse q, clamped to avoid division by zero.
        inv_q = 1.0 / q_wall.clamp(min=1e-6)  # (19, nz, ny, nx)
    else:
        inv_q = 1.0 / q_wall.clamp(min=1e-6)  # (nz, ny, nx)

    fx = torch.tensor(0.0, device=device, dtype=f.dtype)
    fy = torch.tensor(0.0, device=device, dtype=f.dtype)
    fz = torch.tensor(0.0, device=device, dtype=f.dtype)

    for i in range(1, 19):
        opp_i = int(opp[i].item())
        ci = c[i]
        di = int(ci[0].item())
        dj = int(ci[1].item())
        dk = int(ci[2].item())

        crossing = _crossing_mask(near, solid, di, dj, dk)
        if not crossing.any():
            continue

        f_opp_solid = torch.roll(f[opp_i], (-dk, -dj, -di), dims=(0, 1, 2))

        # Weight by 1/q at each crossing cell.
        if per_direction:
            weight = crossing.float() * inv_q[i]
        else:
            weight = (crossing.float() * inv_q)
        contrib = ((f[i] + f_opp_solid) * weight).sum()
        fx = fx + float(ci[0].item()) * contrib
        fy = fy + float(ci[1].item()) * contrib
        fz = fz + float(ci[2].item()) * contrib

    return float(fx.item()), float(fy.item()), float(fz.item())


# ---------------------------------------------------------------------------
# 4. Stress integration (delegates to drag_pressure)
# ---------------------------------------------------------------------------
def momentum_exchange_stress(
    f: torch.Tensor,
    mesh,
    dpS: float,
    nu: float,
    formula: str = "standard",
    q_wall: torch.Tensor | None = None,
) -> tuple[float, float, float]:
    """Viscous stress integration: F = ∮ τ · n dA.

    Computes the friction drag by integrating the wall shear stress over
    the body surface.  This is the control-surface friction method,
    not a momentum-exchange method, but is included here for comparison.

    Delegates to :func:`tensorlbm.drag_pressure.drag_friction_integration`.

    Args:
        f:       Distribution tensor ``(19, nz, ny, nx)``.
        mesh:    SurfaceMesh with normals and near-wall mask.
        dpS:     Normalisation factor (dynamic pressure × reference area).
        nu:      Kinematic viscosity in lattice units.
        formula: Friction formula: 'standard', '2nd_order', 'central',
                 'lagrange', or 'bfl'.
        q_wall:  Required if formula='bfl'.

    Returns:
        ``(fx, fy, fz)`` — friction drag coefficient components.
    """
    from .drag_pressure import drag_friction_integration
    return drag_friction_integration(f, mesh, dpS, nu, q_wall=q_wall,
                                     formula=formula)


# ---------------------------------------------------------------------------
# 5. Pressure + friction integration (legacy control-surface method)
# ---------------------------------------------------------------------------
def momentum_exchange_pressure_friction(
    f: torch.Tensor,
    mesh,
    dpS: float,
    nu: float,
    extrap: str = "none",
    p0_method: str = "near_wall",
    solid: torch.Tensor | None = None,
    friction_formula: str = "standard",
    q_wall: torch.Tensor | None = None,
) -> dict[str, float]:
    """Pressure + friction integration (legacy control-surface method).

    F_total = ∮ p·n dA + ∮ τ·n dA

    Delegates to :func:`tensorlbm.drag_pressure.drag_pressure_integration`
    and :func:`tensorlbm.drag_pressure.drag_friction_integration`.

    Args:
        f:               Distribution tensor ``(19, nz, ny, nx)``.
        mesh:            SurfaceMesh with normals and near-wall mask.
        dpS:             Normalisation factor.
        nu:              Kinematic viscosity in lattice units.
        extrap:          Wall-pressure extrapolation: 'none', 'linear',
                         'quadratic'.
        p0_method:       Background pressure method: 'near_wall',
                         'far_field', 'domain_avg', 'inlet'.
        solid:           Solid mask (required for non-'near_wall' p0).
        friction_formula: Friction formula: 'standard', '2nd_order',
                         'central', 'lagrange', 'bfl'.
        q_wall:          Required if friction_formula='bfl'.

    Returns:
        Dict with keys: ``cd_p_x, cd_p_y, cd_p_z, cd_f_x, cd_f_y,
        cd_f_z, cd_tot_x, cd_tot_y, cd_tot_z``.
    """
    from .drag_pressure import drag_pressure_integration, drag_friction_integration

    px, py, pz = drag_pressure_integration(
        f, mesh, dpS, extrap=extrap, p0_method=p0_method, solid=solid
    )
    fx, fy, fz = drag_friction_integration(
        f, mesh, dpS, nu, q_wall=q_wall, formula=friction_formula
    )
    return {
        "cd_p_x": px, "cd_p_y": py, "cd_p_z": pz,
        "cd_f_x": fx, "cd_f_y": fy, "cd_f_z": fz,
        "cd_tot_x": px + fx, "cd_tot_y": py + fy, "cd_tot_z": pz + fz,
    }


# ---------------------------------------------------------------------------
# Comparison: all methods side-by-side
# ---------------------------------------------------------------------------
def compare_all_methods(
    f: torch.Tensor,
    solid: torch.Tensor,
    near: torch.Tensor,
    mesh,
    dpS: float,
    nu: float,
    tau: float,
    q_wall: torch.Tensor | None = None,
    extrap: str = "none",
    p0_method: str = "near_wall",
    friction_formula: str = "standard",
) -> dict[str, float]:
    """Compare all force-measurement methods on the same distribution.

    Computes the x-component (drag) from each method and returns them
    in a single dictionary for easy comparison.

    Args:
        f:        Distribution tensor ``(19, nz, ny, nx)`` — post-streaming.
        solid:    Boolean solid mask.
        near:     Near-wall fluid mask.
        mesh:     SurfaceMesh for pressure/friction integration.
        dpS:      Normalisation factor.
        nu:       Kinematic viscosity (lattice units).
        tau:      Relaxation time.
        q_wall:   Fractional wall distance (for BFL MEM).
        extrap:   Pressure extrapolation method.
        p0_method: Background pressure method.
        friction_formula: Friction formula.

    Returns:
        Dict with keys:
        ``cd_mem_standard, cd_mem_galilean, cd_mem_bfl,
        cd_stress, cd_pressure, cd_friction, cd_pf_total``.
    """
    # MEM variants (raw force, normalised by dpS)
    me_std = momentum_exchange_standard(f, solid, near)
    me_gal = momentum_exchange_galilean(f, solid, near, tau)

    if q_wall is not None:
        me_bfl = momentum_exchange_bfl(f, solid, near, q_wall)
        cd_bfl = me_bfl[0] / dpS
    else:
        cd_bfl = float("nan")

    # Stress integration (friction only)
    cd_stress = momentum_exchange_stress(
        f, mesh, dpS, nu, formula=friction_formula, q_wall=q_wall
    )[0]

    # Pressure + friction
    pf = momentum_exchange_pressure_friction(
        f, mesh, dpS, nu,
        extrap=extrap, p0_method=p0_method, solid=solid,
        friction_formula=friction_formula, q_wall=q_wall,
    )

    return {
        "cd_mem_standard": me_std[0] / dpS,
        "cd_mem_galilean": me_gal[0] / dpS,
        "cd_mem_bfl": cd_bfl,
        "cd_stress": cd_stress,
        "cd_pressure": pf["cd_p_x"],
        "cd_friction": pf["cd_f_x"],
        "cd_pf_total": pf["cd_tot_x"],
    }


__all__ = [
    "momentum_exchange_standard",
    "momentum_exchange_galilean",
    "momentum_exchange_background_subtracted",
    "momentum_exchange_bfl",
    "momentum_exchange_stress",
    "momentum_exchange_pressure_friction",
    "compare_all_methods",
]
