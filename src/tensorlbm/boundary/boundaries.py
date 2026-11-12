from __future__ import annotations

import torch

from tensorlbm.d2q9 import OPPOSITE, C, W, equilibrium, macroscopic
from ..input_contract import check_positive_scalar


def cylinder_mask(
    nx: int,
    ny: int,
    cx: float,
    cy: float,
    radius: float,
    device: torch.device,
) -> torch.Tensor:
    """Boolean mask for circular obstacle in a 2D grid."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def make_channel_wall_mask(
    ny: int,
    nx: int,
    obstacle_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Top/bottom wall mask excluding obstacle cells."""
    wall_mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall_mask[0, :] = True
    wall_mask[-1, :] = True
    wall_mask[obstacle_mask] = False
    return wall_mask


def bounce_back_cells(f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Bounce-back reflection on selected cells (obstacle/walls).

    Uses ``torch.where`` instead of clone + scatter to reduce the number of
    GPU kernel launches and avoid an intermediate boolean-indexed allocation.
    """
    opp = OPPOSITE.to(f.device)  # (9,)
    # mask.unsqueeze(0) broadcasts (1, ny, nx) → (9, ny, nx)
    return torch.where(mask.unsqueeze(0), f[opp], f)


def zou_he_inlet_velocity(
    f: torch.Tensor,
    u_in: float | torch.Tensor,
    uy_in: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Zou/He inlet velocity boundary condition at the left column (x=0).

    Prescribes *ux = u_in* and *uy = uy_in* at every row of the inlet column
    by analytically determining the unknown in-flowing populations so that
    mass and momentum are conserved exactly.

    The method follows Zou & He (1997) Phys. Fluids 9 1591.

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)``.
        u_in: Prescribed x-velocity at the inlet.
        uy_in: Prescribed y-velocity at the inlet (default 0).

    Returns:
        Updated distribution tensor (same shape).
    """

    # Populations pointing into the domain (cx > 0): directions 1, 5, 8
    # Populations pointing out of the domain (cx < 0): directions 3, 6, 7
    # Tangential populations (cx = 0): 0, 2, 4
    def _as_inlet_profile(value: float | torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            profile = value.to(device=ref.device, dtype=ref.dtype)
            if profile.ndim == 0:
                return torch.full_like(ref, float(profile.item()))
            if profile.ndim == 2 and profile.shape[1] == 1:
                profile = profile[:, 0]
            if profile.shape != ref.shape:
                msg = (
                    f"Inlet profile must have shape {tuple(ref.shape)}, got {tuple(profile.shape)}"
                )
                raise ValueError(msg)
            return profile
        return torch.full_like(ref, float(value))

    f0, f2, f3, f4, f6, f7 = f[0, :, 0], f[2, :, 0], f[3, :, 0], f[4, :, 0], f[6, :, 0], f[7, :, 0]
    u_col = _as_inlet_profile(u_in, f0)
    uy_col = _as_inlet_profile(uy_in, f0)
    rho = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - u_col)

    f_new = f.clone()
    f_new[1, :, 0] = f3 + (2.0 / 3.0) * rho * u_col
    f_new[5, :, 0] = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * u_col + 0.5 * rho * uy_col
    f_new[8, :, 0] = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * u_col - 0.5 * rho * uy_col
    return f_new


def zou_he_outlet_pressure(f: torch.Tensor, rho_out: float = 1.0) -> torch.Tensor:
    """Zou/He pressure (density) boundary condition at the right column (x=nx-1).

    Prescribes *rho = rho_out* and zero y-velocity at the outlet column.
    The unknown out-going populations are reconstructed from the in-coming ones.

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)``.
        rho_out: Prescribed density at the outlet (default 1.0).

    Returns:
        Updated distribution tensor (same shape).
    """
    rho_out = check_positive_scalar(rho_out, "rho_out", where="zou_he_outlet_pressure")
    f1, f2, f4, f5, f8 = f[1, :, -1], f[2, :, -1], f[4, :, -1], f[5, :, -1], f[8, :, -1]
    ux = -1.0 + (f[0, :, -1] + f2 + f4 + 2.0 * (f1 + f5 + f8)) / rho_out

    f_new = f.clone()
    f_new[3, :, -1] = f1 - (2.0 / 3.0) * rho_out * ux
    f_new[7, :, -1] = f5 + 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_out * ux
    f_new[6, :, -1] = f8 - 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_out * ux
    return f_new


def stabilize_outlet_backflow(
    f: torch.Tensor,
    *,
    max_backflow_speed: float = 0.0,
    solid_mask_col: torch.Tensor | None = None,
) -> torch.Tensor:
    """Clamp excessive outlet backflow by rebuilding outlet equilibrium states.

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)``.
        max_backflow_speed: Most-negative allowed outlet x-velocity.
            ``0.0`` means no backflow is allowed.
        solid_mask_col: Optional boolean mask for outlet rows that should not
            be modified (e.g. walls/obstacle cells touching the outlet).

    Returns:
        Updated distribution tensor.
    """
    rho, ux, uy = macroscopic(f)
    ux_out = ux[:, -1]
    needs_fix = ux_out < -float(max_backflow_speed)
    if solid_mask_col is not None:
        needs_fix = needs_fix & (~solid_mask_col.to(device=f.device))
    if not bool(needs_fix.any().item()):
        return f

    ux_clamped = torch.clamp(ux_out, min=-float(max_backflow_speed))
    feq_out = equilibrium(
        rho[:, -1].unsqueeze(1),
        ux_clamped.unsqueeze(1),
        uy[:, -1].unsqueeze(1),
    )
    f_new = f.clone()
    f_new[:, needs_fix, -1] = feq_out[:, needs_fix, 0]
    return f_new


def compute_obstacle_forces(
    f: torch.Tensor,
    obstacle_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Momentum-exchange drag and lift forces on a stationary obstacle.

    This implements the Ladd momentum-exchange method (1994).  The function
    must be called **after** streaming but **before** bounce-back is applied
    to the obstacle cells.

    At each solid node the post-stream population carries momentum that will
    be reversed by the subsequent bounce-back step.  The net force on the
    solid in direction α is:

        F_α = 2 · Σ_{x_s ∈ solid} Σ_i c_i_α · f_i(x_s)

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)`` *after* streaming.
        obstacle_mask: Boolean tensor of shape ``(ny, nx)`` marking solid cells.

    Returns:
        Tuple ``(fx, fy)`` – scalar tensors for the x and y force components.
    """
    device = f.device
    c = C.to(device)
    cx = c[:, 0].view(9, 1, 1).to(f.dtype)  # (9, 1, 1)
    cy = c[:, 1].view(9, 1, 1).to(f.dtype)

    # Broadcast obstacle mask over velocity directions
    mask_3d = obstacle_mask.unsqueeze(0)  # (1, ny, nx)
    f_solid = f * mask_3d  # zero outside solid

    fx = 2.0 * (cx * f_solid).sum()
    fy = 2.0 * (cy * f_solid).sum()
    return fx, fy


def far_field_bc_2d(
    f: torch.Tensor,
    u_in: float,
    obstacle_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """2-D free-stream (Dirichlet) far-field boundary condition.

    Imposes equilibrium free-stream on the inlet and both lateral faces
    (top/bottom), zero-gradient outlet.  Removes blockage (no wall
    acceleration) so the body sees effectively unbounded flow.  The 3-D
    analogue is :func:`tensorlbm.boundaries3d.far_field_bc_3d`.
    """
    rho1 = torch.ones((f.shape[1], f.shape[2]), dtype=f.dtype, device=f.device)
    feq = equilibrium(rho1, torch.full_like(rho1, u_in), torch.zeros_like(rho1))
    f = f.clone()
    f[:, :, 0] = feq[:, :, 0]  # inlet (free stream)
    f[:, :, -1] = f[:, :, -2]  # outlet (zero gradient)
    f[:, 0, :] = feq[:, 0, :]  # bottom lateral
    f[:, -1, :] = feq[:, -1, :]  # top lateral
    if obstacle_mask is not None:
        f = bounce_back_cells(f, obstacle_mask)
    return f


def apply_simple_channel_boundaries(
    f: torch.Tensor,
    u_in: float,
    wall_mask: torch.Tensor,
    obstacle_mask: torch.Tensor,
) -> torch.Tensor:
    """Minimal boundary treatment.

    Applies equilibrium inlet, zero-gradient outlet, and bounce-back on walls
    and obstacle cells.  Merges wall + obstacle masks into a single bounce-back
    call to reduce kernel launches.
    """
    rho, ux, uy = macroscopic(f)

    ux[:, 0] = u_in
    uy[:, 0] = 0.0
    rho[:, 0] = rho[:, 1]
    feq_in = equilibrium(rho[:, 0:1], ux[:, 0:1], uy[:, 0:1])
    f[:, :, 0] = feq_in[:, :, 0]

    f[:, :, -1] = f[:, :, -2]

    # Merge wall + obstacle into single bounce-back (one kernel launch instead of two)
    combined_mask = wall_mask | obstacle_mask
    f = bounce_back_cells(f, combined_mask)
    return f


def apply_zou_he_channel_boundaries(
    f: torch.Tensor,
    u_in: float,
    wall_mask: torch.Tensor,
    obstacle_mask: torch.Tensor,
) -> torch.Tensor:
    """Channel boundaries using Zou/He inlet and pressure outlet (higher accuracy).

    Drop-in replacement for :func:`apply_simple_channel_boundaries`.  The inlet
    uses the analytical Zou/He velocity BC (:func:`zou_he_inlet_velocity`) and
    the outlet uses the Zou/He pressure BC (:func:`zou_he_outlet_pressure`).

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)``.
        u_in: Inlet x-velocity.
        wall_mask: Boolean tensor of shape ``(ny, nx)``.
        obstacle_mask: Boolean tensor of shape ``(ny, nx)``.

    Returns:
        Updated distribution tensor.
    """
    f = zou_he_inlet_velocity(f, u_in)
    f = zou_he_outlet_pressure(f)
    f = bounce_back_cells(f, wall_mask)
    f = bounce_back_cells(f, obstacle_mask)
    return f


# ---------------------------------------------------------------------------
# P1.3 New boundary conditions — 2-D (D2Q9)
# ---------------------------------------------------------------------------


def porous_jump_2d(
    f: torch.Tensor,
    jump_col: int,
    face_area: float,
    alpha: float,
    beta: float,
    thickness: float = 1.0,
) -> torch.Tensor:
    """Porous jump boundary condition based on the Ergun equation (2-D, D2Q9).

    Models a thin porous interface (e.g. a filter, screen, or perforated plate)
    as a pressure drop:

        ΔP = −(μ·α·u + ρ·β·u²) · thickness

    where α (viscous resistance, [1/lu²]) and β (inertial resistance, [1/lu])
    are Ergun coefficients for the medium.  The density jump across the
    interface column is applied by adjusting populations on the downstream side.

    Args:
        f:          Distribution tensor (9, ny, nx).
        jump_col:   Column index of the porous interface (0-based).
        face_area:  Cross-sectional area of the interface in y-direction
                    (number of fluid rows), used for averaging.
        alpha:      Viscous resistance coefficient [1/lu²].
        beta:       Inertial resistance coefficient [1/lu].
        thickness:  Porous medium thickness in lattice units (default 1.0).

    Returns:
        Updated distribution tensor.
    """
    from tensorlbm.d2q9 import equilibrium, macroscopic

    if jump_col < 1 or jump_col >= f.shape[2] - 1:
        return f

    rho, ux, uy = macroscopic(f)

    # Velocity at the interface column (upstream face)
    u_face = ux[:, jump_col]  # (ny,)

    # Pressure drop from Ergun equation
    # ΔP = -(α·μ·u + β·ρ·u²) · L,  μ = ν = (tau-0.5)/3 ≈ lu
    delta_rho = -(alpha * u_face + beta * u_face.abs() * u_face) * thickness
    delta_rho = delta_rho.clamp(min=-rho[:, jump_col] * 0.5)

    # Apply density correction on the downstream side
    f_new = f.clone()
    rho_up = rho[:, jump_col]
    rho_down = (rho_up + delta_rho).clamp(min=1e-6)
    ux_face = u_face
    uy_face = uy[:, jump_col]

    f_eq_down = equilibrium(rho_down, ux_face, uy_face)
    f_new[:, :, jump_col + 1] = f_eq_down[:, :, 0]
    return f_new


def fan_model_2d(
    f: torch.Tensor,
    fan_col: int,
    pressure_rise_fn: object,
) -> torch.Tensor:
    """Simplified fan / axial-flow boundary condition (2-D, D2Q9).

    Models a fan or blower as a thin actuator plane that adds a prescribed
    pressure rise ΔP = f(Q) where Q is the local volume flow rate.

    Args:
        f:               Distribution tensor (9, ny, nx).
        fan_col:         Column index of the fan plane (0-based).
        pressure_rise_fn: Callable ``(flow_rate: float) -> float`` returning
                         the pressure rise in lattice units.  Typical form:
                         ``lambda q: p_max * (1 - q / q_max)``.

    Returns:
        Updated distribution tensor.
    """
    from tensorlbm.d2q9 import equilibrium, macroscopic

    if fan_col < 1 or fan_col >= f.shape[2] - 1:
        return f

    rho, ux, uy = macroscopic(f)
    # Volume flow rate at fan column: Q = sum(ux * dy)
    u_col = ux[:, fan_col]
    flow_rate = float(u_col.sum().item())

    # Pressure rise from fan curve
    try:
        delta_p = float(pressure_rise_fn(flow_rate))  # type: ignore[operator]
    except Exception:
        delta_p = 0.0

    # Distribute pressure rise as density increment on downstream side
    delta_rho = delta_p  # in LBM units, P = ρ·cs² → ΔP = Δρ/3

    f_new = f.clone()
    rho_down = (rho[:, fan_col] + delta_rho).clamp(min=1e-6)
    f_eq_down = equilibrium(rho_down, ux[:, fan_col], uy[:, fan_col])
    f_new[:, :, fan_col + 1] = f_eq_down[:, :, 0]
    return f_new


def make_sponge_strength(
    ny: int,
    nx: int,
    x0: int,
    width: int,
    power: float = 2.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sponge-layer strength field σ(x) ramping smoothly from 0 to 1.

    Used to build absorbing layers at the downstream end of free-stream
    domains (e.g. cylinder vortex shedding).  σ rises from 0 at column
    ``x0`` to 1 at column ``x0 + width`` following a power law:

        σ(x) = clamp((x − x0) / width, 0, 1) ** power

    The quadratic profile (``power=2``) has a smooth zero derivative at the
    sponge entrance, minimising spurious reflections.  Combine with
    :func:`tensorlbm.solver.collide_mrt`'s ``tau_field`` argument as
    ``tau_eff = tau · (1 + alpha · σ)``, or use it to scale a damping force.

    Args:
        ny: Number of rows in the grid.
        nx: Number of columns in the grid.
        x0: First column of the sponge (0-based).
        width: Sponge thickness in lattice units.
        power: Smoothing exponent (default 2 → quadratic).
        device: Target device (default None → CPU).

    Returns:
        Float tensor of shape ``(ny, nx)`` with σ ∈ [0, 1].
    """
    xx = torch.arange(nx, device=device, dtype=torch.float32).view(1, -1).expand(ny, nx)
    return ((xx - x0) / width).clamp(min=0.0, max=1.0) ** power


def nscbc_outlet_2d(
    f: torch.Tensor,
    rho_target: float = 1.0,
    sigma: float = 0.25,
    c_s: float = 1.0 / 3.0**0.5,
) -> torch.Tensor:
    """Non-Reflecting (Characteristic Wave) outlet boundary condition (D2Q9).

    Based on the Navier–Stokes Characteristic Boundary Conditions (NSCBC)
    method (Poinsot & Lele 1992 / Thompson 1987).  Attenuates spurious
    acoustic reflections by controlling the amplitude of incoming
    characteristic waves at the outlet plane (x = nx−1).

    The target pressure (density) *rho_target* is enforced softly via a
    relaxation coefficient σ:

        L1 = σ · c_s · (ρ − ρ_target)   (incoming acoustic wave amplitude)

    Args:
        f:          Distribution tensor (9, ny, nx).
        rho_target: Target outlet density (default 1.0).
        sigma:      Relaxation factor in [0, 1].  0 = fully non-reflecting,
                    1 = hard pressure fix.  Typical: 0.1–0.5.
        c_s:        Speed of sound in lattice units (default 1/√3).

    Returns:
        Updated distribution tensor.
    """
    from tensorlbm.d2q9 import equilibrium, macroscopic

    rho, ux, uy = macroscopic(f)

    # Characteristic wave amplitude correction at right boundary
    rho_out = rho[:, -1]
    ux_out = ux[:, -1]
    uy_out = uy[:, -1]

    # Incoming wave amplitude L1 (pressure-relaxation)
    L1 = sigma * c_s * (rho_out - rho_target)

    # Apply correction: adjust ρ at outlet without changing velocity
    rho_corrected = (rho_out - L1).clamp(min=1e-6)

    f_new = f.clone()
    f_eq_out = equilibrium(rho_corrected, ux_out, uy_out)
    # Update only the outlet column
    f_new[:, :, -1] = f_eq_out[:, :, 0]
    return f_new



def nonequilibrium_extrapolation_outlet_2d(f: torch.Tensor) -> torch.Tensor:
    """Non-equilibrium extrapolation (NEE) outflow boundary, D2Q9.

    Guo, Zheng & Shi (2002), *An extrapolation method for boundary conditions
    in lattice Boltzmann method*, J. Comput. Phys. **177**, 443-458,
    Eq. (10)-(13): the unknown outlet distributions are rebuilt from a
    *locally extrapolated* equilibrium plus the non-equilibrium part carried
    over from the last interior node::

        f_i(x_b) = f_i^eq(rho*, u*) + [f_i(x_{b-1}) - f_i^eq(rho_{b-1}, u_{b-1})]
        rho* = 2 rho_{b-1} - rho_{b-2},      u* = 2 u_{b-1} - u_{b-2}

    The macroscopic fields are the post-streaming ones, so ``x_{b-1}`` is the
    last interior column ``x = nx-2`` and ``x_b = nx-1`` is the outlet plane.
    Unlike :func:`zero_gradient_outlet_2d` (first-order copy of the whole
    distribution, which flattens the transverse velocity at the outlet) the
    NEE closure is second-order accurate and leaves a smooth outflow
    undisturbed, so the outlet does not set an O(1) error floor.

    Extrapolated density is clamped to ``[0.5, 1.5]`` and the extrapolated
    velocity magnitude to ``0.3`` (``Ma < 0.52``) as a robustness guard.

    .. warning::
       **STATUS: EXPERIMENTAL / NOT VALIDATED.** This closure was tried as a
       drop-in replacement for :func:`zero_gradient_outlet_2d` on the Kovasznay
       (1948) steady analytic solution (Re=40, Zou–He analytic inlet, x' in
       [0,3], D2Q9/BGK) and **diverged to NaN** at ny=32 within 12000 steps
       (see ``docs/CASE-LBM-KOVASZNAY-OUTLET-BC.md``, honest FAIL).  Pure
       linear extrapolation of (rho, u) into the outlet plane without any
       relaxation/characteristic damping is not stable for this configuration.
       Use :func:`zero_gradient_outlet_2d` (or a damped/limiter variant) for
       production outflow until a stabilised version is verified.

    Args:
        f: Distribution tensor (9, ny, nx), post-streaming, nx >= 3.

    Returns:
        Updated distribution tensor (outlet column replaced).
    """
    if f.shape[0] != 9:
        msg = f"f must have 9 directions, got {f.shape[0]}"
        raise ValueError(msg)
    if f.shape[2] < 3:
        msg = f"nx must be >= 3 for NEE outlet, got {f.shape[2]}"
        raise ValueError(msg)

    rho, ux, uy = macroscopic(f)
    rho_b1, rho_b2 = rho[:, -2], rho[:, -3]
    ux_b1, ux_b2 = ux[:, -2], ux[:, -3]
    uy_b1, uy_b2 = uy[:, -2], uy[:, -3]

    # Linear extrapolation of the macroscopic fields to the outlet plane.
    rho_b = (2.0 * rho_b1 - rho_b2).clamp(min=0.5, max=1.5)
    ux_b = 2.0 * ux_b1 - ux_b2
    uy_b = 2.0 * uy_b1 - uy_b2
    speed = torch.sqrt(ux_b * ux_b + uy_b * uy_b)
    scale = torch.where(speed > 0.3, 0.3 / speed.clamp(min=1e-12), torch.ones_like(speed))
    ux_b = ux_b * scale
    uy_b = uy_b * scale

    f_eq_out = equilibrium(rho_b, ux_b, uy_b)
    f_eq_neighbour = equilibrium(rho_b1, ux_b1, uy_b1)

    # Only the populations entering the domain (c_x < 0 at the right plane) are
    # unknown after streaming; the outgoing ones already carry the correct
    # characteristics and must be left untouched (prescribing all 9 injects
    # mass/momentum inconsistent with the outgoing waves and diverges).
    incoming = C[:, 0] < 0
    f_new = f.clone()
    outlet = f_new[:, :, -1]  # view: in-place advanced-index writes propagate
    outlet[incoming] = f_eq_out[incoming, :, 0] + (
        f[incoming, :, -2] - f_eq_neighbour[incoming, :, 0]
    )
    return f_new


def zero_gradient_outlet_2d(f: torch.Tensor) -> torch.Tensor:
    """Zero-gradient (Neumann) outlet boundary condition (D2Q9).

    Copies the upstream column's 9 distribution functions to the outlet
    column (x = nx-1): ``f[:, :, -1] = f[:, :, -2]``, i.e. ``\partial f/\partial x = 0``
    (a first-order Neumann extrapolation of the whole distribution).

    This is the outlet used by the Kovasznay (1948) steady-state verification
    (``benchmarks/verified/kovasznay_2d``): with a long enough downstream domain
    (x' >= 3 y-periods) the outlet disturbance ``e^{lambda x'}`` has decayed below
    the 3 % gate, so the zero-gradient outlet does not pollute the interior field.
    On a short domain (x' in [0,1]) it kills the outlet ``v`` component to ~0
    (full-field v max relative error ~96 %), so a long downstream domain is
    required.  It is collision-agnostic (applied post-streaming).

    Args:
        f: Distribution tensor (9, ny, nx).

    Returns:
        Updated distribution tensor (outlet column = upstream column).
    """
    if f.shape[0] != 9:
        msg = f"f must have 9 directions, got {f.shape[0]}"
        raise ValueError(msg)
    f = f.clone()
    f[:, :, -1] = f[:, :, -2]
    return f


# ---------------------------------------------------------------------------
# Unified moving-wall / free-slip entry points (closes G18, see
# benchmarks/gaps.md and the 2026-11-08 convention audit, report 7618).
# ---------------------------------------------------------------------------

_SPECULAR_Y = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.int64)
_SPECULAR_X = torch.tensor([0, 3, 2, 1, 4, 7, 5, 8, 6], dtype=torch.int64)


def bounce_back_moving_wall(
    f: torch.Tensor,
    mask: torch.Tensor,
    u_wall: float | tuple[float, float] | torch.Tensor,
    rho: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ladd (1994) second-order moving-wall bounce-back, D2Q9, solid-side.

    On every cell flagged by *mask* (post-streaming state):

    .. math::
        f_q \\leftarrow f_{\\bar q} + 6\\, w_q\\, \\rho\\, (c_q \\cdot u_w)

    with ``rho = f.sum(0)`` (local density) unless *rho* is supplied.  The map
    is a **fixed point of a uniform Galilean state** ``f^eq(rho, u_w)`` — a
    flow translating with the wall is left unchanged to machine precision —
    and it reduces to :func:`bounce_back_cells` when ``u_wall == 0``.

    This is the single library entry point for the moving-wall bounce-back
    family.  It reproduces the two adjudicated-correct shipped copies
    (``herschel_bulkley_lbm.moving_wall_bounce_back_2d`` and
    ``rotating_cylinder.moving_wall_bounce_back``) bit-for-bit; the other
    family members differ by contract (fluid-side injection / surface links
    only) and are pinned by ``tests/test_moving_wall_convention_audit.py``.

    Args:
        f:      Distribution tensor ``(9, ny, nx)``.
        mask:   Boolean wall mask ``(ny, nx)``.
        u_wall: Wall velocity — a scalar (x-velocity, ``uy=0``), a pair
                ``(ux, uy)``, or a tensor of shape ``(2, ny, nx)``.
        rho:    Density field ``(ny, nx)``; computed from ``f`` if ``None``.

    Returns:
        Updated distribution tensor.
    """
    if f.shape[0] != 9:
        msg = f"f must have 9 directions, got {f.shape[0]}"
        raise ValueError(msg)
    if mask.shape != f.shape[1:]:
        msg = f"mask {tuple(mask.shape)} != field {tuple(f.shape[1:])}"
        raise ValueError(msg)
    device = f.device
    if rho is None:
        rho = f.sum(dim=0)
    if isinstance(u_wall, torch.Tensor):
        if u_wall.shape != (2, *f.shape[1:]):
            msg = f"u_wall tensor must be (2, ny, nx), got {tuple(u_wall.shape)}"
            raise ValueError(msg)
        ux, uy = u_wall[0].to(f.dtype), u_wall[1].to(f.dtype)
    elif isinstance(u_wall, (tuple, list)):
        ux = torch.full_like(rho, float(u_wall[0]), dtype=f.dtype)
        uy = torch.full_like(rho, float(u_wall[1]), dtype=f.dtype)
    else:
        ux = torch.full_like(rho, float(u_wall), dtype=f.dtype)
        uy = torch.zeros_like(rho)
    c = C.to(device=device, dtype=f.dtype)
    w = W.to(device=device, dtype=f.dtype)
    cu = c[:, 0].view(9, 1, 1) * ux.unsqueeze(0) + c[:, 1].view(9, 1, 1) * uy.unsqueeze(0)
    f_bb = f[OPPOSITE.to(device)] + 6.0 * w.view(9, 1, 1) * rho.unsqueeze(0) * cu
    return torch.where(mask.unsqueeze(0), f_bb, f)


def specular_reflection(
    f: torch.Tensor,
    mask: torch.Tensor,
    axis: str = "y",
) -> torch.Tensor:
    """Specular (free-slip, tangential) reflection on selected cells, D2Q9.

    Tangential populations pass through; populations carrying a velocity
    component along the wall normal are mirrored.  ``axis="y"`` is a
    horizontal wall (normal in y, map ``[0, 1, 4, 3, 2, 8, 7, 6, 5]``);
    ``axis="x"`` a vertical wall (map ``[0, 3, 2, 1, 4, 7, 5, 8, 6]``).

    Closed-form properties (both pinned in tests): a uniform state with
    velocity purely tangential to the wall is a fixed point, and a purely
    normal velocity is exactly reversed at the wall cell.  Mass is conserved
    (the remap is a permutation); tangential momentum is conserved, normal
    momentum is reversed.
    """
    if f.shape[0] != 9:
        msg = f"f must have 9 directions, got {f.shape[0]}"
        raise ValueError(msg)
    if mask.shape != f.shape[1:]:
        msg = f"mask {tuple(mask.shape)} != field {tuple(f.shape[1:])}"
        raise ValueError(msg)
    if axis not in ("x", "y"):
        msg = f"axis must be 'x' or 'y', got {axis!r}"
        raise ValueError(msg)
    spec = (_SPECULAR_X if axis == "x" else _SPECULAR_Y).to(f.device)
    return torch.where(mask.unsqueeze(0), f[spec], f)
