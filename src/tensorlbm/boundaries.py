from __future__ import annotations

import torch

from .d2q9 import OPPOSITE, C, equilibrium, macroscopic


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
                    f"Inlet profile must have shape {tuple(ref.shape)}, "
                    f"got {tuple(profile.shape)}"
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
    f1, f2, f4, f5, f8 = f[1, :, -1], f[2, :, -1], f[4, :, -1], f[5, :, -1], f[8, :, -1]
    ux = -1.0 + (f[0, :, -1] + f2 + f4 + 2.0 * (f1 + f5 + f8)) / rho_out

    f_new = f.clone()
    f_new[3, :, -1] = f1 - (2.0 / 3.0) * rho_out * ux
    f_new[7, :, -1] = f5 + 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_out * ux
    f_new[6, :, -1] = f8 - 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_out * ux
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
    cx = c[:, 0].view(9, 1, 1).float()  # (9, 1, 1)
    cy = c[:, 1].view(9, 1, 1).float()

    # Broadcast obstacle mask over velocity directions
    mask_3d = obstacle_mask.unsqueeze(0)  # (1, ny, nx)
    f_solid = f * mask_3d  # zero outside solid

    fx = 2.0 * (cx * f_solid).sum()
    fy = 2.0 * (cy * f_solid).sum()
    return fx, fy


def apply_simple_channel_boundaries(
    f: torch.Tensor,
    u_in: float,
    wall_mask: torch.Tensor,
    obstacle_mask: torch.Tensor,
) -> torch.Tensor:
    """Minimal boundary treatment.

    Applies equilibrium inlet, zero-gradient outlet, and bounce-back on walls
    and obstacle cells.
    """
    rho, ux, uy = macroscopic(f)

    ux[:, 0] = u_in
    uy[:, 0] = 0.0
    rho[:, 0] = rho[:, 1]
    feq_in = equilibrium(rho[:, 0:1], ux[:, 0:1], uy[:, 0:1])
    f[:, :, 0] = feq_in[:, :, 0]

    f[:, :, -1] = f[:, :, -2]

    f = bounce_back_cells(f, wall_mask)
    f = bounce_back_cells(f, obstacle_mask)
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
    from .d2q9 import equilibrium, macroscopic

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
    rho_up   = rho[:, jump_col]
    rho_down = (rho_up + delta_rho).clamp(min=1e-6)
    ux_face  = u_face
    uy_face  = uy[:, jump_col]

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
    from .d2q9 import equilibrium, macroscopic

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


def nscbc_outlet_2d(
    f: torch.Tensor,
    rho_target: float = 1.0,
    sigma: float = 0.25,
    c_s: float = 1.0 / 3.0 ** 0.5,
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
    from .d2q9 import equilibrium, macroscopic

    rho, ux, uy = macroscopic(f)

    # Characteristic wave amplitude correction at right boundary
    rho_out = rho[:, -1]
    ux_out  = ux[:, -1]
    uy_out  = uy[:, -1]

    # Incoming wave amplitude L1 (pressure-relaxation)
    L1 = sigma * c_s * (rho_out - rho_target)

    # Apply correction: adjust ρ at outlet without changing velocity
    rho_corrected = (rho_out - L1).clamp(min=1e-6)

    f_new = f.clone()
    f_eq_out = equilibrium(rho_corrected, ux_out, uy_out)
    # Update only the outlet column
    f_new[:, :, -1] = f_eq_out[:, :, 0]
    return f_new
