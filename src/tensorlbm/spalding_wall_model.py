"""Off-lattice Spalding wall model with an exchange-location velocity.

The model follows the reusable algorithm documented by OpenLB:

1. apply the curved-wall population reconstruction separately;
2. sample velocity at an exchange location ``y2`` in the fluid;
3. solve Spalding's unified wall law for friction velocity ``u_tau``;
4. evaluate the modelled tangential velocity at the boundary node ``y1``;
5. assimilate that velocity while retaining a configurable fraction of the
   local non-equilibrium populations.

It is geometry- and collision-agnostic and operates only on sparse boundary
nodes.  D3Q19 and D3Q27 are supported.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


SPALDING_KAPPA = 0.4
SPALDING_A = 0.1108


def spalding_y_plus(u_plus: torch.Tensor) -> torch.Tensor:
    """Spalding unified law ``y+ = f(u+)``."""
    ku = SPALDING_KAPPA * u_plus.clamp(min=0.0, max=60.0)
    return u_plus + SPALDING_A * (
        torch.exp(ku) - 1.0 - ku - 0.5 * ku.square() - ku.pow(3) / 6.0
    )


def spalding_u_plus_from_y_plus(y_plus: torch.Tensor, iterations: int = 40) -> torch.Tensor:
    """Invert Spalding's monotone law by vectorised bisection."""
    target = y_plus.clamp_min(0.0)
    low = torch.zeros_like(target)
    high = torch.full_like(target, 60.0)
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        below = spalding_y_plus(mid) < target
        low = torch.where(below, mid, low)
        high = torch.where(below, high, mid)
    return 0.5 * (low + high)


def solve_spalding_friction_velocity(
    tangential_speed: torch.Tensor,
    wall_distance: torch.Tensor | float,
    nu: float,
    *,
    iterations: int = 48,
) -> torch.Tensor:
    """Solve ``u2/u_tau`` and ``y2*u_tau/nu`` on the Spalding curve."""
    if nu <= 0.0:
        raise ValueError("nu must be positive")
    speed = tangential_speed.clamp_min(0.0)
    distance = torch.as_tensor(
        wall_distance, dtype=speed.dtype, device=speed.device,
    ).expand_as(speed)
    if bool((distance <= 0.0).any()):
        raise ValueError("wall_distance must be positive")
    active = speed > 1e-14
    # g(u_tau)=Spalding(u/u_tau)-y*u_tau/nu decreases monotonically.
    low = torch.full_like(speed, 1e-12)
    high = torch.maximum(speed, torch.sqrt(nu * speed / distance).clamp_min(1e-10))
    # The physical root lies below u for ordinary wall flows.  Expansion also
    # covers viscous low-speed states without a Newton initial-guess failure.
    for _ in range(8):
        g_high = spalding_y_plus(speed / high.clamp_min(1e-20)) - distance * high / nu
        high = torch.where(g_high > 0.0, 2.0 * high, high)
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        g_mid = spalding_y_plus(speed / mid.clamp_min(1e-20)) - distance * mid / nu
        low = torch.where(g_mid > 0.0, mid, low)
        high = torch.where(g_mid > 0.0, high, mid)
    return torch.where(active, 0.5 * (low + high), torch.zeros_like(speed))


def effective_bfl_wall_distance(
    fluid_boundary_mask: torch.Tensor,
    q_field: torch.Tensor,
    normals: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Estimate normal wall distance ``y1`` from all crossing BFL links."""
    if fluid_boundary_mask.shape != q_field.shape or fluid_boundary_mask.ndim != 4:
        raise ValueError("BFL mask and q field must share shape (Q,nz,ny,nx)")
    q_count = fluid_boundary_mask.shape[0]
    if q_count == 19:
        from .d3q19 import C
    elif q_count == 27:
        from .d3q27 import C
    else:
        raise ValueError("only D3Q19 and D3Q27 are supported")
    nx_n, ny_n, nz_n = normals
    if any(component.shape != q_field.shape[1:] for component in normals):
        raise ValueError("normal fields must have the spatial grid shape")
    c = C.to(device=q_field.device, dtype=q_field.dtype)
    projection = (
        c[:, 0, None, None, None] * nx_n
        + c[:, 1, None, None, None] * ny_n
        + c[:, 2, None, None, None] * nz_n
    ).abs()
    weights = projection * fluid_boundary_mask.to(q_field.dtype)
    denominator = weights.sum(dim=0)
    distance = (q_field * weights).sum(dim=0) / denominator.clamp_min(1e-12)
    return torch.where(denominator > 0.0, distance.clamp_min(1e-4), torch.zeros_like(distance))


def _sample_sparse_trilinear(
    field: torch.Tensor,
    z: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    nz, ny, nx = field.shape
    z = z.clamp(0.0, nz - 1.000001)
    y = y.clamp(0.0, ny - 1.000001)
    x = x.clamp(0.0, nx - 1.000001)
    z0, y0, x0 = z.floor().long(), y.floor().long(), x.floor().long()
    z1, y1, x1 = (z0 + 1).clamp_max(nz - 1), (y0 + 1).clamp_max(ny - 1), (x0 + 1).clamp_max(nx - 1)
    wz, wy, wx = z - z0, y - y0, x - x0
    result = torch.zeros_like(z)
    for zz, az in ((z0, 1.0 - wz), (z1, wz)):
        for yy, ay in ((y0, 1.0 - wy), (y1, wy)):
            for xx, ax in ((x0, 1.0 - wx), (x1, wx)):
                result = result + field[zz, yy, xx] * az * ay * ax
    return result


def _lattice(q: int, device: torch.device, dtype: torch.dtype):
    if q == 19:
        from .d3q19 import C, W, macroscopic3d
        return C.to(device=device, dtype=dtype), W.to(device=device, dtype=dtype), macroscopic3d
    if q == 27:
        from .d3q27 import C, W, macroscopic27
        return C.to(device=device, dtype=dtype), W.to(device=device, dtype=dtype), macroscopic27
    raise ValueError("only D3Q19 and D3Q27 are supported")


def _equilibrium_sparse(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    c: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    cu = c[:, 0, None] * ux + c[:, 1, None] * uy + c[:, 2, None] * uz
    u2 = ux.square() + uy.square() + uz.square()
    return w[:, None] * rho[None, :] * (1.0 + 3.0 * cu + 4.5 * cu.square() - 1.5 * u2)


@dataclass(frozen=True)
class SpaldingWallDiagnostics:
    boundary_nodes: int
    mean_y1: float
    mean_y2_plus: float
    mean_u_tau: float
    shear_force: tuple[float, float, float]


def apply_spalding_exchange_wall_model(
    f: torch.Tensor,
    fluid_boundary_mask: torch.Tensor,
    q_field: torch.Tensor,
    normals: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    nu: float,
    *,
    exchange_distance: float = 3.0,
    nonequilibrium_scale: float = 0.5,
    area_weight: torch.Tensor | None = None,
    activation: float = 1.0,
) -> tuple[torch.Tensor, SpaldingWallDiagnostics]:
    """Assimilate a Spalding-modelled boundary velocity after curved BFL."""
    if exchange_distance <= 0.0:
        raise ValueError("exchange_distance must be positive")
    if not 0.0 <= nonequilibrium_scale <= 1.0:
        raise ValueError("nonequilibrium_scale must be in [0,1]")
    if not 0.0 <= activation <= 1.0:
        raise ValueError("activation must be in [0,1]")
    c, w, macro = _lattice(f.shape[0], f.device, f.dtype)
    rho, ux, uy, uz = macro(f)
    nx_n, ny_n, nz_n = (component.to(device=f.device, dtype=f.dtype) for component in normals)
    y1_field = effective_bfl_wall_distance(
        fluid_boundary_mask, q_field, (nx_n, ny_n, nz_n),
    )
    boundary = fluid_boundary_mask.any(dim=0) & (y1_field > 0.0)
    indices = boundary.nonzero(as_tuple=False)
    count = int(indices.shape[0])
    if count == 0:
        zero = (0.0, 0.0, 0.0)
        return f, SpaldingWallDiagnostics(0, 0.0, 0.0, 0.0, zero)
    iz, iy, ix = indices[:, 0], indices[:, 1], indices[:, 2]
    nxb, nyb, nzb = nx_n[boundary], ny_n[boundary], nz_n[boundary]
    y1 = y1_field[boundary]
    # y2 is measured from the wall and must remain beyond the boundary node.
    y2 = torch.full_like(y1, exchange_distance)
    y2 = torch.maximum(y2, y1 + 0.5)
    offset = y2 - y1
    sample_z = iz.to(f.dtype) + offset * nzb
    sample_y = iy.to(f.dtype) + offset * nyb
    sample_x = ix.to(f.dtype) + offset * nxb
    u2x = _sample_sparse_trilinear(ux, sample_z, sample_y, sample_x)
    u2y = _sample_sparse_trilinear(uy, sample_z, sample_y, sample_x)
    u2z = _sample_sparse_trilinear(uz, sample_z, sample_y, sample_x)
    u2n = u2x * nxb + u2y * nyb + u2z * nzb
    t2x, t2y, t2z = u2x - u2n * nxb, u2y - u2n * nyb, u2z - u2n * nzb
    u2mag = torch.sqrt(t2x.square() + t2y.square() + t2z.square()).clamp_min(1e-20)
    tx, ty, tz = t2x / u2mag, t2y / u2mag, t2z / u2mag
    u_tau = solve_spalding_friction_velocity(u2mag, y2, nu)
    y1_plus = y1 * u_tau / nu
    u1 = spalding_u_plus_from_y_plus(y1_plus) * u_tau

    uxb, uyb, uzb = ux[boundary], uy[boundary], uz[boundary]
    ubn = uxb * nxb + uyb * nyb + uzb * nzb
    target_x = ubn * nxb + u1 * tx
    target_y = ubn * nyb + u1 * ty
    target_z = ubn * nzb + u1 * tz
    rho_b = rho[boundary]
    old_eq = _equilibrium_sparse(rho_b, uxb, uyb, uzb, c, w)
    new_eq = _equilibrium_sparse(rho_b, target_x, target_y, target_z, c, w)
    old_values = f[:, boundary]
    modelled_values = new_eq + nonequilibrium_scale * (old_values - old_eq)
    # Ramp the complete population assimilation.  In particular, activation
    # zero must not damp non-equilibrium content left by BFL/streaming.
    values = old_values + activation * (modelled_values - old_values)
    out = f.clone()
    out[:, boundary] = values

    tau_w = u_tau.square() * activation
    if area_weight is None:
        area = torch.ones_like(tau_w)
    else:
        if area_weight.shape != boundary.shape:
            raise ValueError("area_weight must have the spatial grid shape")
        area = area_weight.to(device=f.device, dtype=f.dtype)[boundary]
    shear = torch.stack(((tau_w * tx * area).sum(), (tau_w * ty * area).sum(), (tau_w * tz * area).sum()))
    diagnostics = SpaldingWallDiagnostics(
        boundary_nodes=count,
        mean_y1=float(y1.mean().item()),
        mean_y2_plus=float((y2 * u_tau / nu).mean().item()),
        mean_u_tau=float(u_tau.mean().item()),
        shear_force=tuple(float(value) for value in shear.tolist()),
    )
    return out, diagnostics


__all__ = [
    "SPALDING_A",
    "SPALDING_KAPPA",
    "SpaldingWallDiagnostics",
    "apply_spalding_exchange_wall_model",
    "effective_bfl_wall_distance",
    "solve_spalding_friction_velocity",
    "spalding_u_plus_from_y_plus",
    "spalding_y_plus",
]
