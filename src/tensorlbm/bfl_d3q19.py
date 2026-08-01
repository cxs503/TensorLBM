"""D3Q19 BFL q-value computation for 2D extruded geometries.

Extends compute_q_circle from D2Q9 to D3Q19 by computing q for all
directions with c_z=0 (8 directions) and setting q=0.5 for z-containing
directions (10 directions, no intersection with extruded cylinder).
"""
import math
import torch
from .d3q19 import C


def compute_q_cylinder_d3q19(
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    radius: float,
    device: torch.device,
    axis: str = 'z',
    cz: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-direction BFL q-values for a 2D extruded cylinder using D3Q19.

    The cylinder cross-section lies in the plane perpendicular to *axis*;
    q-values are computed in that plane and broadcast across all layers
    along *axis*.

    =========  =================  ===================================================
    axis      cross-section      quadratic solve
    =========  =================  ===================================================
    ``'z'``   x-y (default)      ``(x+cx·t-cx_c)² + (y+cy·t-cy_c)² = R²``
    ``'y'``   x-z                ``(x+cx·t-cx_c)² + (z+cz·t-cz_c)² = R²``
    ``'x'``   y-z                ``(y+cy·t-cy_c)² + (z+cz·t-cz_c)² = R²``
    =========  =================  ===================================================

    Args:
        nx, ny, nz: Grid dimensions.
        cx, cy: Cylinder centre in the cross-section plane.
            For axis='y' or 'x', *cz* specifies the centre along z
            (defaults to ``nz / 2``).
        radius: Cylinder radius.
        device: Torch device.
        axis: Extrusion axis (default ``'z'``).

    Returns:
        fluid_boundary_mask: (19, nz, ny, nx) bool
        q_field: (19, nz, ny, nx) float, fractional distance per direction
    """
    c = C.to(device).float()  # (19, 3)

    # Map axis → in-plane coordinate arrays, centres, velocity indices, and
    # the broadcast dimension for expanding the 2D result to (nz, ny, nx).
    if axis == 'z':
        coord2, coord1 = torch.meshgrid(
            torch.arange(ny, device=device, dtype=torch.float64),
            torch.arange(nx, device=device, dtype=torch.float64),
            indexing="ij",
        )  # coord1=x (ny,nx), coord2=y (ny,nx)
        c1, c2 = cx, cy
        vi1, vi2, vai = 0, 1, 2   # in-plane=(cx,cy), axis=cz
        bdim = 0                  # unsqueeze dim → (1,ny,nx)→expand(nz,ny,nx)
    elif axis == 'y':
        cz_c = cz if cz is not None else nz / 2.0
        coord2, coord1 = torch.meshgrid(
            torch.arange(nz, device=device, dtype=torch.float64),
            torch.arange(nx, device=device, dtype=torch.float64),
            indexing="ij",
        )  # coord1=x (nz,nx), coord2=z (nz,nx)
        c1, c2 = cx, cz_c
        vi1, vi2, vai = 0, 2, 1   # in-plane=(cx,cz), axis=cy
        bdim = 1                  # (nz,1,nx)→expand(nz,ny,nx)
    elif axis == 'x':
        cz_c = cz if cz is not None else nz / 2.0
        coord2, coord1 = torch.meshgrid(
            torch.arange(nz, device=device, dtype=torch.float64),
            torch.arange(ny, device=device, dtype=torch.float64),
            indexing="ij",
        )  # coord1=y (nz,ny), coord2=z (nz,ny)
        c1, c2 = cy, cz_c
        vi1, vi2, vai = 1, 2, 0   # in-plane=(cy,cz), axis=cx
        bdim = 2                  # (nz,ny,1)→expand(nz,ny,nx)
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")

    fluid_boundary_mask = torch.zeros((19, nz, ny, nx), dtype=torch.bool, device=device)
    q_field = torch.full((19, nz, ny, nx), 0.5, dtype=torch.float32, device=device)

    for d in range(19):
        dv1 = float(c[d, vi1].item())  # in-plane velocity component 1
        dv2 = float(c[d, vi2].item())  # in-plane velocity component 2
        dva = float(c[d, vai].item())  # axis velocity component

        if dv1 == 0.0 and dv2 == 0.0 and dva == 0.0:
            continue  # rest direction

        # For axis-containing directions, no intersection with extruded cylinder
        if dva != 0.0:
            if dv1 == 0.0 and dv2 == 0.0:
                continue  # pure axis direction, no crossing
            # Fall through: compute q from in-plane components only

        # Neighbour in direction d (in-plane)
        dist_nb = (coord1 + dv1 - c1) ** 2 + (coord2 + dv2 - c2) ** 2
        nb_is_solid = dist_nb <= radius ** 2

        # Current node is fluid
        dist_self = (coord1 - c1) ** 2 + (coord2 - c2) ** 2
        self_is_fluid = dist_self > radius ** 2

        boundary = self_is_fluid & nb_is_solid  # 2D

        if not boundary.any():
            continue

        # Solve quadratic: |x + t*c - centre|^2 = r^2
        d1 = coord1 - c1
        d2 = coord2 - c2
        a_coef = dv1 ** 2 + dv2 ** 2  # |c_in-plane|^2
        if a_coef < 1e-10:
            continue
        b_coef = 2.0 * (dv1 * d1 + dv2 * d2)
        c_coef = d1 ** 2 + d2 ** 2 - radius ** 2

        discriminant = b_coef ** 2 - 4.0 * a_coef * c_coef
        safe_disc = torch.where(
            boundary & (discriminant >= 0.0),
            discriminant,
            torch.zeros_like(discriminant),
        )
        sqrt_disc = torch.sqrt(safe_disc)

        t1 = (-b_coef - sqrt_disc) / (2.0 * a_coef)
        t2 = (-b_coef + sqrt_disc) / (2.0 * a_coef)

        # q = t (fractional distance: 0=fluid cell, 1=solid cell)
        q1 = t1
        q2 = t2

        valid1 = (t1 > 1e-10) & (q1 <= 1.0 + 1e-10)
        valid2 = (t2 > 1e-10) & (q2 <= 1.0 + 1e-10)

        q_val = torch.where(
            valid1 & valid2,
            torch.min(q1, q2),
            torch.where(valid1, q1, torch.where(valid2, q2, torch.full_like(q1, 0.5))),
        ).clamp(1e-6, 1.0).float()

        # Broadcast 2D result to all layers along axis
        boundary_3d = boundary.unsqueeze(bdim).expand(nz, ny, nx)
        q_val_3d = q_val.unsqueeze(bdim).expand(nz, ny, nx)
        fluid_boundary_mask[d] = boundary_3d
        q_field[d] = torch.where(boundary_3d, q_val_3d, q_field[d])

    return fluid_boundary_mask, q_field


def bouzidi_bounce_back_d3q19(
    f: torch.Tensor,
    f_prev: torch.Tensor,
    fluid_boundary_mask: torch.Tensor,
    q_field: torch.Tensor,
) -> torch.Tensor:
    """Apply BFL interpolated bounce-back for ALL D3Q19 directions.
    
    Per-direction q-values (not per-cell). Uses OPPOSITE array.
    
    Args:
        f: Post-stream distribution (19, nz, ny, nx)
        f_prev: Pre-stream distribution (19, nz, ny, nx)
        fluid_boundary_mask: (19, nz, ny, nx) bool
        q_field: (19, nz, ny, nx) float, per-direction fractional distance
    
    Returns:
        Updated distribution tensor.
    """
    from .d3q19 import OPPOSITE
    opp = OPPOSITE.to(f.device)
    f_out = f.clone()
    
    for d in range(1, 19):  # skip rest
        opp_d = int(opp[d].item())

        mask = fluid_boundary_mask[d]
        if not mask.any():
            continue

        q_cell = q_field[d][mask]
        mask_lin = q_cell < 0.5
        mask_quad = ~mask_lin

        # Pre-stream populations (post-collision, before streaming).  With
        # pull streaming the unknown population is f_opp(x_f,t+1), whose
        # source lies inside the solid.  It must be reconstructed from the
        # known outgoing f_d populations; the post-stream value from the
        # solid is not physical boundary data.
        fp_opp = f_prev[opp_d][mask]
        fp_d = f_prev[d][mask]

        dcx, dcy, dcz = (int(v) for v in C[d].tolist())
        fp_d_upstream_field = torch.roll(
            f_prev[d], shifts=(dcz, dcy, dcx), dims=(0, 1, 2),
        )
        fp_d_upstream = fp_d_upstream_field[mask]

        # Wall closer than half-link: interpolate the two outgoing fluid
        # populations at x_f and x_f-c_d.
        f_bc_lin = (
            2.0 * q_cell * fp_d
            + (1.0 - 2.0 * q_cell) * fp_d_upstream
        )

        # Wall farther than half-link: interpolate outgoing and opposite
        # post-collision populations at the boundary fluid node.
        safe_q = torch.where(mask_quad, q_cell, torch.ones_like(q_cell))
        f_bc_quad = (
            fp_d / (2.0 * safe_q)
            + (2.0 * safe_q - 1.0) / (2.0 * safe_q) * fp_opp
        )

        f_bc = torch.where(mask_lin, f_bc_lin, f_bc_quad)

        # Set f[opp_d] (the UNKNOWN population, from solid toward fluid),
        # NOT f[d] (the known population, from fluid toward solid).
        # The unknown is the one whose streaming source is the solid cell.
        target = f_out[opp_d].clone()
        target[mask] = f_bc
        f_out[opp_d] = target
    
    return f_out
