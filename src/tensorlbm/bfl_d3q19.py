"""D3Q19 BFL q-value computation for 2D extruded geometries.

Extends compute_q_circle from D2Q9 to D3Q19 by computing q for all
directions with c_z=0 (8 directions) and setting q=0.5 for z-containing
directions (10 directions, no intersection with extruded cylinder).
"""
from dataclasses import dataclass
import math
import torch
from .d3q19 import C, W


@dataclass(frozen=True)
class BFLLinkForceDecomposition:
    """Geometry-normal and tangential parts of the actual link impulse."""

    force_frame: str
    active_links: int
    decomposed_links: int
    undecomposed_links: int
    coverage_fraction: float
    total_force: tuple[float, float, float]
    normal_force: tuple[float, float, float]
    tangential_force: tuple[float, float, float]
    unresolved_force: tuple[float, float, float]
    stationary_interpolation_force: tuple[float, float, float]
    moving_wall_population_correction_force: tuple[float, float, float]
    frame_correction_force: tuple[float, float, float]
    maximum_closure_error: float
    maximum_relative_closure_error: float
    maximum_component_closure_error: float
    maximum_relative_component_closure_error: float


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
    *,
    wall_velocity: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    wall_density: torch.Tensor | None = None,
    boundary_fraction: float = 1.0,
    return_force: bool = False,
    force_frame: str = "laboratory",
    force_normals: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    return_force_decomposition: bool = False,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, tuple[float, float, float]]
    | tuple[
        torch.Tensor,
        tuple[float, float, float],
        BFLLinkForceDecomposition,
    ]
):
    """Apply BFL interpolated bounce-back for ALL D3Q19 directions.
    
    Per-direction q-values (not per-cell). Uses OPPOSITE array.
    
    Args:
        f: Post-stream distribution (19, nz, ny, nx)
        f_prev: Pre-stream distribution (19, nz, ny, nx)
        fluid_boundary_mask: (19, nz, ny, nx) bool
        q_field: (19, nz, ny, nx) float, per-direction fractional distance
        wall_velocity: Optional wall velocity fields ``(ux, uy, uz)``.  A
            local tangential velocity gives an impermeable slip wall suitable
            for wall-stress models; ``None`` is a stationary no-slip wall.
        wall_density: Density field used by the moving-wall correction.
            Required when ``wall_velocity`` is provided.
        boundary_fraction: Smooth activation in ``[0,1]``.  Zero leaves the
            streamed population untouched; one applies the complete BFL wall.
        return_force: Also return the link momentum-exchange force on the wall.
        force_frame: ``"laboratory"`` returns the discrete force that closes
            a fixed laboratory-frame control-volume balance. ``"wall"``
            returns the Galilean-invariant moving-wall-frame diagnostic.  A
            numerical tangential slip velocity is not a physical body motion,
            so stationary-body force validation must use ``"laboratory"``.
        force_normals: Optional outward unit-normal fields ``(nx, ny, nz)``.
            Required with ``return_force_decomposition``.  The diagnostic
            projects the *actual population impulse* after any moving-wall
            correction; it does not reconstruct pressure from density.
        return_force_decomposition: Return a third result containing the
            geometry-normal and tangential link-impulse sums.  This is a
            diagnostic decomposition, not by itself a pressure/shear model.
    
    Returns:
        Updated distribution tensor.
    """
    from .d3q19 import OPPOSITE
    if wall_velocity is not None and wall_density is None:
        raise ValueError("wall_density is required with wall_velocity")
    if not 0.0 <= boundary_fraction <= 1.0:
        raise ValueError("boundary_fraction must be in [0,1]")
    if force_frame not in {"laboratory", "wall"}:
        raise ValueError("force_frame must be 'laboratory' or 'wall'")
    if return_force_decomposition and not return_force:
        raise ValueError("force decomposition requires return_force=True")
    if return_force_decomposition and force_normals is None:
        raise ValueError("force_normals are required for force decomposition")
    if force_normals is not None and any(
        component.shape != f.shape[1:] for component in force_normals
    ):
        raise ValueError("force_normals must share the spatial grid shape")
    opp = OPPOSITE.to(f.device)
    weights = W.to(device=f.device, dtype=f.dtype)
    f_out = f.clone()
    force_x = torch.zeros((), device=f.device, dtype=f.dtype)
    force_y = torch.zeros_like(force_x)
    force_z = torch.zeros_like(force_x)
    normal_force_x = torch.zeros_like(force_x)
    normal_force_y = torch.zeros_like(force_x)
    normal_force_z = torch.zeros_like(force_x)
    tangential_force_x = torch.zeros_like(force_x)
    tangential_force_y = torch.zeros_like(force_x)
    tangential_force_z = torch.zeros_like(force_x)
    unresolved_force_x = torch.zeros_like(force_x)
    unresolved_force_y = torch.zeros_like(force_x)
    unresolved_force_z = torch.zeros_like(force_x)
    interpolation_force_x = torch.zeros_like(force_x)
    interpolation_force_y = torch.zeros_like(force_x)
    interpolation_force_z = torch.zeros_like(force_x)
    moving_correction_force_x = torch.zeros_like(force_x)
    moving_correction_force_y = torch.zeros_like(force_x)
    moving_correction_force_z = torch.zeros_like(force_x)
    frame_correction_force_x = torch.zeros_like(force_x)
    frame_correction_force_y = torch.zeros_like(force_x)
    frame_correction_force_z = torch.zeros_like(force_x)
    active_force_links = 0
    decomposed_links = 0
    
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
        f_bc_stationary = torch.where(mask_lin, f_bc_lin, f_bc_quad)

        if wall_velocity is not None:
            uwx, uwy, uwz = wall_velocity
            c_dot_uw = (
                float(dcx) * uwx[mask]
                + float(dcy) * uwy[mask]
                + float(dcz) * uwz[mask]
            )
            rho_w = wall_density[mask]
            moving_base = weights[d] * rho_w * c_dot_uw
            # Bouzidi moving-wall correction.  With our convention c_d
            # points from fluid to solid, the sign is negative.  At q=.5
            # both branches reduce to standard moving half-way bounce-back:
            # f_opp = f_d - 6*w*rho*(c_d·u_wall).
            f_bc_lin = f_bc_lin - 6.0 * moving_base
            f_bc_quad = f_bc_quad - (3.0 / safe_q) * moving_base

        f_bc = torch.where(mask_lin, f_bc_lin, f_bc_quad)

        if return_force and boundary_fraction > 0.0:
            # Laboratory-frame discrete momentum exchange is the population
            # impulse that exactly closes a fixed control-volume balance:
            # c_d*f_d - c_opp*f_opp = c_d*(f_d+f_opp).
            stationary_exchange = fp_d + f_bc_stationary
            moving_exchange = f_bc - f_bc_stationary
            interpolation_link_fx = float(dcx) * stationary_exchange
            interpolation_link_fy = float(dcy) * stationary_exchange
            interpolation_link_fz = float(dcz) * stationary_exchange
            moving_link_fx = float(dcx) * moving_exchange
            moving_link_fy = float(dcy) * moving_exchange
            moving_link_fz = float(dcz) * moving_exchange
            frame_link_fx = torch.zeros_like(interpolation_link_fx)
            frame_link_fy = torch.zeros_like(interpolation_link_fy)
            frame_link_fz = torch.zeros_like(interpolation_link_fz)
            if force_frame == "wall" and wall_velocity is not None:
                # Galilean-invariant momentum exchange in the moving-wall
                # frame.  This is a useful physical diagnostic for a genuinely
                # moving body, but it does not equal the laboratory-frame
                # population impulse when u_w is an artificial slip closure.
                exchange_diff = f_bc - fp_d
                frame_link_fx = exchange_diff * uwx[mask]
                frame_link_fy = exchange_diff * uwy[mask]
                frame_link_fz = exchange_diff * uwz[mask]
            # Preserve the original conservative population-impulse reduction
            # exactly; the component sums below are an independent ledger and
            # may differ by floating-point reduction roundoff only.
            exchange_sum = fp_d + f_bc
            link_fx = float(dcx) * exchange_sum + frame_link_fx
            link_fy = float(dcy) * exchange_sum + frame_link_fy
            link_fz = float(dcz) * exchange_sum + frame_link_fz
            link_fx = boundary_fraction * link_fx
            link_fy = boundary_fraction * link_fy
            link_fz = boundary_fraction * link_fz
            force_x = force_x + link_fx.sum()
            force_y = force_y + link_fy.sum()
            force_z = force_z + link_fz.sum()
            if return_force_decomposition:
                interpolation_force_x += (
                    boundary_fraction * interpolation_link_fx.sum()
                )
                interpolation_force_y += (
                    boundary_fraction * interpolation_link_fy.sum()
                )
                interpolation_force_z += (
                    boundary_fraction * interpolation_link_fz.sum()
                )
                moving_correction_force_x += (
                    boundary_fraction * moving_link_fx.sum()
                )
                moving_correction_force_y += (
                    boundary_fraction * moving_link_fy.sum()
                )
                moving_correction_force_z += (
                    boundary_fraction * moving_link_fz.sum()
                )
                frame_correction_force_x += (
                    boundary_fraction * frame_link_fx.sum()
                )
                frame_correction_force_y += (
                    boundary_fraction * frame_link_fy.sum()
                )
                frame_correction_force_z += (
                    boundary_fraction * frame_link_fz.sum()
                )
                assert force_normals is not None
                local_normals = tuple(
                    component.to(device=f.device, dtype=f.dtype)[mask]
                    for component in force_normals
                )
                normal_magnitude = torch.sqrt(sum(
                    component * component for component in local_normals
                ))
                if not bool(torch.isfinite(normal_magnitude).all()):
                    raise ValueError(
                        "force_normals must be finite on active links",
                    )
                valid_normal = normal_magnitude > 1.0e-12
                unit_normals = tuple(
                    torch.where(
                        valid_normal,
                        component / normal_magnitude.clamp_min(1.0e-12),
                        torch.zeros_like(component),
                    )
                    for component in local_normals
                )
                normal_scalar = (
                    link_fx * unit_normals[0]
                    + link_fy * unit_normals[1]
                    + link_fz * unit_normals[2]
                )
                normal_components = tuple(
                    normal_scalar * component for component in unit_normals
                )
                tangential_components = (
                    torch.where(
                        valid_normal,
                        link_fx - normal_components[0],
                        torch.zeros_like(link_fx),
                    ),
                    torch.where(
                        valid_normal,
                        link_fy - normal_components[1],
                        torch.zeros_like(link_fy),
                    ),
                    torch.where(
                        valid_normal,
                        link_fz - normal_components[2],
                        torch.zeros_like(link_fz),
                    ),
                )
                unresolved_components = (
                    torch.where(valid_normal, torch.zeros_like(link_fx), link_fx),
                    torch.where(valid_normal, torch.zeros_like(link_fy), link_fy),
                    torch.where(valid_normal, torch.zeros_like(link_fz), link_fz),
                )
                normal_force_x += normal_components[0].sum()
                normal_force_y += normal_components[1].sum()
                normal_force_z += normal_components[2].sum()
                tangential_force_x += tangential_components[0].sum()
                tangential_force_y += tangential_components[1].sum()
                tangential_force_z += tangential_components[2].sum()
                unresolved_force_x += unresolved_components[0].sum()
                unresolved_force_y += unresolved_components[1].sum()
                unresolved_force_z += unresolved_components[2].sum()
                active_force_links += int(mask.sum().item())
                decomposed_links += int(valid_normal.sum().item())

        # Set f[opp_d] (the UNKNOWN population, from solid toward fluid),
        # NOT f[d] (the known population, from fluid toward solid).
        # The unknown is the one whose streaming source is the solid cell.
        target = f_out[opp_d].clone()
        if boundary_fraction == 1.0:
            target[mask] = f_bc
        elif boundary_fraction > 0.0:
            target[mask] = (
                (1.0 - boundary_fraction) * target[mask]
                + boundary_fraction * f_bc
            )
        f_out[opp_d] = target
    
    if return_force:
        total_force = (
            float(force_x.item()),
            float(force_y.item()),
            float(force_z.item()),
        )
        if return_force_decomposition:
            normal_force = (
                float(normal_force_x.item()),
                float(normal_force_y.item()),
                float(normal_force_z.item()),
            )
            tangential_force = (
                float(tangential_force_x.item()),
                float(tangential_force_y.item()),
                float(tangential_force_z.item()),
            )
            unresolved_force = (
                float(unresolved_force_x.item()),
                float(unresolved_force_y.item()),
                float(unresolved_force_z.item()),
            )
            stationary_interpolation_force = (
                float(interpolation_force_x.item()),
                float(interpolation_force_y.item()),
                float(interpolation_force_z.item()),
            )
            moving_wall_population_correction_force = (
                float(moving_correction_force_x.item()),
                float(moving_correction_force_y.item()),
                float(moving_correction_force_z.item()),
            )
            frame_correction_force = (
                float(frame_correction_force_x.item()),
                float(frame_correction_force_y.item()),
                float(frame_correction_force_z.item()),
            )
            closure = max(
                abs(total - normal - tangential - unresolved)
                for total, normal, tangential, unresolved in zip(
                    total_force,
                    normal_force,
                    tangential_force,
                    unresolved_force,
                    strict=True,
                )
            )
            closure_scale = max(
                *(abs(value) for value in total_force),
                *(abs(value) for value in normal_force),
                *(abs(value) for value in tangential_force),
                *(abs(value) for value in unresolved_force),
                torch.finfo(f.dtype).tiny,
            )
            component_closure = max(
                abs(total - stationary - moving - frame)
                for total, stationary, moving, frame in zip(
                    total_force,
                    stationary_interpolation_force,
                    moving_wall_population_correction_force,
                    frame_correction_force,
                    strict=True,
                )
            )
            component_scale = max(
                *(abs(value) for value in total_force),
                *(abs(value) for value in stationary_interpolation_force),
                *(
                    abs(value)
                    for value in moving_wall_population_correction_force
                ),
                *(abs(value) for value in frame_correction_force),
                torch.finfo(f.dtype).tiny,
            )
            return f_out, total_force, BFLLinkForceDecomposition(
                force_frame=force_frame,
                active_links=active_force_links,
                decomposed_links=decomposed_links,
                undecomposed_links=active_force_links - decomposed_links,
                coverage_fraction=(
                    decomposed_links / active_force_links
                    if active_force_links else 1.0
                ),
                total_force=total_force,
                normal_force=normal_force,
                tangential_force=tangential_force,
                unresolved_force=unresolved_force,
                stationary_interpolation_force=stationary_interpolation_force,
                moving_wall_population_correction_force=(
                    moving_wall_population_correction_force
                ),
                frame_correction_force=frame_correction_force,
                maximum_closure_error=closure,
                maximum_relative_closure_error=closure / closure_scale,
                maximum_component_closure_error=component_closure,
                maximum_relative_component_closure_error=(
                    component_closure / component_scale
                ),
            )
        return f_out, total_force
    return f_out
