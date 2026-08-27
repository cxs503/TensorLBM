"""D3Q19 BFL q-value computation for 2D extruded geometries.

Extends compute_q_circle from D2Q9 to D3Q19 by computing q for all
directions with c_z=0 (8 directions) and setting q=0.5 for z-containing
directions (10 directions, no intersection with extruded cylinder).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .bfl_common import bfl_bounce_back_common
from .d3q19 import C


def compute_q_cylinder_d3q19(
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    radius: float,
    device: torch.device,
    axis: str = "z",
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
    if axis == "z":
        coord2, coord1 = torch.meshgrid(
            torch.arange(ny, device=device, dtype=torch.float64),
            torch.arange(nx, device=device, dtype=torch.float64),
            indexing="ij",
        )  # coord1=x (ny,nx), coord2=y (ny,nx)
        c1, c2 = cx, cy
        vi1, vi2, vai = 0, 1, 2  # in-plane=(cx,cy), axis=cz
        bdim = 0  # unsqueeze dim → (1,ny,nx)→expand(nz,ny,nx)
    elif axis == "y":
        cz_c = cz if cz is not None else nz / 2.0
        coord2, coord1 = torch.meshgrid(
            torch.arange(nz, device=device, dtype=torch.float64),
            torch.arange(nx, device=device, dtype=torch.float64),
            indexing="ij",
        )  # coord1=x (nz,nx), coord2=z (nz,nx)
        c1, c2 = cx, cz_c
        vi1, vi2, vai = 0, 2, 1  # in-plane=(cx,cz), axis=cy
        bdim = 1  # (nz,1,nx)→expand(nz,ny,nx)
    elif axis == "x":
        cz_c = cz if cz is not None else nz / 2.0
        coord2, coord1 = torch.meshgrid(
            torch.arange(nz, device=device, dtype=torch.float64),
            torch.arange(ny, device=device, dtype=torch.float64),
            indexing="ij",
        )  # coord1=y (nz,ny), coord2=z (nz,ny)
        c1, c2 = cy, cz_c
        vi1, vi2, vai = 1, 2, 0  # in-plane=(cy,cz), axis=cx
        bdim = 2  # (nz,ny,1)→expand(nz,ny,nx)
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
        nb_is_solid = dist_nb <= radius**2

        # Current node is fluid
        dist_self = (coord1 - c1) ** 2 + (coord2 - c2) ** 2
        self_is_fluid = dist_self > radius**2

        boundary = self_is_fluid & nb_is_solid  # 2D

        if not boundary.any():
            continue

        # Solve quadratic: |x + t*c - centre|^2 = r^2
        d1 = coord1 - c1
        d2 = coord2 - c2
        a_coef = dv1**2 + dv2**2  # |c_in-plane|^2
        if a_coef < 1e-10:
            continue
        b_coef = 2.0 * (dv1 * d1 + dv2 * d2)
        c_coef = d1**2 + d2**2 - radius**2

        discriminant = b_coef**2 - 4.0 * a_coef * c_coef
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

        q_val = (
            torch.where(
                valid1 & valid2,
                torch.min(q1, q2),
                torch.where(valid1, q1, torch.where(valid2, q2, torch.full_like(q1, 0.5))),
            )
            .clamp(1e-6, 1.0)
            .float()
        )

        # Broadcast 2D result to all layers along axis
        boundary_3d = boundary.unsqueeze(bdim).expand(nz, ny, nx)
        q_val_3d = q_val.unsqueeze(bdim).expand(nz, ny, nx)
        fluid_boundary_mask[d] = boundary_3d
        q_field[d] = torch.where(boundary_3d, q_val_3d, q_field[d])

    return fluid_boundary_mask, q_field


@dataclass(frozen=True)
class BFLForceDecomposition:
    """Per-link force ledger of a BFL bounce-back application.

    ``total_force`` is the conservative laboratory-frame momentum exchange
    (it closes a fixed control volume).  It decomposes *exactly* into the
    stationary interpolation part plus the moving-wall population
    correction (the frame correction is zero by construction) and splits
    into normal/tangential components along the provided link normals.
    """

    active_links: int
    decomposed_links: int
    undecomposed_links: int
    coverage_fraction: float
    unresolved_force: tuple[float, float, float]
    stationary_interpolation_force: tuple[float, float, float]
    moving_wall_population_correction_force: tuple[float, float, float]
    frame_correction_force: tuple[float, float, float]
    total_force: tuple[float, float, float]
    normal_force: tuple[float, float, float]
    tangential_force: tuple[float, float, float]
    maximum_closure_error: float
    maximum_relative_closure_error: float
    maximum_component_closure_error: float


def bouzidi_bounce_back_d3q19(
    f: torch.Tensor,
    f_prev: torch.Tensor,
    fluid_boundary_mask: torch.Tensor,
    q_field: torch.Tensor,
    *,
    wall_velocity: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    wall_density: torch.Tensor | None = None,
    boundary_fraction: float | torch.Tensor | None = None,
    return_force: bool = False,
    force_frame: str = "laboratory",
    force_normals: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    return_force_decomposition: bool = False,
) -> torch.Tensor | tuple:
    """Apply BFL interpolated bounce-back for ALL D3Q19 directions.

    Bouzidi (2001) convention, expressed purely on pre-stream
    (post-collision) populations and bit-identical on the default argument
    path to :func:`tensorlbm.bfl_common.bfl_bounce_back_common`:

    * ``q < 0.5``:  ``f_bc = 2q*fp[d](x) + (1-2q)*fp[d](x - c_d)``
    * ``q >= 0.5``: ``f_bc = fp[d](x)/(2q) + (2q-1)/(2q)*fp[opp_d](x)``

    The unknown population ``f[opp_d]`` at the fluid boundary cell is set
    to ``f_bc`` (any streamed-from-solid value is discarded).

    Args:
        f: Post-stream distribution (19, nz, ny, nx).
        f_prev: Pre-stream distribution (19, nz, ny, nx).
        fluid_boundary_mask: (19, nz, ny, nx) bool.
        q_field: (19, nz, ny, nx) float, per-direction fractional distance.
        wall_velocity: Optional per-component (nz, ny, nx) wall velocity;
            requires ``wall_density``.
        wall_density: (nz, ny, nx) wall density for the moving-wall
            momentum correction ``2*rho*w*(c.u_w)/cs^2``.
        boundary_fraction: Optional scalar or (nz, ny, nx) field in [0, 1]
            blending the BFL result against the streamed field (smooth body
            insertion ramp).  ``0.0`` is exactly transparent.
        return_force: Return ``(f_out, force)``; force is a 3-tuple of
            0-dim tensors.
        force_frame: ``"laboratory"`` (conservative ledger, closes a fixed
            control volume) or ``"wall"`` (removes the moving-wall
            population correction, i.e. the background flux of a co-moving
            transparent wall).
        force_normals: Optional per-component (nz, ny, nx) link unit-normal
            fields used by the force decomposition.
        return_force_decomposition: Additionally return a
            :class:`BFLForceDecomposition` (requires ``return_force`` and
            ``force_normals``).

    Returns:
        Updated distribution tensor, ``(f_out, force)`` when
        ``return_force``, or ``(f_out, force, decomposition)`` when
        ``return_force_decomposition``.
    """
    if return_force_decomposition and not return_force:
        msg = "return_force_decomposition requires return_force=True"
        raise ValueError(msg)
    if return_force_decomposition and force_normals is None:
        msg = "return_force_decomposition requires force_normals"
        raise ValueError(msg)
    if wall_velocity is not None and wall_density is None:
        msg = "wall_velocity requires wall_density for the moving-wall correction"
        raise ValueError(msg)
    if force_frame not in {"laboratory", "wall"}:
        msg = f"force_frame must be 'laboratory' or 'wall', got {force_frame!r}"
        raise ValueError(msg)

    fraction = 1.0 if boundary_fraction is None else boundary_fraction
    if isinstance(fraction, (int, float)) and float(fraction) == 0.0:
        # Exactly transparent: the streamed field passes through untouched.
        zero3 = (torch.zeros((), dtype=f.dtype, device=f.device),) * 3
        if return_force_decomposition:
            zero = (0.0, 0.0, 0.0)
            return (
                f,
                zero3,
                BFLForceDecomposition(
                    active_links=0,
                    decomposed_links=0,
                    undecomposed_links=0,
                    coverage_fraction=0.0,
                    unresolved_force=zero,
                    stationary_interpolation_force=zero,
                    moving_wall_population_correction_force=zero,
                    frame_correction_force=zero,
                    total_force=zero,
                    normal_force=zero,
                    tangential_force=zero,
                    maximum_closure_error=0.0,
                    maximum_relative_closure_error=0.0,
                    maximum_component_closure_error=0.0,
                ),
            )
        if return_force:
            return f, zero3
        return f

    # --- populations: delegate to the canonical shared implementation so
    # the default path stays bit-identical to the vectorized/common forms.
    wall_correction = None
    if wall_velocity is not None:
        from .d3q19 import W_EXACT64
        from .d3q19 import C as C19

        # The equilibrium that produced f_prev uses W_EXACT64 (see
        # d3q19._w_on), so the correction must use the same exact weights —
        # promoting the float32 W would leave a ~1e-10 force residual.
        c19 = C19.to(device=f.device)  # (19, 3), integer-valued
        w19 = W_EXACT64.to(device=f.device, dtype=f.dtype).view(19, 1, 1, 1)
        cu = (
            c19[:, 0].view(19, 1, 1, 1) * wall_velocity[0].unsqueeze(0)
            + c19[:, 1].view(19, 1, 1, 1) * wall_velocity[1].unsqueeze(0)
            + c19[:, 2].view(19, 1, 1, 1) * wall_velocity[2].unsqueeze(0)
        )
        # 2/cs^2 = 6 for D3Q19.  bfl_common adds wall_correction[d] to
        # f_bc[d], which lands in f_out[opp_d]; the correction for the
        # unknown opp_d population is corr[opp_d] = 2*rho*w*(c_opp.u_w)/cs^2
        # = -2*rho*w_d*(c_d.u_w)/cs^2, hence the negation.
        wall_correction = -(6.0 * wall_density.unsqueeze(0) * w19 * cu)

    f_out = bfl_bounce_back_common(
        f,
        f_prev,
        fluid_boundary_mask,
        q_field,
        lattice="D3Q19",
        wall_correction=wall_correction,
    )
    if isinstance(fraction, torch.Tensor):
        frac3 = fraction.unsqueeze(0).to(dtype=f_out.dtype)
        f_out = (1.0 - frac3) * f + frac3 * f_out
    elif float(fraction) != 1.0:
        f_out = (1.0 - float(fraction)) * f + float(fraction) * f_out

    if not (return_force or return_force_decomposition):
        return f_out

    # --- force ledger: recompute per-direction with the same convention.
    from .d3q19 import OPPOSITE as OPP19
    from .d3q19 import C as C19

    stat_acc = torch.zeros(3, dtype=f.dtype, device=f.device)
    corr_acc = torch.zeros(3, dtype=f.dtype, device=f.device)
    transparent_acc = torch.zeros(3, dtype=f.dtype, device=f.device)
    normal_acc = torch.zeros(3, dtype=f.dtype, device=f.device)
    tangential_acc = torch.zeros(3, dtype=f.dtype, device=f.device)
    unresolved_acc = torch.zeros(3, dtype=f.dtype, device=f.device)
    active_links = 0
    decomposed_links = 0
    decomposed_cell: torch.Tensor | None = None
    if force_normals is not None:
        nx_f, ny_f, nz_f = force_normals
        decomposed_cell = (nx_f != 0) | (ny_f != 0) | (nz_f != 0)

    for d in range(1, 19):
        mask_d = fluid_boundary_mask[d]
        if not mask_d.any():
            continue
        od = int(OPP19[d].item())
        cx, cy, cz = (int(v) for v in C19[d].tolist())
        # Promote float32 q to the population dtype, mirroring
        # bfl_bounce_back_common so the force ledger stays bit-consistent
        # with the populations it accounts for (identity for float32 runs).
        q_cell = q_field[d][mask_d].to(dtype=f.dtype)
        lin = q_cell < 0.5
        fp_d = f_prev[d][mask_d]
        fp_od = f_prev[od][mask_d]
        # Replicate bfl_bounce_back_common's arithmetic bit-for-bit
        # (including its torch.roll upstream gather and its
        # reciprocal-first quadratic form) so the force ledger matches the
        # populations it accounts for.
        fp_up = torch.roll(f_prev[d], shifts=(cz, cy, cx), dims=(0, 1, 2))[mask_d]
        q_safe = torch.where(lin, torch.ones_like(q_cell), q_cell)
        inv_2q = 1.0 / (2.0 * q_safe)
        f_bc_stat = torch.where(
            lin,
            2.0 * q_cell * fp_d + (1.0 - 2.0 * q_cell) * fp_up,
            fp_d * inv_2q + (2.0 * q_safe - 1.0) * inv_2q * fp_od,
        )
        corr_d = (
            wall_correction[d][mask_d] if wall_correction is not None else torch.zeros_like(q_cell)
        )
        f_streamed_unk = f[od][mask_d]
        if isinstance(fraction, torch.Tensor):
            frac_cell = fraction[mask_d]
        else:
            frac_cell = float(fraction)
        stat_link = fp_d + (1.0 - frac_cell) * f_streamed_unk + frac_cell * f_bc_stat
        corr_link = frac_cell * corr_d
        c_vec = torch.tensor([cx, cy, cz], dtype=f.dtype, device=f.device)
        active_links += int(mask_d.sum().item())
        stat_acc += c_vec * stat_link.sum()
        corr_acc += c_vec * corr_link.sum()
        transparent_acc += c_vec * (fp_d + f_streamed_unk).sum()

        if decomposed_cell is not None:
            dec = decomposed_cell[mask_d]
            total_link = stat_link + corr_link
            if bool(dec.any()):
                n_vec = torch.stack(
                    [nx_f[mask_d][dec], ny_f[mask_d][dec], nz_f[mask_d][dec]],
                    dim=0,
                ).to(dtype=f.dtype)
                n_hat = n_vec / n_vec.norm(dim=0, keepdim=True)
                link_vec = c_vec.unsqueeze(1) * total_link[dec].unsqueeze(0)
                fn_scalar = (link_vec * n_hat).sum(dim=0, keepdim=True)
                normal_acc += (fn_scalar * n_hat).sum(dim=1)
                tangential_acc += (link_vec - fn_scalar * n_hat).sum(dim=1)
                decomposed_links += int(dec.sum().item())
            if bool((~dec).any()):
                unresolved_acc += c_vec * total_link[~dec].sum()

    total_force = stat_acc + corr_acc
    # Wall frame: remove the background flux of a co-moving transparent wall
    # (free streaming would have written f[opp_d]; a wall moving with the
    # local flow exchanges no momentum).  Laboratory frame: the conservative
    # ledger that closes a fixed control volume.
    returned_force = total_force - transparent_acc if force_frame == "wall" else total_force
    force = (
        returned_force[0].detach(),
        returned_force[1].detach(),
        returned_force[2].detach(),
    )
    if not return_force_decomposition:
        return f_out, force

    frame_acc = total_force - stat_acc - corr_acc
    closure_vec = total_force - (stat_acc + corr_acc + frame_acc)
    max_component = float(closure_vec.abs().max().item())
    max_closure = float(closure_vec.norm().item())
    total_norm = float(total_force.norm().item())
    max_relative = max_closure / total_norm if total_norm > 0.0 else 0.0

    def _trip(t: torch.Tensor) -> tuple[float, float, float]:
        return (float(t[0].item()), float(t[1].item()), float(t[2].item()))

    decomposition = BFLForceDecomposition(
        active_links=active_links,
        decomposed_links=decomposed_links,
        undecomposed_links=active_links - decomposed_links,
        coverage_fraction=(decomposed_links / active_links if active_links else 0.0),
        unresolved_force=_trip(unresolved_acc),
        stationary_interpolation_force=_trip(stat_acc),
        moving_wall_population_correction_force=_trip(corr_acc),
        frame_correction_force=_trip(frame_acc),
        total_force=_trip(total_force),
        normal_force=_trip(normal_acc),
        tangential_force=_trip(tangential_acc),
        maximum_closure_error=max_closure,
        maximum_relative_closure_error=max_relative,
        maximum_component_closure_error=max_component,
    )
    return f_out, force, decomposition


__all__ = [
    "BFLForceDecomposition",
    "bouzidi_bounce_back_d3q19",
    "compute_q_cylinder_d3q19",
]
