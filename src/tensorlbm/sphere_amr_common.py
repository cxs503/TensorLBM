"""Shared sphere AMR assembly for the AMR sphere validation runners.

Everything the four ``examples/amr_sphere_*_validate.py`` runners need to
build the sphere geometry on the coarse and fine grids, convert the sphere
centre between levels, assemble the control volume, and advance one level
with the frozen-solid + Bouzidi BFL treatment.

Coordinate conventions (mirroring ``examples/amr_sphere_drag_validate.py``):

* a level's fine tensor is ``physical_extent * ratio`` cells with a one-cell
  ghost layer on every side; the fine sphere centre in with-ghost indices is
  ``center * ratio - box_origin * ratio + ghost``
  (:func:`fine_center_l1`).
* the L2 box of a nested hierarchy is expressed in the L1 fine *with-ghost*
  tensor's indices (that tensor is interface 1's parent), so converting one
  more level applies the same formula with the parent coordinate already in
  the parent's with-ghost fine grid (:func:`fine_center_l2`).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tensorlbm.bfl_d3q19 import bouzidi_bounce_back_d3q19
from tensorlbm.boundaries3d import sphere_mask
from tensorlbm.control_volume_force import (
    box_control_volume,
    observe_control_volume_force,
)
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.evidence_io import stationarity_dict
from tensorlbm.external_open_boundary import non_equilibrium_far_field_bc_3d
from tensorlbm.force_convergence import assess_force_stationarity
from tensorlbm.refinement import BoxRegion
from tensorlbm.sphere_bfl_control_volume import schiller_naumann_cd
from tensorlbm.sponge_layer import apply_equilibrium_difference_sponge


@dataclass(frozen=True)
class FineSphere:
    """Sphere geometry on one fine level (physical, with-ghost, BFL)."""

    solid: torch.Tensor    # (nz, ny, nx) physical cells, no ghost
    solid_g: torch.Tensor  # with one-cell ghost layer
    solid_q: torch.Tensor  # (19, nz+2, ny+2, nx+2) collision freeze mask
    bfl_mask: torch.Tensor
    bfl_q: torch.Tensor


def ramp_activation(step: int, steps: int) -> float:
    """Linear ramp from 0 at step 0 to 1 at step == steps (1.0 when disabled)."""
    if steps <= 0:
        return 1.0
    return min(1.0, step / steps)


# ---------------------------------------------------------------------------
# Lattice dispatch helpers (D3Q19 keeps the exact legacy code path; D3Q27
# selects the D3Q27 kernel set).  The D3Q27 imports are lazy so that the
# D3Q19-only callers pay no import cost.
# ---------------------------------------------------------------------------

def _q_channels(lattice: str) -> int:
    """Number of population channels for a lattice name."""
    if lattice == "D3Q27":
        return 27
    return 19


def _equilibrium(
    rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor,
    device: torch.device, lattice: str,
) -> torch.Tensor:
    """Lattice-dispatched Maxwell-Boltzmann equilibrium."""
    if lattice == "D3Q27":
        from tensorlbm.d3q27 import equilibrium27
        return equilibrium27(rho, ux, uy, uz, device=device)
    from tensorlbm.d3q19 import equilibrium3d
    return equilibrium3d(rho, ux, uy, uz, device=device)


def _macroscopic(
    f: torch.Tensor, lattice: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lattice-dispatched macroscopic recovery ``(rho, ux, uy, uz)``."""
    if lattice == "D3Q27":
        from tensorlbm.d3q27 import macroscopic27
        return macroscopic27(f)
    from tensorlbm.d3q19 import macroscopic3d
    return macroscopic3d(f)


def _stream(f: torch.Tensor, lattice: str) -> torch.Tensor:
    """Lattice-dispatched streaming."""
    if lattice == "D3Q27":
        from tensorlbm.d3q27 import stream27_roll
        return stream27_roll(f)
    from tensorlbm.solver3d import stream3d
    return stream3d(f)


def _compute_q_sphere(
    nx: int, ny: int, nz: int,
    cx: float, cy: float, cz: float,
    radius: float, device: torch.device, lattice: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lattice-dispatched spherical BFL q-field ``(mask, q)``."""
    if lattice == "D3Q27":
        from tensorlbm.interpolated_bc_common import compute_q_sphere_27
        return compute_q_sphere_27(nx, ny, nz, cx, cy, cz, radius, device)
    from tensorlbm.interpolated_bc import compute_q_sphere
    return compute_q_sphere(nx, ny, nz, cx, cy, cz, radius, device)


# ---------------------------------------------------------------------------
# Geometry assembly
# ---------------------------------------------------------------------------

def build_sphere_geometry(
    nx: int, ny: int, nz: int,
    cx: float, cy: float, cz: float,
    radius: float, device: torch.device,
    lattice: str = "D3Q19",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coarse-grid sphere mask and its freeze mask ``(solid, solid_q)``.

    ``solid`` has shape ``(nz, ny, nx)``; ``solid_q`` is the
    ``_q_channels(lattice)``-channel expansion used to freeze solid cells at
    their pre-collision state.
    """
    solid = sphere_mask(nx, ny, nz, cx, cy, cz, radius, device=device)
    if not bool(solid.any()):
        raise ValueError("no sphere cells on the coarse grid")
    solid_q = solid.unsqueeze(0).expand(_q_channels(lattice), nz, ny, nx).contiguous()
    return solid, solid_q


def build_fine_sphere(
    fine_shape: tuple[int, int, int],
    fine_center: tuple[float, float, float],
    radius: float,
    device: torch.device,
    ghost: int = 1,
    lattice: str = "D3Q19",
) -> FineSphere:
    """Build the sphere mask + ghost freeze + BFL fields on one fine level.

    The mask is evaluated on the fine block's own coordinates (with-ghost
    convention ``fine_center``), so the returned solid only ever marks sphere
    cells the block actually covers — robust whether or not the block box
    contains the sphere centre (partial-shell case).
    """
    nz, ny, nx = fine_shape  # physical (nz, ny, nx), no ghost
    solid = sphere_mask(nx, ny, nz, *fine_center, radius, device=device)
    solid_g = torch.zeros(
        (nz + 2 * ghost, ny + 2 * ghost, nx + 2 * ghost),
        dtype=torch.bool, device=device,
    )
    solid_g[ghost:-ghost, ghost:-ghost, ghost:-ghost] = solid
    solid_q = solid_g.unsqueeze(0).expand(_q_channels(lattice), *solid_g.shape).contiguous()
    bfl_mask, bfl_q = _compute_q_sphere(
        nx + 2 * ghost, ny + 2 * ghost, nz + 2 * ghost,
        *fine_center, radius, device=device, lattice=lattice,
    )
    return FineSphere(solid, solid_g, solid_q, bfl_mask, bfl_q)


def fine_center_l1(
    center: float, box_origin: float, ratio: int, ghost: int = 1,
) -> float:
    """Sphere-centre coordinate on a level's fine with-ghost grid.

    The fine block covers coarse interval ``[box_origin, box_origin +
    extent)``; its physical fine cells start at fine index
    ``box_origin * ratio`` and the with-ghost tensor shifts every index by
    ``ghost``.  A coarse point ``center`` therefore lands at
    ``center * ratio - box_origin * ratio + ghost`` in with-ghost fine
    indices.
    """
    return center * ratio - box_origin * ratio + ghost


def fine_center_l2(
    parent_fine_center: float, box2_origin: float,
    ratio: int, ghost: int = 1,
) -> float:
    """Sphere-centre coordinate on the L2 fine grid, from the L1 with-ghost centre.

    The L2 box is expressed in the L1 fine *with-ghost* tensor's indices
    (that tensor is the L2 interface's parent).  Converting one more level is
    the same formula as :func:`fine_center_l1` with the parent coordinate
    already in the parent's with-ghost fine grid:
    ``parent_fine_center * ratio - box2_origin * ratio + ghost``.
    """
    return parent_fine_center * ratio - box2_origin * ratio + ghost


def build_fine_block_geometry(
    box: BoxRegion,
    center: tuple[float, float, float],
    radius_coarse: float,
    ratio: int,
    ghost: int = 1,
    device: torch.device = torch.device("cpu"),
    lattice: str = "D3Q19",
) -> tuple[tuple[int, int, int], tuple[float, float, float], float, FineSphere]:
    """Assemble one fine block's geometry from its coarse box.

    Returns ``(shape, fine_center, radius, sphere)`` with ``shape`` the
    physical ``(nz, ny, nx)`` extent, ``fine_center`` the sphere centre in
    with-ghost fine indices, ``radius = radius_coarse * ratio`` and
    ``sphere`` the :class:`FineSphere` assembly.
    """
    shape = (
        (box.z1 - box.z0) * ratio,
        (box.y1 - box.y0) * ratio,
        (box.x1 - box.x0) * ratio,
    )
    fine_center = (
        fine_center_l1(center[0], box.x0, ratio, ghost),
        fine_center_l1(center[1], box.y0, ratio, ghost),
        fine_center_l1(center[2], box.z0, ratio, ghost),
    )
    radius = radius_coarse * ratio
    sphere = build_fine_sphere(
        shape, fine_center, radius, device, ghost=ghost, lattice=lattice,
    )
    return shape, fine_center, radius, sphere


def build_l2_shell_geometry(
    c1_w: tuple[float, float, float],
    parent_shape: tuple[int, int, int],
    radius1: float,
    l2_margin: int,
    ratio: int,
    ghost: int = 1,
    device: torch.device = torch.device("cpu"),
    lattice: str = "D3Q19",
) -> tuple[BoxRegion, tuple[int, int, int], tuple[float, float, float], FineSphere]:
    """L2 surface-hugging shell geometry from the L1 with-ghost centre.

    ``c1_w`` is the sphere centre in the L1 fine *with-ghost* tensor (the
    L2 interface's parent) and ``parent_shape`` that tensor's
    ``(nz, ny, nx)`` shape.  Returns ``(box2, s2, fc2, sphere)`` where
    ``s2`` is the physical L2 shape, ``fc2`` the L2 with-ghost fine centre
    and ``sphere`` the :class:`FineSphere` assembly at radius
    ``radius1 * ratio``.
    """
    box2 = l2_shell_box(c1_w, parent_shape, radius1, l2_margin)
    s2 = (
        (box2.z1 - box2.z0) * ratio,
        (box2.y1 - box2.y0) * ratio,
        (box2.x1 - box2.x0) * ratio,
    )
    fc2 = (
        fine_center_l2(c1_w[0], box2.x0, ratio, ghost),
        fine_center_l2(c1_w[1], box2.y0, ratio, ghost),
        fine_center_l2(c1_w[2], box2.z0, ratio, ghost),
    )
    sphere = build_fine_sphere(
        s2, fc2, radius1 * ratio, device, ghost=ghost, lattice=lattice,
    )
    return box2, s2, fc2, sphere


def distinct_level_shapes(
    level_populations: Sequence[torch.Tensor], expected: int,
) -> tuple[torch.Size, ...]:
    """Population shapes of a nested solver, asserted pairwise distinct."""
    shapes = tuple(level.shape for level in level_populations)
    if len(set(shapes)) != expected:
        raise ValueError(
            "level population shapes must be distinct for shape-based advance "
            f"dispatch, got {shapes}",
        )
    return shapes


def level_index_of(f: torch.Tensor, level_shapes: tuple[torch.Size, ...]) -> int:
    """Index of the level whose population tensor shape matches ``f``."""
    for index, shape in enumerate(level_shapes):
        if f.shape == shape:
            return index
    raise ValueError(
        f"advance received unexpected population shape {tuple(f.shape)}",
    )


def _clamp_axis(lo: int, hi: int, limit: int, axis: str) -> tuple[int, int]:
    """Keep an L2 box strictly interior to its parent (L1 with-ghost) tensor."""
    lo = max(1, lo)
    hi = min(limit - 2, hi)
    if hi - lo < 3:
        raise ValueError(
            f"L2 box degenerate on {axis} axis ({lo}, {hi}) in parent limit {limit} "
            "-- reduce --l2-margin or enlarge the L1 block",
        )
    return lo, hi


def l2_shell_box(
    c1_w: tuple[float, float, float],
    parent_shape: tuple[int, int, int],
    radius1: float,
    l2_margin: int,
) -> BoxRegion:
    """Surface-hugging L2 box around the sphere, in L1 with-ghost coordinates.

    ``parent_shape`` is the L1 with-ghost tensor shape ``(nz, ny, nx)`` and
    ``c1_w`` the sphere centre in those coordinates.  The box spans
    ``radius1 + l2_margin`` around the centre, clamped strictly interior to
    the parent tensor.
    """
    half2 = int(math.floor(radius1 + l2_margin))
    x0_2, x1_2 = _clamp_axis(
        int(math.floor(c1_w[0] - half2)), int(math.ceil(c1_w[0] + half2)),
        parent_shape[2], "x",
    )
    y0_2, y1_2 = _clamp_axis(
        int(math.floor(c1_w[1] - half2)), int(math.ceil(c1_w[1] + half2)),
        parent_shape[1], "y",
    )
    z0_2, z1_2 = _clamp_axis(
        int(math.floor(c1_w[2] - half2)), int(math.ceil(c1_w[2] + half2)),
        parent_shape[0], "z",
    )
    return BoxRegion(x0_2, x1_2, y0_2, y1_2, z0_2, z1_2)


def build_control_volume(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radius: float,
    margin: int,
    device: torch.device,
) -> torch.Tensor:
    """Control volume around the sphere on the fine grid (with-ghost offset).

    Bounds are clamped to the strictly-interior range required by
    :func:`~tensorlbm.control_volume_force.box_control_volume` and an
    enclosure guard verifies the volume fully contains the sphere.
    """
    def _cv_lo(centre: float) -> int:
        return int(max(1, math.floor(centre - radius) - margin))

    def _cv_hi(centre: float, extent: int) -> int:
        return int(min(extent - 2, math.ceil(centre + radius) + margin + 1))

    nz, ny, nx = shape
    cv = box_control_volume(
        tuple(shape),
        x0=_cv_lo(center[0]), x1=_cv_hi(center[0], nx),
        y0=_cv_lo(center[1]), y1=_cv_hi(center[1], ny),
        z0=_cv_lo(center[2]), z1=_cv_hi(center[2], nz),
        device=device,
    )
    cv_bounds = {
        "x": (int(_cv_lo(center[0])), int(_cv_hi(center[0], nx))),
        "y": (int(_cv_lo(center[1])), int(_cv_hi(center[1], ny))),
        "z": (int(_cv_lo(center[2])), int(_cv_hi(center[2], nz))),
    }
    for name, (cv0, cv1) in cv_bounds.items():
        axis = 0 if name == "x" else 1 if name == "y" else 2
        centre = center[axis]
        if cv0 > math.floor(centre - radius) or \
           cv1 < math.ceil(centre + radius) + 1:
            raise ValueError(
                f"control volume does not enclose the sphere along {name} "
                f"(cv=[{cv0},{cv1})); enlarge the refined block or shrink "
                f"--cv-margin",
            )
    return cv


# ---------------------------------------------------------------------------
# Advance pieces shared by the static-block runners
# ---------------------------------------------------------------------------

def _collide(
    f: torch.Tensor, tau: float, collision: str | None, lattice: str,
    *,
    les_model: str | None = None,
    cs_smag: float = 0.05,
    cw_wale: float = 0.5,
) -> torch.Tensor:
    """Dispatch collision operator by name and lattice.

    ``collision``: "cumulant" | "cascaded" (explicit, no LES) or None to
    fall through to the LES dispatch (``les_model`` wale/smagorinsky) —
    required for high-Reynolds runs.
    """
    if collision is not None and collision == "cascaded":
        if lattice == "D3Q27":
            from tensorlbm.cascaded_collision import collide_cascaded_d3q27
            return collide_cascaded_d3q27(f, tau)
        from tensorlbm.cascaded_collision import collide_cascaded_d3q19
        return collide_cascaded_d3q19(f, tau)
    if collision is not None and collision == "cumulant":
        if lattice == "D3Q27":
            from tensorlbm.cumulant import collide_cumulant_d3q27
            return collide_cumulant_d3q27(f, tau, C_s=0.0)
        from tensorlbm.cumulant import collide_cumulant_d3q19
        return collide_cumulant_d3q19(f, tau, C_s=0.0)
    # LES dispatch (collision is None)
    if lattice == "D3Q27":
        if les_model == "wale":
            from tensorlbm.turbulence import collide_wale_bgk27
            return collide_wale_bgk27(f, tau, C_w=cw_wale)
        from tensorlbm.turbulence import collide_smagorinsky_bgk27
        return collide_smagorinsky_bgk27(f, tau, C_s=cs_smag)
    if les_model == "wale":
        from tensorlbm.turbulence import collide_wale_mrt3d
        return collide_wale_mrt3d(f, tau, C_w=cw_wale)
    from tensorlbm.turbulence import collide_smagorinsky_mrt3d
    return collide_smagorinsky_mrt3d(f, tau, C_s=cs_smag)


def root_advance(
    f: torch.Tensor, tau: float,
    solid_q: torch.Tensor, sigma: torch.Tensor, lattice_speed: float,
    *,
    collision: str | None = "cumulant",
    lattice: str = "D3Q19",
    les_model: str | None = None,
    cs_smag: float = 0.05,
    cw_wale: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Root-level advance: collide, freeze coarse solid, stream, far-field + sponge.

    Returns ``(out, post_collision, collided)`` where ``post_collision`` is
    the frozen-solid state and ``collided`` the unfrozen one — the runners
    disagree on which is passed to ``AMRAdvanceResult``, so both are
    returned.
    """
    before = f
    collided = _collide(
        f, tau, collision, lattice,
        les_model=les_model, cs_smag=cs_smag, cw_wale=cw_wale,
    )
    post_collision = torch.where(solid_q, before, collided)
    out = _stream(post_collision, lattice)
    out = non_equilibrium_far_field_bc_3d(out, u_in=lattice_speed)
    out = apply_equilibrium_difference_sponge(
        out, sigma, velocity_target=(lattice_speed, 0.0, 0.0),
    )
    out = non_equilibrium_far_field_bc_3d(out, u_in=lattice_speed)
    return out, post_collision, collided


def _bouzidi_bounce_back_d3q27(
    f: torch.Tensor,
    f_prev: torch.Tensor,
    fluid_boundary_mask: torch.Tensor,
    q_field: torch.Tensor,
    *,
    wall_velocity: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    wall_density: torch.Tensor | None = None,
    boundary_fraction: float = 1.0,
    return_force: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[float, float, float]]:
    """D3Q27 full-stencil Bouzidi BFL reconstruction (all 27 directions).

    Mirrors ``tensorlbm.bfl_d3q19.bouzidi_bounce_back_d3q19`` exactly —
    same linear (q < 0.5) / quadratic (q >= 0.5) interpolation, same
    moving-wall population correction (cs² = 1/3, so the 6.0 / 3.0 factors
    are lattice-independent) and the same laboratory-frame link momentum
    exchange — but uses the D3Q27 OPPOSITE map, velocity set and weights,
    looping over the 26 non-rest directions.  Kept module-private so the
    read-only core library (``bfl_d3q19.py``) is untouched.
    """
    from tensorlbm.d3q27 import OPPOSITE, W, C
    if wall_velocity is not None and wall_density is None:
        raise ValueError("wall_density is required with wall_velocity")
    if not 0.0 <= boundary_fraction <= 1.0:
        raise ValueError("boundary_fraction must be in [0,1]")
    opp = OPPOSITE.to(f.device)
    weights = W.to(device=f.device, dtype=f.dtype)
    f_out = f.clone()
    force_x = torch.zeros((), device=f.device, dtype=f.dtype)
    force_y = torch.zeros_like(force_x)
    force_z = torch.zeros_like(force_x)

    for d in range(1, 27):  # skip rest direction
        opp_d = int(opp[d].item())

        mask = fluid_boundary_mask[d]
        if not mask.any():
            continue

        q_cell = q_field[d][mask]
        mask_lin = q_cell < 0.5
        mask_quad = ~mask_lin

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
            assert wall_density is not None  # guaranteed by the guard above
            uwx, uwy, uwz = wall_velocity
            c_dot_uw = (
                float(dcx) * uwx[mask]
                + float(dcy) * uwy[mask]
                + float(dcz) * uwz[mask]
            )
            rho_w = wall_density[mask]
            moving_base = weights[d] * rho_w * c_dot_uw
            # Same moving-wall correction as D3Q19 (cs² = 1/3): at q=.5 both
            # branches reduce to f_opp = f_d - 6*w*rho*(c_d·u_wall).
            f_bc_lin = f_bc_lin - 6.0 * moving_base
            f_bc_quad = f_bc_quad - (3.0 / safe_q) * moving_base

        f_bc = torch.where(mask_lin, f_bc_lin, f_bc_quad)

        if return_force and boundary_fraction > 0.0:
            # Laboratory-frame discrete momentum exchange: c_d*f_d - c_opp*f_opp
            # = c_d*(f_d + f_opp) with the unknown population set to f_bc.
            exchange_sum = fp_d + f_bc
            link_fx = float(dcx) * exchange_sum
            link_fy = float(dcy) * exchange_sum
            link_fz = float(dcz) * exchange_sum
            link_fx = boundary_fraction * link_fx
            link_fy = boundary_fraction * link_fy
            link_fz = boundary_fraction * link_fz
            force_x = force_x + link_fx.sum()
            force_y = force_y + link_fy.sum()
            force_z = force_z + link_fz.sum()

        # Set f[opp_d] (the UNKNOWN population, from solid toward fluid).
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
        return f_out, total_force
    return f_out


def bfl_sphere_advance(
    out: torch.Tensor, post_collision: torch.Tensor,
    bfl_mask: torch.Tensor, bfl_q: torch.Tensor,
    step: int, ramp_steps: int,
    lattice: str = "D3Q19",
) -> torch.Tensor:
    """Bouzidi BFL curved-wall reconstruction after streaming (ramped wall)."""
    rho_post, ux_post, uy_post, uz_post = _macroscopic(post_collision, lattice)
    activation = ramp_activation(step, ramp_steps)
    wall_velocity = (
        (1.0 - activation) * ux_post,
        (1.0 - activation) * uy_post,
        (1.0 - activation) * uz_post,
    )
    if lattice == "D3Q27":
        out, _bfl_force = _bouzidi_bounce_back_d3q27(
            out, post_collision, bfl_mask, bfl_q,
            wall_velocity=wall_velocity, wall_density=rho_post,
            return_force=True,
        )
        return out
    out, _bfl_force = bouzidi_bounce_back_d3q19(
        out, post_collision, bfl_mask, bfl_q,
        wall_velocity=wall_velocity, wall_density=rho_post,
        return_force=True,
    )
    return out


def fine_sphere_advance(
    f: torch.Tensor, tau: float,
    *,
    solid_q: torch.Tensor, bfl_mask: torch.Tensor, bfl_q: torch.Tensor,
    step: int, ramp_steps: int,
    sample_cv: bool = False,
    cv: torch.Tensor | None = None,
    solid_g: torch.Tensor | None = None,
    collision: str | None = "cumulant",
    lattice: str = "D3Q19",
    les_model: str | None = None,
    cs_smag: float = 0.05,
    cw_wale: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, float | None]:
    """Fine-level advance: collide (freeze solid), stream, BFL, optional CV force.

    Returns ``(out, post_collision, cv_force)`` where ``cv_force`` is a float
    when ``sample_cv`` is true and ``None`` otherwise.
    """
    before = f
    collided = _collide(
        f, tau, collision, lattice,
        les_model=les_model, cs_smag=cs_smag, cw_wale=cw_wale,
    )
    post_collision = torch.where(solid_q, before, collided)
    out = _stream(post_collision, lattice)
    out = bfl_sphere_advance(
        out, post_collision, bfl_mask, bfl_q, step, ramp_steps, lattice=lattice,
    )
    cv_force = None
    if sample_cv:
        cv_force = cv_force_from(before, out, post_collision, cv, solid_g)
    return out, post_collision, cv_force


def cv_force_from(
    f_before: torch.Tensor,
    f_after: torch.Tensor,
    f_post_collision: torch.Tensor,
    cv: torch.Tensor,
    solid: torch.Tensor,
) -> float:
    """Streamwise control-volume force on the body over one LBM step."""
    return float(observe_control_volume_force(
        f_before, f_after, f_post_collision, cv, solid=solid,
    ).force_on_body[0].item())


# ---------------------------------------------------------------------------
# Final statistics
# ---------------------------------------------------------------------------

def summarize_force_history(
    force_samples: list[float],
    dynamic_area: float,
    reynolds: float,
    statistics_window_steps: int = 0,
) -> dict[str, object]:
    """Cd, Schiller-Naumann reference, error and stationarity from samples."""
    statistics_window = statistics_window_steps or len(force_samples)
    selected = force_samples[-statistics_window:]
    mean_force = sum(selected) / len(selected)
    cd = mean_force / dynamic_area
    reference = schiller_naumann_cd(reynolds)
    cd_history = [f_ / dynamic_area for f_ in selected]
    stationarity = assess_force_stationarity(
        cd_history, block_size=max(1, len(cd_history) // 8),
    )
    reference_error = abs(cd - reference) / reference * 100.0
    return {
        "cd": cd,
        "reference_cd": reference,
        "reference_error_pct": reference_error,
        "mean_force_lu": mean_force,
        "stationarity": stationarity_dict(stationarity),
    }


__all__ = [
    "FineSphere",
    "_bouzidi_bounce_back_d3q27",
    "bfl_sphere_advance",
    "build_control_volume",
    "build_fine_block_geometry",
    "build_fine_sphere",
    "build_l2_shell_geometry",
    "build_sphere_geometry",
    "cv_force_from",
    "distinct_level_shapes",
    "fine_center_l1",
    "fine_center_l2",
    "fine_sphere_advance",
    "l2_shell_box",
    "level_index_of",
    "ramp_activation",
    "root_advance",
    "summarize_force_history",
]
