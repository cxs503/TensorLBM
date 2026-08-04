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
from tensorlbm.d3q19 import macroscopic3d
from tensorlbm.evidence_io import stationarity_dict
from tensorlbm.external_open_boundary import non_equilibrium_far_field_bc_3d
from tensorlbm.force_convergence import assess_force_stationarity
from tensorlbm.interpolated_bc import compute_q_sphere
from tensorlbm.refinement import BoxRegion
from tensorlbm.solver3d import stream3d
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
# Geometry assembly
# ---------------------------------------------------------------------------

def build_sphere_geometry(
    nx: int, ny: int, nz: int,
    cx: float, cy: float, cz: float,
    radius: float, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coarse-grid sphere mask and its D3Q19 freeze mask ``(solid, solid_q)``.

    ``solid`` has shape ``(nz, ny, nx)``; ``solid_q`` is the 19-channel
    expansion used to freeze solid cells at their pre-collision state.
    """
    solid = sphere_mask(nx, ny, nz, cx, cy, cz, radius, device=device)
    if not bool(solid.any()):
        raise ValueError("no sphere cells on the coarse grid")
    solid_q = solid.unsqueeze(0).expand(19, nz, ny, nx).contiguous()
    return solid, solid_q


def build_fine_sphere(
    fine_shape: tuple[int, int, int],
    fine_center: tuple[float, float, float],
    radius: float,
    device: torch.device,
    ghost: int = 1,
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
    solid_q = solid_g.unsqueeze(0).expand(19, *solid_g.shape).contiguous()
    bfl_mask, bfl_q = compute_q_sphere(
        nx + 2 * ghost, ny + 2 * ghost, nz + 2 * ghost,
        *fine_center, radius, device=device,
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
    sphere = build_fine_sphere(shape, fine_center, radius, device, ghost=ghost)
    return shape, fine_center, radius, sphere


def build_l2_shell_geometry(
    c1_w: tuple[float, float, float],
    parent_shape: tuple[int, int, int],
    radius1: float,
    l2_margin: int,
    ratio: int,
    ghost: int = 1,
    device: torch.device = torch.device("cpu"),
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
    sphere = build_fine_sphere(s2, fc2, radius1 * ratio, device, ghost=ghost)
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

def root_advance(
    f: torch.Tensor, tau: float,
    solid_q: torch.Tensor, sigma: torch.Tensor, lattice_speed: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Root-level advance: collide, freeze coarse solid, stream, far-field + sponge.

    Returns ``(out, post_collision, collided)`` where ``post_collision`` is
    the frozen-solid state and ``collided`` the unfrozen one — the runners
    disagree on which is passed to ``AMRAdvanceResult``, so both are
    returned.
    """
    before = f
    collided = collide_cumulant_d3q19(f, tau, C_s=0.0)
    post_collision = torch.where(solid_q, before, collided)
    out = stream3d(post_collision)
    out = non_equilibrium_far_field_bc_3d(out, u_in=lattice_speed)
    out = apply_equilibrium_difference_sponge(
        out, sigma, velocity_target=(lattice_speed, 0.0, 0.0),
    )
    out = non_equilibrium_far_field_bc_3d(out, u_in=lattice_speed)
    return out, post_collision, collided


def bfl_sphere_advance(
    out: torch.Tensor, post_collision: torch.Tensor,
    bfl_mask: torch.Tensor, bfl_q: torch.Tensor,
    step: int, ramp_steps: int,
) -> torch.Tensor:
    """Bouzidi BFL curved-wall reconstruction after streaming (ramped wall)."""
    rho_post, ux_post, uy_post, uz_post = macroscopic3d(post_collision)
    activation = ramp_activation(step, ramp_steps)
    wall_velocity = (
        (1.0 - activation) * ux_post,
        (1.0 - activation) * uy_post,
        (1.0 - activation) * uz_post,
    )
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
) -> tuple[torch.Tensor, torch.Tensor, float | None]:
    """Fine-level advance: collide (freeze solid), stream, BFL, optional CV force.

    Returns ``(out, post_collision, cv_force)`` where ``cv_force`` is a float
    when ``sample_cv`` is true and ``None`` otherwise.
    """
    before = f
    collided = collide_cumulant_d3q19(f, tau, C_s=0.0)
    post_collision = torch.where(solid_q, before, collided)
    out = stream3d(post_collision)
    out = bfl_sphere_advance(out, post_collision, bfl_mask, bfl_q, step, ramp_steps)
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
