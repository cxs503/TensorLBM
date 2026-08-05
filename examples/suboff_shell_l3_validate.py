#!/usr/bin/env python3
"""DARPA SUBOFF three-level body-fitted shell AMR resistance candidate.

Route-C variant of :file:`suboff_shell_amr_validate.py` extended to a
three-level strictly nested hierarchy (:class:`NestedStaticBlockAMR3D`):

  level 0  root coarse grid                 — external domain, far-field BC,
                                              sponge, coarse hull frozen
  level 1  L1 body-fitted shell + wake      — hull (length x2) + shell_margin
                                              + downstream wake, regenerated
                                              from CAD in block-local fine
                                              coordinates
  level 2  L2 surface-hugging shell         — hull-proximity shell around the
      (2:1)                                  L1 fine hull (margin l2_margin),
                                              hull length x4
  level 3  L3 super-hugging shell           — hull-proximity shell around the
      (2:1)                                  L2 fine hull (margin l3_margin),
                                              hull length x8

Every level's hull is the DARPA CAD profile re-voxelised in that level's
block-local fine coordinates (``build_suboff_mask`` with the length doubled
per level), so the nested blocks never repeat a coarser voxelisation.

Wall treatment is deliberately **finest-level only**: the BFL interpolated
bounce-back + wall function (``bfl_wall_function_3d``, with the analytical
SUBOFF surface normals and the BFL q-field of the L3 hull) is applied on L3
only, together with the control-volume drag sampling.  L1 and L2 use the
plain half-way bounce-back (``bounce_back_cells_3d``) — this avoids the
double wall treatment warned about in the WALL_MODEL skill (a wall model on
two levels would double wall effects).

Coordinate conventions (mirroring ``examples/amr_sphere_shell_l3_validate.py``
and the single-level ``suboff_shell_amr_validate.py``):

* interface ``i``'s box lives in its parent tensor's allocated coordinates;
  the parent of interface 1 is L1's fine tensor *with* its one-cell ghost
  layer, so the L2 box is expressed in L1 with-ghost indices (and likewise
  the L3 box in L2 with-ghost indices).
* hull centre on the L1 fine grid: fc1 = (cx*2 - box.x0*2 + g, ...).
* hull centre in L1 with-ghost coordinates: c1_w = fc1 + g (the centre of
  the frozen-solid hull in the L1 fine tensor).
* hull centre on the L2 fine grid: fc2 = (c1_w*2 - box2.x0*2 + g, ...),
  hull length x4; then c2_w = fc2 + g and fc3 likewise for L3 (length x8).

Run (CPU smoke):::

    PYTHONPATH=src .venv/bin/python examples/suboff_shell_l3_validate.py \\
        --device cpu --hull-length 50 --speed-knots 5.92 --steps 60 \\
        --warmup-steps 15 --ramp-steps 10 --shell-margin 5 --wake-cells 15 \\
        --l2-margin 4 --l3-margin 3 --sponge-width 8 --collision cumulant \\
        --report-interval 20 --output /tmp/suboff_l3_smoke.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from suboff_experimental_resistance import (
    MODEL_LENGTH_M,
    experimental_point,
    force_scale_newton,
    smooth_ramp_factor,
)
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.cascaded_collision import collide_cascaded_d3q19
from tensorlbm.control_volume_force import (
    box_control_volume,
    observe_control_volume_force,
)
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.drag_pressure import SurfaceMesh, get_near_wall_3d
from tensorlbm.external_open_boundary import non_equilibrium_far_field_bc_3d
from tensorlbm.force_convergence import assess_force_stationarity
from tensorlbm.interpolated_bc_suboff import compute_q_suboff
from tensorlbm.population_positivity import limit_nonequilibrium_for_positivity
from tensorlbm.refinement import BoxRegion, HullProximityRegion, WakeRegion
from tensorlbm.solver3d import stream3d
from tensorlbm.sponge_layer import (
    apply_equilibrium_difference_sponge,
    build_sponge_sigma_3d,
)
from tensorlbm.static_block_amr import (
    AMRAdvanceResult,
    NestedStaticBlockAMR3D,
    StaticBlockAMRConfig,
)
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask
from tensorlbm.surface_area_weights import bfl_surface_area_weights
from tensorlbm.turbulence import collide_smagorinsky_mrt3d, collide_wale_mrt3d
from tensorlbm.wall_model import (
    WALL_TRACTION_SOURCE_SCHEME,
    bfl_wall_function_3d,
    physical_wall_lattice_viscosity,
)

SUBOFF_L_D = 8.57  # DARPA SUBOFF L/D (used for the fine hull radius)
RATIO = 2
GHOST = 1


def _json_safe(value: object) -> object:
    """Recursively map non-finite floats to None for strict JSON validity.

    ``ForceStationarityReport`` may hold ``math.inf`` (autocorrelation
    fields) and tuples; ``json.dumps`` would emit non-standard ``Infinity``
    tokens.
    """
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _clamp_axis(lo: int, hi: int, limit: int, axis: str) -> tuple[int, int]:
    """Keep a nested box strictly interior to its parent tensor.

    Mirrors ``sphere_amr_common._clamp_axis``: the box must stay inside
    ``[1, limit - 2]`` of the parent (with-ghost) tensor so the coarse-fine
    interface never touches the domain edge or the hull.
    """
    lo = max(1, lo)
    hi = min(limit - 2, hi)
    if hi - lo < 3:
        raise ValueError(
            f"nested box degenerate on {axis} axis ({lo}, {hi}) in parent "
            f"limit {limit} -- reduce --l2-margin/--l3-margin or enlarge the "
            f"parent block",
        )
    return lo, hi


def _level_shapes(
    level_populations: tuple[torch.Tensor, ...],
) -> tuple[torch.Size, ...]:
    """Population shapes of the 4-level hierarchy, asserted pairwise distinct."""
    shapes = tuple(level.shape for level in level_populations)
    if len(set(shapes)) != 4:
        raise ValueError(
            "level population shapes must be distinct for shape-based advance "
            f"dispatch, got {shapes}",
        )
    return shapes


def _level_index_of(f: torch.Tensor, shapes: tuple[torch.Size, ...]) -> int:
    """Index of the level whose population tensor shape matches ``f``."""
    for index, shape in enumerate(shapes):
        if f.shape == shape:
            return index
    raise ValueError(
        f"advance received unexpected population shape {tuple(f.shape)}",
    )


def _with_ghost(solid: torch.Tensor, ghost: int = GHOST) -> torch.Tensor:
    """Embed a physical ``(nz, ny, nx)`` bool mask into a with-ghost tensor."""
    solid_g = torch.zeros(
        tuple(size + 2 * ghost for size in solid.shape),
        dtype=torch.bool,
        device=solid.device,
    )
    solid_g[ghost:-ghost, ghost:-ghost, ghost:-ghost] = solid
    return solid_g


def _shell_box(
    solid_g: torch.Tensor,
    margin: int,
    tag: str,
) -> BoxRegion:
    """Bounding box of the hull-proximity shell in the parent with-ghost grid."""
    shell_mask = HullProximityRegion(solid_g, margin=margin).expand_mask()
    indices = shell_mask.nonzero(as_tuple=False)
    if indices.numel() == 0:
        raise ValueError(
            f"empty {tag} surface-hugging shell; reduce the {tag} margin",
        )
    z0, y0, x0 = (
        int(indices[:, axis].min().item()) for axis in range(3)
    )
    z1, y1, x1 = (
        int(indices[:, axis].max().item()) + 1 for axis in range(3)
    )
    nz, ny, nx = solid_g.shape
    x0, x1 = _clamp_axis(x0, x1, nx, "x")
    y0, y1 = _clamp_axis(y0, y1, ny, "y")
    z0, z1 = _clamp_axis(z0, z1, nz, "z")
    return BoxRegion(x0, x1, y0, y1, z0, z1)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hull-type", choices=("bare_hull", "full"), default="bare_hull")
    p.add_argument("--speed-knots", type=float, default=5.92)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--nx", type=int, default=200)
    p.add_argument("--ny", type=int, default=80)
    p.add_argument("--nz", type=int, default=80)
    p.add_argument("--hull-length", type=float, default=80.0)
    p.add_argument("--center-x-fraction", type=float, default=0.35)
    p.add_argument(
        "--shell-margin",
        type=int,
        default=8,
        help="L1 body-fitted shell thickness in coarse cells around the hull "
        "surface (HullProximityRegion margin)",
    )
    p.add_argument(
        "--wake-cells",
        type=int,
        default=40,
        help="downstream wake extension in coarse cells behind the hull",
    )
    p.add_argument(
        "--l2-margin",
        type=int,
        default=8,
        help="L2 surface-hugging shell thickness (L1 fine coords)",
    )
    p.add_argument(
        "--l3-margin",
        type=int,
        default=6,
        help="L3 surface-hugging shell thickness (L2 fine coords)",
    )
    p.add_argument(
        "--cv-margin",
        type=int,
        default=4,
        help="control-volume margin around the finest (L3) hull; clamped to "
        "the largest margin that fits strictly inside the L3 block",
    )
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--warmup-steps", type=int, default=1500)
    p.add_argument("--ramp-steps", type=int, default=800)
    p.add_argument("--report-interval", type=int, default=100)
    p.add_argument("--statistics-window-steps", type=int, default=0)
    p.add_argument(
        "--ghost-interpolation",
        choices=("injection", "trilinear"),
        default="injection",
    )
    p.add_argument("--sponge-width", type=int, default=12)
    p.add_argument("--sponge-strength", type=float, default=0.2)
    p.add_argument("--lattice-speed", type=float, default=0.06)
    p.add_argument("--resolved-reynolds", type=float, default=2.0e6)
    p.add_argument("--nu-water", type=float, default=1.004e-6)
    p.add_argument("--rho-water", type=float, default=998.2)
    p.add_argument(
        "--collision",
        choices=("cumulant", "cascaded"),
        default="cumulant",
        help="D3Q19 collision operator applied on every level (default cumulant)",
    )
    p.add_argument(
        "--collision-model",
        choices=("mrt_les", "cumulant_smagorinsky"),
        default=None,
        help="legacy LES collision model (used only when --collision is unset)",
    )
    p.add_argument("--les-model", choices=("wale", "smagorinsky"), default="wale")
    p.add_argument("--cs-smag", type=float, default=0.05)
    p.add_argument("--cw-wale", type=float, default=0.5)
    p.add_argument(
        "--wall-law", choices=("log", "reichardt", "musker"), default="reichardt",
    )
    p.add_argument("--wall-distance", type=float, default=0.5)
    p.add_argument("--stress-exchange-distance", type=float, default=0.0)
    p.add_argument("--disable-reflux", action="store_true")
    p.add_argument("--disable-positivity-limiter", action="store_true")
    p.add_argument("--maximum-reflux-correction-fraction", type=float, default=0.2)
    p.add_argument("--output", required=True)
    return p


def run(args: argparse.Namespace) -> dict:
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("warmup-steps must lie in [0, steps)")
    if args.shell_margin < 1:
        raise ValueError("shell-margin must be positive")
    if args.wake_cells < 0:
        raise ValueError("wake-cells must be non-negative")
    if args.l2_margin < 1:
        raise ValueError("l2-margin must be positive")
    if args.l3_margin < 1:
        raise ValueError("l3-margin must be positive")
    if args.cv_margin < 1:
        raise ValueError("cv-margin must be positive")
    if args.disable_reflux:
        raise ValueError(
            "nested static-block AMR requires reflux on every interface; "
            "remove --disable-reflux",
        )
    if not 0.0 < args.maximum_reflux_correction_fraction <= 1.0:
        raise ValueError("maximum-reflux-correction-fraction must lie in (0,1]")
    if not 0.0 < args.center_x_fraction < 1.0:
        raise ValueError("center-x-fraction must lie in (0,1)")

    device = torch.device(args.device)
    point = experimental_point(args.hull_type, args.speed_knots)
    shape = (args.nz, args.ny, args.nx)
    center = (args.nx * args.center_x_fraction, args.ny / 2.0, args.nz / 2.0)
    if (
        center[0] - args.hull_length / 2.0 <= 1
        or center[0] + args.hull_length / 2.0 >= args.nx - 1
    ):
        raise ValueError("SUBOFF hull does not fit inside the streamwise domain")
    config = SuboffConfig()

    # ---- coarse hull (level-0 freeze mask for coarse cells the fine block
    # ---- does not own).
    solid_coarse, coarse_geometry = build_suboff_mask(
        args.hull_type, args.nx, args.ny, args.nz,
        cx=center[0], cy=center[1], cz=center[2], length=args.hull_length,
        config=config, device=device,
    )
    if not bool(solid_coarse.any()):
        raise ValueError("no SUBOFF cells on the coarse grid")
    solid_coarse_q = solid_coarse.unsqueeze(0).expand(19, *shape).contiguous()

    # ---- L1 (level 1) body-fitted shell refinement region: hull-proximity
    # ---- surface shell + downstream wake, clipped laterally to the shell
    # ---- extent (identical to the single-level runner).
    shell_mask = HullProximityRegion(
        solid_coarse, margin=args.shell_margin,
    ).expand_mask()
    wake_mask = WakeRegion(
        solid_coarse, extend_x=args.wake_cells,
    ).expand_mask()
    shell_idx = shell_mask.nonzero(as_tuple=False)
    if shell_idx.numel() == 0:
        raise ValueError(
            "empty shell refinement region; adjust --shell-margin",
        )
    sz0, sy0 = int(shell_idx[:, 0].min().item()), int(shell_idx[:, 1].min().item())
    sz1, sy1 = int(shell_idx[:, 0].max().item()), int(shell_idx[:, 1].max().item())
    wake_mask[:sz0, :, :] = False
    wake_mask[sz1 + 1:, :, :] = False
    wake_mask[:, :sy0, :] = False
    wake_mask[:, sy1 + 1:, :] = False
    refine_mask = shell_mask | wake_mask
    indices = refine_mask.nonzero(as_tuple=False)
    if indices.numel() == 0:
        raise ValueError(
            "empty shell+wake refinement region; adjust --shell-margin/--wake-cells",
        )
    z_min, y_min, x_min = (int(indices[:, a].min().item()) for a in range(3))
    z_max, y_max, x_max = (int(indices[:, a].max().item()) + 1 for a in range(3))
    pad = 2  # keep the coarse-fine interface off the hull surface
    x0 = max(1, x_min - pad)
    x1 = min(args.nx - 1, x_max + pad)
    y0 = max(1, y_min - pad)
    y1 = min(args.ny - 1, y_max + pad)
    z0 = max(1, z_min - pad)
    z1 = min(args.nz - 1, z_max + pad)
    if min(x1 - x0, y1 - y0, z1 - z0) < 3:
        raise ValueError(
            "refinement box too small: try a larger --wake-cells or a smaller "
            "--shell-margin",
        )
    box1 = BoxRegion(x0, x1, y0, y1, z0, z1)

    # ---- physics (identical similarity map to suboff_static_amr_resistance).
    physical_re = point.speed_mps * MODEL_LENGTH_M / args.nu_water
    collision_re = args.resolved_reynolds or physical_re
    nu_coarse = args.lattice_speed * args.hull_length / collision_re
    wall_nu_fine3 = physical_wall_lattice_viscosity(
        args.lattice_speed, args.hull_length * 8.0, physical_re,
    )
    tau_coarse = 0.5 + 3.0 * nu_coarse

    rho = torch.ones(shape, device=device)
    ux = torch.full_like(rho, args.lattice_speed)
    zero = torch.zeros_like(rho)
    coarse_f = equilibrium3d(rho, ux, zero, zero, device=device)

    # ---- L1 (level 1) hull geometry: CAD re-voxelised in block-local fine
    # ---- coordinates, hull length x2 (never a coarse voxel repeat).
    s1 = (
        (z1 - z0) * RATIO,
        (y1 - y0) * RATIO,
        (x1 - x0) * RATIO,
    )
    fc1 = (
        center[0] * RATIO - x0 * RATIO + GHOST,
        center[1] * RATIO - y0 * RATIO + GHOST,
        center[2] * RATIO - z0 * RATIO + GHOST,
    )
    l1_solid, l1_geometry = build_suboff_mask(
        args.hull_type, s1[2], s1[1], s1[0],
        cx=fc1[0], cy=fc1[1], cz=fc1[2],
        length=args.hull_length * 2.0, config=config, device=device,
    )
    if not bool(l1_solid.any()):
        raise RuntimeError("L1 block contains no SUBOFF cells")
    l1_solid_g = _with_ghost(l1_solid)
    c1_w = tuple(value + GHOST for value in fc1)

    # ---- L2 (level 2) surface-hugging shell, planned in L1 with-ghost
    # ---- coordinates from the L1 fine hull.
    box2 = _shell_box(l1_solid_g, args.l2_margin, "L2")
    x0_2, x1_2, y0_2, y1_2, z0_2, z1_2 = (
        box2.x0, box2.x1, box2.y0, box2.y1, box2.z0, box2.z1,
    )
    s2 = (
        (z1_2 - z0_2) * RATIO,
        (y1_2 - y0_2) * RATIO,
        (x1_2 - x0_2) * RATIO,
    )
    fc2 = (
        c1_w[0] * RATIO - x0_2 * RATIO + GHOST,
        c1_w[1] * RATIO - y0_2 * RATIO + GHOST,
        c1_w[2] * RATIO - z0_2 * RATIO + GHOST,
    )
    l2_solid, l2_geometry = build_suboff_mask(
        args.hull_type, s2[2], s2[1], s2[0],
        cx=fc2[0], cy=fc2[1], cz=fc2[2],
        length=args.hull_length * 4.0, config=config, device=device,
    )
    if not bool(l2_solid.any()):
        raise RuntimeError("L2 block contains no SUBOFF cells")
    l2_solid_g = _with_ghost(l2_solid)
    c2_w = tuple(value + GHOST for value in fc2)

    # ---- L3 (level 3) super-hugging shell, planned in L2 with-ghost
    # ---- coordinates from the L2 fine hull.
    box3 = _shell_box(l2_solid_g, args.l3_margin, "L3")
    x0_3, x1_3, y0_3, y1_3, z0_3, z1_3 = (
        box3.x0, box3.x1, box3.y0, box3.y1, box3.z0, box3.z1,
    )
    s3 = (
        (z1_3 - z0_3) * RATIO,
        (y1_3 - y0_3) * RATIO,
        (x1_3 - x0_3) * RATIO,
    )
    fc3 = (
        c2_w[0] * RATIO - x0_3 * RATIO + GHOST,
        c2_w[1] * RATIO - y0_3 * RATIO + GHOST,
        c2_w[2] * RATIO - z0_3 * RATIO + GHOST,
    )
    l3_solid, l3_geometry = build_suboff_mask(
        args.hull_type, s3[2], s3[1], s3[0],
        cx=fc3[0], cy=fc3[1], cz=fc3[2],
        length=args.hull_length * 8.0, config=config, device=device,
    )
    if not bool(l3_solid.any()):
        raise RuntimeError("L3 block contains no SUBOFF cells")
    l3_solid_g = _with_ghost(l3_solid)

    # ---- tau chain: interface i+1's tau_coarse must equal interface i's
    # ---- tau_fine (0.5 + 2*(tau-0.5) recursion, convective scaling).
    config1 = StaticBlockAMRConfig(
        box1, tau_coarse=tau_coarse, reflux=True,
        maximum_reflux_correction_fraction=(
            args.maximum_reflux_correction_fraction
        ),
        ghost_interpolation=args.ghost_interpolation,
    )
    config2 = StaticBlockAMRConfig(
        box2, tau_coarse=config1.tau_fine, reflux=True,
        maximum_reflux_correction_fraction=(
            args.maximum_reflux_correction_fraction
        ),
        ghost_interpolation=args.ghost_interpolation,
    )
    config3 = StaticBlockAMRConfig(
        box3, tau_coarse=config2.tau_fine, reflux=True,
        maximum_reflux_correction_fraction=(
            args.maximum_reflux_correction_fraction
        ),
        ghost_interpolation=args.ghost_interpolation,
    )
    tau_fine3 = config3.tau_fine

    amr = NestedStaticBlockAMR3D(
        coarse_f,
        (config1, config2, config3),
        fine_solids=(l1_solid, l2_solid, l3_solid),
    )
    level_shapes = _level_shapes(amr.level_populations)

    # ---- finest-level (L3) BFL wall-function geometry: analytical SUBOFF
    # ---- normals + BFL q-field evaluated on the L3 with-ghost grid.
    print("building L3 BFL link distances", flush=True)
    bfl_mask3, bfl_q3 = compute_q_suboff(
        l3_solid_g.shape[2], l3_solid_g.shape[1], l3_solid_g.shape[0],
        *fc3, args.hull_length * 8.0,
        hull_type=args.hull_type, config=config, device=device,
        solid_mask=l3_solid_g,
    )
    l3_near = get_near_wall_3d(l3_solid_g)
    l3_surface = SurfaceMesh.from_suboff(
        l3_solid_g, l3_near, *fc3,
        args.hull_length * 8.0, args.hull_length * 8.0 / (2.0 * SUBOFF_L_D),
        config=config,
    )
    l3_area_weight, surface_area_diagnostics = bfl_surface_area_weights(
        bfl_mask3,
        (l3_surface.nx_n, l3_surface.ny_n, l3_surface.nz_n),
        reference_area=float(l3_geometry["wetted_area_lu2"]),
        boundary_mask=l3_near,
    )
    l3_surface.dA = l3_area_weight
    solid_q1 = l1_solid_g.unsqueeze(0).expand(19, *l1_solid_g.shape).contiguous()
    solid_q2 = l2_solid_g.unsqueeze(0).expand(19, *l2_solid_g.shape).contiguous()
    solid_q3 = l3_solid_g.unsqueeze(0).expand(19, *l3_solid_g.shape).contiguous()

    # ---- control volume on the finest level (L3 with-ghost tensor).  The
    # ---- L3 block is a thin shell around the hull, so the requested margin
    # ---- is clamped to the largest margin that keeps the CV strictly
    # ---- interior while still enclosing the hull.
    l3_indices = l3_solid_g.nonzero(as_tuple=False)
    if l3_indices.numel() == 0:
        raise RuntimeError("L3 block contains no SUBOFF cells")
    z_min_f, y_min_f, x_min_f = (
        int(l3_indices[:, axis].min().item()) for axis in range(3)
    )
    z_max_f, y_max_f, x_max_f = (
        int(l3_indices[:, axis].max().item()) + 1 for axis in range(3)
    )
    nz_f, ny_f, nx_f = l3_solid_g.shape
    resolved_margin = args.cv_margin
    for lower, upper, size in (
        (x_min_f, x_max_f, nx_f),
        (y_min_f, y_max_f, ny_f),
        (z_min_f, z_max_f, nz_f),
    ):
        available = min(lower - GHOST - 1, size - GHOST - 1 - upper)
        resolved_margin = min(resolved_margin, available)
    if resolved_margin < 1:
        raise ValueError(
            "no control-volume margin fits inside the L3 shell block; "
            "reduce --cv-margin or enlarge --l3-margin",
        )
    if resolved_margin < args.cv_margin:
        print(
            f"note: clamped cv-margin {args.cv_margin} -> {resolved_margin} "
            f"(thin-shell L3 block)",
            flush=True,
        )
    cv = box_control_volume(
        tuple(l3_solid_g.shape),
        x0=x_min_f - resolved_margin, x1=x_max_f + resolved_margin,
        y0=y_min_f - resolved_margin, y1=y_max_f + resolved_margin,
        z0=z_min_f - resolved_margin, z1=z_max_f + resolved_margin,
        device=device,
    )

    sponge_faces = ("x+", "y-", "y+", "z-", "z+")
    sponge = build_sponge_sigma_3d(
        shape, width=args.sponge_width, max_strength=args.sponge_strength,
        device=device, faces=sponge_faces,
    )

    dx_fine_m = MODEL_LENGTH_M / (8.0 * args.hull_length)
    scale = force_scale_newton(
        rho_water=args.rho_water, dx_m=dx_fine_m,
        speed_mps=point.speed_mps, lattice_speed=args.lattice_speed,
    )

    print(
        f"coarse={list(shape)} "
        f"L1_box={[x0, x1, y0, y1, z0, z1]} L1_shape={list(s1)} fc1={tuple(fc1)} "
        f"L2_box_l1fine={[x0_2, x1_2, y0_2, y1_2, z0_2, z1_2]} "
        f"L2_shape={list(s2)} fc2={tuple(fc2)} "
        f"L3_box_l2fine={[x0_3, x1_3, y0_3, y1_3, z0_3, z1_3]} "
        f"L3_shape={list(s3)} fc3={tuple(fc3)} "
        f"tau=[{tau_coarse:.6f},{config1.tau_fine:.6f},{config2.tau_fine:.6f},"
        f"{tau_fine3:.6f}] Re={physical_re:.4e}",
        flush=True,
    )
    print(
        f"allocated_cells={amr.total_allocated_cells} "
        f"uniform_finest_equivalent={amr.uniform_finest_equivalent_cells} "
        f"cell_saving_fraction={amr.cell_saving_fraction:.4f}",
        flush=True,
    )

    force_samples: list[float] = []
    max_reflux_residual = 0.0
    reflux_residual_by_level = [0.0, 0.0, 0.0]
    maximum_positivity_limited_fraction = 0.0
    current_step = 0
    started = time.time()

    def collide(state: torch.Tensor, tau: float) -> torch.Tensor:
        nonlocal maximum_positivity_limited_fraction
        if args.collision == "cascaded":
            result = collide_cascaded_d3q19(state, tau)
        elif args.collision == "cumulant":
            result = collide_cumulant_d3q19(state, tau, C_s=0.0)
        elif args.collision_model == "cumulant_smagorinsky":
            result = collide_cumulant_d3q19(state, tau, C_s=args.cs_smag)
        elif args.les_model == "wale":
            result = collide_wale_mrt3d(state, tau, C_w=args.cw_wale)
        else:
            result = collide_smagorinsky_mrt3d(state, tau, C_s=args.cs_smag)
        if not args.disable_positivity_limiter:
            result, diagnostic = limit_nonequilibrium_for_positivity(result)
            maximum_positivity_limited_fraction = max(
                maximum_positivity_limited_fraction,
                diagnostic.limited_fraction,
            )
        return result

    def advance(
        f: torch.Tensor, tau: float, level: int, substep: int,
    ) -> AMRAdvanceResult:
        nonlocal max_reflux_residual, maximum_positivity_limited_fraction
        level_index = _level_index_of(f, level_shapes)
        if level != level_index:
            raise ValueError(
                f"runtime level {level} disagrees with shape-derived level "
                f"{level_index}",
            )
        if level_index == 0:
            # root: coarse hull frozen, far-field + sponge (patch-free region)
            before = f
            collided = collide(f, tau)
            post_collision = torch.where(solid_coarse_q, before, collided)
            out = stream3d(post_collision)
            out = non_equilibrium_far_field_bc_3d(
                out, u_in=args.lattice_speed,
            )
            if args.sponge_width > 0 and args.sponge_strength > 0.0:
                out = apply_equilibrium_difference_sponge(
                    out, sponge,
                    velocity_target=(args.lattice_speed, 0.0, 0.0),
                )
            out = non_equilibrium_far_field_bc_3d(
                out, u_in=args.lattice_speed,
            )
            return AMRAdvanceResult(out, post_collision)

        if level_index in (1, 2):
            # intermediate levels: frozen hull + plain half-way bounce-back
            # (no wall model — avoids double wall treatment; the wall model
            # lives exclusively on the finest level L3).
            solid_g, solid_q = (
                (l1_solid_g, solid_q1) if level_index == 1
                else (l2_solid_g, solid_q2)
            )
            before = f
            collided = collide(f, tau)
            post_collision = torch.where(solid_q, before, collided)
            out = stream3d(post_collision)
            out = bounce_back_cells_3d(out, solid_g, f_pre=post_collision)
            return AMRAdvanceResult(out, post_collision)

        # level 3 (finest): frozen hull + BFL wall function with the
        # analytical SUBOFF normals, and control-volume drag sampling.
        before = f
        collided = collide(f, tau)
        post_collision = torch.where(solid_q3, before, collided)
        out = stream3d(post_collision)
        activation = smooth_ramp_factor(current_step, args.ramp_steps)
        wall_result = bfl_wall_function_3d(
            out, post_collision, l3_solid_g, wall_nu_fine3,
            bfl_mask3, bfl_q3, y_val=args.wall_distance,
            wall_law=args.wall_law, near_mask=l3_near,
            bfl_wall_mode="wall_model_slip", wall_activation=activation,
            stress_exchange_distance=(
                args.stress_exchange_distance
                if args.stress_exchange_distance > 0.0 else None
            ),
            wall_normals=(
                l3_surface.nx_n, l3_surface.ny_n, l3_surface.nz_n,
            ),
            area_weight=l3_area_weight,
            apply_wall_stress=True,
        )
        out, _friction, _pressure = wall_result
        if not args.disable_positivity_limiter:
            out, diagnostic = limit_nonequilibrium_for_positivity(out)
            maximum_positivity_limited_fraction = max(
                maximum_positivity_limited_fraction,
                diagnostic.limited_fraction,
            )
        if substep == 0 and current_step > args.warmup_steps:
            # one sample per coarse step, post-warmup only (same statistics
            # semantics as suboff_static_amr_resistance.py)
            cv_force = float(observe_control_volume_force(
                before, out, post_collision, cv, solid=l3_solid_g,
            ).force_on_body[0].item())
            force_samples.append(cv_force)
        return AMRAdvanceResult(out, post_collision)

    for current_step in range(1, args.steps + 1):
        ledgers = amr.step(advance)
        for index, ledger in enumerate(ledgers):
            residual = float(ledger.residual.abs().max().item())
            reflux_residual_by_level[index] = max(
                reflux_residual_by_level[index], residual,
            )
        max_reflux_residual = max(max_reflux_residual, *reflux_residual_by_level)
        if current_step % args.report_interval == 0:
            if not all(
                bool(torch.isfinite(level).all())
                for level in amr.level_populations
            ):
                raise FloatingPointError(
                    f"SUBOFF shell L3 AMR run diverged (non-finite "
                    f"populations) at step {current_step}",
                )
            recent = force_samples[-min(len(force_samples), args.report_interval):]
            recent_n = (
                sum(recent) / len(recent) * scale if recent else math.nan
            )
            elapsed = time.time() - started
            print(
                f"step={current_step}/{args.steps} recent_Rt={recent_n:.3f} N "
                f"exp={point.resistance_n:.2f} N "
                f"steps/s={current_step / elapsed:.2f} "
                f"max_ref_res={max_reflux_residual:.2e} "
                f"ref_res_by_level="
                f"{[f'{r:.2e}' for r in reflux_residual_by_level]}",
                flush=True,
            )

    statistics_window = args.statistics_window_steps or len(force_samples)
    selected = force_samples[-statistics_window:]
    mean_force = sum(selected) / len(selected) if selected else math.nan
    resistance_n = mean_force * scale
    resistance_history_n = [value * scale for value in selected]
    stationarity = assess_force_stationarity(
        resistance_history_n,
        block_size=max(1, len(resistance_history_n) // 8),
    )
    stationarity_dict = (
        asdict(stationarity)
        if hasattr(stationarity, "__dataclass_fields__")
        else stationarity
    )
    reference_error_pct = (
        abs(resistance_n - point.resistance_n) / point.resistance_n * 100.0
        if math.isfinite(resistance_n) else math.nan
    )
    final_window = resistance_history_n[-min(len(resistance_history_n), args.report_interval):]
    final_window_resistance_n = (
        sum(final_window) / len(final_window) if final_window else math.nan
    )
    finite = (
        all(
            bool(torch.isfinite(level).all())
            for level in amr.level_populations
        )
        and math.isfinite(resistance_n)
    )
    wall_time_s = time.time() - started

    result = {
        "schema": "tensorlbm-suboff-shell-l3-amr-cv-v1",
        "status": "measured_candidate",
        "physical_validation": False,
        "case": (
            f"nested AMR body-fitted shell SUBOFF {args.hull_type} @ "
            f"{args.speed_knots:.2f} kn: coarse {list(shape)} + L1 shell "
            f"{[x0, x1, y0, y1, z0, z1]} + L2 surface shell "
            f"{[x0_2, x1_2, y0_2, y1_2, z0_2, z1_2]} in L1 fine coords + "
            f"L3 super shell {[x0_3, x1_3, y0_3, y1_3, z0_3, z1_3]} in L2 "
            f"fine coords, BFL wall function on L3 only"
        ),
        "configuration": {
            "coarse_shape_zyx": list(shape),
            "lattice": "D3Q19",
            "hull_type": args.hull_type,
            "speed_knots": args.speed_knots,
            "reynolds": physical_re,
            "collision_reynolds": collision_re,
            "lattice_speed": args.lattice_speed,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "ramp_steps": args.ramp_steps,
            "sponge_width": args.sponge_width,
            "sponge_strength": args.sponge_strength,
            "cv_margin": args.cv_margin,
            "cv_margin_resolved": resolved_margin,
            "shell_margin": args.shell_margin,
            "wake_cells": args.wake_cells,
            "l2_margin": args.l2_margin,
            "l3_margin": args.l3_margin,
            "hull_length_coarse": args.hull_length,
            "ratio": RATIO,
            "ghost": GHOST,
            "reflux": True,
            "reflux_method": "face_local_conserved_moment_flux",
            "maximum_reflux_correction_fraction": (
                args.maximum_reflux_correction_fraction
            ),
            "ghost_interpolation": args.ghost_interpolation,
            "collision": args.collision,
            "collision_model": args.collision_model,
            "les_model": args.les_model,
            "les_constant": (
                args.cw_wale if args.les_model == "wale" else args.cs_smag
            ),
            "wall_law": args.wall_law,
            "wall_distance": args.wall_distance,
            "stress_exchange_distance": (
                args.stress_exchange_distance
                if args.stress_exchange_distance > 0.0 else None
            ),
            "wall_traction_source_scheme": WALL_TRACTION_SOURCE_SCHEME,
            "positivity_limiter_enabled": not args.disable_positivity_limiter,
            "far_field_mode": "non_equilibrium_extrapolation",
            "boundary_treatment": "wall_function_l3_only_bounce_back_l1_l2",
            "wall_viscosity_basis": "physical_reynolds",
            "levels": [
                {
                    "level": 0,
                    "box": None,
                    "fine_shape_zyx": list(shape),
                    "hull_length_cells": args.hull_length,
                    "tau": tau_coarse,
                },
                {
                    "level": 1,
                    "box_l0_coarse_coords": [x0, x1, y0, y1, z0, z1],
                    "fine_shape_zyx": list(s1),
                    "fine_center_physical": list(fc1),
                    "hull_length_cells": args.hull_length * 2.0,
                    "tau": config1.tau_fine,
                    "boundary": "bounce_back",
                },
                {
                    "level": 2,
                    "box_l1_fine_coords": [x0_2, x1_2, y0_2, y1_2, z0_2, z1_2],
                    "fine_shape_zyx": list(s2),
                    "fine_center_physical": list(fc2),
                    "hull_length_cells": args.hull_length * 4.0,
                    "tau": config2.tau_fine,
                    "boundary": "bounce_back",
                },
                {
                    "level": 3,
                    "box_l2_fine_coords": [x0_3, x1_3, y0_3, y1_3, z0_3, z1_3],
                    "fine_shape_zyx": list(s3),
                    "fine_center_physical": list(fc3),
                    "hull_length_cells": args.hull_length * 8.0,
                    "tau": tau_fine3,
                    "boundary": "bfl_wall_function",
                },
            ],
            "tau_chain": [
                tau_coarse, config1.tau_fine, config2.tau_fine, tau_fine3,
            ],
            "cell_saving_fraction": amr.cell_saving_fraction,
        },
        "geometry": {
            "coarse_solid_cells": int(solid_coarse.sum().item()),
            "l1_solid_cells": int(l1_solid.sum().item()),
            "l2_solid_cells": int(l2_solid.sum().item()),
            "l3_solid_cells": int(l3_solid.sum().item()),
            "l3_wetted_area_lu2": float(l3_geometry["wetted_area_lu2"]),
            "l2_wetted_area_lu2": float(l2_geometry["wetted_area_lu2"]),
            "l1_wetted_area_lu2": float(l1_geometry["wetted_area_lu2"]),
            "coarse_wetted_area_lu2": float(coarse_geometry["wetted_area_lu2"]),
            "surface_area_diagnostics": vars(surface_area_diagnostics),
        },
        "mesh": {
            "coarse_cells": int(coarse_f[0].numel()),
            "l1_allocated_cells": int(amr.level_populations[1][0].numel()),
            "l2_allocated_cells": int(amr.level_populations[2][0].numel()),
            "l3_allocated_cells": int(amr.level_populations[3][0].numel()),
            "total_allocated_cells": amr.total_allocated_cells,
            "uniform_finest_equivalent_cells": amr.uniform_finest_equivalent_cells,
            "cell_saving_fraction": amr.cell_saving_fraction,
        },
        "result": {
            "resistance_n": resistance_n,
            "experimental_n": point.resistance_n,
            "reference_error_pct": reference_error_pct,
            "final_window_resistance_n": final_window_resistance_n,
            "resistance_history_n_tail": resistance_history_n[-25:],
            "mean_force_lu": mean_force,
            "force_scale_newton_per_lu": scale,
            "stationarity": stationarity_dict,
            "max_reflux_residual": max_reflux_residual,
            "max_reflux_residual_by_level": reflux_residual_by_level,
            "maximum_positivity_limited_fraction": (
                maximum_positivity_limited_fraction
            ),
            "force_samples": len(force_samples),
            "statistics_window_steps": statistics_window,
            "finite": finite,
            "wall_time_s": wall_time_s,
        },
        "artifacts": {
            "output": args.output,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    print(json.dumps(_json_safe(result["result"]), indent=2), flush=True)
    return result


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
