#!/usr/bin/env python3
"""DARPA SUBOFF body-fitted shell AMR resistance candidate (route C).

Route-C variant of :file:`suboff_static_amr_resistance.py`: instead of
refining one fat box around the whole hull (hull + uniform wall margin +
wake), the fine 2:1 block is built from a **body-fitted surface shell** —
the coarse cells within ``shell_margin`` of the hull surface
(:class:`~tensorlbm.refinement.HullProximityRegion`) plus a downstream wake
slab (:class:`~tensorlbm.refinement.WakeRegion`) clipped laterally to the
shell extent — and its bounding box.

For a slender body like SUBOFF the boundary layer is thin and the flow
develops mostly along the surface, so the shell is expected to keep the
fine-grid accuracy of the fat-block runner while saving the fine cells far
from the hull (the sphere-shell probe saved 86–97 % but degraded the blunt-
body accuracy; SUBOFF is the case the shell scheme is actually for).

Two independent solid freeze layers keep the geometry exact:

* coarse level 0: ``solid_coarse_q`` freezes every coarse hull cell that the
  fine block does *not* own (the fine block is restricted back over its box
  every root step, so coarse cells inside the box are fine-owned);
* fine level 1: ``fine_solid`` (the CAD hull regenerated in block-local
  fine coordinates) freezes fine solid cells at collision, and the Bouzidi
  BFL interpolated bounce-back + wall-function enforces the curved surface
  after streaming with q-values evaluated on the fine grid (BFL
  :func:`~tensorlbm.interpolated_bc_suboff.compute_q_suboff`).

The drag is measured with an independent control volume on the fine grid
(including ghost), converted to Newtons with the same similarity map as the
static-AMR runner and compared with the AFF-1 tow-tank measurement
(5.92 kn = 87.4 N, Liu & Huang Table 14).

Run (CPU smoke):::

    PYTHONPATH=src .venv/bin/python examples/suboff_shell_amr_validate.py \\
        --device cpu --hull-length 60 --speed-knots 5.92 --steps 100 \\
        --warmup-steps 30 --ramp-steps 15 --shell-margin 6 --wake-cells 20 \\
        --sponge-width 10 --report-interval 25 --output /tmp/suboff_shell_smoke.json

D3Q27 variant (GPU smoke):::

    PYTHONPATH=src .venv/bin/python examples/suboff_shell_amr_validate.py \\
        --lattice D3Q27 --hull-length 60 --speed-knots 5.92 --steps 60 \\
        --warmup-steps 15 --ramp-steps 10 --shell-margin 6 --wake-cells 20 \\
        --sponge-width 10 --report-interval 20 --output /tmp/suboff_shell_d3q27_smoke.json

``--lattice D3Q27`` switches the equilibrium, streaming, collision (D3Q27
cumulant / cascaded), the full-stencil D3Q27 BFL bounce-back and the
:func:`~tensorlbm.wall_model.bfl_wall_function_d3q27` wall traction; the
result schema gains a ``-d3q27`` suffix.  The D3Q19 path is bit-for-bit the
legacy single-shell run.
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

from tensorlbm.cascaded_collision import collide_cascaded_d3q27
from tensorlbm.control_volume_force import (
    box_control_volume,
    observe_control_volume_force,
)
from tensorlbm.cumulant import collide_cumulant_d3q19, collide_cumulant_d3q27
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.d3q27 import equilibrium27, macroscopic27, stream27_roll
from tensorlbm.drag_pressure import SurfaceMesh, get_near_wall_3d
from tensorlbm.external_open_boundary import non_equilibrium_far_field_bc_3d
from tensorlbm.force_convergence import assess_force_stationarity
from tensorlbm.interpolated_bc_suboff import compute_q_suboff
from tensorlbm.interpolated_bc_suboff_d3q27 import compute_q_suboff_27
from tensorlbm.population_positivity import limit_nonequilibrium_for_positivity
from tensorlbm.refinement import BoxRegion, HullProximityRegion, WakeRegion
from tensorlbm.solver3d import stream3d
from tensorlbm.sphere_amr_common import _bouzidi_bounce_back_d3q27
from tensorlbm.sponge_layer import (
    apply_equilibrium_difference_sponge,
    build_sponge_sigma_3d,
)
from tensorlbm.static_block_amr import (
    AMRAdvanceResult,
    StaticBlockAMR3D,
    StaticBlockAMRConfig,
)
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask
from tensorlbm.surface_area_weights import bfl_surface_area_weights
from tensorlbm.turbulence import collide_smagorinsky_mrt3d, collide_wale_mrt3d
from tensorlbm.wall_model import (
    WALL_TRACTION_SOURCE_SCHEME,
    bfl_wall_function_3d,
    bfl_wall_function_d3q27,
    physical_wall_lattice_viscosity,
)

SUBOFF_L_D = 8.57  # DARPA SUBOFF L/D (used for the fine hull radius)


def _q_channels(lattice: str) -> int:
    """Number of population channels for a lattice name."""
    return 27 if lattice == "D3Q27" else 19


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
        help="body-fitted shell thickness in coarse cells around the hull "
        "surface (HullProximityRegion margin)",
    )
    p.add_argument(
        "--wake-cells",
        type=int,
        default=40,
        help="downstream wake extension in coarse cells behind the hull",
    )
    p.add_argument(
        "--cv-margin",
        type=int,
        default=4,
        help="control-volume margin around the fine hull; the fine block is a "
        "thin shell, so this is clamped to the largest margin that fits "
        "strictly inside the fine block",
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
        "--collision-model",
        choices=("mrt_les", "cumulant_smagorinsky"),
        default="mrt_les",
        help="D3Q19: MRT-LES (wale/smagorinsky) or cumulant+Smagorinsky. "
        "D3Q27: mrt_les maps to the D3Q27 cascaded operator (no separate "
        "MRT-LES wrapper exists for D3Q27) and cumulant_smagorinsky maps to "
        "collide_cumulant_d3q27 with the Smagorinsky constant.",
    )
    p.add_argument(
        "--lattice",
        choices=("D3Q19", "D3Q27"),
        default="D3Q19",
        help="lattice stencil (D3Q19 keeps the legacy path exactly; D3Q27 uses "
        "the D3Q27 equilibrium/collision/stream/BFL kernels)",
    )
    p.add_argument("--les-model", choices=("wale", "smagorinsky"), default="wale")
    p.add_argument("--cs-smag", type=float, default=0.05)
    p.add_argument("--cw-wale", type=float, default=0.5)
    p.add_argument(
        "--wall-law",
        choices=("log", "reichardt", "musker"),
        default="reichardt",
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
    if args.cv_margin < 1:
        raise ValueError("cv-margin must be positive")
    if not 0.0 < args.maximum_reflux_correction_fraction <= 1.0:
        raise ValueError("maximum-reflux-correction-fraction must lie in (0,1]")
    if not 0.0 < args.center_x_fraction < 1.0:
        raise ValueError("center-x-fraction must lie in (0,1)")

    device = torch.device(args.device)
    point = experimental_point(args.hull_type, args.speed_knots)
    shape = (args.nz, args.ny, args.nx)
    center = (args.nx * args.center_x_fraction, args.ny / 2.0, args.nz / 2.0)
    if center[0] - args.hull_length / 2.0 <= 1 or center[0] + args.hull_length / 2.0 >= args.nx - 1:
        raise ValueError("SUBOFF hull does not fit inside the streamwise domain")
    config = SuboffConfig()

    # ---- coarse hull (level-0 freeze mask for coarse cells the fine block
    # ---- does not own).
    solid_coarse, coarse_geometry = build_suboff_mask(
        args.hull_type,
        args.nx,
        args.ny,
        args.nz,
        cx=center[0],
        cy=center[1],
        cz=center[2],
        length=args.hull_length,
        config=config,
        device=device,
    )
    if not bool(solid_coarse.any()):
        raise ValueError("no SUBOFF cells on the coarse grid")
    q_channels = _q_channels(args.lattice)
    solid_coarse_q = (
        solid_coarse.unsqueeze(0)
        .expand(
            q_channels,
            *shape,
        )
        .contiguous()
    )

    # ---- body-fitted shell refinement region: hull-proximity surface shell
    # ---- + downstream wake, clipped laterally to the shell extent so the
    # ---- fine block stays body-fitted instead of spanning the full coarse
    # ---- domain height/depth (same pattern as amr_sphere_shell_validate.py).
    shell_mask = HullProximityRegion(
        solid_coarse,
        margin=args.shell_margin,
    ).expand_mask()
    wake_mask = WakeRegion(
        solid_coarse,
        extend_x=args.wake_cells,
    ).expand_mask()
    shell_idx = shell_mask.nonzero(as_tuple=False)
    if shell_idx.numel() == 0:
        raise ValueError(
            "empty shell refinement region; adjust --shell-margin",
        )
    sz0, sy0 = int(shell_idx[:, 0].min().item()), int(shell_idx[:, 1].min().item())
    sz1, sy1 = int(shell_idx[:, 0].max().item()), int(shell_idx[:, 1].max().item())
    wake_mask[:sz0, :, :] = False
    wake_mask[sz1 + 1 :, :, :] = False
    wake_mask[:, :sy0, :] = False
    wake_mask[:, sy1 + 1 :, :] = False
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
            "refinement box too small: try a larger --wake-cells or a smaller --shell-margin",
        )
    box = BoxRegion(x0, x1, y0, y1, z0, z1)
    ratio, g = 2, 1

    # ---- physics (identical similarity map to suboff_static_amr_resistance).
    physical_re = point.speed_mps * MODEL_LENGTH_M / args.nu_water
    collision_re = args.resolved_reynolds or physical_re
    nu_coarse = args.lattice_speed * args.hull_length / collision_re
    wall_nu_fine = physical_wall_lattice_viscosity(
        args.lattice_speed,
        args.hull_length * 2.0,
        physical_re,
    )
    tau_coarse = 0.5 + 3.0 * nu_coarse

    rho = torch.ones(shape, device=device)
    ux = torch.full_like(rho, args.lattice_speed)
    zero = torch.zeros_like(rho)
    if args.lattice == "D3Q27":
        coarse_f = equilibrium27(rho, ux, zero, zero, device=device)
    else:
        coarse_f = equilibrium3d(rho, ux, zero, zero, device=device)

    # ---- fine hull geometry, regenerated from CAD in block-local fine
    # ---- coordinates (never a coarse voxel repeat).
    fine_shape = ((z1 - z0) * ratio, (y1 - y0) * ratio, (x1 - x0) * ratio)
    fine_center = (
        center[0] * ratio - x0 * ratio + g,
        center[1] * ratio - y0 * ratio + g,
        center[2] * ratio - z0 * ratio + g,
    )
    fine_solid, fine_geometry = build_suboff_mask(
        args.hull_type,
        fine_shape[2],
        fine_shape[1],
        fine_shape[0],
        cx=fine_center[0],
        cy=fine_center[1],
        cz=fine_center[2],
        length=args.hull_length * 2.0,
        config=config,
        device=device,
    )
    if not bool(fine_solid.any()):
        raise RuntimeError("fine block contains no SUBOFF cells")

    amr = StaticBlockAMR3D(
        coarse_f,
        StaticBlockAMRConfig(
            box,
            tau_coarse=tau_coarse,
            reflux=not args.disable_reflux,
            maximum_reflux_correction_fraction=(args.maximum_reflux_correction_fraction),
            ghost_interpolation=args.ghost_interpolation,
        ),
        fine_solid=fine_solid,
    )
    fine_solid_g = amr.fine_solid_with_ghost
    assert fine_solid_g is not None

    print("building fine-grid BFL link distances", flush=True)
    if args.lattice == "D3Q27":
        bfl_mask, bfl_q = compute_q_suboff_27(
            fine_solid_g.shape[2],
            fine_solid_g.shape[1],
            fine_solid_g.shape[0],
            *fine_center,
            args.hull_length * 2.0,
            hull_type=args.hull_type,
            config=config,
            device=device,
            solid_mask=fine_solid_g,
        )
    else:
        bfl_mask, bfl_q = compute_q_suboff(
            fine_solid_g.shape[2],
            fine_solid_g.shape[1],
            fine_solid_g.shape[0],
            *fine_center,
            args.hull_length * 2.0,
            hull_type=args.hull_type,
            config=config,
            device=device,
            solid_mask=fine_solid_g,
        )
    fine_near = get_near_wall_3d(fine_solid_g)
    fine_surface = SurfaceMesh.from_suboff(
        fine_solid_g,
        fine_near,
        *fine_center,
        args.hull_length * 2.0,
        args.hull_length * 2.0 / (2.0 * SUBOFF_L_D),
        config=config,
    )
    fine_area_weight, surface_area_diagnostics = bfl_surface_area_weights(
        bfl_mask,
        (fine_surface.nx_n, fine_surface.ny_n, fine_surface.nz_n),
        reference_area=float(fine_geometry["wetted_area_lu2"]),
        boundary_mask=fine_near,
    )
    fine_surface.dA = fine_area_weight
    solid_q = (
        fine_solid_g.unsqueeze(0)
        .expand(
            q_channels,
            *fine_solid_g.shape,
        )
        .contiguous()
    )

    # ---- control volume on the fine grid (with ghost offset).  The fine
    # ---- block is a thin shell, so the requested margin is clamped to the
    # ---- largest margin that keeps the CV strictly interior while still
    # ---- enclosing the hull.
    fine_indices = fine_solid_g.nonzero(as_tuple=False)
    if fine_indices.numel() == 0:
        raise RuntimeError("fine block contains no SUBOFF cells")
    z_min_f, y_min_f, x_min_f = (int(fine_indices[:, axis].min().item()) for axis in range(3))
    z_max_f, y_max_f, x_max_f = (int(fine_indices[:, axis].max().item()) + 1 for axis in range(3))
    nz_f, ny_f, nx_f = fine_solid_g.shape
    resolved_margin = args.cv_margin
    for lower, upper, size in (
        (x_min_f, x_max_f, nx_f),
        (y_min_f, y_max_f, ny_f),
        (z_min_f, z_max_f, nz_f),
    ):
        # original runner guard: lower-margin > g and upper+margin < size-g
        available = min(lower - g - 1, size - g - 1 - upper)
        resolved_margin = min(resolved_margin, available)
    if resolved_margin < 1:
        raise ValueError(
            "no control-volume margin fits inside the body-fitted fine block; "
            "reduce --cv-margin or enlarge the shell via --shell-margin/--wake-cells",
        )
    if resolved_margin < args.cv_margin:
        print(
            f"note: clamped cv-margin {args.cv_margin} -> {resolved_margin} "
            f"(thin-shell fine block)",
            flush=True,
        )
    cv = box_control_volume(
        tuple(fine_solid_g.shape),
        x0=x_min_f - resolved_margin,
        x1=x_max_f + resolved_margin,
        y0=y_min_f - resolved_margin,
        y1=y_max_f + resolved_margin,
        z0=z_min_f - resolved_margin,
        z1=z_max_f + resolved_margin,
        device=device,
    )

    sponge_faces = ("x+", "y-", "y+", "z-", "z+")
    sponge = build_sponge_sigma_3d(
        shape,
        width=args.sponge_width,
        max_strength=args.sponge_strength,
        device=device,
        faces=sponge_faces,
    )

    dx_fine_m = MODEL_LENGTH_M / (2.0 * args.hull_length)
    scale = force_scale_newton(
        rho_water=args.rho_water,
        dx_m=dx_fine_m,
        speed_mps=point.speed_mps,
        lattice_speed=args.lattice_speed,
    )

    print(
        f"coarse={list(shape)} fine_box={[x0, x1, y0, y1, z0, z1]} "
        f"fine_shape={list(fine_shape)} fine_center={tuple(fine_center)} "
        f"shell_margin={args.shell_margin} wake_cells={args.wake_cells} "
        f"refine_cells={int(refine_mask.sum().item())} "
        f"tau_coarse={tau_coarse:.6f} Re={physical_re:.4e}",
        flush=True,
    )
    print(
        f"allocated_cells={amr.total_allocated_cells} "
        f"uniform_fine_equivalent={amr.uniform_fine_equivalent_cells} "
        f"cell_saving_fraction={amr.cell_saving_fraction:.4f}",
        flush=True,
    )

    force_samples: list[float] = []
    max_reflux_residual = 0.0
    maximum_positivity_limited_fraction = 0.0
    current_step = 0
    started = time.time()

    def collide(state: torch.Tensor, tau: float) -> torch.Tensor:
        nonlocal maximum_positivity_limited_fraction
        if args.lattice == "D3Q27":
            if args.collision_model == "cumulant_smagorinsky":
                result = collide_cumulant_d3q27(
                    state,
                    tau=tau,
                    C_s=args.cs_smag,
                )
            else:
                result = collide_cascaded_d3q27(state, tau)
        elif args.collision_model == "cumulant_smagorinsky":
            result = collide_cumulant_d3q19(state, tau=tau, C_s=args.cs_smag)
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

    def _stream(f: torch.Tensor) -> torch.Tensor:
        """Lattice-dispatched streaming."""
        if args.lattice == "D3Q27":
            return stream27_roll(f)
        return stream3d(f)

    def advance(
        f: torch.Tensor,
        tau: float,
        level: int,
        substep: int,
    ) -> AMRAdvanceResult:
        nonlocal max_reflux_residual, maximum_positivity_limited_fraction
        if level == 0:
            before = f
            collided = collide(f, tau)
            post_collision = torch.where(solid_coarse_q, before, collided)
            out = _stream(post_collision)
            out = non_equilibrium_far_field_bc_3d(
                out,
                u_in=args.lattice_speed,
            )
            if args.sponge_width > 0 and args.sponge_strength > 0.0:
                out = apply_equilibrium_difference_sponge(
                    out,
                    sponge,
                    velocity_target=(args.lattice_speed, 0.0, 0.0),
                )
            out = non_equilibrium_far_field_bc_3d(
                out,
                u_in=args.lattice_speed,
            )
            return AMRAdvanceResult(out, collided)

        before = f
        collided = collide(f, tau)
        post_collision = torch.where(solid_q, before, collided)
        out = _stream(post_collision)
        activation = smooth_ramp_factor(current_step, args.ramp_steps)
        if args.lattice == "D3Q27":
            # bfl_wall_function_d3q27 only applies the wall-law Guo traction
            # (it does not include BFL bounce-back), so reconstruct the curved
            # hull with the full-stencil D3Q27 BFL first.  Wall-velocity uses
            # the same wall_model_slip semantics as the D3Q19 path: smoothly
            # introduce the body in the fluid frame (activation 0 → wall moves
            # with local fluid, no impulse; 1 → only the normal component is
            # removed, tangential slip preserved).  The analytical SUBOFF
            # surface normal (SurfaceMesh.from_suboff) replaces the D3Q19
            # link-normal fallback.
            rho_post, ux_post, uy_post, uz_post = macroscopic27(post_collision)
            u_dot_n = (
                ux_post * fine_surface.nx_n
                + uy_post * fine_surface.ny_n
                + uz_post * fine_surface.nz_n
            )
            bfl_out = _bouzidi_bounce_back_d3q27(
                out,
                post_collision,
                bfl_mask,
                bfl_q,
                wall_velocity=(
                    ux_post - activation * u_dot_n * fine_surface.nx_n,
                    uy_post - activation * u_dot_n * fine_surface.ny_n,
                    uz_post - activation * u_dot_n * fine_surface.nz_n,
                ),
                wall_density=rho_post,
            )
            assert isinstance(bfl_out, torch.Tensor)
            out = bfl_out
            out, _friction, _pressure = bfl_wall_function_d3q27(
                out,
                post_collision,
                fine_solid_g,
                wall_nu_fine,
                bfl_mask,
                bfl_q,
                y_val=args.wall_distance,
                wall_law=args.wall_law,
                near_mask=fine_near,
                area_weight=fine_area_weight,
                wall_activation=activation,
                apply_wall_stress=True,
            )
        else:
            wall_result = bfl_wall_function_3d(
                out,
                post_collision,
                fine_solid_g,
                wall_nu_fine,
                bfl_mask,
                bfl_q,
                y_val=args.wall_distance,
                wall_law=args.wall_law,
                near_mask=fine_near,
                bfl_wall_mode="wall_model_slip",
                wall_activation=activation,
                stress_exchange_distance=(
                    args.stress_exchange_distance if args.stress_exchange_distance > 0.0 else None
                ),
                wall_normals=(
                    fine_surface.nx_n,
                    fine_surface.ny_n,
                    fine_surface.nz_n,
                ),
                area_weight=fine_area_weight,
                apply_wall_stress=True,
            )
            if isinstance(wall_result, tuple) and len(wall_result) == 4:
                out, _friction, _pressure, _wall_diag = wall_result
            else:
                out, _friction, _pressure = wall_result  # type: ignore[misc]
        if not args.disable_positivity_limiter:
            out, diagnostic = limit_nonequilibrium_for_positivity(out)
            maximum_positivity_limited_fraction = max(
                maximum_positivity_limited_fraction,
                diagnostic.limited_fraction,
            )
        if substep == 1 and current_step > args.warmup_steps:
            # one sample per coarse step, post-warmup only (same statistics
            # semantics as suboff_static_amr_resistance.py)
            cv_force = float(
                observe_control_volume_force(
                    before,
                    out,
                    post_collision,
                    cv,
                    solid=fine_solid_g,
                )
                .force_on_body[0]
                .item()
            )
            force_samples.append(cv_force)
        return AMRAdvanceResult(out, post_collision)

    for current_step in range(1, args.steps + 1):
        ledger = amr.step(advance)
        max_reflux_residual = max(
            max_reflux_residual,
            float(ledger.residual.abs().max().item()),
        )
        if current_step % args.report_interval == 0:
            coarse_finite = bool(torch.isfinite(amr.coarse_f).all())
            fine_finite = bool(torch.isfinite(amr.fine_f).all())
            if not (coarse_finite and fine_finite):
                raise FloatingPointError(
                    f"SUBOFF shell AMR run diverged (non-finite populations) "
                    f"at step {current_step}",
                )
            recent = force_samples[-min(len(force_samples), args.report_interval) :]
            recent_n = sum(recent) / len(recent) * scale if recent else math.nan
            elapsed = time.time() - started
            print(
                f"step={current_step}/{args.steps} recent_Rt={recent_n:.3f} N "
                f"exp={point.resistance_n:.2f} N "
                f"steps/s={current_step / elapsed:.2f} "
                f"max_ref_res={max_reflux_residual:.2e}",
                flush=True,
            )

    statistics_window = args.statistics_window_steps or len(force_samples)
    selected = force_samples[-statistics_window:]
    mean_force = sum(selected) / len(selected)
    resistance_n = mean_force * scale
    resistance_history_n = [value * scale for value in selected]
    stationarity = assess_force_stationarity(
        resistance_history_n,
        block_size=max(1, len(resistance_history_n) // 8),
    )
    stationarity_dict = (
        asdict(stationarity) if hasattr(stationarity, "__dataclass_fields__") else stationarity
    )
    reference_error_pct = abs(resistance_n - point.resistance_n) / point.resistance_n * 100.0
    final_window = resistance_history_n[-min(len(resistance_history_n), args.report_interval) :]
    final_window_resistance_n = sum(final_window) / len(final_window) if final_window else math.nan
    finite = (
        bool(torch.isfinite(amr.coarse_f).all())
        and bool(torch.isfinite(amr.fine_f).all())
        and math.isfinite(resistance_n)
    )
    wall_time_s = time.time() - started

    result = {
        "schema": (
            "tensorlbm-suboff-shell-amr-cv-v1" + ("-d3q27" if args.lattice == "D3Q27" else "")
        ),
        "status": "measured_candidate",
        "physical_validation": False,
        "case": (
            f"AMR body-fitted shell SUBOFF {args.hull_type} @ "
            f"{args.speed_knots:.2f} kn: coarse {list(shape)} + 2:1 shell block "
            f"{[x0, x1, y0, y1, z0, z1]} shell_margin={args.shell_margin} "
            f"wake_cells={args.wake_cells} hull_length={args.hull_length}"
        ),
        "configuration": {
            "lattice": args.lattice,
            "coarse_shape_zyx": list(shape),
            "fine_shape_zyx": list(fine_shape),
            "fine_box": [x0, x1, y0, y1, z0, z1],
            "shell_margin": args.shell_margin,
            "wake_cells": args.wake_cells,
            "refine_cells_coarse": int(refine_mask.sum().item()),
            "cell_saving_fraction": amr.cell_saving_fraction,
            "hull_length": args.hull_length,
            "fine_hull_length_cells": args.hull_length * 2.0,
            "hull_type": args.hull_type,
            "speed_knots": args.speed_knots,
            "reynolds": physical_re,
            "collision_reynolds": collision_re,
            "lattice_speed": args.lattice_speed,
            "tau_coarse": tau_coarse,
            "tau_fine": amr.config.tau_fine,
            "wall_nu_fine": wall_nu_fine,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "ramp_steps": args.ramp_steps,
            "sponge_width": args.sponge_width,
            "sponge_strength": args.sponge_strength,
            "cv_margin": args.cv_margin,
            "cv_margin_resolved": resolved_margin,
            "reflux": not args.disable_reflux,
            "reflux_method": "face_local_conserved_moment_flux",
            "maximum_reflux_correction_fraction": (args.maximum_reflux_correction_fraction),
            "ghost_interpolation": args.ghost_interpolation,
            "collision_model": args.collision_model,
            "les_model": args.les_model,
            "les_constant": (args.cw_wale if args.les_model == "wale" else args.cs_smag),
            "wall_law": args.wall_law,
            "wall_distance": args.wall_distance,
            "stress_exchange_distance": (
                args.stress_exchange_distance if args.stress_exchange_distance > 0.0 else None
            ),
            "wall_traction_source_scheme": WALL_TRACTION_SOURCE_SCHEME,
            "positivity_limiter_enabled": not args.disable_positivity_limiter,
            "far_field_mode": "non_equilibrium_extrapolation",
            "boundary_treatment": "bfl_wall_model",
            "refinement_ratio": ratio,
            "ghost": g,
            "wall_viscosity_basis": "physical_reynolds",
        },
        "geometry": {
            "coarse_solid_cells": int(solid_coarse.sum().item()),
            "fine_solid_cells": int(fine_solid.sum().item()),
            "fine_wetted_area_lu2": float(fine_geometry["wetted_area_lu2"]),
            "surface_area_diagnostics": vars(surface_area_diagnostics),
            "coarse_wetted_area_lu2": float(coarse_geometry["wetted_area_lu2"]),
        },
        "mesh": {
            "coarse_cells": int(coarse_f[0].numel()),
            "fine_allocated_cells": int(amr.fine_f[0].numel()),
            "total_allocated_cells": amr.total_allocated_cells,
            "uniform_fine_equivalent_cells": amr.uniform_fine_equivalent_cells,
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
            "maximum_positivity_limited_fraction": (maximum_positivity_limited_fraction),
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
