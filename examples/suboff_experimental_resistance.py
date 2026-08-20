#!/usr/bin/env python3
"""DARPA SUBOFF resistance benchmark against the original tow-tank table.

Unlike the older demonstration scripts, this runner does not use the
ITTC-1957 friction line as though it were an experimental total-resistance
coefficient.  It compares the dimensional force predicted by the LBM run
with Table 14 of Liu & Huang (1998), CRDKNSWC/HD-1298-11 (ADA359226).

The report's configuration 1 is the bare hull (AFF-1); configuration 8 is
the fully appended hull (AFF-8).  The runner preserves the measured force as
the primary acceptance quantity, and also reports the nearly speed-matched
AFF-8/AFF-1 force ratio (1.169 around both 5.92 and 11.85 knots).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.checkpoint_io import atomic_torch_save
from tensorlbm.control_volume_force import (
    assess_nested_control_volume_invariance,
    box_control_volume,
    observe_control_volume_force,
)
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    drag_pressure_integration,
    get_near_wall_3d,
)
from tensorlbm.external_open_boundary import non_equilibrium_far_field_bc_3d
from tensorlbm.force_convergence import assess_force_stationarity
from tensorlbm.interpolated_bc_suboff import (
    SUBOFF_APPENDAGE_LINK_SCHEME,
    compute_q_suboff,
    refine_q_suboff_appendages,
)
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.sponge_layer import (
    apply_equilibrium_difference_sponge,
    build_anisotropic_sponge_sigma_3d,
)
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask
from tensorlbm.suboff_reference_data import (
    SUBOFF_TOW_TANK_RESISTANCE_TABLE14,
    SuboffTowTankResistancePoint,
)
from tensorlbm.surface_area_weights import bfl_surface_area_weights
from tensorlbm.turbulence import collide_smagorinsky_mrt3d, collide_wale_mrt3d
from tensorlbm.wall_model import (
    bfl_wall_function_3d,
    compute_wall_normal,
    wall_function_3d,
)

MODEL_LENGTH_M = 4.356
KNOT_TO_MPS = 0.514444
PRIMARY_SOURCE = {
    "title": "Summary of DARPA Suboff Experimental Program Data",
    "authors": "Han-Lieh Liu and Thomas T. Huang",
    "report": "CRDKNSWC/HD-1298-11 (ADA359226)",
    "doi": "10.21236/ADA359226",
    "table": "Table 14, report page 23",
    "url": "https://archive.org/details/DTIC_ADA359226",
}


TowTankPoint = SuboffTowTankResistancePoint
TOW_TANK_POINTS = {
    hull_type: tuple(
        point for point in SUBOFF_TOW_TANK_RESISTANCE_TABLE14 if point.hull_type == hull_type
    )
    for hull_type in ("bare_hull", "full")
}


def experimental_point(hull_type: str, speed_knots: float) -> TowTankPoint:
    """Return the exact table point at *speed_knots* (within 0.02 knot)."""
    candidates = TOW_TANK_POINTS[hull_type]
    point = min(candidates, key=lambda item: abs(item.speed_knots - speed_knots))
    if abs(point.speed_knots - speed_knots) > 0.02:
        raise ValueError("speed must match a primary Table 14 measurement")
    return point


def force_scale_newton(
    *, rho_water: float, dx_m: float, speed_mps: float, lattice_speed: float
) -> float:
    """Return Newton per lattice-force unit for a 3-D LBM similarity map."""
    if min(rho_water, dx_m, speed_mps, lattice_speed) <= 0:
        raise ValueError("force-scale inputs must be positive")
    return rho_water * dx_m**2 * (speed_mps / lattice_speed) ** 2


def smooth_ramp_factor(step: int, ramp_steps: int) -> float:
    """Raised-cosine wall activation without endpoint impulses."""
    if ramp_steps <= 0 or step >= ramp_steps:
        return 1.0
    if step <= 0:
        return 0.0
    phase = float(step) / float(ramp_steps)
    return 0.5 * (1.0 - math.cos(math.pi * phase))


def build_far_field_sponge(
    nx: int,
    ny: int,
    nz: int,
    width: int,
    strength: float,
    device: torch.device,
) -> torch.Tensor:
    """Quadratic target-field sponge at all six far-field faces."""
    if width < 0:
        raise ValueError("sponge width must be non-negative")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("sponge strength must be in [0, 1]")
    sponge = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    if width == 0 or strength == 0.0:
        return sponge

    wx = min(width, max(nx // 4, 1))
    wy = min(width, max(ny // 4, 1))
    wz = min(width, max(nz // 4, 1))
    x = torch.arange(nx, device=device, dtype=torch.float32)
    y = torch.arange(ny, device=device, dtype=torch.float32)
    z = torch.arange(nz, device=device, dtype=torch.float32)
    x_edge = torch.minimum(x, (nx - 1) - x)
    streamwise = torch.clamp((wx - x_edge) / wx, 0.0, 1.0).square()
    y_edge = torch.minimum(y, (ny - 1) - y)
    z_edge = torch.minimum(z, (nz - 1) - z)
    lateral_y = torch.clamp((wy - y_edge) / wy, 0.0, 1.0).square()
    lateral_z = torch.clamp((wz - z_edge) / wz, 0.0, 1.0).square()
    sponge = torch.maximum(sponge, streamwise.view(1, 1, nx))
    sponge = torch.maximum(sponge, lateral_y.view(1, ny, 1))
    sponge = torch.maximum(sponge, lateral_z.view(nz, 1, 1))
    return sponge * strength


def project_no_penetration(
    f: torch.Tensor,
    solid: torch.Tensor,
    near: torch.Tensor,
) -> torch.Tensor:
    """Remove near-wall normal velocity while preserving tangential flow."""
    rho, ux, uy, uz = macroscopic3d(f)
    nx_n, ny_n, nz_n = compute_wall_normal(solid, near)
    u_normal = ux * nx_n + uy * ny_n + uz * nz_n
    ux_wall = ux - u_normal * nx_n
    uy_wall = uy - u_normal * ny_n
    uz_wall = uz - u_normal * nz_n
    near_f = near.to(f.dtype)
    ux_target = ux + near_f * (ux_wall - ux)
    uy_target = uy + near_f * (uy_wall - uy)
    uz_target = uz + near_f * (uz_wall - uz)
    return (
        f
        + equilibrium3d(
            rho,
            ux_target,
            uy_target,
            uz_target,
            device=f.device,
        )
        - equilibrium3d(rho, ux, uy, uz, device=f.device)
    )


def run_case(args: argparse.Namespace) -> dict:
    if args.surface_force_interval < 1:
        raise ValueError("surface-force-interval must be positive")
    if args.checkpoint_interval < 0:
        raise ValueError("checkpoint-interval must be non-negative")
    if args.wall_diagnostic_interval < 1:
        raise ValueError("wall-diagnostic-interval must be positive")
    if args.stress_exchange_distance < 0.0:
        raise ValueError("stress-exchange-distance must be non-negative")
    if args.resume and not args.checkpoint:
        raise ValueError("resume requires --checkpoint")
    aux_cv_margins = tuple(
        sorted({int(value) for value in args.aux_cv_margins.split(",") if value.strip()})
    )
    if any(margin < 1 for margin in aux_cv_margins):
        raise ValueError("aux-cv-margins must contain positive integers")
    point = experimental_point(args.hull_type, args.speed_knots)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    length_lu = float(args.hull_length)
    radius_lu = length_lu / (2.0 * 8.57)
    dx_m = MODEL_LENGTH_M / length_lu
    re = point.speed_mps * MODEL_LENGTH_M / args.nu_water
    nu_lu = args.lattice_speed * length_lu / re
    resolved_re = args.resolved_reynolds if args.resolved_reynolds > 0 else re
    collision_nu_lu = args.lattice_speed * length_lu / resolved_re
    tau = 0.5 + 3.0 * collision_nu_lu
    cx, cy, cz = args.nx * args.center_x_fraction, args.ny / 2.0, args.nz / 2.0
    if not 0.0 < args.center_x_fraction < 1.0:
        raise ValueError("center-x-fraction must lie in (0,1)")
    if cx - length_lu / 2.0 <= 1 or cx + length_lu / 2.0 >= args.nx - 1:
        raise ValueError("SUBOFF hull does not fit inside the streamwise domain")
    tag = f"[{args.hull_type} L={length_lu:g} Re={re:.4e}]"

    print(
        f"{tag} grid={args.nx}x{args.ny}x{args.nz} device={device} "
        f"experiment={point.speed_knots:.2f} kn/{point.resistance_n:.2f} N "
        f"u_lu={args.lattice_speed} wall_nu_lu={nu_lu:.8e} "
        f"collision_Re={resolved_re:.4e} tau={tau:.9f}",
        flush=True,
    )
    started = time.time()
    solid, geometry = build_suboff_mask(
        args.hull_type,
        args.nx,
        args.ny,
        args.nz,
        cx=cx,
        cy=cy,
        cz=cz,
        length=length_lu,
        radius=radius_lu,
        config=SuboffConfig(),
        device=device,
    )
    near = get_near_wall_3d(solid)
    if args.hull_type == "bare_hull":
        pressure_mesh = SurfaceMesh.from_suboff(
            solid,
            near,
            cx,
            cy,
            cz,
            length_lu,
            radius_lu,
            config=SuboffConfig(),
        )
        pressure_method = "analytical SUBOFF normals"
    else:
        pressure_mesh = SurfaceMesh.from_gradient(solid, near)
        pressure_method = "voxel-gradient normals"

    bfl_mask = None
    bfl_q = None
    wall_area_weight = None
    surface_area_diagnostics = None
    appendage_links = 0
    appendage_link_diagnostics = None
    bare = None
    if args.boundary in {"bfl_wall", "bfl_wall_model", "bfl_spalding"}:
        print(f"{tag} building BFL link-distance field", flush=True)
        bfl_mask, bfl_q = compute_q_suboff(
            args.nx,
            args.ny,
            args.nz,
            cx,
            cy,
            cz,
            length_lu,
            hull_type=args.hull_type,
            config=SuboffConfig(),
            device=device,
        )
        if args.hull_type == "full":
            bare, _ = build_suboff_mask(
                "bare_hull",
                args.nx,
                args.ny,
                args.nz,
                cx=cx,
                cy=cy,
                cz=cz,
                length=length_lu,
                radius=radius_lu,
                config=SuboffConfig(),
                device=device,
            )
            bfl_q, appendage_link_diagnostics = refine_q_suboff_appendages(
                bfl_mask,
                bfl_q,
                solid,
                bare,
                center=(cx, cy, cz),
                length=length_lu,
                inplace=True,
            )
            appendage_links = appendage_link_diagnostics.target_links
            print(
                f"{tag} BFL links={int(bfl_mask.sum().item())} "
                f"appendage_boundary_links={appendage_links} "
                f"scheme={SUBOFF_APPENDAGE_LINK_SCHEME}",
                flush=True,
            )
        else:
            print(f"{tag} BFL links={int(bfl_mask.sum().item())}", flush=True)
        if args.hull_type == "bare_hull":
            wall_area_weight, surface_area_diagnostics = bfl_surface_area_weights(
                bfl_mask,
                (
                    pressure_mesh.nx_n,
                    pressure_mesh.ny_n,
                    pressure_mesh.nz_n,
                ),
                reference_area=float(geometry["wetted_area_lu2"]),
                boundary_mask=near,
            )
        else:
            assert bare is not None
            bare_near = get_near_wall_3d(bare)
            bare_surface = SurfaceMesh.from_gradient(bare, bare_near)
            bare_bfl_mask, _ = compute_q_suboff(
                args.nx,
                args.ny,
                args.nz,
                cx,
                cy,
                cz,
                length_lu,
                hull_type="bare_hull",
                config=SuboffConfig(),
                device=device,
            )
            _, bare_area_diagnostics = bfl_surface_area_weights(
                bare_bfl_mask,
                (
                    bare_surface.nx_n,
                    bare_surface.ny_n,
                    bare_surface.nz_n,
                ),
                reference_area=float(geometry["wetted_area_lu2"]),
                boundary_mask=bare_near,
            )
            wall_area_weight, surface_area_diagnostics = bfl_surface_area_weights(
                bfl_mask,
                (
                    pressure_mesh.nx_n,
                    pressure_mesh.ny_n,
                    pressure_mesh.nz_n,
                ),
                calibration_factor=(bare_area_diagnostics.calibration_factor),
                boundary_mask=near,
            )

    rho0 = torch.ones((args.nz, args.ny, args.nx), device=device)
    ux0 = torch.full_like(rho0, args.lattice_speed)
    if not (args.boundary in {"bfl_wall_model", "bfl_spalding"} and args.ramp_steps > 0):
        ux0[solid] = 0.0
    zeros = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, ux0, zeros, zeros, device=device)
    free_stream_f = equilibrium3d(
        rho0,
        torch.full_like(rho0, args.lattice_speed),
        zeros,
        zeros,
        device=device,
    )
    if args.sponge_mode == "equilibrium_difference":
        outlet_width = args.outlet_sponge_width or args.sponge_width
        face_widths = {
            "x+": outlet_width,
            "y-": args.sponge_width,
            "y+": args.sponge_width,
            "z-": args.sponge_width,
            "z+": args.sponge_width,
        }
        if args.sponge_inlet:
            face_widths["x-"] = args.sponge_width
        sponge = build_anisotropic_sponge_sigma_3d(
            (args.nz, args.ny, args.nx),
            face_widths=face_widths,
            max_strength=args.sponge_strength,
            device=device,
        )
    else:
        sponge = build_far_field_sponge(
            args.nx,
            args.ny,
            args.nz,
            args.sponge_width,
            args.sponge_strength,
            device,
        )
    initial_mass = float(rho0.sum().item())
    solid_mask = solid.unsqueeze(0).expand(19, args.nz, args.ny, args.nx)
    body_indices = solid.nonzero(as_tuple=False)
    z_min, y_min, x_min = (int(body_indices[:, axis].min().item()) for axis in range(3))
    z_max, y_max, x_max = (int(body_indices[:, axis].max().item()) + 1 for axis in range(3))
    cv = box_control_volume(
        (args.nz, args.ny, args.nx),
        x0=max(1, x_min - args.cv_margin),
        x1=min(args.nx - 1, x_max + args.cv_margin),
        y0=max(1, y_min - args.cv_margin),
        y1=min(args.ny - 1, y_max + args.cv_margin),
        z0=max(1, z_min - args.cv_margin),
        z1=min(args.nz - 1, z_max + args.cv_margin),
        device=device,
    )
    if bool(solid[cv.logical_not() & solid].any()):
        raise RuntimeError("control volume does not contain the complete body")
    if args.sponge_width > 0 and float(sponge[cv].max().item()) > 0.0:
        raise ValueError(
            "control volume overlaps the sponge; enlarge the domain or reduce cv-margin"
        )
    auxiliary_cvs: dict[int, torch.Tensor] = {}
    for margin in aux_cv_margins:
        if margin == args.cv_margin:
            continue
        candidate = box_control_volume(
            (args.nz, args.ny, args.nx),
            x0=max(1, x_min - margin),
            x1=min(args.nx - 1, x_max + margin),
            y0=max(1, y_min - margin),
            y1=min(args.ny - 1, y_max + margin),
            z0=max(1, z_min - margin),
            z1=min(args.nz - 1, z_max + margin),
            device=device,
        )
        if bool(solid[candidate.logical_not() & solid].any()):
            raise ValueError(f"auxiliary control volume margin {margin} does not contain body")
        if args.sponge_width > 0 and float(sponge[candidate].max().item()) > 0.0:
            raise ValueError(f"auxiliary control volume margin {margin} overlaps sponge")
        auxiliary_cvs[margin] = candidate

    force_scale = force_scale_newton(
        rho_water=args.rho_water,
        dx_m=dx_m,
        speed_mps=point.speed_mps,
        lattice_speed=args.lattice_speed,
    )
    block_p: list[float] = []
    block_f: list[float] = []
    snapshots: list[dict] = []
    all_p: list[float] = []
    all_p_voxel: list[float] = []
    all_f: list[float] = []
    all_cv: list[float] = []
    all_bfl_total: list[float] = []
    surface_pressure_samples: list[tuple[int, float]] = []
    primary_cv_samples: list[tuple[int, float]] = []
    auxiliary_cv_samples: dict[int, list[tuple[int, float]]] = {
        margin: [] for margin in auxiliary_cvs
    }
    wall_applicability: dict[str, float | int | None] = {
        "samples": 0,
        "y_plus_min": None,
        "y_plus_mean_sum": 0.0,
        "y_plus_max": None,
        "maximum_rejected_fraction": 0.0,
    }
    diverged = False
    force_method = args.force_method
    if force_method == "auto":
        force_method = (
            "control_volume"
            if args.boundary in {"bfl_wall_model", "bfl_spalding"}
            else "surface_pressure"
        )

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    checkpoint_signature = {
        "grid_nx_ny_nz": [args.nx, args.ny, args.nz],
        "hull_type": args.hull_type,
        "hull_length": args.hull_length,
        "speed_knots": args.speed_knots,
        "center_x_fraction": args.center_x_fraction,
        "lattice_speed": args.lattice_speed,
        "resolved_reynolds": args.resolved_reynolds,
        "boundary": args.boundary,
        "collision_model": args.collision_model,
        "les_model": args.les_model,
        "cs_smag": args.cs_smag,
        "cw_wale": args.cw_wale,
        "far_field_mode": args.far_field_mode,
        "ramp_steps": args.ramp_steps,
        "wall_law": args.wall_law,
        "wall_distance": args.wall_distance,
        "exchange_distance": args.exchange_distance,
        "wall_nonequilibrium_scale": args.wall_nonequilibrium_scale,
        "sponge_width": args.sponge_width,
        "outlet_sponge_width": args.outlet_sponge_width,
        "sponge_strength": args.sponge_strength,
        "sponge_inlet": args.sponge_inlet,
        "sponge_mode": args.sponge_mode,
        "cv_margin": args.cv_margin,
        "aux_cv_margins": list(auxiliary_cvs),
        "force_method": force_method,
        "pressure_reference": args.pressure_reference,
        "mass_interval": args.mass_interval,
        "surface_force_interval": args.surface_force_interval,
        "report_interval": args.report_interval,
        "average_window": args.average_window,
        "rho_water": args.rho_water,
        "nu_water": args.nu_water,
        "stress_exchange_distance": (
            args.stress_exchange_distance if args.stress_exchange_distance > 0.0 else None
        ),
        "surface_area_weighting": (
            "bfl_axial_projection_calibrated_v1" if bfl_mask is not None else None
        ),
        "wall_diagnostic_interval": args.wall_diagnostic_interval,
    }
    if args.hull_type == "full" and bfl_mask is not None:
        checkpoint_signature["appendage_link_scheme"] = SUBOFF_APPENDAGE_LINK_SCHEME
    start_step = 0
    if args.resume:
        assert checkpoint is not None
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state.get("configuration") != checkpoint_signature:
            raise ValueError("checkpoint configuration does not match SUBOFF run")
        start_step = int(state["step"])
        if start_step >= args.steps:
            raise ValueError("checkpoint already reached or exceeded requested steps")
        f = state["populations"].to(device=device)
        all_p = state["pressure_history"].tolist()
        all_p_voxel = state["bfl_pressure_history"].tolist()
        all_f = state["friction_history"].tolist()
        all_cv = state["control_volume_history"].tolist()
        all_bfl_total = state["bfl_total_history"].tolist()
        block_p = state["pending_pressure_block"].tolist()
        block_f = state["pending_friction_block"].tolist()
        surface_pressure_samples = [
            (int(sample_step), float(value))
            for sample_step, value in state["surface_pressure_samples"].tolist()
        ]
        primary_cv_samples = [
            (int(sample_step), float(value))
            for sample_step, value in state["primary_cv_samples"].tolist()
        ]
        auxiliary_cv_samples = {
            int(margin): [
                (int(sample_step), float(value)) for sample_step, value in samples.tolist()
            ]
            for margin, samples in state["auxiliary_cv_samples"].items()
        }
        snapshots = list(state["snapshots"])
        wall_applicability = dict(state["wall_applicability"])

    def save_checkpoint(step: int) -> None:
        if checkpoint is None:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(
            {
                "schema": "tensorlbm-suboff-direct-checkpoint-v4",
                "configuration": checkpoint_signature,
                "step": step,
                "populations": f.detach().cpu(),
                "pressure_history": torch.tensor(all_p, dtype=torch.float64),
                "bfl_pressure_history": torch.tensor(all_p_voxel, dtype=torch.float64),
                "friction_history": torch.tensor(all_f, dtype=torch.float64),
                "control_volume_history": torch.tensor(all_cv, dtype=torch.float64),
                "bfl_total_history": torch.tensor(all_bfl_total, dtype=torch.float64),
                "pending_pressure_block": torch.tensor(block_p, dtype=torch.float64),
                "pending_friction_block": torch.tensor(block_f, dtype=torch.float64),
                "surface_pressure_samples": torch.tensor(
                    surface_pressure_samples,
                    dtype=torch.float64,
                ).reshape(-1, 2),
                "primary_cv_samples": torch.tensor(
                    primary_cv_samples,
                    dtype=torch.float64,
                ).reshape(-1, 2),
                "auxiliary_cv_samples": {
                    str(margin): torch.tensor(samples, dtype=torch.float64).reshape(-1, 2)
                    for margin, samples in auxiliary_cv_samples.items()
                },
                "snapshots": snapshots,
                "wall_applicability": wall_applicability,
            },
            checkpoint,
        )

    def apply_outer_boundary(state: torch.Tensor) -> torch.Tensor:
        if args.far_field_mode == "non_equilibrium_extrapolation":
            return non_equilibrium_far_field_bc_3d(
                state,
                u_in=args.lattice_speed,
            )
        return far_field_bc_3d(state, u_in=args.lattice_speed)

    for step in range(start_step + 1, args.steps + 1):
        ramp_factor = smooth_ramp_factor(step, args.ramp_steps)
        f_step_old = f.clone()
        f_pre_collision = f_step_old
        if args.collision_model == "cumulant_smagorinsky":
            f = collide_cumulant_d3q19(f, tau=tau, C_s=args.cs_smag)
        elif args.les_model == "wale":
            f = collide_wale_mrt3d(f, tau=tau, C_w=args.cw_wale)
        else:
            f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=args.cs_smag)
        # NoDynamics: do not collide populations located inside the body.
        f = torch.where(solid_mask, f_pre_collision, f)
        f_post_collision = f.clone()
        f = stream3d(f)
        f = apply_outer_boundary(f)
        wall_diagnostics = None
        if args.boundary in {"bfl_wall", "bfl_wall_model", "bfl_spalding"}:
            collect_wall_diagnostics = step % args.wall_diagnostic_interval == 0
            wall_result = bfl_wall_function_3d(
                f,
                f_post_collision,
                solid,
                nu_lu,
                bfl_mask,
                bfl_q,
                y_val=args.wall_distance,
                wall_law=args.wall_law,
                near_mask=near,
                apply_bfl=True,
                use_guo=True,
                bfl_wall_mode=(
                    "spalding_exchange"
                    if args.boundary == "bfl_spalding"
                    else ("wall_model_slip" if args.boundary == "bfl_wall_model" else "stationary")
                ),
                wall_activation=(
                    ramp_factor if args.boundary in {"bfl_wall_model", "bfl_spalding"} else 1.0
                ),
                exchange_distance=args.exchange_distance,
                stress_exchange_distance=(
                    args.stress_exchange_distance if args.stress_exchange_distance > 0.0 else None
                ),
                nonequilibrium_scale=args.wall_nonequilibrium_scale,
                wall_normals=(
                    pressure_mesh.nx_n,
                    pressure_mesh.ny_n,
                    pressure_mesh.nz_n,
                ),
                area_weight=wall_area_weight,
                return_wall_diagnostics=collect_wall_diagnostics,
            )
            if collect_wall_diagnostics:
                f, friction_lu, pressure_voxel_lu, wall_diagnostics = wall_result
                wall_applicability["samples"] = int(wall_applicability["samples"]) + 1
                wall_applicability["maximum_rejected_fraction"] = max(
                    float(wall_applicability["maximum_rejected_fraction"]),
                    wall_diagnostics.rejected_fraction,
                )
                if wall_diagnostics.y_plus_mean is not None:
                    wall_applicability["y_plus_mean_sum"] = (
                        float(wall_applicability["y_plus_mean_sum"]) + wall_diagnostics.y_plus_mean
                    )
                if wall_diagnostics.y_plus_min is not None:
                    current_min = wall_applicability["y_plus_min"]
                    wall_applicability["y_plus_min"] = (
                        wall_diagnostics.y_plus_min
                        if current_min is None
                        else min(float(current_min), wall_diagnostics.y_plus_min)
                    )
                if wall_diagnostics.y_plus_max is not None:
                    current_max = wall_applicability["y_plus_max"]
                    wall_applicability["y_plus_max"] = (
                        wall_diagnostics.y_plus_max
                        if current_max is None
                        else max(float(current_max), wall_diagnostics.y_plus_max)
                    )
            else:
                f, friction_lu, pressure_voxel_lu = wall_result
        else:
            if args.boundary == "projected_wall":
                f = project_no_penetration(f, solid, near)
            f, friction_lu, pressure_voxel_lu = wall_function_3d(
                f,
                solid,
                nu_lu,
                y_val=args.wall_distance,
                wall_law=args.wall_law,
                near_mask=near,
            )
        if args.sponge_width > 0 and args.sponge_strength > 0.0:
            if args.sponge_mode == "equilibrium_difference":
                f = apply_equilibrium_difference_sponge(
                    f,
                    sponge,
                    velocity_target=(args.lattice_speed, 0.0, 0.0),
                )
            else:
                sponge_4d = sponge.unsqueeze(0)
                f = (1.0 - sponge_4d) * f + sponge_4d * free_stream_f
        # Re-assert outer faces because the wall-force operation computes and
        # updates the full tensor, while the physical far field is prescribed.
        f = apply_outer_boundary(f)
        cv_force_lu = float(
            observe_control_volume_force(
                f_step_old,
                f,
                f_post_collision,
                cv,
                solid=solid,
            )
            .force_on_body[0]
            .item()
        )
        if force_method == "bfl_momentum":
            pressure_lu = pressure_voxel_lu
        elif force_method == "control_volume":
            # The CV balance already contains pressure and wall-model shear.
            pressure_lu = cv_force_lu - friction_lu
        else:
            pressure_lu = drag_pressure_integration(
                f,
                pressure_mesh,
                1.0,
                extrap="none",
                p0_method=args.pressure_reference,
                solid=solid,
            )[0]
        if step % args.mass_interval == 0:
            f = correct_mass3d(f, initial_mass)

        all_p.append(pressure_lu)
        all_p_voxel.append(pressure_voxel_lu)
        all_f.append(friction_lu)
        all_cv.append(cv_force_lu)
        all_bfl_total.append(pressure_voxel_lu + friction_lu)
        if step % args.surface_force_interval == 0:
            primary_cv_samples.append((step, cv_force_lu))
            surface_pressure_lu = (
                pressure_lu
                if force_method == "surface_pressure"
                else drag_pressure_integration(
                    f,
                    pressure_mesh,
                    1.0,
                    extrap="none",
                    p0_method=args.pressure_reference,
                    solid=solid,
                )[0]
            )
            surface_pressure_samples.append(
                (
                    step,
                    surface_pressure_lu,
                )
            )
            for margin, auxiliary_cv in auxiliary_cvs.items():
                auxiliary_force = float(
                    observe_control_volume_force(
                        f_step_old,
                        f,
                        f_post_collision,
                        auxiliary_cv,
                        solid=solid,
                    )
                    .force_on_body[0]
                    .item()
                )
                auxiliary_cv_samples[margin].append((step, auxiliary_force))
        block_p.append(pressure_lu)
        block_f.append(friction_lu)

        if not bool(torch.isfinite(f).all()):
            diverged = True
            print(f"{tag} DIVERGED step={step}", flush=True)
            break

        if step % args.report_interval == 0:
            p_lu = sum(block_p) / len(block_p)
            fr_lu = sum(block_f) / len(block_f)
            predicted_n = (p_lu + fr_lu) * force_scale
            snapshot = {
                "step": step,
                "pressure_force_lu": p_lu,
                "friction_force_lu": fr_lu,
                "predicted_resistance_n": predicted_n,
                "control_volume_resistance_n": (
                    sum(all_cv[-len(block_p) :]) / len(block_p) * force_scale
                ),
                "bfl_link_plus_wall_stress_n": (
                    sum(all_bfl_total[-len(block_p) :]) / len(block_p) * force_scale
                ),
                "surface_pressure_plus_wall_stress_n": (
                    (surface_pressure_samples[-1][1] + fr_lu) * force_scale
                    if surface_pressure_samples
                    else math.nan
                ),
                "error_pct": abs(predicted_n - point.resistance_n) / point.resistance_n * 100.0,
                "elapsed_s": time.time() - started,
                "wall_stress_diagnostics": (
                    asdict(wall_diagnostics) if wall_diagnostics is not None else None
                ),
            }
            snapshots.append(snapshot)
            print(
                f"{tag} step={step}/{args.steps} Rp={p_lu * force_scale:.3f} N "
                f"Rf={fr_lu * force_scale:.3f} N Rt={predicted_n:.3f} N "
                f"err={snapshot['error_pct']:.2f}% ({snapshot['elapsed_s']:.1f}s)",
                flush=True,
            )
            block_p.clear()
            block_f.clear()
        if (
            checkpoint is not None
            and args.checkpoint_interval
            and step % args.checkpoint_interval == 0
        ):
            save_checkpoint(step)

    completed = len(all_p)
    window = min(args.average_window, completed)
    p_final = sum(all_p[-window:]) / window if window else math.nan
    f_final = sum(all_f[-window:]) / window if window else math.nan
    predicted_n = (p_final + f_final) * force_scale
    surface_window = [
        value for sample_step, value in surface_pressure_samples if sample_step > completed - window
    ]
    primary_cv_window = [
        value for sample_step, value in primary_cv_samples if sample_step > completed - window
    ]
    primary_cv_paired_final = (
        sum(primary_cv_window) / len(primary_cv_window) if primary_cv_window else math.nan
    )
    surface_pressure_final = (
        sum(surface_window) / len(surface_window) if surface_window else math.nan
    )
    auxiliary_cv_final: dict[int, float] = {}
    for margin, samples in auxiliary_cv_samples.items():
        selected = [value for sample_step, value in samples if sample_step > completed - window]
        auxiliary_cv_final[margin] = sum(selected) / len(selected) if selected else math.nan
    auxiliary_items = list(auxiliary_cv_final.items())
    nested_cv_assessment = assess_nested_control_volume_invariance(
        primary_cv_paired_final,
        [value for _, value in auxiliary_items],
    )
    auxiliary_cv_difference_pct = {
        str(margin): difference
        for (margin, _), difference in zip(
            auxiliary_items,
            nested_cv_assessment.differences_pct,
            strict=True,
        )
    }
    error_pct = abs(predicted_n - point.resistance_n) / point.resistance_n * 100.0
    force_stationarity = assess_force_stationarity(
        [
            (pressure + friction) * force_scale
            for pressure, friction in zip(
                all_p[-window:],
                all_f[-window:],
                strict=True,
            )
        ],
        block_size=args.report_interval,
    )
    drift_pct = max(
        force_stationarity.relative_range_pct,
        force_stationarity.half_mean_drift_pct,
        force_stationarity.linear_trend_pct,
    )
    finite = not diverged and math.isfinite(predicted_n)
    reference_area_m2 = float(geometry["wetted_area_lu2"]) * dx_m**2
    dynamic_pressure_area = 0.5 * args.rho_water * point.speed_mps**2 * reference_area_m2
    experimental_ct = point.resistance_n / dynamic_pressure_area
    predicted_ct = predicted_n / dynamic_pressure_area
    ittc_cf = 0.075 / (math.log10(re) - 2.0) ** 2
    wall_samples = int(wall_applicability["samples"])
    maximum_exchange_rejected_fraction = float(
        wall_applicability["maximum_rejected_fraction"],
    )
    exchange_sampling_acceptable = (
        (wall_samples > 0 and maximum_exchange_rejected_fraction <= 0.01)
        if args.boundary in {"bfl_wall", "bfl_wall_model", "bfl_spalding"}
        else True
    )
    surface_area_acceptable = (
        surface_area_diagnostics is not None
        and surface_area_diagnostics.unweighted_nodes == 0
        and surface_area_diagnostics.calibrated_area > 0.0
    )
    production_boundary_acceptable = (
        args.boundary == "bfl_wall_model" and force_method == "control_volume"
    )
    result = {
        "schema": "tensorlbm-suboff-experimental-resistance-v4",
        "status": "measured_candidate" if finite else "failed",
        "physical_validation": False,
        "primary_source": PRIMARY_SOURCE,
        "experimental_point": asdict(point) | {"speed_mps": point.speed_mps},
        "assumptions": {
            "model_length_m": MODEL_LENGTH_M,
            "rho_water_kg_m3": args.rho_water,
            "nu_water_m2_s": args.nu_water,
            "water_note": (
                "Fresh water properties supplied by CLI; original Table 14 "
                "page does not state them."
            ),
        },
        "configuration": {
            "hull_type": args.hull_type,
            "device": str(device),
            "grid_nx_ny_nz": [args.nx, args.ny, args.nz],
            "center_x_fraction": args.center_x_fraction,
            "hull_length_lu": length_lu,
            "hull_radius_lu": radius_lu,
            "dx_m": dx_m,
            "reynolds_number": re,
            "lattice_speed": args.lattice_speed,
            "wall_nu_lu": nu_lu,
            "resolved_reynolds_number": resolved_re,
            "collision_nu_lu": collision_nu_lu,
            "tau": tau,
            "collision": (
                f"D3Q19 cumulant+Smagorinsky(Cs={args.cs_smag})"
                if args.collision_model == "cumulant_smagorinsky"
                else (
                    f"D3Q19 MRT+WALE(Cw={args.cw_wale})"
                    if args.les_model == "wale"
                    else f"D3Q19 MRT+Smagorinsky(Cs={args.cs_smag})"
                )
            ),
            "wall_treatment": (
                f"{args.wall_law}(exchange_y={args.stress_exchange_distance})"
                if args.stress_exchange_distance > 0.0 and args.boundary != "bfl_spalding"
                else f"{args.wall_law}(y={args.wall_distance})"
            ),
            "stress_exchange_distance": (
                args.stress_exchange_distance if args.stress_exchange_distance > 0.0 else None
            ),
            "pressure_force_method": (
                "conservative BFL link momentum exchange"
                if force_method == "bfl_momentum"
                else (
                    "discrete internal control-volume momentum balance"
                    if force_method == "control_volume"
                    else f"{pressure_method}; p0={args.pressure_reference}"
                )
            ),
            "boundary": (
                "far_field + target sponge + NoDynamics + BFL + Spalding exchange wall model"
                if args.boundary == "bfl_spalding"
                else (
                    "far_field + target sponge + NoDynamics + BFL slip + Guo wall stress"
                    if args.boundary == "bfl_wall_model"
                    else (
                        "far_field + target sponge + NoDynamics + stationary "
                        "BFL + Guo wall function"
                        if args.boundary == "bfl_wall"
                        else (
                            "far_field + target sponge + NoDynamics + "
                            "normal-velocity projection + wall function"
                            if args.boundary == "projected_wall"
                            else "far_field + target sponge + NoDynamics + legacy "
                            "wall-function body force"
                        )
                    )
                )
            ),
            "sponge_width": args.sponge_width,
            "outlet_sponge_width": args.outlet_sponge_width or args.sponge_width,
            "sponge_strength": args.sponge_strength,
            "sponge_mode": args.sponge_mode,
            "sponge_inlet_enabled": args.sponge_inlet,
            "far_field_mode": args.far_field_mode,
            "steps_requested": args.steps,
            "steps_completed": completed,
            "resumed_from_step": start_step,
            "checkpoint_path": str(checkpoint) if checkpoint else None,
            "wall_activation_ramp_steps": (
                args.ramp_steps if args.boundary in {"bfl_wall_model", "bfl_spalding"} else 0
            ),
            "average_window": window,
            "surface_force_interval": args.surface_force_interval,
            "wall_diagnostic_interval": args.wall_diagnostic_interval,
            "appendage_link_scheme": (
                SUBOFF_APPENDAGE_LINK_SCHEME
                if args.hull_type == "full" and bfl_mask is not None
                else None
            ),
        },
        "geometry": geometry
        | {
            "appendage_boundary_links": appendage_links,
            "appendage_halfway_links": 0,
            "appendage_link_intersection": (
                appendage_link_diagnostics.to_dict()
                if appendage_link_diagnostics is not None
                else None
            ),
            "surface_area_weighting": (
                vars(surface_area_diagnostics) if surface_area_diagnostics is not None else None
            ),
        },
        "result": {
            "pressure_resistance_n": p_final * force_scale,
            "boundary_force_resistance_n_diagnostic": (
                sum(all_p_voxel[-window:]) / window * force_scale
            ),
            "control_volume_resistance_n_diagnostic": (
                sum(all_cv[-window:]) / window * force_scale
            ),
            "bfl_link_plus_wall_stress_n_diagnostic": (
                sum(all_bfl_total[-window:]) / window * force_scale
            ),
            "surface_pressure_resistance_n_diagnostic": (surface_pressure_final * force_scale),
            "surface_pressure_plus_wall_stress_n_diagnostic": (
                (surface_pressure_final + f_final) * force_scale
            ),
            "surface_pressure_samples_in_window": len(surface_window),
            "auxiliary_control_volume_resistance_n": {
                str(margin): value * force_scale for margin, value in auxiliary_cv_final.items()
            },
            "paired_primary_control_volume_resistance_n": (primary_cv_paired_final * force_scale),
            "paired_control_volume_samples_in_window": len(primary_cv_window),
            "auxiliary_control_volume_difference_pct": auxiliary_cv_difference_pct,
            "nested_control_volume_invariance": {
                "auxiliary_count": nested_cv_assessment.auxiliary_count,
                "maximum_difference_pct": (nested_cv_assessment.maximum_difference_pct),
                "finite": nested_cv_assessment.finite,
            },
            "friction_resistance_n": f_final * force_scale,
            "total_resistance_n": predicted_n,
            "experimental_resistance_n": point.resistance_n,
            "error_pct": error_pct,
            "last_three_block_drift_pct": drift_pct,
            "maximum_stationarity_metric_pct": drift_pct,
            "force_stationarity": force_stationarity.to_dict(),
            "finite": finite,
            "diverged": diverged,
            "wall_stress_applicability": {
                "samples": wall_samples,
                "y_plus_min": wall_applicability["y_plus_min"],
                "y_plus_mean": (
                    float(wall_applicability["y_plus_mean_sum"]) / wall_samples
                    if wall_samples
                    else None
                ),
                "y_plus_max": wall_applicability["y_plus_max"],
                "maximum_rejected_fraction": (maximum_exchange_rejected_fraction),
            },
        },
        "coefficients": {
            "reference_wetted_area_m2": reference_area_m2,
            "area_note": (
                "Analytical bare-body wetted area; full appendage area is not yet included."
                if args.hull_type == "full"
                else "Analytical bare-body wetted area from the DARPA profile."
            ),
            "experimental_ct": experimental_ct,
            "predicted_ct": predicted_ct,
            "predicted_friction_cf": (f_final * force_scale / dynamic_pressure_area),
            "predicted_pressure_ct": (p_final * force_scale / dynamic_pressure_area),
            "ittc_1957_cf_context_only": ittc_cf,
            "wall_model_vs_ittc_friction_error_pct": (
                abs(f_final * force_scale / dynamic_pressure_area - ittc_cf) / ittc_cf * 100.0
            ),
            "experimental_ct_over_ittc_cf": experimental_ct / ittc_cf,
        },
        "snapshots": snapshots,
        "elapsed_s": time.time() - started,
        "acceptance": {
            "force_error_target_pct": args.error_target,
            "steady_drift_target_pct": args.drift_target,
            "maximum_exchange_rejected_fraction": 0.01,
            "nested_control_volume_target_pct": 1.0,
            "minimum_auxiliary_control_volumes": 2,
            "force_target_met": error_pct <= args.error_target,
            "steady_target_met": force_stationarity.meets(args.drift_target),
            "exchange_sampling_target_met": exchange_sampling_acceptable,
            "nested_control_volume_target_met": (
                nested_cv_assessment.meets(1.0, minimum_auxiliary_count=2)
            ),
            "surface_area_target_met": surface_area_acceptable,
            "production_boundary_target_met": production_boundary_acceptable,
            "admitted": (
                error_pct <= args.error_target
                and force_stationarity.meets(args.drift_target)
                and exchange_sampling_acceptable
                and nested_cv_assessment.meets(
                    1.0,
                    minimum_auxiliary_count=2,
                )
                and surface_area_acceptable
                and production_boundary_acceptable
                and finite
            ),
            "claim_boundary": (
                "One grid/time candidate; grid and time convergence plus "
                "paired AFF-1/AFF-8 ratio are required for physical validation."
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if checkpoint is not None:
        save_checkpoint(completed)
    print(
        f"{tag} DONE Rt={predicted_n:.3f} N experiment={point.resistance_n:.3f} N "
        f"error={error_pct:.2f}% drift={drift_pct:.2f}% output={output}",
        flush=True,
    )
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hull-type", choices=("bare_hull", "full"), required=True)
    p.add_argument("--speed-knots", type=float, default=5.92)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--nx", type=int, default=200)
    p.add_argument("--ny", type=int, default=80)
    p.add_argument("--nz", type=int, default=80)
    p.add_argument("--hull-length", type=float, default=80.0)
    p.add_argument("--center-x-fraction", type=float, default=0.35)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--average-window", type=int, default=500)
    p.add_argument("--report-interval", type=int, default=250)
    p.add_argument("--wall-diagnostic-interval", type=int, default=50)
    p.add_argument("--mass-interval", type=int, default=200)
    p.add_argument("--cv-margin", type=int, default=8)
    p.add_argument(
        "--aux-cv-margins",
        default="4,12",
        help="Comma-separated nested control-volume margins sampled independently.",
    )
    p.add_argument("--ramp-steps", type=int, default=1000)
    p.add_argument("--lattice-speed", type=float, default=0.06)
    p.add_argument("--cs-smag", type=float, default=0.05)
    p.add_argument("--cw-wale", type=float, default=0.5)
    p.add_argument("--les-model", choices=("wale", "smagorinsky"), default="wale")
    p.add_argument(
        "--collision-model",
        choices=("mrt_les", "cumulant_smagorinsky"),
        default="mrt_les",
    )
    p.add_argument(
        "--resolved-reynolds",
        type=float,
        default=0.0,
        help="Optional bulk-flow Re for wall-model sensitivity; 0 uses physical Re.",
    )
    p.add_argument("--wall-law", choices=("log", "reichardt", "musker"), default="log")
    p.add_argument("--wall-distance", type=float, default=0.5)
    p.add_argument("--exchange-distance", type=float, default=3.0)
    p.add_argument(
        "--stress-exchange-distance",
        type=float,
        default=0.0,
        help=(
            "Wall-normal velocity sampling distance for BFL slip + Guo stress; "
            "0 keeps the legacy boundary-node stress input."
        ),
    )
    p.add_argument("--wall-nonequilibrium-scale", type=float, default=0.5)
    p.add_argument(
        "--boundary",
        choices=("bfl_spalding", "bfl_wall_model", "bfl_wall", "projected_wall", "legacy_wall"),
        default="bfl_wall_model",
        help=(
            "bfl_wall_model is the production BFL+exchange-stress path; "
            "other modes are diagnostics."
        ),
    )
    p.add_argument("--sponge-width", type=int, default=12)
    p.add_argument(
        "--outlet-sponge-width",
        type=int,
        default=0,
        help="Independent x+ sponge thickness; 0 uses --sponge-width.",
    )
    p.add_argument("--sponge-strength", type=float, default=0.2)
    p.add_argument(
        "--sponge-inlet",
        action="store_true",
        help="Also damp the prescribed inlet; disabled by default to avoid upstream reflection.",
    )
    p.add_argument(
        "--sponge-mode",
        choices=("equilibrium_difference", "legacy_distribution_blend"),
        default="equilibrium_difference",
    )
    p.add_argument(
        "--far-field-mode",
        choices=("non_equilibrium_extrapolation", "legacy_hard_equilibrium"),
        default="non_equilibrium_extrapolation",
    )
    p.add_argument(
        "--pressure-reference",
        choices=("near_wall", "far_field", "domain_avg", "inlet"),
        default="inlet",
    )
    p.add_argument(
        "--force-method",
        choices=("auto", "control_volume", "bfl_momentum", "surface_pressure"),
        default="auto",
    )
    p.add_argument(
        "--surface-force-interval",
        type=int,
        default=50,
        help="Cadence for the independent surface-pressure force observer.",
    )
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--checkpoint-interval", type=int, default=2000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--rho-water", type=float, default=998.2)
    p.add_argument("--nu-water", type=float, default=1.004e-6)
    p.add_argument("--error-target", type=float, default=5.0)
    p.add_argument("--drift-target", type=float, default=1.0)
    p.add_argument("--output", required=True)
    return p


if __name__ == "__main__":
    run_case(parser().parse_args())
