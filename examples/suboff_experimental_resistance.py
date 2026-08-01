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
from tensorlbm.control_volume_force import (
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
from tensorlbm.interpolated_bc_suboff import compute_q_suboff
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.sponge_layer import (
    apply_equilibrium_difference_sponge,
    build_sponge_sigma_3d,
)
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask
from tensorlbm.suboff_reference_data import (
    SUBOFF_TOW_TANK_RESISTANCE_TABLE14,
    SuboffTowTankResistancePoint,
)
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
        point for point in SUBOFF_TOW_TANK_RESISTANCE_TABLE14
        if point.hull_type == hull_type
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
    nx: int, ny: int, nz: int, width: int, strength: float,
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
    f: torch.Tensor, solid: torch.Tensor, near: torch.Tensor,
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
    return f + equilibrium3d(
        rho, ux_target, uy_target, uz_target, device=f.device,
    ) - equilibrium3d(rho, ux, uy, uz, device=f.device)


def run_case(args: argparse.Namespace) -> dict:
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
    cx, cy, cz = args.nx * 0.35, args.ny / 2.0, args.nz / 2.0
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
        args.hull_type, args.nx, args.ny, args.nz,
        cx=cx, cy=cy, cz=cz, length=length_lu, radius=radius_lu,
        config=SuboffConfig(), device=device,
    )
    near = get_near_wall_3d(solid)
    if args.hull_type == "bare_hull":
        pressure_mesh = SurfaceMesh.from_suboff(
            solid, near, cx, cy, cz, length_lu, radius_lu,
            config=SuboffConfig(),
        )
        pressure_method = "analytical SUBOFF normals"
    else:
        pressure_mesh = SurfaceMesh.from_gradient(solid, near)
        pressure_method = "voxel-gradient normals"

    bfl_mask = None
    bfl_q = None
    if args.boundary in {"bfl_wall", "bfl_wall_model", "bfl_spalding"}:
        print(f"{tag} building BFL link-distance field", flush=True)
        bfl_mask, bfl_q = compute_q_suboff(
            args.nx, args.ny, args.nz, cx, cy, cz, length_lu,
            hull_type=args.hull_type, config=SuboffConfig(), device=device,
        )
        if args.hull_type == "full":
            # The analytical q solver describes the axisymmetric main body.
            # Retain those sub-cell distances and use half-way bounce-back for
            # sail/control-surface links, whose geometry is voxelised.
            bare, _ = build_suboff_mask(
                "bare_hull", args.nx, args.ny, args.nz,
                cx=cx, cy=cy, cz=cz, length=length_lu, radius=radius_lu,
                config=SuboffConfig(), device=device,
            )
            from tensorlbm.d3q19 import C as C19
            appendage_links = 0
            for direction in range(1, 19):
                dcx, dcy, dcz = (int(v) for v in C19[direction].tolist())
                full_nb = torch.roll(
                    solid, shifts=(-dcz, -dcy, -dcx), dims=(0, 1, 2),
                )
                bare_nb = torch.roll(
                    bare, shifts=(-dcz, -dcy, -dcx), dims=(0, 1, 2),
                )
                use_halfway = bfl_mask[direction] & full_nb & ~bare_nb
                appendage_links += int(use_halfway.sum().item())
                bfl_q[direction][use_halfway] = 0.5
            print(
                f"{tag} BFL links={int(bfl_mask.sum().item())} "
                f"appendage_halfway_links={appendage_links}", flush=True,
            )
        else:
            print(f"{tag} BFL links={int(bfl_mask.sum().item())}", flush=True)

    rho0 = torch.ones((args.nz, args.ny, args.nx), device=device)
    ux0 = torch.full_like(rho0, args.lattice_speed)
    if not (
        args.boundary in {"bfl_wall_model", "bfl_spalding"}
        and args.ramp_steps > 0
    ):
        ux0[solid] = 0.0
    zeros = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, ux0, zeros, zeros, device=device)
    free_stream_f = equilibrium3d(
        rho0, torch.full_like(rho0, args.lattice_speed), zeros, zeros,
        device=device,
    )
    if args.sponge_mode == "equilibrium_difference":
        sponge = build_sponge_sigma_3d(
            (args.nz, args.ny, args.nx), width=args.sponge_width,
            max_strength=args.sponge_strength, device=device,
        )
    else:
        sponge = build_far_field_sponge(
            args.nx, args.ny, args.nz, args.sponge_width,
            args.sponge_strength, device,
        )
    initial_mass = float(rho0.sum().item())
    solid_mask = solid.unsqueeze(0).expand(19, args.nz, args.ny, args.nx)
    body_indices = solid.nonzero(as_tuple=False)
    z_min, y_min, x_min = (
        int(body_indices[:, axis].min().item()) for axis in range(3)
    )
    z_max, y_max, x_max = (
        int(body_indices[:, axis].max().item()) + 1 for axis in range(3)
    )
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
        raise ValueError("control volume overlaps the sponge; enlarge the domain or reduce cv-margin")

    force_scale = force_scale_newton(
        rho_water=args.rho_water, dx_m=dx_m,
        speed_mps=point.speed_mps, lattice_speed=args.lattice_speed,
    )
    block_p: list[float] = []
    block_f: list[float] = []
    snapshots: list[dict] = []
    all_p: list[float] = []
    all_p_voxel: list[float] = []
    all_f: list[float] = []
    all_cv: list[float] = []
    all_bfl_total: list[float] = []
    diverged = False
    force_method = args.force_method
    if force_method == "auto":
        force_method = (
            "control_volume"
            if args.boundary in {"bfl_wall_model", "bfl_spalding"}
            else "surface_pressure"
        )

    for step in range(1, args.steps + 1):
        ramp_factor = smooth_ramp_factor(step, args.ramp_steps)
        boundary_speed = args.lattice_speed
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
        f = far_field_bc_3d(f, u_in=boundary_speed)
        if args.boundary in {"bfl_wall", "bfl_wall_model", "bfl_spalding"}:
            f, friction_lu, pressure_voxel_lu = bfl_wall_function_3d(
                f, f_post_collision, solid, nu_lu,
                bfl_mask, bfl_q, y_val=args.wall_distance,
                wall_law=args.wall_law, near_mask=near,
                apply_bfl=True, use_guo=True,
                bfl_wall_mode=(
                    "spalding_exchange" if args.boundary == "bfl_spalding"
                    else (
                        "wall_model_slip"
                        if args.boundary == "bfl_wall_model" else "stationary"
                    )
                ),
                wall_activation=(
                    ramp_factor
                    if args.boundary in {"bfl_wall_model", "bfl_spalding"}
                    else 1.0
                ),
                exchange_distance=args.exchange_distance,
                nonequilibrium_scale=args.wall_nonequilibrium_scale,
                wall_normals=(
                    pressure_mesh.nx_n, pressure_mesh.ny_n,
                    pressure_mesh.nz_n,
                ),
                area_weight=(
                    torch.full_like(
                        near,
                        float(geometry["wetted_area_lu2"])
                        / max(float(near.sum().item()), 1.0),
                        dtype=f.dtype,
                    )
                    if args.hull_type == "bare_hull" else None
                ),
            )
        else:
            if args.boundary == "projected_wall":
                f = project_no_penetration(f, solid, near)
            f, friction_lu, pressure_voxel_lu = wall_function_3d(
                f, solid, nu_lu, y_val=args.wall_distance,
                wall_law=args.wall_law, near_mask=near,
            )
        if args.sponge_width > 0 and args.sponge_strength > 0.0:
            if args.sponge_mode == "equilibrium_difference":
                f = apply_equilibrium_difference_sponge(
                    f, sponge,
                    velocity_target=(args.lattice_speed, 0.0, 0.0),
                )
            else:
                sponge_4d = sponge.unsqueeze(0)
                f = (1.0 - sponge_4d) * f + sponge_4d * free_stream_f
        # Re-assert outer faces because the wall-force operation computes and
        # updates the full tensor, while the physical far field is prescribed.
        f = far_field_bc_3d(f, u_in=boundary_speed)
        cv_force_lu = float(observe_control_volume_force(
            f_step_old, f, f_post_collision, cv, solid=solid,
        ).force_on_body[0].item())
        if force_method == "bfl_momentum":
            pressure_lu = pressure_voxel_lu
        elif force_method == "control_volume":
            # The CV balance already contains pressure and wall-model shear.
            pressure_lu = cv_force_lu - friction_lu
        else:
            pressure_lu = drag_pressure_integration(
                f, pressure_mesh, 1.0, extrap="none",
                p0_method=args.pressure_reference,
                solid=solid,
            )[0]
        if step % args.mass_interval == 0:
            f = correct_mass3d(f, initial_mass)

        all_p.append(pressure_lu); all_p_voxel.append(pressure_voxel_lu)
        all_f.append(friction_lu)
        all_cv.append(cv_force_lu)
        all_bfl_total.append(pressure_voxel_lu + friction_lu)
        block_p.append(pressure_lu); block_f.append(friction_lu)

        if not bool(torch.isfinite(f).all()):
            diverged = True
            print(f"{tag} DIVERGED step={step}", flush=True)
            break

        if step % args.report_interval == 0:
            p_lu = sum(block_p) / len(block_p); fr_lu = sum(block_f) / len(block_f)
            predicted_n = (p_lu + fr_lu) * force_scale
            snapshot = {
                "step": step,
                "pressure_force_lu": p_lu,
                "friction_force_lu": fr_lu,
                "predicted_resistance_n": predicted_n,
                "control_volume_resistance_n": (
                    sum(all_cv[-len(block_p):]) / len(block_p) * force_scale
                ),
                "bfl_link_plus_wall_stress_n": (
                    sum(all_bfl_total[-len(block_p):]) / len(block_p) * force_scale
                ),
                "error_pct": abs(predicted_n - point.resistance_n) / point.resistance_n * 100.0,
                "elapsed_s": time.time() - started,
            }
            snapshots.append(snapshot)
            print(
                f"{tag} step={step}/{args.steps} Rp={p_lu*force_scale:.3f} N "
                f"Rf={fr_lu*force_scale:.3f} N Rt={predicted_n:.3f} N "
                f"err={snapshot['error_pct']:.2f}% ({snapshot['elapsed_s']:.1f}s)",
                flush=True,
            )
            block_p.clear(); block_f.clear()

    completed = len(all_p)
    window = min(args.average_window, completed)
    p_final = sum(all_p[-window:]) / window if window else math.nan
    f_final = sum(all_f[-window:]) / window if window else math.nan
    predicted_n = (p_final + f_final) * force_scale
    error_pct = abs(predicted_n - point.resistance_n) / point.resistance_n * 100.0
    recent = [x["predicted_resistance_n"] for x in snapshots[-3:]]
    recent_mean = sum(recent) / len(recent) if recent else 0.0
    drift_pct = (
        (max(recent) - min(recent)) / abs(recent_mean) * 100.0
        if len(recent) >= 3 and abs(recent_mean) > 1e-12 else math.inf
    )
    finite = not diverged and math.isfinite(predicted_n)
    reference_area_m2 = float(geometry["wetted_area_lu2"]) * dx_m**2
    dynamic_pressure_area = (
        0.5 * args.rho_water * point.speed_mps**2 * reference_area_m2
    )
    experimental_ct = point.resistance_n / dynamic_pressure_area
    predicted_ct = predicted_n / dynamic_pressure_area
    ittc_cf = 0.075 / (math.log10(re) - 2.0) ** 2
    result = {
        "schema": "tensorlbm-suboff-experimental-resistance-v1",
        "status": "measured_candidate" if finite else "failed",
        "physical_validation": False,
        "primary_source": PRIMARY_SOURCE,
        "experimental_point": asdict(point) | {"speed_mps": point.speed_mps},
        "assumptions": {
            "model_length_m": MODEL_LENGTH_M,
            "rho_water_kg_m3": args.rho_water,
            "nu_water_m2_s": args.nu_water,
            "water_note": "Fresh water properties supplied by CLI; original Table 14 page does not state them.",
        },
        "configuration": {
            "hull_type": args.hull_type, "device": str(device),
            "grid_nx_ny_nz": [args.nx, args.ny, args.nz],
            "hull_length_lu": length_lu, "hull_radius_lu": radius_lu,
            "dx_m": dx_m, "reynolds_number": re,
            "lattice_speed": args.lattice_speed, "wall_nu_lu": nu_lu,
            "resolved_reynolds_number": resolved_re,
            "collision_nu_lu": collision_nu_lu, "tau": tau,
            "collision": (
                f"D3Q19 cumulant+Smagorinsky(Cs={args.cs_smag})"
                if args.collision_model == "cumulant_smagorinsky" else
                (
                    f"D3Q19 MRT+WALE(Cw={args.cw_wale})"
                    if args.les_model == "wale" else
                    f"D3Q19 MRT+Smagorinsky(Cs={args.cs_smag})"
                )
            ),
            "wall_treatment": f"{args.wall_law}(y={args.wall_distance})",
            "pressure_force_method": (
                "conservative BFL link momentum exchange"
                if force_method == "bfl_momentum" else
                (
                    "discrete internal control-volume momentum balance"
                    if force_method == "control_volume" else
                    f"{pressure_method}; p0={args.pressure_reference}"
                )
            ),
            "boundary": (
                "far_field + target sponge + NoDynamics + BFL + Spalding exchange wall model"
                if args.boundary == "bfl_spalding" else
                (
                    "far_field + target sponge + NoDynamics + BFL slip + Guo wall stress"
                    if args.boundary == "bfl_wall_model" else
                    (
                        "far_field + target sponge + NoDynamics + stationary BFL + Guo wall function"
                        if args.boundary == "bfl_wall" else
                        (
                            "far_field + target sponge + NoDynamics + normal-velocity projection + wall function"
                            if args.boundary == "projected_wall" else
                            "far_field + target sponge + NoDynamics + legacy wall-function body force"
                        )
                    )
                )
            ),
            "sponge_width": args.sponge_width,
            "sponge_strength": args.sponge_strength,
            "sponge_mode": args.sponge_mode,
            "steps_requested": args.steps, "steps_completed": completed,
            "wall_activation_ramp_steps": (
                args.ramp_steps
                if args.boundary in {"bfl_wall_model", "bfl_spalding"} else 0
            ),
            "average_window": window,
        },
        "geometry": geometry,
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
            "friction_resistance_n": f_final * force_scale,
            "total_resistance_n": predicted_n,
            "experimental_resistance_n": point.resistance_n,
            "error_pct": error_pct,
            "last_three_block_drift_pct": drift_pct,
            "finite": finite, "diverged": diverged,
        },
        "coefficients": {
            "reference_wetted_area_m2": reference_area_m2,
            "area_note": (
                "Analytical bare-body wetted area; full appendage area is not yet included."
                if args.hull_type == "full" else
                "Analytical bare-body wetted area from the DARPA profile."
            ),
            "experimental_ct": experimental_ct,
            "predicted_ct": predicted_ct,
            "ittc_1957_cf_context_only": ittc_cf,
            "experimental_ct_over_ittc_cf": experimental_ct / ittc_cf,
        },
        "snapshots": snapshots,
        "elapsed_s": time.time() - started,
        "acceptance": {
            "force_error_target_pct": args.error_target,
            "steady_drift_target_pct": args.drift_target,
            "force_target_met": error_pct <= args.error_target,
            "steady_target_met": drift_pct <= args.drift_target,
            "admitted": error_pct <= args.error_target and drift_pct <= args.drift_target and finite,
            "claim_boundary": "One grid/time candidate; grid and time convergence plus paired AFF-1/AFF-8 ratio are required for physical validation.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
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
    p.add_argument("--nx", type=int, default=200); p.add_argument("--ny", type=int, default=80); p.add_argument("--nz", type=int, default=80)
    p.add_argument("--hull-length", type=float, default=80.0)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--average-window", type=int, default=500)
    p.add_argument("--report-interval", type=int, default=250)
    p.add_argument("--mass-interval", type=int, default=200)
    p.add_argument("--cv-margin", type=int, default=8)
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
        "--resolved-reynolds", type=float, default=0.0,
        help="Optional bulk-flow Re for wall-model sensitivity; 0 uses physical Re.",
    )
    p.add_argument("--wall-law", choices=("log", "reichardt", "musker"), default="log")
    p.add_argument("--wall-distance", type=float, default=0.5)
    p.add_argument("--exchange-distance", type=float, default=3.0)
    p.add_argument("--wall-nonequilibrium-scale", type=float, default=0.5)
    p.add_argument(
        "--boundary",
        choices=("bfl_spalding", "bfl_wall_model", "bfl_wall", "projected_wall", "legacy_wall"),
        default="bfl_spalding",
        help="bfl_spalding is the exchange-location validation path; other modes are diagnostics.",
    )
    p.add_argument("--sponge-width", type=int, default=12)
    p.add_argument("--sponge-strength", type=float, default=0.2)
    p.add_argument(
        "--sponge-mode",
        choices=("equilibrium_difference", "legacy_distribution_blend"),
        default="equilibrium_difference",
    )
    p.add_argument(
        "--pressure-reference",
        choices=("near_wall", "far_field", "domain_avg", "inlet"),
        default="near_wall",
    )
    p.add_argument(
        "--force-method",
        choices=("auto", "control_volume", "bfl_momentum", "surface_pressure"),
        default="auto",
    )
    p.add_argument("--rho-water", type=float, default=998.2)
    p.add_argument("--nu-water", type=float, default=1.004e-6)
    p.add_argument("--error-target", type=float, default=5.0)
    p.add_argument("--drift-target", type=float, default=1.0)
    p.add_argument("--output", required=True)
    return p


if __name__ == "__main__":
    run_case(parser().parse_args())
