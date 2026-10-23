#!/usr/bin/env python3
"""New benchmarks: Cylinder Re=40/3900 + Sphere Re=300/10000.

Uses the verified main loop (lbm_step_correct) with:
  - NoDynamics at solid cells
  - Half-way bounce-back with f_pre (Bug 27 fix)
  - Far-field BC
  - MRT / MRT+Smagorinsky collision

Universal flow (common interface ONLY):
  solid → get_near_wall_3d → SurfaceMesh.from_xxx → lbm_step_correct
  → drag_pressure_integration + drag_friction_integration → detect_strouhal

Benchmarks:
  1. cyl_re40    — Cylinder Re=40,  steady separation (SDAA:0)
  2. cyl_re3900  — Cylinder Re=3900, turbulent wake   (SDAA:1)
  3. sph_re300   — Sphere Re=300,   steady ring vortex (SDAA:2)
  4. sph_re10000 — Sphere Re=10000, turbulent wake     (SDAA:3)

Usage:
  python new_benchmarks_worker.py <benchmark> <device_id> <output_path>
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.solver3d import collide_mrt3d
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    get_near_wall_3d,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.postprocess import detect_strouhal


# ---------------------------------------------------------------------------
#  Geometry builders
# ---------------------------------------------------------------------------
def build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    """Boolean solid mask for a cylinder extruded along z-axis."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


def build_sphere_mask(nx, ny, nz, cx, cy, cz, R, device):
    """Boolean solid mask for a 3D sphere."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) < R ** 2


# ---------------------------------------------------------------------------
#  Separation angle (cylinder)
# ---------------------------------------------------------------------------
def measure_separation_angle(ux, uy, near, cx, cy, R, mid_z):
    """Measure separation angle from rear stagnation point (upper half).

    The tangential velocity in the flow direction (front→rear, clockwise
    on upper half) is  u_t = ux·sin(φ) − uy·cos(φ)  where φ is the polar
    angle from the +x axis measured at the cylinder centre.

    Separation occurs where u_t changes sign (positive→negative).
    The separation angle from the rear stagnation point = φ (degrees).
    """
    # Work on the mid-z slice (2D)
    near_2d = near[mid_z].cpu()
    ux_2d = ux[mid_z].cpu()
    uy_2d = uy[mid_z].cpu()
    ny, nx = near_2d.shape

    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )

    # Polar angle from center (degrees), upper half only (y > cy)
    dx = xx - cx
    dy = yy - cy
    phi = torch.atan2(dy, dx)  # radians, -pi to pi
    phi_deg = torch.rad2deg(phi)

    # Upper half: y > cy  →  phi in (0, 180)
    upper = near_2d & (dy > 0)

    if upper.sum() < 4:
        return float("nan"), []

    # Tangential velocity in flow direction
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    u_t = ux_2d * sin_phi - uy_2d * cos_phi

    # Extract upper-half near-wall cells
    idx = upper.nonzero(as_tuple=False).squeeze(1)
    phi_vals = phi_deg[idx[:, 0], idx[:, 1]].numpy()
    ut_vals = u_t[idx[:, 0], idx[:, 1]].numpy()

    # Sort by phi descending (front=180 → rear=0)
    order = np.argsort(-phi_vals)
    phi_sorted = phi_vals[order]
    ut_sorted = ut_vals[order]

    # Find zero crossing (positive → negative)
    sep_angle = float("nan")
    for i in range(len(ut_sorted) - 1):
        if ut_sorted[i] > 0 and ut_sorted[i + 1] <= 0:
            # Linear interpolation
            t = ut_sorted[i] / (ut_sorted[i] - ut_sorted[i + 1])
            sep_angle = float(phi_sorted[i] + t * (phi_sorted[i + 1] - phi_sorted[i]))
            break

    return sep_angle, list(zip(phi_sorted.tolist(), ut_sorted.tolist()))


# ---------------------------------------------------------------------------
#  Recirculation length (cylinder)
# ---------------------------------------------------------------------------
def measure_recirculation_length(ux, solid, cx, cy, R, mid_z, u_in):
    """Measure recirculation length along centerline (y=cy).

    L_r = distance from rear of cylinder to where ux returns to positive
    along the centerline y=cy.
    """
    ux_line = ux[mid_z, int(cy), :].cpu().numpy()
    x_rear = int(cx + R)
    # Search from x_rear forward for sign change (negative → positive)
    x_start = min(x_rear + 1, len(ux_line) - 1)
    for i in range(x_start, len(ux_line) - 1):
        if ux_line[i] <= 0 and ux_line[i + 1] > 0:
            # Linear interpolation
            t = -ux_line[i] / (ux_line[i + 1] - ux_line[i])
            x_sep = i + t
            return float((x_sep - x_rear) / (2 * R))
    return float("nan")


# ---------------------------------------------------------------------------
#  Main run function
# ---------------------------------------------------------------------------
def run_benchmark(benchmark, device_id, output_path):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # ---- Benchmark configurations ----
    if benchmark == "cyl_re40":
        config = dict(
            shape="cylinder", D=48.0, nx=400, ny=160, nz=4,
            u_in=0.08, Re=40.0, n_steps=10000,
            collision="mrt", Cs=0.0,
            Cd_ref=1.50, sep_ref=53.0, Cl_ref=0.0,
            ref_name="steady separation (Re=40)",
        )
    elif benchmark == "cyl_re3900":
        config = dict(
            shape="cylinder", D=48.0, nx=400, ny=160, nz=4,
            u_in=0.08, Re=3900.0, n_steps=20000,
            collision="mrt_smag", Cs=0.1,
            Cd_ref=0.98, sep_ref=None, Cl_ref=None,
            recirc_ref=1.4,
            ref_name="turbulent wake (Re=3900)",
        )
    elif benchmark == "sph_re300":
        config = dict(
            shape="sphere", D=40.0, nx=180, ny=180, nz=180,
            u_in=0.08, Re=300.0, n_steps=5000,
            collision="mrt_smag", Cs=0.05,
            Cd_ref=0.44, sep_ref=None, Cl_ref=None,
            ref_name="steady ring vortex (Re=300)",
        )
    elif benchmark == "sph_re10000":
        config = dict(
            shape="sphere", D=40.0, nx=180, ny=180, nz=180,
            u_in=0.08, Re=10000.0, n_steps=5000,
            collision="mrt_smag", Cs=0.1,
            Cd_ref=0.40, sep_ref=None, Cl_ref=None,
            ref_name="turbulent wake (Re=10000)",
        )
    else:
        print(f"Unknown benchmark: {benchmark}", flush=True)
        sys.exit(1)

    D = config["D"]
    R = D / 2.0
    nx, ny, nz = config["nx"], config["ny"], config["nz"]
    u_in = config["u_in"]
    Re = config["Re"]
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = config["n_steps"]
    Cs = config["Cs"]
    shape = config["shape"]

    tag = f"[{benchmark} SDAA:{device_id}]"
    print(
        f"{tag} shape={shape} D={D} nx={nx} ny={ny} nz={nz} "
        f"u_in={u_in} Re={Re} nu={nu:.6e} tau={tau:.6f} "
        f"collision={config['collision']} Cs={Cs} n_steps={n_steps}",
        flush=True,
    )

    t0 = time.time()

    # ---- Build geometry ----
    cx = nx * 0.25
    cy = ny * 0.5
    cz = nz * 0.5

    if shape == "cylinder":
        solid = build_cylinder_mask(nx, ny, nz, cx, cy, R, device)
        # Universal flow: get_near_wall_3d for all geometries
        near = get_near_wall_3d(solid)
        mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, R, axis="z")
        # Frontal area for 2D extruded cylinder: D * nz
        A_frontal = D * nz
    else:
        solid = build_sphere_mask(nx, ny, nz, cx, cy, cz, R, device)
        # Universal flow: get_near_wall_3d for all geometries
        near = get_near_wall_3d(solid)
        mesh = SurfaceMesh.from_sphere(solid, near, cx, cy, cz, R)
        # Frontal area for sphere: pi * R^2
        A_frontal = math.pi * R ** 2

    dpS = 0.5 * u_in ** 2 * A_frontal
    n_solid = int(solid.sum().item())
    n_near = int(near.sum().item())
    print(f"{tag} solid={n_solid} near={n_near} dpS={dpS:.6e}", flush=True)

    # ---- Initialise flow field ----
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s) mass={initial_mass}", flush=True)

    # ---- Collision function ----
    if config["collision"] == "mrt":
        collide_fn = collide_mrt3d
        collide_kwargs = {}
    else:
        collide_fn = collide_smagorinsky_mrt3d
        collide_kwargs = {"C_s": Cs}

    # ---- History accumulators ----
    cd_p_hist = []
    cd_f_hist = []
    cd_tot_hist = []
    cl_hist = []
    warmup = n_steps // 5  # 20% warmup

    # ---- Main loop using lbm_step_correct (common interface) ----
    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f,
            collide_fn,
            tau,
            solid,
            u_in,
            far_field_bc_3d,
            correct_mass_fn=correct_mass3d,
            target_mass=initial_mass,
            step=step,
            mass_interval=200,
            **collide_kwargs,
        )

        # Drag computation via common interface (drag_pressure + drag_friction)
        # Use p0_method='far_field' for stable free-stream pressure reference
        fx_p, fy_p, fz_p = drag_pressure_integration(
            f, mesh, dpS, solid=solid, p0_method='far_field'
        )
        fx_f, fy_f, fz_f = drag_friction_integration(f, mesh, dpS, nu)
        cd_p = float(fx_p)
        cd_f = float(fx_f)
        cd_tot = cd_p + cd_f
        cl = float(fy_p + fy_f)

        # Check divergence
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        # Record post-warmup
        if step > warmup:
            if math.isfinite(cd_tot):
                cd_p_hist.append(cd_p)
                cd_f_hist.append(cd_f)
                cd_tot_hist.append(cd_tot)
                cl_hist.append(cl)

        if step % 500 == 0 or step == n_steps:
            n_avg = min(500, len(cd_tot_hist))
            cd_avg = sum(cd_tot_hist[-n_avg:]) / max(n_avg, 1)
            cl_avg = sum(cl_hist[-n_avg:]) / max(n_avg, 1)
            elapsed = time.time() - t0
            print(
                f"{tag} step={step}/{n_steps} Cd={cd_avg:.4f} Cl={cl_avg:.6f} "
                f"({elapsed:.0f}s, {elapsed/step:.3f}s/step)",
                flush=True,
            )

    elapsed = time.time() - t0

    # ---- Final statistics ----
    n_final = min(max(n_steps // 10, 200), len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / max(n_final, 1)
    cd_f_final = sum(cd_f_hist[-n_final:]) / max(n_final, 1)
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / max(n_final, 1)
    cl_final = sum(cl_hist[-n_final:]) / max(n_final, 1)

    # Cl RMS
    if len(cl_hist) > 10:
        cl_mean = sum(cl_hist) / len(cl_hist)
        cl_rms = math.sqrt(sum((c - cl_mean) ** 2 for c in cl_hist) / len(cl_hist))
    else:
        cl_rms = 0.0

    # ---- Strouhal number via detect_strouhal (common interface) ----
    st_result = detect_strouhal(
        cl_hist,
        sample_rate=1.0,
        u_ref=u_in,
        length_ref=D,
        min_cycles=2,
        method='auto',
    )
    st = float(st_result) if st_result is not None else None

    # ---- Separation angle (cylinder only) ----
    sep_angle = float("nan")
    sep_data = []
    if shape == "cylinder":
        rho_f, ux_f, uy_f, uz_f = macroscopic3d(f)
        mid_z = nz // 2
        sep_angle, sep_data = measure_separation_angle(
            ux_f, uy_f, near, cx, cy, R, mid_z
        )

    # ---- Recirculation length (cylinder Re=3900) ----
    recirc_len = float("nan")
    if shape == "cylinder" and "recirc_ref" in config:
        rho_f, ux_f, uy_f, uz_f = macroscopic3d(f)
        mid_z = nz // 2
        recirc_len = measure_recirculation_length(
            ux_f, solid, cx, cy, R, mid_z, u_in
        )

    # ---- Reference comparison ----
    cd_ref = config["Cd_ref"]
    cd_err = abs(cd_tot_final - cd_ref) / cd_ref * 100 if cd_ref > 0 else float("nan")

    sep_ref = config.get("sep_ref")
    sep_err = (
        abs(sep_angle - sep_ref) / sep_ref * 100
        if sep_ref is not None and math.isfinite(sep_angle)
        else float("nan")
    )

    recirc_ref = config.get("recirc_ref")
    recirc_err = (
        abs(recirc_len - recirc_ref) / recirc_ref * 100
        if recirc_ref is not None and math.isfinite(recirc_len)
        else float("nan")
    )

    # ---- Print final results ----
    print(f"\n{tag} === FINAL RESULTS ===", flush=True)
    print(f"{tag} Cd_p   = {cd_p_final:.4f}", flush=True)
    print(f"{tag} Cd_f   = {cd_f_final:.4f}", flush=True)
    print(f"{tag} Cd_tot = {cd_tot_final:.4f}  (ref={cd_ref}, err={cd_err:.1f}%)", flush=True)
    print(f"{tag} Cl     = {cl_final:.6f}", flush=True)
    print(f"{tag} Cl_rms = {cl_rms:.6f}", flush=True)
    print(f"{tag} St     = {st:.4f}" if st is not None else f"{tag} St     = None", flush=True)
    if not math.isnan(sep_angle):
        print(
            f"{tag} Sep_angle = {sep_angle:.1f}°  "
            f"(ref={sep_ref}°, err={sep_err:.1f}%)",
            flush=True,
        )
    if not math.isnan(recirc_len):
        print(
            f"{tag} Recirc_len = {recirc_len:.2f}D  "
            f"(ref={recirc_ref}D, err={recirc_err:.1f}%)",
            flush=True,
        )
    print(f"{tag} time = {elapsed:.0f}s ({elapsed/n_steps:.3f}s/step)", flush=True)

    # ---- Save results (all values converted to Python floats) ----
    result = {
        "case": benchmark,
        "device": f"sdaa:{device_id}",
        "shape": shape,
        "lattice": "D3Q19",
        "collision": config["collision"],
        "Cs": float(Cs),
        "boundary": "halfway_BB(f_pre)+farfield",
        "grid": f"{nx}x{ny}x{nz}",
        "D": float(D),
        "u_in": float(u_in),
        "Re": float(Re),
        "nu": float(nu),
        "tau": float(tau),
        "n_steps": int(n_steps),
        "warmup": int(warmup),
        "n_solid": int(n_solid),
        "n_near": int(n_near),
        "Cd_pressure": float(cd_p_final),
        "Cd_friction": float(cd_f_final),
        "Cd_total": float(cd_tot_final),
        "Cd_ref": float(cd_ref),
        "Cd_err_pct": float(cd_err),
        "Cl": float(cl_final),
        "Cl_rms": float(cl_rms),
        "St": st,
        "separation_angle_deg": float(sep_angle) if math.isfinite(sep_angle) else None,
        "separation_angle_ref": float(sep_ref) if sep_ref is not None else None,
        "separation_angle_err_pct": float(sep_err) if math.isfinite(sep_err) else None,
        "recirculation_length_D": float(recirc_len) if math.isfinite(recirc_len) else None,
        "recirculation_length_ref": float(recirc_ref) if recirc_ref is not None else None,
        "recirculation_length_err_pct": float(recirc_err) if math.isfinite(recirc_err) else None,
        "ref_name": config["ref_name"],
        "n_samples": int(len(cd_tot_hist)),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": float(elapsed),
        "time_per_step_s": float(elapsed / n_steps),
    }

    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} results saved to {output_path}", flush=True)
    return result


def main():
    if len(sys.argv) < 4:
        print("Usage: python new_benchmarks_worker.py <benchmark> <device_id> <output_path>")
        print("  benchmark: cyl_re40 | cyl_re3900 | sph_re300 | sph_re10000")
        sys.exit(1)

    benchmark = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]
    run_benchmark(benchmark, device_id, output_path)


if __name__ == "__main__":
    main()
