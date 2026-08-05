#!/usr/bin/env python3
"""IBM common module test worker — SDAA cards 8-11.

Tests the Immersed Boundary Method (IBM) as a common module on stationary,
moving, and rotating bodies.  Uses ONLY the common interface modules:

  - Geometry:  generate_cylinder_markers / generate_sphere_markers
  - Force:     compute_ibm_drag_from_markers (IBM momentum exchange)
               + drag_pressure_integration + drag_friction_integration (BB-style)
  - Main loop: ibm_step_correct (IBM force addition)
  - St:        detect_strouhal

Benchmarks (one per SDAA card):
  8. cyl_stationary:  Cylinder Re=200, D=48, stationary IBM
                       Compare Cd with BB result (7.6% error target)
  9. cyl_oscillating: Oscillating cylinder, A=0.1D, f=0.2 (VIV)
                       Measure Cd, Cl oscillation amplitude
  10. cyl_rotating:   Rotating cylinder, omega=0.1 (Magnus effect)
                       Compare Cl with analytical 2*pi*omega*D/u
  11. sph_moving:     Sphere Re=100, rising velocity v_rise=0.01
                       Measure drag during ascent

Usage:
  python _ibm_common_worker.py <benchmark> <device_id> [output_path]
"""
from __future__ import annotations

import functools
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

# ---- Common interface imports (ONLY these modules) ----
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_mrt3d, correct_mass3d
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.ibm_common import (
    ibm_step_correct,
    generate_cylinder_markers,
    generate_sphere_markers,
    update_moving_markers,
    update_rotating_markers,
    compute_ibm_drag_from_markers,
)
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.postprocess import detect_strouhal


# ------------------------------------------------------------------ #
#  Geometry builders (mask construction — for drag computation)      #
# ------------------------------------------------------------------ #

def build_cylinder_solid(nx, ny, nz, cx, cy, radius, device):
    """Boolean solid mask for a cylinder extruded along the z-axis."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    return circle.unsqueeze(0).expand(nz, ny, nx).clone()


def build_sphere_solid(nx, ny, nz, cx, cy, cz, R, device):
    """Boolean solid mask for a sphere."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) < R ** 2


# ------------------------------------------------------------------ #
#  Benchmark configurations                                          #
# ------------------------------------------------------------------ #

BENCHMARKS = {
    "cyl_stationary": {
        "shape": "cylinder",
        "Re": 200,
        "D": 48,
        "nx": 640, "ny": 256, "nz": 4,
        "u_in": 0.1,
        "collision": "mrt_smag",
        "Cs": 0.1,
        "n_steps": 8000,
        "avg_window": 2000,
        "Cd_ref": 1.33,  # Re=200 cylinder Cd ~1.33 (experimental)
        "Cd_bb": 1.43,  # BB result from common interface (7.6% high)
        "note": "Stationary cylinder, IBM replaces BB. Compare Cd with BB.",
        "bc_config": {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]},
        "n_theta": 48,
    },
    "cyl_oscillating": {
        "shape": "cylinder",
        "Re": 200,
        "D": 48,
        "nx": 640, "ny": 256, "nz": 4,
        "u_in": 0.1,
        "collision": "mrt_smag",
        "Cs": 0.1,
        "n_steps": 8000,
        "avg_window": 2000,
        # Non-dimensional: A_vel/u_in = 0.1, St = f*D/u_in = 0.2
        "A_vel": 0.01,     # velocity amplitude = 0.1 * u_in
        "f_reduced": 0.2,  # St = f*D/U = 0.2 → f = 0.2*U/D
        "note": "Oscillating cylinder (VIV). u_solid = A_vel*sin(2*pi*f*t).",
        "bc_config": {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]},
        "n_theta": 48,
    },
    "cyl_rotating": {
        "shape": "cylinder",
        "Re": 200,
        "D": 48,
        "nx": 640, "ny": 256, "nz": 4,
        "u_in": 0.1,
        "collision": "mrt_smag",
        "Cs": 0.1,
        "n_steps": 8000,
        "avg_window": 2000,
        # Non-dimensional: alpha = omega*R/u_in = 0.5 (surface vel = 0.5*U)
        "alpha": 0.5,      # rotation rate ratio
        "note": "Rotating cylinder (Magnus). Cl_analytical = 2*pi*alpha.",
        "bc_config": {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]},
        "n_theta": 48,
    },
    "sph_moving": {
        "shape": "sphere",
        "Re": 100,
        "D": 40,
        "nx": 160, "ny": 160, "nz": 160,
        "u_in": 0.08,
        "collision": "mrt_smag",
        "Cs": 0.05,
        "n_steps": 3000,
        "avg_window": 1000,
        "v_rise": 0.01,  # rising velocity
        "Cd_ref": 1.09,  # Sphere Re=100 Cd ~1.09
        "note": "Moving sphere, rising at v_rise=0.01. IBM handles 3D moving body.",
        "bc_config": {"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []},
        "n_theta": 32,
        "n_phi": 16,
    },
}


# ------------------------------------------------------------------ #
#  Target velocity functions for different body motions              #
# ------------------------------------------------------------------ #

def make_stationary_target(n_markers, device):
    """Stationary body: u_target = 0 everywhere."""
    zero = torch.zeros(n_markers, dtype=torch.float32, device=device)
    def fn(step):
        return zero.clone(), zero.clone(), zero.clone()
    return fn


def make_oscillating_target(n_markers, device, A_vel, f_reduced, D, u_in):
    """Oscillating cylinder: u_solid = A_vel*sin(2*pi*f*t) in y-direction.
    
    A_vel = velocity amplitude (lattice units)
    f_reduced = St = f*D/U → f = f_reduced*U/D (per lattice step)
    """
    f_lattice = f_reduced * u_in / D
    def fn(step):
        t = float(step)
        u_y = A_vel * math.sin(2.0 * math.pi * f_lattice * t)
        ux = torch.zeros(n_markers, dtype=torch.float32, device=device)
        uy = torch.full((n_markers,), u_y, dtype=torch.float32, device=device)
        uz = torch.zeros(n_markers, dtype=torch.float32, device=device)
        return ux, uy, uz
    return fn


def make_rotating_target(markers_x, markers_y, markers_z, device, alpha, cx, cy, cz, u_in, R, axis='x'):
    """Rotating body: u = omega × r (rigid-body rotation about axis).
    
    alpha = omega*R/U (surface velocity ratio)
    omega = alpha*U/R (lattice angular velocity)
    """
    omega = alpha * u_in / R
    mx = markers_x.to(device)
    my = markers_y.to(device)
    mz = markers_z.to(device)
    def fn(step):
        dx = mx - cx
        dy = my - cy
        dz = mz - cz
        if axis == 'x':
            ux = torch.zeros_like(mx)
            uy = -omega * dz
            uz = omega * dy
        elif axis == 'z':
            ux = -omega * dy
            uy = omega * dx
            uz = torch.zeros_like(mx)
        else:
            ux = torch.zeros_like(mx)
            uy = torch.zeros_like(mx)
            uz = torch.zeros_like(mx)
        return ux, uy, uz
    return fn


def make_moving_sphere_target(n_markers, device, v_rise):
    """Moving sphere: uniform rising velocity in x-direction."""
    def fn(step):
        ux = torch.full((n_markers,), v_rise, dtype=torch.float32, device=device)
        uy = torch.zeros(n_markers, dtype=torch.float32, device=device)
        uz = torch.zeros(n_markers, dtype=torch.float32, device=device)
        return ux, uy, uz
    return fn


# ------------------------------------------------------------------ #
#  Main benchmark runner                                             #
# ------------------------------------------------------------------ #

def run_benchmark(bench_name: str, device_id: int, output_path: str | None = None):
    """Run a single IBM benchmark on the specified SDAA card."""
    cfg = BENCHMARKS[bench_name]
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    tag = f"[{bench_name} SDAA:{device_id}]"

    # --- Physical parameters ---
    D = cfg["D"]
    R = D / 2.0
    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]
    u_in = cfg["u_in"]
    Re = cfg["Re"]
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = cfg["n_steps"]
    avg_window = cfg["avg_window"]

    # Frontal area → dynamic-pressure × area (dpS)
    if cfg["shape"] == "cylinder":
        A_frontal = D * nz
    else:
        A_frontal = math.pi * R ** 2
    dpS = 0.5 * u_in ** 2 * A_frontal

    # --- Collision operator ---
    if cfg["collision"] == "mrt":
        collide_fn = collide_mrt3d
        collide_kwargs: dict = {}
    else:
        collide_fn = collide_smagorinsky_mrt3d
        collide_kwargs = {"C_s": cfg["Cs"]}

    # --- Far-field BC wrapper ---
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=cfg["bc_config"])

    print(
        f"{tag} Re={Re} D={D} grid={nx}x{ny}x{nz} u_in={u_in} "
        f"nu={nu:.6e} tau={tau:.6f} dpS={dpS:.6e}",
        flush=True,
    )
    print(
        f"{tag} collision={cfg['collision']} Cs={cfg.get('Cs')} "
        f"n_steps={n_steps} avg_window={avg_window}",
        flush=True,
    )

    t0 = time.time()

    # === Step 1: Build geometry (solid mask for drag computation) ===
    if cfg["shape"] == "cylinder":
        cx = nx * 0.25
        cy = ny * 0.5
        cz = nz * 0.5
        solid = build_cylinder_solid(nx, ny, nz, cx, cy, R, device)
    else:
        cx = nx * 0.25
        cy = ny * 0.5
        cz = nz * 0.5
        solid = build_sphere_solid(nx, ny, nz, cx, cy, cz, R, device)

    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    # === Step 2: Near-wall mask + SurfaceMesh (for pressure/friction drag) ===
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    if cfg["shape"] == "cylinder":
        mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, R, axis="z")
    else:
        mesh = SurfaceMesh.from_sphere(solid, near, cx, cy, cz, R)
    print(f"{tag} SurfaceMesh.from_{cfg['shape']} built", flush=True)

    # === Step 3: Generate IBM markers (analytical surface points) ===
    if cfg["shape"] == "cylinder":
        marker_x0, marker_y0, marker_z0 = generate_cylinder_markers(
            cx, cy, cz, R, nz, axis="z", n_theta=cfg.get("n_theta", 48),
            device=device,
        )
    else:
        marker_x0, marker_y0, marker_z0 = generate_sphere_markers(
            cx, cy, cz, R, n_theta=cfg.get("n_theta", 32),
            n_phi=cfg.get("n_phi", 16), device=device,
        )
    n_markers = marker_x0.shape[0]
    print(f"{tag} IBM markers: {n_markers}", flush=True)

    # === Step 4: Set up target velocity function ===
    if bench_name == "cyl_stationary":
        u_target_fn = make_stationary_target(n_markers, device)
        markers = (marker_x0, marker_y0, marker_z0)
    elif bench_name == "cyl_oscillating":
        A_vel = cfg["A_vel"]
        f_reduced = cfg["f_reduced"]
        u_target_fn = make_oscillating_target(n_markers, device, A_vel, f_reduced, D, u_in)
        markers = (marker_x0, marker_y0, marker_z0)
        f_lattice = f_reduced * u_in / D
        print(f"{tag} oscillating: A_vel={A_vel} St={f_reduced} f_lattice={f_lattice:.6e}", flush=True)
    elif bench_name == "cyl_rotating":
        alpha = cfg["alpha"]
        omega = alpha * u_in / R
        u_target_fn = make_rotating_target(
            marker_x0, marker_y0, marker_z0, device, alpha, cx, cy, cz, u_in, R, axis='x',
        )
        markers = (marker_x0, marker_y0, marker_z0)
        Re_omega = omega * D**2 / nu
        Cl_ref = 2 * math.pi * alpha
        print(f"{tag} rotating: alpha={alpha} omega={omega:.6e} Re_omega={Re_omega:.1f} Cl_ref={Cl_ref:.4f}", flush=True)
    elif bench_name == "sph_moving":
        v_rise = cfg["v_rise"]
        u_target_fn = make_moving_sphere_target(n_markers, device, v_rise)
        markers = (marker_x0, marker_y0, marker_z0)
        print(f"{tag} moving sphere: v_rise={v_rise}", flush=True)
    else:
        u_target_fn = make_stationary_target(n_markers, device)
        markers = (marker_x0, marker_y0, marker_z0)

    # --- Initialise flow field ---
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s) mass={initial_mass}", flush=True)

    # --- History accumulators ---
    cd_ibm_hist: list[float] = []     # IBM momentum-exchange drag
    cd_p_hist: list[float] = []       # pressure drag (BB-style)
    cd_f_hist: list[float] = []       # friction drag (BB-style)
    cd_tot_hist: list[float] = []     # total (pressure + friction)
    cl_hist: list[float] = []
    cl_ibm_hist: list[float] = []

    # === Step 5: Main loop via ibm_step_correct ===
    for step in range(1, n_steps + 1):
        f, (mfx, mfy, mfz) = ibm_step_correct(
            f,
            collide_fn,
            tau,
            solid,
            u_in,
            far_field_fn,
            markers,
            u_target_fn,
            lattice="D3Q19",
            kernel="hat",
            correct_mass_fn=correct_mass3d,
            target_mass=initial_mass,
            step=step,
            mass_interval=200,
            **collide_kwargs,
        )

        # === Step 6: Force computation (both IBM and BB-style) ===
        # IBM drag (momentum exchange from marker forces)
        cd_ibm, cl_ibm, _ = compute_ibm_drag_from_markers(mfx, mfy, mfz, dpS)

        # BB-style pressure + friction drag (for comparison)
        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS, extrap="none")
        fx_f, fy_f, _ = drag_friction_integration(
            f, mesh, dpS, nu, q_wall=None, formula="standard"
        )

        cd_p = fx_p
        cd_f = fx_f
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f

        cd_ibm_hist.append(cd_ibm)
        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        cl_hist.append(cl)
        cl_ibm_hist.append(cl_ibm)

        # Divergence guard
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        # Progress
        if step % 500 == 0:
            n_avg = min(500, len(cd_tot_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            cd_ibm_avg = sum(cd_ibm_hist[-n_avg:]) / n_avg
            elapsed = time.time() - t0
            print(
                f"{tag} step={step}/{n_steps} "
                f"Cd_ibm={cd_ibm_avg:.4f} "
                f"Cd_p={cd_p_avg:.4f} Cd_f={cd_f_avg:.4f} "
                f"Cd_tot={cd_tot_avg:.4f} ({elapsed:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0

    # --- Final time-averaged forces ---
    n_final = min(avg_window, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_final = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final
    cl_final = sum(cl_hist[-n_final:]) / n_final
    cd_ibm_final = sum(cd_ibm_hist[-n_final:]) / n_final
    cl_ibm_final = sum(cl_ibm_hist[-n_final:]) / n_final

    # Cl oscillation amplitude (for oscillating/rotating cases)
    if len(cl_hist) > 100:
        cl_max = max(cl_hist[-n_final:])
        cl_min = min(cl_hist[-n_final:])
        cl_amp = (cl_max - cl_min) / 2.0
    else:
        cl_amp = 0.0

    # === Step 7: Strouhal number ===
    st = detect_strouhal(
        cl_hist, sample_rate=1.0, u_ref=u_in, length_ref=D, min_cycles=3
    )

    # --- Comparison with reference ---
    cd_ref = cfg.get("Cd_ref", 0.0)
    cd_bb = cfg.get("Cd_bb", 0.0)
    err_pct = abs(cd_tot_final - cd_ref) / cd_ref * 100 if cd_ref > 0 else float("nan")
    err_vs_bb = abs(cd_tot_final - cd_bb) / cd_bb * 100 if cd_bb > 0 else float("nan")

    # Rotating: compare Cl with analytical
    if bench_name == "cyl_rotating":
        alpha = cfg["alpha"]
        cl_ref = 2 * math.pi * alpha
    else:
        cl_ref = cfg.get("Cl_ref", 0.0)
    cl_err = abs(cl_final - cl_ref) / cl_ref * 100 if cl_ref > 0 else float("nan")

    result = {
        "case": bench_name,
        "device": f"sdaa:{device_id}",
        "shape": cfg["shape"],
        "Re": Re,
        "D": D,
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "collision": cfg["collision"],
        "Cs": cfg.get("Cs"),
        "n_steps": n_steps,
        "n_solid": n_solid,
        "n_near": n_near,
        "n_markers": n_markers,
        "Cd_ibm": cd_ibm_final,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl": cl_final,
        "Cl_ibm": cl_ibm_final,
        "Cl_amplitude": cl_amp,
        "St": st,
        "Cd_ref": cd_ref,
        "Cd_bb": cd_bb,
        "error_pct": err_pct,
        "error_vs_bb_pct": err_vs_bb,
        "Cl_ref": cl_ref,
        "Cl_error_pct": cl_err,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
        "note": cfg.get("note", ""),
        "modules_used": [
            "ibm_common.generate_cylinder_markers",
            "ibm_common.ibm_step_correct",
            "ibm_common.compute_ibm_drag_from_markers",
            "drag_pressure.get_near_wall_3d",
            "drag_pressure.SurfaceMesh.from_" + cfg["shape"],
            "drag_pressure.drag_pressure_integration",
            "drag_pressure.drag_friction_integration",
            "boundaries3d.far_field_bc_3d",
            "postprocess.detect_strouhal",
        ],
    }

    print(
        f"{tag} DONE  Cd_ibm={cd_ibm_final:.4f}  Cd_p={cd_p_final:.4f}  "
        f"Cd_f={cd_f_final:.4f}  Cd_tot={cd_tot_final:.4f}  "
        f"Cl={cl_final:.4f}  St={st}  "
        f"(ref={cd_ref:.2f}, bb={cd_bb:.2f})  "
        f"err={err_pct:.1f}%  vs_bb={err_vs_bb:.1f}%  "
        f"time={elapsed:.0f}s",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results → {output_path}", flush=True)

    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: python _ibm_common_worker.py <benchmark> <device_id> [output_path]")
        print(f"Benchmarks: {', '.join(BENCHMARKS.keys())}")
        sys.exit(1)

    bench_name = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    if bench_name not in BENCHMARKS:
        print(f"Unknown benchmark: {bench_name}")
        print(f"Available: {', '.join(BENCHMARKS.keys())}")
        sys.exit(1)

    run_benchmark(bench_name, device_id, output_path)


if __name__ == "__main__":
    main()
