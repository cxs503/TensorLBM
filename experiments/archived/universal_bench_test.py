#!/usr/bin/env python3
"""Universal benchmark test: D3Q19 MRT+Smag Cs=0.05 + wall_fn + far_field.

Tests:
  1. Sphere drag at Re=100, 1000, 10000
  2. Flat plate Cf at Re=2e6
  3. Ahmed body at slant=25° (adapted to D3Q19)

Uses wall_function_3d for friction drag. Computes pressure drag separately
with correct sign convention (p_front - p_back for drag in flow direction).

All results saved to /tmp/universal_bench_results.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.boundaries3d import sphere_mask, far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.wall_model import wall_function_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d

# ─── Configuration ───────────────────────────────────────────────────────────
CS = 0.05

# ─── Helpers ─────────────────────────────────────────────────────────────────

def compute_pressure_drag_x(f, solid):
    """Compute pressure drag in +x direction with correct sign convention.

    drag_x = Σ_F p_F * (n_x from fluid to solid)
    For solid at F+1: outward normal from solid is +x → n_x = +1 → +p_F
    For solid at F-1: outward normal from solid is -x → n_x = -1 → -p_F

    Returns drag_force_x (positive = force on solid in +x, i.e., drag opposing flow).
    """
    rho, _, _, _ = macroscopic3d(f)
    p = (rho - 1.0) / 3.0  # gauge pressure
    fluid = ~solid

    # solid at +x neighbor (to the right of fluid cell)
    solid_px = torch.roll(solid, -1, dims=2)
    # solid at -x neighbor (to the left of fluid cell)
    solid_mx = torch.roll(solid, 1, dims=2)

    # Drag = +p where solid is to the right (front face pushes body +x)
    #       -p where solid is to the left (rear face pushes body -x)
    drag = float((p * (solid_px.to(f.dtype) - solid_mx.to(f.dtype)) * fluid.to(f.dtype)).sum().item())
    return drag


# ─── Sphere Drag ─────────────────────────────────────────────────────────────

def run_sphere_drag(re: float, nx=120, ny=60, nz=60, radius=12.0,
                    u_in=0.06, n_steps=2000, warmup=500, device="sdaa:8"):
    """Run D3Q19 MRT+Smag Cs=0.05 sphere drag test."""
    dev = torch.device(device)
    nu = u_in * 2.0 * radius / re
    tau = 3.0 * nu + 0.5

    cx, cy, cz = nx * 0.25, ny * 0.5, nz * 0.5
    solid = sphere_mask(nx, ny, nz, cx, cy, cz, radius, device=dev)

    rho0 = torch.ones((nz, ny, nx), device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=dev)
    initial_mass = float(rho0.sum().item())

    # Reference area: cross-sectional area = pi * r^2
    ref_area = math.pi * radius ** 2
    dynamic_press = 0.5 * 1.0 * u_in ** 2 * ref_area

    cd_ref_map = {100: 1.09, 1000: 0.47, 10000: 0.40}
    cd_ref = cd_ref_map.get(int(re), None)

    samples_fric = []
    samples_pres = []
    step_log = []

    print(f"\n{'='*60}")
    print(f"Sphere Re={re}  grid={nx}x{ny}x{nz}  radius={radius}  Cs={CS}")
    print(f"nu={nu:.6e}  tau={tau:.5f}  ref_area={ref_area:.1f}  Cd_ref={cd_ref}")
    print(f"{'='*60}")

    t0 = time.time()
    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=CS)
        f = stream3d(f)
        # Use wall_function_3d for log-law body force (friction)
        f, drag_fric_wf, _ = wall_function_3d(f, solid, nu, y_val=0.5)
        # Compute pressure drag with correct sign convention
        drag_pres = compute_pressure_drag_x(f, solid)
        f = far_field_bc_3d(f, u_in=u_in)
        f = bounce_back_cells_3d(f, solid)  # restore solid after far_field
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > warmup and math.isfinite(drag_fric_wf):
            samples_fric.append(drag_fric_wf)
            samples_pres.append(drag_pres)

        if step % 500 == 0 or step == n_steps:
            cf = sum(samples_fric) / max(len(samples_fric), 1) / dynamic_press if samples_fric else 0
            cp = sum(samples_pres) / max(len(samples_pres), 1) / dynamic_press if samples_pres else 0
            cd = cf + cp
            cd_err = (abs(cd - cd_ref) / cd_ref * 100) if cd_ref else None
            print(f"  step {step:5d}: Cf={cf:.5f} Cp={cp:.5f} Cd={cd:.5f} "
                  f"(ref={cd_ref}, err={cd_err:.1f}%)" if cd_err else f"  step {step:5d}: Cd={cd:.5f}")
            step_log.append({"step": step, "Cf": round(cf, 6), "Cp": round(cp, 6),
                             "Cd": round(cd, 6)})

    dt = time.time() - t0
    cf = sum(samples_fric) / max(len(samples_fric), 1) / dynamic_press if samples_fric else 0
    cp = sum(samples_pres) / max(len(samples_pres), 1) / dynamic_press if samples_pres else 0
    cd = cf + cp
    cd_err = (abs(cd - cd_ref) / cd_ref * 100) if cd_ref else None

    result = {
        "benchmark": f"sphere_Re{int(re)}",
        "grid": f"{nx}x{ny}x{nz}",
        "Cs": CS,
        "Re": re,
        "radius": radius,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "warmup": warmup,
        "n_samples": len(samples_fric),
        "Cd_friction": round(cf, 6),
        "Cd_pressure": round(cp, 6),
        "Cd_total": round(cd, 6),
        "Cd_ref": cd_ref,
        "Cd_err_pct": round(cd_err, 2) if cd_err else None,
        "wall_time_s": round(dt, 1),
        "step_log": step_log,
    }
    print(f"Final Sphere Re={re}: Cd={cd:.5f}  ref={cd_ref}  err={cd_err:.1f}%  time={dt:.0f}s")
    return result


# ─── Flat Plate Cf ───────────────────────────────────────────────────────────

def run_flat_plate(re_L=2e6, L=160.0, nx=200, ny=40, nz=40, u_in=0.06,
                   n_steps=2000, warmup=500, device="sdaa:8"):
    """Run D3Q19 MRT+Smag Cs=0.05 flat plate friction test."""
    dev = torch.device(device)
    nu = u_in * L / re_L
    tau = 3.0 * nu + 0.5

    # Plate: solid at y=0, from x=4 to end
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid[:, 0, 4:] = True
    plate_area = (nx - 4) * nz
    dyn_p_A = 0.5 * 1.0 * u_in ** 2 * plate_area

    # ITTC-1957: Cf = 0.075/(log10(Re)-2)^2
    cf_ittc = 0.075 / (math.log10(re_L) - 2) ** 2

    rho0 = torch.ones((nz, ny, nx), device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=dev)
    initial_mass = float(rho0.sum().item())

    samples_fric = []
    samples_pres = []
    step_log = []

    print(f"\n{'='*60}")
    print(f"Flat Plate Re_L={re_L:.0e}  grid={nx}x{ny}x{nz}  L={L}  Cs={CS}")
    print(f"nu={nu:.6e}  tau={tau:.5f}  Cf_ITTC={cf_ittc:.5f}")
    print(f"{'='*60}")

    t0 = time.time()
    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=CS)
        f = stream3d(f)
        f, drag_fric, _ = wall_function_3d(f, solid, nu, y_val=0.5)
        drag_pres = compute_pressure_drag_x(f, solid)
        f = far_field_bc_3d(f, u_in=u_in)
        f = bounce_back_cells_3d(f, solid)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > warmup and math.isfinite(drag_fric):
            samples_fric.append(drag_fric)
            samples_pres.append(drag_pres)

        if step % 500 == 0 or step == n_steps:
            cf = sum(samples_fric) / max(len(samples_fric), 1) / dyn_p_A if samples_fric else float('nan')
            cp = sum(samples_pres) / max(len(samples_pres), 1) / dyn_p_A if samples_pres else 0
            cf_err = abs(cf - cf_ittc) / cf_ittc * 100 if samples_fric else None
            ct = cf + cp
            print(f"  step {step:5d}: Cf={cf:.5f} Cp={cp:.5f} Ct={ct:.5f} "
                  f"(ITTC={cf_ittc:.5f}, err={cf_err:.1f}%)" if samples_fric else f"  step {step:5d}")
            step_log.append({"step": step, "Cf": round(cf, 6), "Cp": round(cp, 6),
                             "Ct": round(ct, 6)})

    dt = time.time() - t0
    cf = sum(samples_fric) / max(len(samples_fric), 1) / dyn_p_A if samples_fric else float('nan')
    cp = sum(samples_pres) / max(len(samples_pres), 1) / dyn_p_A if samples_pres else 0
    ct = cf + cp
    cf_err = abs(cf - cf_ittc) / cf_ittc * 100 if samples_fric else None

    result = {
        "benchmark": f"flat_plate_Re{re_L:.0e}",
        "grid": f"{nx}x{ny}x{nz}",
        "Cs": CS,
        "Re_L": re_L,
        "L": L,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "warmup": warmup,
        "n_samples": len(samples_fric),
        "Cf_friction": round(cf, 6),
        "Cp_pressure": round(cp, 6),
        "Ct_total": round(ct, 6),
        "Cf_ref_ITTC": round(cf_ittc, 6),
        "Cf_err_pct": round(cf_err, 2) if cf_err else None,
        "wall_time_s": round(dt, 1),
        "step_log": step_log,
    }
    print(f"Final Flat Plate: Cf={cf:.5f}  ITTC={cf_ittc:.5f}  err={cf_err:.1f}%  time={dt:.0f}s")
    return result


# ─── Ahmed Body ──────────────────────────────────────────────────────────────

def build_ahmed_body(nx, ny, nz, slant_deg=25.0, device='cpu'):
    """Build Ahmed body mask. Body centered in domain."""
    L = int(nx * 0.35)
    W = int(ny * 0.4)
    H = int(nz * 0.35)
    cx, cy = nx * 0.35, ny / 2
    cz = nz * 0.35
    slant_len = int(L * 0.3)
    body_len = L - slant_len
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz), torch.arange(ny), torch.arange(nx), indexing='ij')
    x0 = cx; x1 = cx + body_len; x2 = cx + L
    in_body = (xx >= x0) & (xx < x1) & (yy >= cy - W/2) & (yy < cy + W/2) & (zz >= cz) & (zz < cz + H)
    slant_h = H * 0.6
    slant_t = (xx - x1).clamp(min=0) / max(slant_len, 1)
    slant_z = cz + H - slant_t * (H - slant_h)
    in_slant = (xx >= x1) & (xx < x2) & (yy >= cy - W/2) & (yy < cy + W/2) & (zz >= cz) & (zz < slant_z)
    front_r = W * 0.3
    dx = (xx - x0).clamp(min=0)
    dy = torch.minimum((yy - (cy - W/2)).clamp(min=0), (cy + W/2 - yy).clamp(min=0))
    in_front = (xx < x0) & (dx**2 + dy**2 < front_r**2) & (zz >= cz) & (zz < cz + H)
    solid = in_body | in_slant | in_front
    return solid.to(device)


def run_ahmed_body(slant_deg=25.0, re=1e6, u_in=0.06,
                   nx=320, ny=128, nz=96, n_steps=2000, warmup=500,
                   device="sdaa:8"):
    """Run D3Q19 MRT+Smag Cs=0.05 Ahmed body test."""
    dev = torch.device(device)
    nu = u_in * (nx * 0.35) / re
    tau = 3.0 * nu + 0.5

    solid = build_ahmed_body(nx, ny, nz, slant_deg, device='cpu').to(dev)

    # Frontal area
    W = ny * 0.4; H_ah = nz * 0.35; S = W * H_ah
    dyn_p_S = 0.5 * 1.0 * u_in ** 2 * S

    rho0 = torch.ones((nz, ny, nx), device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=dev)
    initial_mass = float(rho0.sum().item())

    cd_ref = 0.28 if slant_deg < 30 else 0.43

    samples_fric = []
    samples_pres = []
    step_log = []

    print(f"\n{'='*60}")
    print(f"Ahmed Body {slant_deg}°  Re={re:.0e}  grid={nx}x{ny}x{nz}  Cs={CS}")
    print(f"nu={nu:.6e}  tau={tau:.5f}  frontal_area={S:.0f}  Cd_ref={cd_ref}")
    print(f"{'='*60}")

    t0 = time.time()
    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=CS)
        f = stream3d(f)
        f, drag_fric, _ = wall_function_3d(f, solid, nu, y_val=0.5)
        drag_pres = compute_pressure_drag_x(f, solid)
        f = far_field_bc_3d(f, u_in=u_in)
        f = bounce_back_cells_3d(f, solid)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > warmup and math.isfinite(drag_fric):
            samples_fric.append(drag_fric)
            samples_pres.append(drag_pres)

        if step % 500 == 0 or step == n_steps:
            cf = sum(samples_fric) / max(len(samples_fric), 1) / dyn_p_S if samples_fric else 0
            cp = sum(samples_pres) / max(len(samples_pres), 1) / dyn_p_S if samples_pres else 0
            cd = cf + cp
            cd_err = abs(cd - cd_ref) / cd_ref * 100 if cd_ref else None
            print(f"  step {step:5d}: Cf={cf:.4f} Cp={cp:.4f} Cd={cd:.4f} "
                  f"(ref={cd_ref}, err={cd_err:.1f}%)" if cd_err else f"  step {step:5d}: Cd={cd:.4f}")
            step_log.append({"step": step, "Cf": round(cf, 6), "Cp": round(cp, 6),
                             "Cd": round(cd, 6)})

    dt = time.time() - t0
    cf = sum(samples_fric) / max(len(samples_fric), 1) / dyn_p_S if samples_fric else 0
    cp = sum(samples_pres) / max(len(samples_pres), 1) / dyn_p_S if samples_pres else 0
    cd = cf + cp
    cd_err = abs(cd - cd_ref) / cd_ref * 100 if cd_ref else None

    result = {
        "benchmark": f"ahmed_body_{int(slant_deg)}deg",
        "grid": f"{nx}x{ny}x{nz}",
        "Cs": CS,
        "Re": re,
        "slant_deg": slant_deg,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "warmup": warmup,
        "n_samples": len(samples_fric),
        "Cd_friction": round(cf, 6),
        "Cd_pressure": round(cp, 6),
        "Cd_total": round(cd, 6),
        "Cd_ref": cd_ref,
        "Cd_err_pct": round(cd_err, 2) if cd_err else None,
        "wall_time_s": round(dt, 1),
        "step_log": step_log,
    }
    print(f"Final Ahmed: Cd={cd:.4f}  ref={cd_ref}  err={cd_err:.1f}%  time={dt:.0f}s")
    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description="Universal Cs=0.05 benchmark")
    p.add_argument("--device", default="sdaa:8", help="SDAA device")
    p.add_argument("--skip-sphere", action="store_true")
    p.add_argument("--skip-plate", action="store_true")
    p.add_argument("--skip-ahmed", action="store_true")
    p.add_argument("--sphere-res", nargs=3, type=int, default=[120, 60, 60])
    p.add_argument("--plate-res", nargs=3, type=int, default=[200, 40, 40])
    p.add_argument("--ahmed-res", nargs=3, type=int, default=[320, 128, 96])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=500)
    args = p.parse_args()

    device = args.device
    all_results = {
        "config": {
            "lattice": "D3Q19",
            "collision": "MRT+Smagorinsky",
            "Cs": CS,
            "wall_model": "wall_function_3d (log-law body force)",
            "far_field": "far_field_bc_3d",
            "device": device,
            "note": "Pressure drag computed with corrected sign convention (p_front - p_back for drag). "
                    "wall_function_3d friction drag is used as-is.",
        },
        "results": [],
        "summary": {},
    }

    # --- Sphere tests ---
    if not args.skip_sphere:
        nx, ny, nz = args.sphere_res
        for re in [100, 1000, 10000]:
            result = run_sphere_drag(
                re=float(re), nx=nx, ny=ny, nz=nz,
                n_steps=args.steps, warmup=args.warmup, device=device)
            all_results["results"].append(result)

    # --- Flat plate ---
    if not args.skip_plate:
        nx, ny, nz = args.plate_res
        result = run_flat_plate(
            re_L=2e6, L=160.0, nx=nx, ny=ny, nz=nz,
            n_steps=args.steps, warmup=args.warmup, device=device)
        all_results["results"].append(result)

    # --- Ahmed body ---
    if not args.skip_ahmed:
        nx, ny, nz = args.ahmed_res
        result = run_ahmed_body(
            slant_deg=25.0, re=1e6, nx=nx, ny=ny, nz=nz,
            n_steps=args.steps, warmup=args.warmup, device=device)
        all_results["results"].append(result)

    # --- Summary ---
    summary = {}
    for r in all_results["results"]:
        name = r["benchmark"]
        if "Cd_total" in r:
            summary[name] = {
                "Cd": r["Cd_total"],
                "Cd_ref": r.get("Cd_ref"),
                "err_pct": r.get("Cd_err_pct"),
                "Cf": r.get("Cd_friction"),
                "Cp": r.get("Cd_pressure"),
            }
        elif "Ct_total" in r:
            summary[name] = {
                "Ct": r["Ct_total"],
                "Cf_ref": r.get("Cf_ref_ITTC"),
                "err_pct": r.get("Cf_err_pct"),
            }

    # Answer the key question
    errs = [v.get("err_pct") for v in summary.values() if v.get("err_pct") is not None]
    if errs:
        avg_err = sum(errs) / len(errs)
        max_err = max(errs)
        universal = avg_err < 20 and max_err < 50
        summary["universal_assessment"] = {
            "Cs": CS,
            "avg_error_pct": round(avg_err, 2),
            "max_error_pct": round(max_err, 2),
            "is_universal": universal,
            "verdict": (
                f"Cs={CS} is approximately UNIVERSAL across all benchmarks "
                f"(avg err {avg_err:.1f}%, max {max_err:.1f}%)"
                if universal else
                f"Cs={CS} shows CASE-DEPENDENT variation "
                f"(avg err {avg_err:.1f}%, max {max_err:.1f}%) — needs per-case tuning"
            ),
        }

    all_results["summary"] = summary

    # Save
    out_path = "/tmp/universal_bench_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*60}")
    print(f"Results saved to {out_path}")
    print(f"Summary:")
    for k, v in summary.items():
        if k == "universal_assessment":
            print(f"\n  UNIVERSAL ASSESSMENT:")
            print(f"    {v['verdict']}")
        else:
            print(f"  {k}: {v}")

    return all_results


if __name__ == "__main__":
    main()
