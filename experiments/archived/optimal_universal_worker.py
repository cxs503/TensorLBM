#!/usr/bin/env python3
"""Universal benchmark worker — optimal settings on three canonical cases.

Optimal settings (determined from prior sweeps):
  - Collision: Cumulant (collide_cumulant_d3q19) — Galilean-invariant,
    numerically stable, no LES constant to tune.
  - Normal: from_gradient (analytical where available, gradient otherwise).
  - Pressure extrapolation: quadratic (2nd-order wall-pressure extrap).
  - Domain: large (low blockage).
  - Steps: 10000.

Benchmarks:
  1. NACA 0012 Re=1000  — chord=100, nx=1200, ny=400, nz=4 (12L domain)
     u_in=0.05, tau=0.515, ref Cd≈0.05
  2. Square prism Re=1000 — D=24, nx=400, ny=160, nz=4 (16D domain)
     u_in=0.08, tau=0.5058, ref Cd=2.10, St=0.14
  3. Backward-facing step Re=1000 — H=10, nx=400, ny=100, nz=4
     u_in=0.05, tau=0.5015, ref x_r/H=6.0

Usage:
  PYTHONPATH=src python optimal_universal_worker.py <device_id> <benchmark> <output_json>
  benchmark: naca | square_prism | backward_facing_step
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from tensorlbm.boundaries3d import (
    bounce_back_cells_3d,
    far_field_bc_3d,
    zou_he_inlet_velocity_3d,
)
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
    get_near_wall_2d,
    get_near_wall_3d,
)
from tensorlbm.backward_facing_step import make_bfs_solid_mask

EXTRAP = "quadratic"  # optimal pressure extrapolation


# ---------------------------------------------------------------------------
# NACA 0012
# ---------------------------------------------------------------------------

def build_naca(chord, nx, ny, x_le, y_c, device):
    """Build NACA 0012 solid mask (2D extruded in z, nz=4)."""
    nz = 4
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    for k in range(nz):
        for i in range(nx):
            xc = (i - x_le) / chord
            if 0 <= xc <= 1:
                yt = 0.6 * (
                    0.2969 * math.sqrt(xc)
                    - 0.1260 * xc
                    - 0.3516 * xc ** 2
                    + 0.2843 * xc ** 3
                    - 0.1015 * xc ** 4
                )
                j_lo = max(0, int(y_c - yt * chord))
                j_hi = min(ny - 1, int(y_c + yt * chord))
                solid[k, j_lo : j_hi + 1, i] = True
    return solid


def run_naca(device, output_path, tag):
    """NACA 0012 at Re=1000 with optimal settings (12L domain, cumulant, quadratic extrap)."""
    chord = 100
    nx = 1200  # 12 chord
    ny = 400   # 4 chord
    nz = 4
    u_in = 0.05
    Re = 1000
    nu = u_in * chord / Re  # 0.005
    tau = 3.0 * nu + 0.5    # 0.515
    n_steps = 10000
    ref_cd = 0.05

    x_le = int(nx * 0.25)  # 3 chords from inlet
    y_c = ny // 2

    dpS = 0.5 * u_in ** 2 * chord * nz

    print(
        f"{tag} [NACA0012] chord={chord} nx={nx} ny={ny} nz={nz} "
        f"u_in={u_in} nu={nu:.6f} tau={tau:.6f} "
        f"x_le={x_le} y_c={y_c} dpS={dpS:.6f} extrap={EXTRAP}",
        flush=True,
    )

    t0 = time.time()

    solid = build_naca(chord, nx, ny, x_le, y_c, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} [NACA0012] solid cells: {n_solid}", flush=True)

    near = get_near_wall_2d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} [NACA0012] near-wall cells: {n_near}", flush=True)

    mesh = SurfaceMesh.from_gradient(solid, near)

    nx_n_vals = mesh.nx_n[near]
    ny_n_vals = mesh.ny_n[near]
    print(
        f"{tag} [NACA0012] normal stats: "
        f"nx_n=[{float(nx_n_vals.min()):.3f}, {float(nx_n_vals.max()):.3f}] "
        f"ny_n=[{float(ny_n_vals.min()):.3f}, {float(ny_n_vals.max()):.3f}]",
        flush=True,
    )

    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    im = float(rho0.sum().item())

    print(f"{tag} [NACA0012] init done ({time.time() - t0:.1f}s), initial_mass={im}", flush=True)

    cd_p_hist = []
    cd_f_hist = []
    cd_tot_hist = []
    cl_hist = []
    diverged = False
    last_step = 0

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_cumulant_d3q19(f, tau=tau)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid)
        f = stream3d(f)
        f = far_field_bc_3d(f, u_in)
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS, extrap=EXTRAP)
        fx_f, fy_f, _ = drag_friction_integration(f, mesh, dpS, nu)

        cd_p = fx_p
        cd_f = fx_f
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f

        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        cl_hist.append(cl)

        if not torch.isfinite(f).all():
            print(f"{tag} [NACA0012] DIVERGED at step {step}", flush=True)
            diverged = True
            last_step = step
            break
        last_step = step

        if step % 500 == 0:
            n_avg = min(500, len(cd_tot_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            cl_avg = sum(cl_hist[-n_avg:]) / n_avg
            elapsed = time.time() - t0
            print(
                f"{tag} [NACA0012] step={step} Cd_p={cd_p_avg:.4f} Cd_f={cd_f_avg:.4f} "
                f"Cd_tot={cd_tot_avg:.4f} Cl={cl_avg:.6f} ({elapsed:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0

    n_final = min(1000, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_final = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final
    cl_final = sum(cl_hist[-n_final:]) / n_final

    err_pct = abs(cd_tot_final - ref_cd) / ref_cd * 100

    result = {
        "benchmark": "naca0012_optimal",
        "device": str(device),
        "Re": Re,
        "chord": chord,
        "grid": f"{nx}x{ny}x{nz}",
        "domain_ratio": f"{nx/chord:.0f}L",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "collision": "cumulant_d3q19",
        "normal_method": "from_gradient",
        "extrap": EXTRAP,
        "n_steps": n_steps,
        "n_solid": n_solid,
        "n_near": n_near,
        "dpS": dpS,
        "x_le": x_le,
        "y_c": y_c,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl": cl_final,
        "Cd_ref": ref_cd,
        "ref_name": "laminar Re=1000",
        "error_pct": err_pct,
        "previous_result": {
            "config": "chord100, 6L domain, MRT+Smag(Cs=0.05)",
            "Cd_tot": 0.0600,
            "err_pct": 20.1,
        },
        "finite": not diverged,
        "diverged": diverged,
        "last_step": last_step,
        "elapsed_s": elapsed,
    }

    print(
        f"{tag} [NACA0012] DONE Cd_p={cd_p_final:.4f} Cd_f={cd_f_final:.4f} "
        f"Cd_tot={cd_tot_final:.4f} Cl={cl_final:.6f} "
        f"(ref={ref_cd:.4f}) err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )
    print(
        f"{tag} [NACA0012] Previous 6L MRT+Smag: Cd_tot=0.0600 err=20.1% → "
        f"Optimal: Cd_tot={cd_tot_final:.4f} err={err_pct:.1f}%",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} [NACA0012] Results written to {output_path}", flush=True)

    return result


# ---------------------------------------------------------------------------
# Square prism
# ---------------------------------------------------------------------------

def run_square_prism(device, output_path, tag):
    """Square prism at Re=1000 with optimal settings (16D domain, cumulant, quadratic extrap)."""
    nx, ny, nz = 400, 160, 4
    D = 24
    u_in = 0.08
    Re = 1000.0
    nu = u_in * D / Re            # = 0.00192
    tau = 3.0 * nu + 0.5         # = 0.50576
    n_steps = 10000
    warmup = 2000
    ref_cd = 2.10
    ref_st = 0.14

    print(
        f"{tag} [Prism] nx={nx} ny={ny} nz={nz} D={D} u_in={u_in} "
        f"nu={nu:.6e} tau={tau:.6f} Re={Re} extrap={EXTRAP}",
        flush=True,
    )

    t0 = time.time()

    cx = nx // 4          # 100
    cy = ny // 2          # 80
    half = D // 2         # 12
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[:, cy - half:cy + half, cx:cx + D] = True

    n_solid = int(solid.sum().item())
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    near = get_near_wall_2d(solid, axis="z")
    mesh = SurfaceMesh.from_gradient(solid, near)

    n_near = int(near.sum().item())
    print(
        f"{tag} [Prism] solid={n_solid} near={n_near} "
        f"mesh built ({time.time()-t0:.1f}s)",
        flush=True,
    )

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0),
                      device=device)
    initial_mass = float(rho0.sum().item())

    print(f"{tag} [Prism] init done ({time.time()-t0:.1f}s), starting loop...", flush=True)

    A_frontal = D * nz
    dpS = 0.5 * 1.0 * u_in ** 2 * A_frontal

    cd_hist = []
    cl_hist = []
    diverged = False
    last_step = 0

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_cumulant_d3q19(f, tau=tau)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid)
        f = stream3d(f)
        f = far_field_bc_3d(f, u_in=u_in)
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        if not torch.isfinite(f).all():
            print(f"{tag} [Prism] DIVERGED at step {step}", flush=True)
            diverged = True
            last_step = step
            break
        last_step = step

        cd_x, cd_y, _ = drag_pressure_integration(f, mesh, dpS, extrap=EXTRAP)

        if step > warmup:
            if math.isfinite(cd_x):
                cd_hist.append(cd_x)
            if math.isfinite(cd_y):
                cl_hist.append(cd_y)

        if step % 500 == 0 or step == n_steps:
            elapsed = time.time() - t0
            _, ux, uy, uz = macroscopic3d(f)
            ms = float(torch.sqrt(ux * ux + uy * uy + uz * uz).max().item())
            cd_avg = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float("nan")
            print(
                f"{tag} [Prism] step={step} Cd={cd_avg:.4f} "
                f"max|u|={ms:.4f} ({elapsed:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0

    cd_mean = sum(cd_hist) / len(cd_hist) if cd_hist else float("nan")
    cl_mean = sum(cl_hist) / len(cl_hist) if cl_hist else float("nan")
    cl_max = max(cl_hist) if cl_hist else 0.0
    cl_min = min(cl_hist) if cl_hist else 0.0
    cl_amp = (cl_max - cl_min) / 2.0

    # Strouhal number from FFT of Cl
    st = float("nan")
    if len(cl_hist) > 100:
        cl_arr = np.array(cl_hist)
        cl_detrend = cl_arr - cl_arr.mean()
        n_fft = len(cl_detrend)
        freqs = np.fft.rfftfreq(n_fft, d=1.0)
        spectrum = np.abs(np.fft.rfft(cl_detrend))
        min_idx = max(1, int(0.01 * n_fft))
        peak_idx = min_idx + np.argmax(spectrum[min_idx:])
        f_peak = freqs[peak_idx]
        st = f_peak * D / u_in

    cd_err = abs(cd_mean - ref_cd) / ref_cd * 100 if (
        ref_cd > 0 and math.isfinite(cd_mean)) else float("nan")
    st_err = abs(st - ref_st) / ref_st * 100 if (
        ref_st > 0 and math.isfinite(st)) else float("nan")

    result = {
        "benchmark": "square_prism_optimal",
        "device": str(device),
        "grid": f"{nx}x{ny}x{nz}",
        "domain_ratio": f"{nx/D:.0f}D",
        "D": D,
        "u_in": u_in,
        "Re": Re,
        "nu": nu,
        "tau": tau,
        "collision": "cumulant_d3q19",
        "normal_method": "from_gradient",
        "extrap": EXTRAP,
        "n_steps": n_steps,
        "warmup": warmup,
        "Cd": cd_mean,
        "Cl_mean": cl_mean,
        "Cl_amp": cl_amp,
        "St": st,
        "Cd_ref": ref_cd,
        "St_ref": ref_st,
        "Cd_error_pct": cd_err,
        "St_error_pct": st_err,
        "n_samples": len(cd_hist),
        "finite": not diverged,
        "diverged": diverged,
        "last_step": last_step,
        "elapsed_s": elapsed,
    }
    print(
        f"{tag} [Prism] DONE Cd={cd_mean:.4f} (ref={ref_cd}, err={cd_err:.1f}%) "
        f"Cl_amp={cl_amp:.4f} St={st:.4f} (ref={ref_st}, err={st_err:.1f}%) "
        f"({elapsed:.0f}s)",
        flush=True,
    )
    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Backward-facing step
# ---------------------------------------------------------------------------

def bfs_channel_bc_3d(f, u_in, solid):
    """Channel BC for backward-facing step (3D).

    1. Zou/He velocity inlet at x=0.
    2. Zero-gradient outlet at x=nx-1.
    3. Bounce-back on the full solid mask.
    """
    f = zou_he_inlet_velocity_3d(f, u_in)
    f[:, :, :, -1] = f[:, :, :, -2]
    f = bounce_back_cells_3d(f, solid)
    return f


def run_backward_facing_step(device, output_path, tag):
    """BFS at Re=1000 with optimal settings (cumulant, quadratic extrap)."""
    nx = 400
    ny = 100
    nz = 4
    step_h = 10  # H
    x_step = 20
    u_in = 0.05
    Re = 1000.0
    nu = u_in * step_h / Re  # 0.0005
    tau = 3.0 * nu + 0.5     # 0.5015
    n_steps = 10000
    ref_xr = 6.0

    ER = ny / (ny - step_h)

    print(
        f"{tag} [BFS] nx={nx} ny={ny} nz={nz} step_h={step_h} x_step={x_step} "
        f"ER={ER:.2f} u_in={u_in} nu={nu:.6e} tau={tau:.6f} Re={Re} extrap={EXTRAP}",
        flush=True,
    )

    t0 = time.time()

    solid_2d = make_bfs_solid_mask(ny, nx, step_h, x_step, device)
    solid = solid_2d.unsqueeze(0).expand(nz, ny, nx).clone()
    solid[0, :, :] = True
    solid[-1, :, :] = True

    n_solid = int(solid.sum().item())
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Step-only mask for pressure drag measurement
    step_only_2d = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device),
        torch.arange(nx, device=device),
        indexing="ij",
    )
    step_only_2d = (xx < x_step) & (yy < step_h)
    step_only = step_only_2d.unsqueeze(0).expand(nz, ny, nx).clone()

    near_step = get_near_wall_3d(step_only)
    mesh_step = SurfaceMesh.from_gradient(step_only, near_step)

    n_near = int(near_step.sum().item())
    print(
        f"{tag} [BFS] solid={n_solid} step_only={int(step_only.sum().item())} "
        f"near_step={n_near} mesh built ({time.time()-t0:.1f}s)",
        flush=True,
    )

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.zeros((nz, ny, nx), device=device)
    ux0[:, step_h:ny - 1, :] = u_in
    ux0[solid] = 0.0
    uy0 = torch.zeros_like(ux0)
    uz0 = torch.zeros_like(ux0)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)
    initial_mass = float(rho0.sum().item())

    print(f"{tag} [BFS] init done ({time.time()-t0:.1f}s), starting loop...", flush=True)

    A_step = step_h * nz
    dpS = 0.5 * 1.0 * u_in ** 2 * A_step

    xr_hist = []
    cd_step_hist = []
    diverged = False
    last_step = 0

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_cumulant_d3q19(f, tau=tau)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid)
        f = stream3d(f)
        f = bfs_channel_bc_3d(f, u_in, solid)
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        if not torch.isfinite(f).all():
            print(f"{tag} [BFS] DIVERGED at step {step}", flush=True)
            diverged = True
            last_step = step
            break
        last_step = step

        if step % 100 == 0 or step == n_steps:
            rho, ux, uy, uz = macroscopic3d(f)
            ux_zmid = ux[nz // 2]
            ux_zmid = ux_zmid.masked_fill(solid[nz // 2], 0.0)

            centreline = ux_zmid[1, x_step:].cpu()
            xr_star = 0.0
            for i, val in enumerate(centreline.tolist()):
                if val > 0.0:
                    xr_star = float(i) / max(step_h, 1)
                    break
            xr_hist.append(xr_star)

            cd_x, _, _ = drag_pressure_integration(f, mesh_step, dpS, extrap=EXTRAP)
            cd_step_hist.append(cd_x)

            if step % 500 == 0 or step == n_steps:
                elapsed = time.time() - t0
                ms = float(torch.sqrt(ux * ux + uy * uy + uz * uz).max().item())
                print(
                    f"{tag} [BFS] step={step} xr/H={xr_star:.3f} "
                    f"Cd_step={cd_x:.4f} max|u|={ms:.4f} ({elapsed:.0f}s)",
                    flush=True,
                )

    elapsed = time.time() - t0

    rho_f, ux_f, uy_f, uz_f = macroscopic3d(f)
    ux_zmid = ux_f[nz // 2].masked_fill(solid[nz // 2], 0.0)
    centreline = ux_zmid[1, x_step:].cpu()
    final_xr = 0.0
    for i, val in enumerate(centreline.tolist()):
        if val > 0.0:
            final_xr = float(i) / max(step_h, 1)
            break

    tail_xr = xr_hist[-max(len(xr_hist) // 5, 1):] if xr_hist else [0.0]
    xr_mean = sum(tail_xr) / len(tail_xr)
    tail_cd = cd_step_hist[-max(len(cd_step_hist) // 5, 1):] if cd_step_hist else [0.0]
    cd_step_mean = sum(tail_cd) / len(tail_cd)

    p = (rho_f - 1.0) / 3.0
    p_before = float(p[nz // 2, ny - 2, x_step - 1].item())
    p_after = float(p[nz // 2, 1, x_step + 1].item())
    cp_drop = (p_before - p_after) / (0.5 * u_in ** 2)

    err_pct = abs(xr_mean - ref_xr) / ref_xr * 100 if ref_xr > 0 else float("nan")

    result = {
        "benchmark": "backward_facing_step_optimal",
        "device": str(device),
        "grid": f"{nx}x{ny}x{nz}",
        "step_h": step_h,
        "x_step": x_step,
        "expansion_ratio": ER,
        "u_in": u_in,
        "Re": Re,
        "nu": nu,
        "tau": tau,
        "collision": "cumulant_d3q19",
        "normal_method": "from_gradient",
        "extrap": EXTRAP,
        "n_steps": n_steps,
        "xr_H_final": final_xr,
        "xr_H_mean": xr_mean,
        "xr_H_ref": ref_xr,
        "xr_error_pct": err_pct,
        "Cd_step_pressure": cd_step_mean,
        "Cp_drop": cp_drop,
        "finite": not diverged,
        "diverged": diverged,
        "last_step": last_step,
        "elapsed_s": elapsed,
    }
    print(
        f"{tag} [BFS] DONE xr/H={xr_mean:.3f} (ref={ref_xr}, err={err_pct:.1f}%) "
        f"Cd_step={cd_step_mean:.4f} Cp_drop={cp_drop:.4f} "
        f"({elapsed:.0f}s)",
        flush=True,
    )
    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 4:
        print("Usage: python optimal_universal_worker.py <device_id> <benchmark> <output_json>")
        print("  benchmark: naca | square_prism | backward_facing_step")
        sys.exit(1)

    device_id = int(sys.argv[1])
    benchmark = sys.argv[2]
    output_path = sys.argv[3]

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[SDAA:{device_id}]"

    if benchmark == "naca":
        run_naca(device, output_path, tag)
    elif benchmark == "square_prism":
        run_square_prism(device, output_path, tag)
    elif benchmark == "backward_facing_step":
        run_backward_facing_step(device, output_path, tag)
    else:
        print(f"Unknown benchmark: {benchmark}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
