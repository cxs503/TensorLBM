"""Backward-facing step + square prism benchmark worker.

Runs one of two canonical benchmarks using the unified pressure integration
module (drag_pressure.py) with MRT+Smagorinsky (Cs=0.05) on D3Q19:

  1. backward_facing_step  — Re=1000, ER=2, channel BC, measure x_r/H + Cd
  2. square_prism          — Re=1000, far-field BC, measure Cd/Cl_amp/St

Usage:
    PYTHONPATH=src python _bfs_prism_bench_worker.py <device_id> <benchmark> <output_json>
    benchmark: backward_facing_step | square_prism
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
    zou_he_outlet_pressure_3d,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    drag_pressure_integration,
    get_near_wall_2d,
    get_near_wall_3d,
)
from tensorlbm.backward_facing_step import make_bfs_solid_mask


# ---------------------------------------------------------------------------
# BFS boundary condition (3D channel: Zou/He inlet + pressure outlet + BB)
# ---------------------------------------------------------------------------

def bfs_channel_bc_3d(f, u_in, solid):
    """Channel BC for backward-facing step (3D).

    1. Zou/He velocity inlet at x=0 (solid cells at x=0 below step will be
       overwritten by the subsequent bounce-back pass).
    2. Zero-gradient outlet at x=nx-1.
    3. Bounce-back on the full solid mask (step + top/bottom/front/back walls).
    """
    f = zou_he_inlet_velocity_3d(f, u_in)
    # Zero-gradient outlet
    f[:, :, :, -1] = f[:, :, :, -2]
    f = bounce_back_cells_3d(f, solid)
    return f


# ---------------------------------------------------------------------------
# Backward-facing step benchmark
# ---------------------------------------------------------------------------

def run_backward_facing_step(
    device, output_path, tag,
    nx=400, ny=20, nz=4, step_h=10, x_step=20,
    u_in=0.05, Re=1000.0, Cs=0.05, n_steps=10000,
):
    """Backward-facing step: Re=1000, ER=2, 10000 steps.

    Default geometry: step_h=10, ny=20 (ER=2), nx=400, nz=4, x_step=20.
    u_in=0.05, Re=1000 (based on H=step_h), tau=0.5015, Cs=0.05.
    Reference: x_r/H = 6.0 (ER=2, Re=1000).
    """
    nu = u_in * step_h / Re
    tau = 3.0 * nu + 0.5
    ref_xr = 6.0

    ER = ny / (ny - step_h)

    print(
        f"{tag} [BFS] nx={nx} ny={ny} nz={nz} step_h={step_h} x_step={x_step} "
        f"ER={ER:.1f} u_in={u_in} nu={nu:.6e} tau={tau:.6f} Re={Re} Cs={Cs}",
        flush=True,
    )

    t0 = time.time()

    # Build 2D solid mask and extrude to 3D
    solid_2d = make_bfs_solid_mask(ny, nx, step_h, x_step, device)
    solid = solid_2d.unsqueeze(0).expand(nz, ny, nx).clone()
    # Add front/back z-walls
    solid[0, :, :] = True
    solid[-1, :, :] = True

    n_solid = int(solid.sum().item())
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Step-only mask (for pressure drag measurement, without channel walls)
    step_only_2d = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device),
        torch.arange(nx, device=device),
        indexing="ij",
    )
    step_only_2d = (xx < x_step) & (yy < step_h)
    step_only = step_only_2d.unsqueeze(0).expand(nz, ny, nx).clone()

    # Near-wall mask and surface mesh for the step (from_gradient normal)
    near_step = get_near_wall_3d(step_only)
    mesh_step = SurfaceMesh.from_gradient(step_only, near_step)

    n_near = int(near_step.sum().item())
    print(
        f"{tag} [BFS] solid={n_solid} step_only={int(step_only.sum().item())} "
        f"near_step={n_near} mesh built ({time.time()-t0:.1f}s)",
        flush=True,
    )

    # Initialize: uniform flow above the step, rest in solid
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.zeros((nz, ny, nx), device=device)
    # Prescribe inlet velocity above the step
    ux0[:, step_h:ny - 1, :] = u_in
    ux0[solid] = 0.0
    uy0 = torch.zeros_like(ux0)
    uz0 = torch.zeros_like(ux0)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)
    initial_mass = float(rho0.sum().item())

    print(f"{tag} [BFS] init done ({time.time()-t0:.1f}s), starting loop...", flush=True)

    # dpS for pressure drag on step (frontal area = step_h * nz)
    A_step = step_h * nz
    dpS = 0.5 * 1.0 * u_in ** 2 * A_step

    xr_hist = []
    cd_step_hist = []
    diverged = False
    last_step = 0

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision
        f_pre = f.clone()

        # 2. Collision (MRT + Smagorinsky)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=Cs)

        # 3. NoDynamics: restore solid cells
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (BEFORE streaming)
        f = bounce_back_cells_3d(f, solid)

        # 5. Streaming
        f = stream3d(f)

        # 6. Channel BC (inlet + outlet + bounce-back on solid)
        f = bfs_channel_bc_3d(f, u_in, solid)

        # 7. Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # Check divergence
        if not torch.isfinite(f).all():
            print(f"{tag} [BFS] DIVERGED at step {step}", flush=True)
            diverged = True
            last_step = step
            break
        last_step = step

        # Measure reattachment length and step pressure drag
        if step % 100 == 0 or step == n_steps:
            rho, ux, uy, uz = macroscopic3d(f)
            ux_zmid = ux[nz // 2]  # middle z-layer (ny, nx)
            ux_zmid = ux_zmid.masked_fill(solid[nz // 2], 0.0)

            # Reattachment: scan y=1 (first fluid row above bottom wall)
            # downstream of the step for first x where ux > 0
            centreline = ux_zmid[1, x_step:].cpu()
            xr_star = 0.0
            for i, val in enumerate(centreline.tolist()):
                if val > 0.0:
                    xr_star = float(i) / max(step_h, 1)
                    break
            xr_hist.append(xr_star)

            # Pressure drag on step (from_gradient normal)
            cd_x, _, _ = drag_pressure_integration(f, mesh_step, dpS)
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

    # Final measurements
    rho_f, ux_f, uy_f, uz_f = macroscopic3d(f)
    ux_zmid = ux_f[nz // 2].masked_fill(solid[nz // 2], 0.0)
    centreline = ux_zmid[1, x_step:].cpu()
    final_xr = 0.0
    for i, val in enumerate(centreline.tolist()):
        if val > 0.0:
            final_xr = float(i) / max(step_h, 1)
            break

    # Average xr over last 20% of history
    tail_xr = xr_hist[-max(len(xr_hist) // 5, 1):] if xr_hist else [0.0]
    xr_mean = sum(tail_xr) / len(tail_xr)
    tail_cd = cd_step_hist[-max(len(cd_step_hist) // 5, 1):] if cd_step_hist else [0.0]
    cd_step_mean = sum(tail_cd) / len(tail_cd)

    # Pressure drop across step: Cp = (p_before - p_after) / (0.5 * u^2)
    p = (rho_f - 1.0) / 3.0
    p_before = float(p[nz // 2, ny - 2, x_step - 1].item())  # just upstream
    p_after = float(p[nz // 2, 1, x_step + 1].item())        # just downstream (recirc)
    cp_drop = (p_before - p_after) / (0.5 * u_in ** 2)

    err_pct = abs(xr_mean - ref_xr) / ref_xr * 100 if ref_xr > 0 else float("nan")

    result = {
        "benchmark": "backward_facing_step",
        "device": str(device),
        "grid": f"{nx}x{ny}x{nz}",
        "step_h": step_h,
        "x_step": x_step,
        "expansion_ratio": ER,
        "u_in": u_in,
        "Re": Re,
        "nu": nu,
        "tau": tau,
        "Cs": Cs,
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
    Path(output_path).write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Square prism benchmark
# ---------------------------------------------------------------------------

def run_square_prism(
    device, output_path, tag,
    nx=300, ny=120, nz=4, D=24,
    u_in=0.08, Re=1000.0, Cs=0.05, n_steps=10000, warmup=2000,
):
    """Square prism: Re=1000, 10000 steps.

    Default geometry: D=24, nx=300, ny=120, nz=4.
    u_in=0.08, Re=1000, tau=0.5058, Cs=0.05.
    Reference: Cd=2.10, St=0.14 (experimental).
    """
    ref_cd = 2.10
    ref_st = 0.14

    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5

    print(
        f"{tag} [Prism] nx={nx} ny={ny} nz={nz} D={D} u_in={u_in} "
        f"nu={nu:.6e} tau={tau:.6f} Re={Re} Cs={Cs}",
        flush=True,
    )

    t0 = time.time()

    # Build square prism solid mask
    cx = nx // 4          # 75
    cy = ny // 2          # 60
    half = D // 2         # 12
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[:, cy - half:cy + half, cx:cx + D] = True

    n_solid = int(solid.sum().item())
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Near-wall mask and surface mesh (from_gradient normal)
    near = get_near_wall_2d(solid, axis="z")
    mesh = SurfaceMesh.from_gradient(solid, near)

    n_near = int(near.sum().item())
    print(
        f"{tag} [Prism] solid={n_solid} near={n_near} "
        f"mesh built ({time.time()-t0:.1f}s)",
        flush=True,
    )

    # Initialize: uniform free-stream
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0),
                      device=device)
    initial_mass = float(rho0.sum().item())

    print(f"{tag} [Prism] init done ({time.time()-t0:.1f}s), starting loop...", flush=True)

    # dpS for drag (frontal area = D * nz)
    A_frontal = D * nz
    dpS = 0.5 * 1.0 * u_in ** 2 * A_frontal

    cd_hist = []
    cl_hist = []
    diverged = False
    last_step = 0

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision
        f_pre = f.clone()

        # 2. Collision (MRT + Smagorinsky)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=Cs)

        # 3. NoDynamics: restore solid cells
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (BEFORE streaming)
        f = bounce_back_cells_3d(f, solid)

        # 5. Streaming
        f = stream3d(f)

        # 6. Far-field BC (free-stream inlet, zero-gradient outlet, y± far-field)
        f = far_field_bc_3d(f, u_in=u_in)

        # 7. Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # Check divergence
        if not torch.isfinite(f).all():
            print(f"{tag} [Prism] DIVERGED at step {step}", flush=True)
            diverged = True
            last_step = step
            break
        last_step = step

        # Drag / lift (pressure integration)
        cd_x, cd_y, _ = drag_pressure_integration(f, mesh, dpS)

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

    # Statistics
    cd_mean = sum(cd_hist) / len(cd_hist) if cd_hist else float("nan")
    cl_mean = sum(cl_hist) / len(cl_hist) if cl_hist else float("nan")
    cl_max = max(cl_hist) if cl_hist else 0.0
    cl_min = min(cl_hist) if cl_hist else 0.0
    cl_amp = (cl_max - cl_min) / 2.0

    # Strouhal number from FFT of Cl (robust: Hanning window, low-freq only)
    st = float("nan")
    if len(cl_hist) > 500:
        cl_arr = np.array(cl_hist, dtype=np.float64)
        cl_detrend = cl_arr - cl_arr.mean()
        # Hanning window to reduce spectral leakage
        win = np.hanning(len(cl_detrend))
        cl_win = cl_detrend * win
        n_fft = len(cl_win)
        freqs = np.fft.rfftfreq(n_fft, d=1.0)  # d=1 lattice time step
        spectrum = np.abs(np.fft.rfft(cl_win))
        # Only look at low frequencies: St < 3.0 → f < 3.0*u_in/D
        f_max = 3.0 * u_in / D
        valid = freqs < f_max
        valid[0] = False  # skip DC
        if valid.any():
            spectrum_valid = np.where(valid, spectrum, 0.0)
            peak_idx = int(np.argmax(spectrum_valid))
            if spectrum_valid[peak_idx] > 0:
                f_peak = freqs[peak_idx]
                st = f_peak * D / u_in

    cd_err = abs(cd_mean - ref_cd) / ref_cd * 100 if (
        ref_cd > 0 and math.isfinite(cd_mean)) else float("nan")
    st_err = abs(st - ref_st) / ref_st * 100 if (
        ref_st > 0 and math.isfinite(st)) else float("nan")

    result = {
        "benchmark": "square_prism",
        "device": str(device),
        "grid": f"{nx}x{ny}x{nz}",
        "D": D,
        "u_in": u_in,
        "Re": Re,
        "nu": nu,
        "tau": tau,
        "Cs": Cs,
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
    Path(output_path).write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device_id = int(sys.argv[1])
    benchmark = sys.argv[2]
    output_path = sys.argv[3]

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[SDAA:{device_id}]"

    if benchmark == "backward_facing_step":
        # Optional overrides: nx ny nz step_h x_step
        kwargs = {}
        if len(sys.argv) > 4:
            kwargs["nx"] = int(sys.argv[4])
        if len(sys.argv) > 5:
            kwargs["ny"] = int(sys.argv[5])
        if len(sys.argv) > 6:
            kwargs["nz"] = int(sys.argv[6])
        if len(sys.argv) > 7:
            kwargs["step_h"] = int(sys.argv[7])
        if len(sys.argv) > 8:
            kwargs["x_step"] = int(sys.argv[8])
        run_backward_facing_step(device, output_path, tag, **kwargs)
    elif benchmark == "square_prism":
        kwargs = {}
        if len(sys.argv) > 4:
            kwargs["D"] = int(sys.argv[4])
        if len(sys.argv) > 5:
            kwargs["Cs"] = float(sys.argv[5])
        if len(sys.argv) > 6:
            kwargs["nx"] = int(sys.argv[6])
        if len(sys.argv) > 7:
            kwargs["ny"] = int(sys.argv[7])
        run_square_prism(device, output_path, tag, **kwargs)
    else:
        print(f"Unknown benchmark: {benchmark}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
