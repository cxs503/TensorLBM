#!/usr/bin/env python3
"""SUBOFF appendage variants + L=160 + 4L domain worker.

Cases (each on a separate SDAA card):
  1. WITH_SAIL (AFF-3) — SDAA:0  — Cd_ref≈0.046, target <30%
  2. FULL (AFF-8)     — SDAA:1  — Cd_ref≈0.055, target <30%
  3. BARE_HULL L=160   — SDAA:2  — Blasius Cf, target <15% (long warmup)
  4. BARE_HULL 4L dom  — SDAA:3  — Blasius Cf, target <5%  (larger domain)

Uses lbm_step_correct (NoDynamics + half-way BB + far-field BC + MRT+Smag)
with SurfaceMesh.from_suboff analytical normals, drag_pressure_integration,
and drag_friction_integration.

Usage:
  python suboff_appendage_worker.py <case_id> <device_id> <output_path>
"""
import json
import math
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    get_near_wall_3d,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig


def compute_strouhal(cl_hist, D, u_in, warmup=0):
    """Strouhal number from Cl time history via FFT.

    St = f * D / U, where f is the dominant shedding frequency (cycles/step).
    Returns (St, amplitude).  If no clear peak, returns (0.0, 0.0).
    """
    signal = np.array(cl_hist[warmup:], dtype=float)
    n = len(signal)
    if n < 32:
        return 0.0, 0.0

    signal = signal - signal.mean()
    fft_vals = np.fft.rfft(signal)
    mags = np.abs(fft_vals)
    mags[0] = 0.0  # skip DC

    if mags.max() < 1e-12:
        return 0.0, 0.0

    idx = int(np.argmax(mags))
    freq = idx / n  # cycles per time step
    st = freq * D / u_in
    amp = 2.0 * mags[idx] / n
    return float(st), float(amp)


def run_case(case_id, device_id, output_path):
    """Run one SUBOFF case on the specified SDAA card."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    Re = 1000
    u_in = 0.06
    cs_smag = 0.05
    n_steps = 10000
    warmup = 5000
    win = 500  # final averaging window

    config = SuboffConfig()

    # ---- Case-specific parameters -------------------------------------------
    if case_id == 1:
        hull_type = "with_sail"
        L = 80
        nx, ny, nz = 200, 80, 80
        cd_ref = 0.046
        ref_name = "AFF-3 Cd≈0.046 (sail adds ~10% drag)"
    elif case_id == 2:
        hull_type = "full"
        L = 80
        nx, ny, nz = 200, 80, 80
        cd_ref = 0.055
        ref_name = "AFF-8 Cd≈0.055 (sail+fins add ~30% drag)"
    elif case_id == 3:
        hull_type = "bare_hull"
        L = 160
        nx, ny, nz = 300, 120, 120
        cd_ref = 1.328 / math.sqrt(Re)  # Blasius
        ref_name = f"Blasius Cf=1.328/sqrt(Re)={cd_ref:.6f}"
    elif case_id == 4:
        hull_type = "bare_hull"
        L = 80
        nx, ny, nz = 320, 120, 120  # 4L domain
        cd_ref = 1.328 / math.sqrt(Re)  # Blasius
        ref_name = f"Blasius Cf=1.328/sqrt(Re)={cd_ref:.6f}"
    else:
        raise ValueError(f"Unknown case_id: {case_id}")

    radius = config.r_over_l * L
    D = 2.0 * radius
    cx = nx * 0.30
    cy = ny * 0.5
    cz = nz * 0.5
    nu = u_in * L / Re
    tau = 3.0 * nu + 0.5

    # dpS = dynamic pressure × wetted surface area (π·D·L for body of revolution)
    dpS = 0.5 * u_in ** 2 * math.pi * D * L

    tag = f"[SDAA:{device_id} C{case_id} {hull_type} L={L}]"
    print(
        f"{tag} nx={nx} ny={ny} nz={nz} L={L} D={D:.3f} "
        f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} "
        f"dpS={dpS:.6e} Cd_ref={cd_ref:.6f} ref={ref_name}",
        flush=True,
    )

    t0 = time.time()

    # 1. Build geometry
    solid, stats = build_suboff_mask(
        hull_type=hull_type,
        nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz,
        length=L, radius=radius,
        config=config, device=device,
    )
    n_solid = int(solid.sum().item())
    print(
        f"{tag} solid={n_solid} L/D={stats['L_D_ratio']} "
        f"solid_frac={n_solid/(nx*ny*nz)*100:.2f}% ({time.time()-t0:.1f}s)",
        flush=True,
    )

    # 2. Near-wall mask
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall={n_near}", flush=True)

    # 3. Surface mesh with from_suboff analytical normals
    mesh = SurfaceMesh.from_suboff(solid, near, cx, cy, cz, L, radius, config)

    # Normal sign check
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    x_bow = cx - L / 2.0
    xi_field = (xx - x_bow) / L
    bow_mask = near & (xi_field < 0.233)
    stern_mask = near & (xi_field > 0.748)
    if bow_mask.any():
        print(
            f"{tag} bow nx_n mean={float(mesh.nx_n[bow_mask].mean()):.4f} "
            f"(expect < 0)",
            flush=True,
        )
    if stern_mask.any():
        print(
            f"{tag} stern nx_n mean={float(mesh.nx_n[stern_mask].mean()):.4f} "
            f"(expect > 0)",
            flush=True,
        )

    # 4. Initialize flow
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s) mass={im}", flush=True)

    # 5. Main loop using lbm_step_correct (BB fix)
    cd_p_hist = []
    cd_f_hist = []
    cl_hist = []
    fz_hist = []

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f,
            collide_smagorinsky_mrt3d,
            tau,
            solid,
            u_in,
            far_field_bc_3d,
            correct_mass_fn=correct_mass3d,
            target_mass=im,
            step=step,
            mass_interval=200,
            C_s=cs_smag,
        )

        # 6. Drag / lift computation
        fx_p, fy_p, fz_p = drag_pressure_integration(f, mesh, dpS, solid=solid)
        fx_f, fy_f, fz_f = drag_friction_integration(
            f, mesh, dpS, nu, formula="standard"
        )

        cd_p_hist.append(fx_p)
        cd_f_hist.append(fx_f)
        cl_hist.append(fy_p + fy_f)
        fz_hist.append(fz_p + fz_f)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0:
            n_avg = min(500, len(cd_p_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg
            cd_tot_avg = cd_p_avg + cd_f_avg
            cl_avg = sum(cl_hist[-n_avg:]) / n_avg
            print(
                f"{tag} step={step} Cd_p={cd_p_avg:.6f} Cd_f={cd_f_avg:.6f} "
                f"Cd_tot={cd_tot_avg:.6f} Cl={cl_avg:.6f} "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0
    steps_run = len(cd_p_hist)

    # ---- Final averages (post-warmup) ---------------------------------------
    start = min(warmup, max(0, steps_run - win))
    n_final = min(win, steps_run - start)
    if n_final <= 0:
        start = 0
        n_final = min(win, steps_run)

    cd_p_final = sum(cd_p_hist[start : start + n_final]) / n_final
    cd_f_final = sum(cd_f_hist[start : start + n_final]) / n_final
    cd_tot_final = cd_p_final + cd_f_final
    cl_final = sum(cl_hist[start : start + n_final]) / n_final
    fz_final = sum(fz_hist[start : start + n_final]) / n_final

    # Strouhal from post-warmup Cl oscillation
    st, cl_amp = compute_strouhal(cl_hist, D, u_in, warmup=warmup)

    err_pct = (
        abs(cd_tot_final - cd_ref) / cd_ref * 100 if cd_ref > 0 else float("nan")
    )

    result = {
        "case": case_id,
        "hull_type": hull_type,
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "L": L,
        "D": D,
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Cs": cs_smag,
        "n_steps": n_steps,
        "warmup": warmup,
        "win": win,
        "steps_run": steps_run,
        "n_solid": n_solid,
        "n_near": n_near,
        "dpS": dpS,
        "Cd_ref": cd_ref,
        "ref_name": ref_name,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl": cl_final,
        "fz": fz_final,
        "St": st,
        "Cl_amplitude": cl_amp,
        "error_pct": err_pct,
        "normal_method": "from_suboff",
        "step_method": "lbm_step_correct",
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    print(f"\n{tag} === FINAL (Cd_ref={cd_ref:.6f}) ===", flush=True)
    print(f"{tag} Cd_p   = {cd_p_final:.6f}", flush=True)
    print(f"{tag} Cd_f   = {cd_f_final:.6f}", flush=True)
    print(f"{tag} Cd_tot = {cd_tot_final:.6f}", flush=True)
    print(f"{tag} Cl     = {cl_final:.6f}", flush=True)
    print(f"{tag} St     = {st:.6f}", flush=True)
    print(f"{tag} err    = {err_pct:.1f}%  time={elapsed:.0f}s", flush=True)

    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python suboff_appendage_worker.py <case_id> <device_id> <output_path>")
        sys.exit(1)
    case_id = int(sys.argv[1])
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]
    try:
        run_case(case_id, device_id, output_path)
    except Exception as e:
        traceback.print_exc()
        Path(output_path).write_text(
            json.dumps({"error": str(e), "case": case_id, "device": device_id})
        )
