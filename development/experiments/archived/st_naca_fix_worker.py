#!/usr/bin/env python3
"""St detection + NACA 4412 Cl sign validation — three tests on SDAA 8-11.

TEST 1: Cylinder Re=200 St (SDAA:8)
  D=48, 50000 steps, ref St=0.165
  Previous: St=0 (not detected, too few steps)

TEST 2: Square prism Re=1000 St (SDAA:9)
  D=24, Cs=0.1, 10000 steps, ref St=0.14
  Previous: St=3.0 (wrong — spurious high-freq peak)

TEST 3: NACA 4412 Cl fix (SDAA:10)
  chord=100, alpha=5deg, Re=1000, ref Cl≈+0.4, Cd≈0.05
  Previous: Cl=-0.295 (wrong sign — rotation direction was CCW instead of CW)

Usage:
  PYTHONPATH=src python st_naca_fix_worker.py <test_name> <device_id> <output_json>
  test_name: cylinder | square_prism | naca4412
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.boundaries3d import (
    bounce_back_cells_3d,
    far_field_bc_3d,
    zou_he_inlet_velocity_3d,
)
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
    get_near_wall_2d,
    get_near_wall_3d,
)
from tensorlbm.postprocess import detect_strouhal


# ---------------------------------------------------------------------------
# Geometry builders
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


def build_square_prism_mask(nx, ny, nz, cx, cy, D, device):
    """Boolean solid mask for a square prism extruded along z-axis."""
    half = D // 2
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[:, cy - half:cy + half, cx:cx + D] = True
    return solid


def naca4412_camber_line(xc):
    """NACA 4412 mean camber line y_c(x) for x in [0,1]. m=0.04, p=0.40."""
    m = 0.04
    p = 0.40
    return np.where(
        xc < p,
        (m / p ** 2) * (2.0 * p * xc - xc ** 2),
        (m / (1.0 - p) ** 2) * ((1.0 - 2.0 * p) + 2.0 * p * xc - xc ** 2),
    )


def naca_thickness(xc, t=0.12):
    """NACA 4-digit thickness distribution y_t(x) for x in [0,1]."""
    return 5.0 * t * (
        0.2969 * np.sqrt(xc)
        - 0.1260 * xc
        - 0.3516 * xc ** 2
        + 0.2843 * xc ** 3
        - 0.1015 * xc ** 4
    )


def build_naca4412(chord, nx, ny, x_le, y_c_chord, alpha_deg, device):
    """Build NACA 4412 solid mask (2D extruded in z, nz=4) at angle of attack.

    The airfoil is built in body coordinates (chord along x, camber along y),
    then rotated by alpha_deg about the quarter-chord point.

    For positive angle of attack (leading edge UP, trailing edge DOWN)
    with the y-axis pointing up, this is a CLOCKWISE rotation:
      [xr]   [ cos  sin] [dx]
      [yr] = [-sin  cos] [dy]
    """
    nz = 4
    alpha_rad = math.radians(alpha_deg)
    cos_a = math.cos(alpha_rad)
    sin_a = math.sin(alpha_rad)

    x_qc = x_le + 0.25 * chord
    y_qc = y_c_chord

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)

    n_samples = 2000
    xc = np.linspace(0.0, 1.0, n_samples)
    yc = naca4412_camber_line(xc)
    yt = naca_thickness(xc, t=0.12)

    x_upper = x_le + xc * chord
    y_upper = y_c_chord + (yc + yt) * chord
    x_lower = x_le + xc * chord
    y_lower = y_c_chord + (yc - yt) * chord

    def rotate(xp, yp):
        dx = xp - x_qc
        dy = yp - y_qc
        xr = x_qc + cos_a * dx + sin_a * dy
        yr = y_qc - sin_a * dx + cos_a * dy
        return xr, yr

    x_upper_r, y_upper_r = rotate(x_upper, y_upper)
    x_lower_r, y_lower_r = rotate(x_lower, y_lower)

    for k in range(nz):
        for i in range(nx):
            xi = float(i)
            if xi < x_upper_r[0] or xi > x_upper_r[-1]:
                continue
            y_u = np.interp(xi, x_upper_r, y_upper_r)
            y_l = np.interp(xi, x_lower_r, y_lower_r)
            j_lo = max(0, int(math.floor(min(y_u, y_l))))
            j_hi = min(ny - 1, int(math.ceil(max(y_u, y_l))))
            if j_hi >= j_lo:
                solid[k, j_lo:j_hi + 1, i] = True

    return solid


# ---------------------------------------------------------------------------
# TEST 1: Cylinder Re=200 St
# ---------------------------------------------------------------------------

def run_cylinder(device_id, output_path):
    """Cylinder Re=200, D=48, 50000 steps — St detection test."""
    nx, ny, nz = 600, 200, 4
    diameter = 48.0
    radius = diameter / 2.0
    u_in = 0.08
    Re = 200.0
    nu = u_in * diameter / Re  # 0.0192
    tau = 3.0 * nu + 0.5       # 0.5576
    cs_smag = 0.05
    n_steps = 50000
    warmup = 10000

    cx_c = nx * 0.25
    cy_c = ny * 0.5

    A_frontal = diameter * nz
    dpS = 0.5 * 1.0 * u_in ** 2 * A_frontal

    ref_cd = 1.30
    ref_st = 0.165

    tag = f"[CylStFix SDAA:{device_id}]"
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    print(f"{tag} Cylinder Re={Re} D={diameter} nx={nx} ny={ny} nz={nz} "
          f"u_in={u_in} nu={nu:.6f} tau={tau:.6f} Cs={cs_smag} "
          f"n_steps={n_steps} warmup={warmup}", flush=True)

    t0 = time.time()

    solid = build_cylinder_mask(nx, ny, nz, cx_c, cy_c, radius, device)
    n_solid = int(solid.sum().item())
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    near = get_near_wall_2d(solid, axis='z')
    n_near = int(near.sum().item())
    mesh = SurfaceMesh.from_cylinder(solid, near, cx_c, cy_c, radius, axis='z')
    print(f"{tag} solid={n_solid} near={n_near} mesh built ({time.time()-t0:.1f}s)", flush=True)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())

    print(f"{tag} init done ({time.time()-t0:.1f}s), starting loop...", flush=True)

    cd_p_hist = []
    cd_f_hist = []
    cd_tot_hist = []
    cl_hist = []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid)
        f = stream3d(f)
        bc_config = {'far_field_faces': ['y-', 'y+'], 'periodic_faces': ['z-', 'z+']}
        f = far_field_bc_3d(f, u_in=u_in, bc_config=bc_config)
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        cdp_x, cdp_y, _ = drag_pressure_integration(f, mesh, dpS)
        cdf_x, cdf_y, _ = drag_friction_integration(f, mesh, dpS, nu)
        cd_tot = cdp_x + cdf_x
        cl = cdp_y + cdf_y

        if step > warmup:
            if math.isfinite(cd_tot):
                cd_p_hist.append(cdp_x)
                cd_f_hist.append(cdf_x)
                cd_tot_hist.append(cd_tot)
            if math.isfinite(cl):
                cl_hist.append(cl)

        if step % 2000 == 0 or step == n_steps:
            elapsed = time.time() - t0
            _, ux, _, _ = macroscopic3d(f)
            ms = float(torch.sqrt(ux * ux).max().item())
            cd_avg = sum(cd_tot_hist) / max(len(cd_tot_hist), 1)
            cl_avg = sum(cl_hist) / max(len(cl_hist), 1)
            print(f"{tag} step={step} Cd={cd_avg:.4f} Cl={cl_avg:.4f} "
                  f"max|ux|={ms:.4f} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0

    cd_p_mean = sum(cd_p_hist) / max(len(cd_p_hist), 1) if cd_p_hist else float("nan")
    cd_f_mean = sum(cd_f_hist) / max(len(cd_f_hist), 1) if cd_f_hist else float("nan")
    cd_tot_mean = sum(cd_tot_hist) / max(len(cd_tot_hist), 1) if cd_tot_hist else float("nan")

    # --- FIXED St detection: Hanning + band-pass [0.05,0.35] + min cycles ---
    st_fft = detect_strouhal(cl_hist, sample_rate=1.0, u_ref=u_in,
                             length_ref=diameter, method='fft', min_cycles=5)
    st_ac = detect_strouhal(cl_hist, sample_rate=1.0, u_ref=u_in,
                            length_ref=diameter, method='autocorr', min_cycles=5)
    st_auto = detect_strouhal(cl_hist, sample_rate=1.0, u_ref=u_in,
                              length_ref=diameter, method='auto', min_cycles=5)

    # --- OLD (buggy) St detection for comparison ---
    n_old = len(cl_hist)
    st_old = float("nan")
    if n_old > 100:
        cl_arr = np.array(cl_hist)
        cl_detrend = cl_arr - cl_arr.mean()
        spectrum = np.abs(np.fft.rfft(cl_detrend))
        peak_idx = int(np.argmax(spectrum[1:])) + 1
        f_peak = peak_idx / n_old
        st_old = f_peak * diameter / u_in

    err_cd = abs(cd_tot_mean - ref_cd) / ref_cd * 100 if math.isfinite(cd_tot_mean) else float("nan")
    err_st = abs(st_auto - ref_st) / ref_st * 100 if (st_auto is not None and math.isfinite(st_auto)) else float("nan")

    result = {
        "test": "cylinder_Re200_st",
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "D": diameter,
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Cs": cs_smag,
        "n_steps": n_steps,
        "warmup": warmup,
        "n_samples": len(cl_hist),
        "Cd_pressure": cd_p_mean,
        "Cd_friction": cd_f_mean,
        "Cd_total": cd_tot_mean,
        "Cd_ref": ref_cd,
        "Cd_err_pct": err_cd,
        "St_fft": st_fft,
        "St_autocorr": st_ac,
        "St_auto": st_auto,
        "St_old_buggy": st_old,
        "St_ref": ref_st,
        "St_err_pct": err_st,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    print(f"\n{tag} === FINAL RESULTS ===", flush=True)
    print(f"{tag} Cd = {cd_tot_mean:.4f} (ref={ref_cd}, err={err_cd:.1f}%)", flush=True)
    print(f"{tag} St (old buggy) = {st_old:.4f}", flush=True)
    print(f"{tag} St (FFT+Hanning+filter) = {st_fft:.4f}" if st_fft is not None else f"{tag} St (FFT+Hanning+filter) = None", flush=True)
    print(f"{tag} St (autocorr)          = {st_ac:.4f}" if st_ac is not None else f"{tag} St (autocorr)          = None", flush=True)
    print(f"{tag} St (auto)              = {st_auto:.4f} (ref={ref_st}, err={err_st:.1f}%)" if st_auto is not None else f"{tag} St (auto)              = None (ref={ref_st})", flush=True)
    print(f"{tag} time = {elapsed:.0f}s", flush=True)

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


# ---------------------------------------------------------------------------
# TEST 2: Square prism Re=1000 St
# ---------------------------------------------------------------------------

def run_square_prism(device_id, output_path):
    """Square prism Re=1000, D=24, Cs=0.1, 10000 steps — St detection test."""
    nx, ny, nz = 400, 160, 4
    D = 24
    u_in = 0.08
    Re = 1000.0
    nu = u_in * D / Re  # 0.00192
    tau = 3.0 * nu + 0.5  # 0.50576
    Cs = 0.1
    n_steps = 10000
    warmup = 2000

    cx = nx // 4  # 100
    cy = ny // 2  # 80

    A_frontal = D * nz
    dpS = 0.5 * 1.0 * u_in ** 2 * A_frontal

    ref_cd = 2.10
    ref_st = 0.14

    tag = f"[PrismStFix SDAA:{device_id}]"
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    print(f"{tag} Square prism Re={Re} D={D} nx={nx} ny={ny} nz={nz} "
          f"u_in={u_in} nu={nu:.6f} tau={tau:.6f} Cs={Cs} "
          f"n_steps={n_steps} warmup={warmup}", flush=True)

    t0 = time.time()

    solid = build_square_prism_mask(nx, ny, nz, cx, cy, D, device)
    n_solid = int(solid.sum().item())
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    near = get_near_wall_2d(solid, axis='z')
    n_near = int(near.sum().item())
    mesh = SurfaceMesh.from_gradient(solid, near)
    print(f"{tag} solid={n_solid} near={n_near} mesh built ({time.time()-t0:.1f}s)", flush=True)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())

    print(f"{tag} init done ({time.time()-t0:.1f}s), starting loop...", flush=True)

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
            print(f"{tag} DIVERGED at step {step}", flush=True)
            diverged = True
            last_step = step
            break
        last_step = step

        cd_x, cd_y, _ = drag_pressure_integration(f, mesh, dpS)

        if step > warmup:
            if math.isfinite(cd_x):
                cd_hist.append(cd_x)
            if math.isfinite(cd_y):
                cl_hist.append(cd_y)

        if step % 1000 == 0 or step == n_steps:
            elapsed = time.time() - t0
            _, ux, uy, uz = macroscopic3d(f)
            ms = float(torch.sqrt(ux * ux + uy * uy + uz * uz).max().item())
            cd_avg = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float("nan")
            print(f"{tag} step={step} Cd={cd_avg:.4f} max|u|={ms:.4f} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0

    cd_mean = sum(cd_hist) / len(cd_hist) if cd_hist else float("nan")
    cl_mean = sum(cl_hist) / len(cl_hist) if cl_hist else float("nan")
    cl_max = max(cl_hist) if cl_hist else 0.0
    cl_min = min(cl_hist) if cl_hist else 0.0
    cl_amp = (cl_max - cl_min) / 2.0

    # --- FIXED St detection ---
    # Prism: D=24, u_in=0.08, St~0.14 → period ~2143 steps
    # 8000 samples / 2143 = ~3.7 cycles → use min_cycles=3
    st_fft = detect_strouhal(cl_hist, sample_rate=1.0, u_ref=u_in,
                             length_ref=D, method='fft', min_cycles=3)
    st_ac = detect_strouhal(cl_hist, sample_rate=1.0, u_ref=u_in,
                            length_ref=D, method='autocorr', min_cycles=3)
    st_auto = detect_strouhal(cl_hist, sample_rate=1.0, u_ref=u_in,
                              length_ref=D, method='auto', min_cycles=3)

    # --- OLD (buggy) St detection for comparison ---
    n_old = len(cl_hist)
    st_old = float("nan")
    if n_old > 100:
        cl_arr = np.array(cl_hist)
        cl_detrend = cl_arr - cl_arr.mean()
        n_fft = len(cl_detrend)
        freqs = np.fft.rfftfreq(n_fft, d=1.0)
        spectrum = np.abs(np.fft.rfft(cl_detrend))
        min_idx = max(1, int(0.01 * n_fft))
        peak_idx = min_idx + int(np.argmax(spectrum[min_idx:]))
        f_peak = freqs[peak_idx]
        st_old = f_peak * D / u_in

    err_cd = abs(cd_mean - ref_cd) / ref_cd * 100 if (ref_cd > 0 and math.isfinite(cd_mean)) else float("nan")
    err_st = abs(st_auto - ref_st) / ref_st * 100 if (st_auto is not None and math.isfinite(st_auto)) else float("nan")

    result = {
        "test": "square_prism_Re1000_st",
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "D": D,
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Cs": Cs,
        "n_steps": n_steps,
        "warmup": warmup,
        "n_samples": len(cl_hist),
        "Cd": cd_mean,
        "Cl_mean": cl_mean,
        "Cl_amp": cl_amp,
        "St_fft": st_fft,
        "St_autocorr": st_ac,
        "St_auto": st_auto,
        "St_old_buggy": st_old,
        "Cd_ref": ref_cd,
        "St_ref": ref_st,
        "Cd_err_pct": err_cd,
        "St_err_pct": err_st,
        "finite": not diverged,
        "diverged": diverged,
        "last_step": last_step,
        "elapsed_s": elapsed,
    }

    print(f"\n{tag} === FINAL RESULTS ===", flush=True)
    print(f"{tag} Cd = {cd_mean:.4f} (ref={ref_cd}, err={err_cd:.1f}%)", flush=True)
    print(f"{tag} Cl_amp = {cl_amp:.4f}", flush=True)
    print(f"{tag} St (old buggy) = {st_old:.4f}", flush=True)
    print(f"{tag} St (FFT+Hanning+filter) = {st_fft:.4f}" if st_fft is not None else f"{tag} St (FFT+Hanning+filter) = None", flush=True)
    print(f"{tag} St (autocorr)          = {st_ac:.4f}" if st_ac is not None else f"{tag} St (autocorr)          = None", flush=True)
    print(f"{tag} St (auto)              = {st_auto:.4f} (ref={ref_st}, err={err_st:.1f}%)" if st_auto is not None else f"{tag} St (auto)              = None (ref={ref_st})", flush=True)
    print(f"{tag} time = {elapsed:.0f}s", flush=True)

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


# ---------------------------------------------------------------------------
# TEST 3: NACA 4412 Cl fix
# ---------------------------------------------------------------------------

def run_naca4412(device_id, output_path):
    """NACA 4412 cambered airfoil at Re=1000, alpha=5 deg — Cl sign fix test."""
    chord = 100
    nx = 600
    ny = 300
    nz = 4
    u_in = 0.05
    Re = 1000
    nu = u_in * chord / Re  # 0.005
    tau = 3.0 * nu + 0.5    # 0.515
    cs_smag = 0.05
    n_steps = 10000
    alpha_deg = 5.0
    ref_cl = 0.4
    ref_cd = 0.05

    x_le = int(nx * 0.25)
    y_c_chord = ny // 2

    dpS = 0.5 * u_in ** 2 * chord * nz

    tag = f"[NACA4412ClFix SDAA:{device_id}]"
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    print(f"{tag} NACA 4412 Re={Re} chord={chord} alpha={alpha_deg}deg "
          f"nx={nx} ny={ny} nz={nz} u_in={u_in} nu={nu:.6f} tau={tau:.6f} "
          f"Cs={cs_smag} n_steps={n_steps}", flush=True)

    t0 = time.time()

    solid = build_naca4412(chord, nx, ny, x_le, y_c_chord, alpha_deg, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid} ({time.time()-t0:.1f}s)", flush=True)

    near = get_near_wall_2d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    # Use from_naca with camber-aware normals (m=0.04, p=0.40, t=0.12)
    mesh = SurfaceMesh.from_naca(solid, near, x_le, y_c_chord, chord,
                                 m=0.04, p=0.40, t=0.12)

    nx_n_vals = mesh.nx_n[near]
    ny_n_vals = mesh.ny_n[near]
    print(f"{tag} normal stats: "
          f"nx_n=[{float(nx_n_vals.min()):.3f}, {float(nx_n_vals.max()):.3f}] "
          f"ny_n=[{float(ny_n_vals.min()):.3f}, {float(ny_n_vals.max()):.3f}]",
          flush=True)

    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())

    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}", flush=True)

    cd_p_hist = []
    cd_f_hist = []
    cd_tot_hist = []
    cl_hist = []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid)
        f = stream3d(f)
        f = far_field_bc_3d(f, u_in)
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS)
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
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 1000 == 0:
            n_avg = min(1000, len(cd_tot_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            cl_avg = sum(cl_hist[-n_avg:]) / n_avg
            elapsed = time.time() - t0
            print(f"{tag} step={step} Cd_p={cd_p_avg:.4f} Cd_f={cd_f_avg:.4f} "
                  f"Cd_tot={cd_tot_avg:.4f} Cl={cl_avg:.6f} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0

    n_final = min(2000, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_final = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final
    cl_final = sum(cl_hist[-n_final:]) / n_final

    err_cd = abs(cd_tot_final - ref_cd) / ref_cd * 100
    err_cl = abs(cl_final - ref_cl) / ref_cl * 100 if ref_cl > 0 else float("nan")

    result = {
        "test": "naca4412_cl_fix",
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "chord": chord,
        "alpha_deg": alpha_deg,
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Cs": cs_smag,
        "n_steps": n_steps,
        "n_solid": n_solid,
        "n_near": n_near,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl": cl_final,
        "Cd_ref": ref_cd,
        "Cl_ref": ref_cl,
        "error_cd_pct": err_cd,
        "error_cl_pct": err_cl,
        "normal_method": "from_naca (camber-aware)",
        "rotation": "clockwise (fixed)",
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    print(f"\n{tag} === FINAL RESULTS ===", flush=True)
    print(f"{tag} Cd_p = {cd_p_final:.4f}", flush=True)
    print(f"{tag} Cd_f = {cd_f_final:.4f}", flush=True)
    print(f"{tag} Cd   = {cd_tot_final:.4f} (ref={ref_cd}, err={err_cd:.1f}%)", flush=True)
    print(f"{tag} Cl   = {cl_final:.6f} (ref={ref_cl}, err={err_cl:.1f}%)", flush=True)
    print(f"{tag} time = {elapsed:.0f}s", flush=True)

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 4:
        print("Usage: python st_naca_fix_worker.py <test_name> <device_id> <output_json>")
        print("  test_name: cylinder | square_prism | naca4412")
        sys.exit(1)

    test_name = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    if test_name == "cylinder":
        run_cylinder(device_id, output_path)
    elif test_name == "square_prism":
        run_square_prism(device_id, output_path)
    elif test_name == "naca4412":
        run_naca4412(device_id, output_path)
    else:
        print(f"Unknown test: {test_name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
