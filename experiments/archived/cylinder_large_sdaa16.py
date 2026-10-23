"""Cylinder Re=200 large domain (D=96, 12D domain) on SDAA:16.

Reference: Cd=1.30, Cl_amp=0.31, St=0.165
D=96, nx=1200, ny=400, nz=4 (12D domain, low blockage)
u_in=0.08, Re=200, tau=3*0.08*96/200+0.5=0.5152
50000 steps (for vortex shedding)

Uses verified modules:
  - lbm_step_correct (correct main loop: NoDynamics + half-way BB)
  - drag_pressure (SurfaceMesh.from_cylinder, drag_pressure/friction_integration)
  - boundaries3d (far_field_bc_3d)
  - turbulence (collide_smagorinsky_mrt3d, Cs=0.05)
"""
from __future__ import annotations
import sys, json, math, time
from pathlib import Path
import numpy as np
import torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import (
    SurfaceMesh, get_near_wall_2d,
    drag_pressure_integration, drag_friction_integration,
)


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


def main():
    device_id = 16
    output_path = sys.argv[1] if len(sys.argv) > 1 else "cylinder_large_sdaa16.json"

    # Parameters (from task spec)
    nx, ny, nz = 1200, 400, 4
    diameter = 96.0
    radius = diameter / 2.0
    u_in = 0.08
    Re = 200.0
    nu = u_in * diameter / Re  # = 0.0384
    tau = 3.0 * nu + 0.5       # = 0.6152... wait task says 0.5152
    # Task: tau=3*0.08*96/200+0.5 = 3*0.0384+0.5 = 0.6152
    # But task says 0.5152. Let me recheck: 3*0.08*96/200 = 3*7.68/200 = 23.04/200 = 0.1152
    # So tau = 0.1152 + 0.5 = 0.6152. Task says 0.5152, which would be 3*0.08*96/200+0.5
    # = 0.1152+0.5 = 0.6152. The task has a typo (0.5152 should be 0.6152).
    # Actually: 3*u_in*D/Re = 3*0.08*96/200 = 0.1152, tau = 0.1152+0.5 = 0.6152
    # But nu = u_in*D/Re = 0.08*96/200 = 0.0384, tau = 3*nu+0.5 = 3*0.0384+0.5 = 0.6152
    # Task says 0.5152 — this is likely a typo. Use correct value 0.6152.
    cs_smag = 0.05
    n_steps = 50000
    warmup = 10000  # transient steps before averaging (need ~1.4 shedding periods)

    cx_c = nx * 0.25   # cylinder center x (quarter from inlet)
    cy_c = ny * 0.5    # cylinder center y

    A_frontal = diameter * nz
    dpS = 0.5 * 1.0 * u_in ** 2 * A_frontal

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    tag = f"[CylLarge SDAA:{device_id}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} D={diameter} u_in={u_in} "
          f"nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} dpS={dpS:.4f}", flush=True)
    print(f"{tag} n_steps={n_steps} warmup={warmup}", flush=True)

    t0 = time.time()

    # Build cylinder mask
    solid = build_cylinder_mask(nx, ny, nz, cx_c, cy_c, radius, device)
    n_solid = int(solid.sum().item())
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Near-wall mask (2D extruded along z)
    near = get_near_wall_2d(solid, axis='z')
    n_near = int(near.sum().item())
    print(f"{tag} solid cells={n_solid} near-wall cells={n_near}", flush=True)

    # Surface mesh with analytical cylinder normal
    mesh = SurfaceMesh.from_cylinder(solid, near, cx_c, cy_c, radius, axis='z')
    print(f"{tag} SurfaceMesh built (analytical cylinder normal)", flush=True)

    # Initialize flow field
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())

    print(f"{tag} init done ({time.time()-t0:.1f}s), starting loop...", flush=True)

    # Accumulators
    cd_p_hist = []
    cd_f_hist = []
    cd_tot_hist = []
    cl_hist = []  # for St computation (full history after warmup)

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision
        f_pre = f.clone()

        # 2. Collision (MRT + Smagorinsky)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 3. NoDynamics: restore solid cells
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (BEFORE streaming)
        f = bounce_back_cells_3d(f, solid)

        # 5. Stream
        f = stream3d(f)

        # 6. Far-field BC (2D extruded: y± far-field, z± periodic)
        bc_config = {'far_field_faces': ['y-', 'y+'], 'periodic_faces': ['z-', 'z+']}
        f = far_field_bc_3d(f, u_in=u_in, bc_config=bc_config)

        # 7. Drag (post-stream, post-BC — physical state)
        cdp_x, cdp_y, _ = drag_pressure_integration(f, mesh, dpS)
        cdf_x, cdf_y, _ = drag_friction_integration(f, mesh, dpS, nu)
        cd_tot = cdp_x + cdf_x
        cl = cdp_y + cdf_y  # lift = pressure + friction y-component

        # 8. Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # Check divergence
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        # Record post-warmup
        if step > warmup:
            if math.isfinite(cd_tot):
                cd_p_hist.append(cdp_x)
                cd_f_hist.append(cdf_x)
                cd_tot_hist.append(cd_tot)
                cl_hist.append(cl)

        if step % 500 == 0:
            _, ux, _, _ = macroscopic3d(f)
            ms = float(torch.sqrt(ux * ux).max().item())
            elapsed = time.time() - t0
            cd_avg = sum(cd_tot_hist) / max(len(cd_tot_hist), 1)
            cl_avg = sum(cl_hist) / max(len(cl_hist), 1)
            print(f"{tag} step={step} Cd={cd_avg:.4f} Cl={cl_avg:.4f} "
                  f"max|ux|={ms:.4f} ({elapsed:.0f}s, {elapsed/step:.3f}s/step)", flush=True)

    elapsed = time.time() - t0
    ref_cd = 1.30
    ref_cl_amp = 0.31
    ref_st = 0.165

    # Compute statistics
    cd_p_mean = sum(cd_p_hist) / max(len(cd_p_hist), 1) if cd_p_hist else float("nan")
    cd_f_mean = sum(cd_f_hist) / max(len(cd_f_hist), 1) if cd_f_hist else float("nan")
    cd_tot_mean = sum(cd_tot_hist) / max(len(cd_tot_hist), 1) if cd_tot_hist else float("nan")

    def _std(vals):
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))

    cd_p_std = _std(cd_p_hist)
    cd_f_std = _std(cd_f_hist)
    cd_tot_std = _std(cd_tot_hist)

    # Cl amplitude (half of peak-to-peak)
    if cl_hist:
        cl_max = max(cl_hist)
        cl_min = min(cl_hist)
        cl_amp = (cl_max - cl_min) / 2.0
    else:
        cl_amp = float("nan")

    # Strouhal number from FFT of Cl
    st_val = float("nan")
    if len(cl_hist) > 100:
        cl_arr = np.array(cl_hist)
        cl_centered = cl_arr - cl_arr.mean()
        fft = np.fft.rfft(cl_centered)
        freqs = np.fft.rfftfreq(len(cl_centered), d=1.0)
        power = np.abs(fft) ** 2
        # Skip DC component
        power[0] = 0
        peak_idx = np.argmax(power)
        f_shed = freqs[peak_idx]
        st_val = f_shed * diameter / u_in

    err_cd = abs(cd_tot_mean - ref_cd) / ref_cd * 100 if math.isfinite(cd_tot_mean) else float("nan")
    err_cl = abs(cl_amp - ref_cl_amp) / ref_cl_amp * 100 if math.isfinite(cl_amp) else float("nan")
    err_st = abs(st_val - ref_st) / ref_st * 100 if math.isfinite(st_val) else float("nan")

    print(f"\n{tag} === FINAL RESULTS ===", flush=True)
    print(f"{tag} Cd_p   = {cd_p_mean:.4f} ± {cd_p_std:.4f}", flush=True)
    print(f"{tag} Cd_f   = {cd_f_mean:.4f} ± {cd_f_std:.4f}", flush=True)
    print(f"{tag} Cd_tot = {cd_tot_mean:.4f} ± {cd_tot_std:.4f}  (ref={ref_cd}, err={err_cd:.1f}%)", flush=True)
    print(f"{tag} Cl_amp = {cl_amp:.4f}  (ref={ref_cl_amp}, err={err_cl:.1f}%)", flush=True)
    print(f"{tag} St     = {st_val:.4f}  (ref={ref_st}, err={err_st:.1f}%)", flush=True)
    print(f"{tag} time = {elapsed:.0f}s ({elapsed/n_steps:.3f}s/step)", flush=True)

    result = {
        "case": "cylinder_Re200_large_domain",
        "device": f"sdaa:{device_id}",
        "lattice": "D3Q19",
        "collision": f"MRT+Smag(Cs={cs_smag})",
        "boundary": "halfway_BB+farfield",
        "grid": f"{nx}x{ny}x{nz}",
        "diameter": diameter,
        "u_in": u_in,
        "Re": Re,
        "nu": nu,
        "tau": tau,
        "Cs": cs_smag,
        "n_steps": n_steps,
        "warmup": warmup,
        "Cd_p_mean": cd_p_mean,
        "Cd_p_std": cd_p_std,
        "Cd_f_mean": cd_f_mean,
        "Cd_f_std": cd_f_std,
        "Cd_tot_mean": cd_tot_mean,
        "Cd_tot_std": cd_tot_std,
        "Cd_ref": ref_cd,
        "Cd_err_pct": err_cd,
        "Cl_amp": cl_amp,
        "Cl_amp_ref": ref_cl_amp,
        "Cl_amp_err_pct": err_cl,
        "St": st_val,
        "St_ref": ref_st,
        "St_err_pct": err_st,
        "n_samples": len(cd_tot_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
        "time_per_step_s": elapsed / n_steps,
    }

    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} results saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
