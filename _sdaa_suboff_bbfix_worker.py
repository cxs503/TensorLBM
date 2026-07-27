#!/usr/bin/env python3
"""SUBOFF grid convergence with BB fix — uses lbm_step_correct + from_suboff.

SDAA:28 (L=40, L=80) and SDAA:29 (L=160).

Key changes from previous (suboff_grid_conv_re1000_worker.py):
  1. lbm_step_correct() — bounce_back_cells_3d(f, solid, f_pre=f_pre) [BB fix]
  2. SurfaceMesh.from_suboff() — analytical normals (not from_gradient)
  3. drag_friction_integration formula='standard' AND 'lagrange'

Previous (without BB fix): 14.0% → 8.1% → 24.4% (Cd_f diverged)
"""
import json, math, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import (
    SurfaceMesh, get_near_wall_3d,
    drag_pressure_integration, drag_friction_integration,
)
from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig

FORMULAS = ['standard', 'lagrange']


def run_suboff_bbfix(device_id, L, output_path):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    Re = 1000
    u_in = 0.06
    cs_smag = 0.05
    n_steps = 5000
    win = 500

    config = SuboffConfig()
    radius = config.r_over_l * L
    D = 2.0 * radius

    if L == 40:
        nx, ny, nz = 100, 40, 40
    elif L == 80:
        nx, ny, nz = 200, 80, 80
    elif L == 160:
        nx, ny, nz = 300, 120, 120  # reduced from 400³ to avoid OOM
        n_steps = 3000  # reduced for large grid
    else:
        raise ValueError(f"L must be 40/80/160, got {L}")

    cx = nx * 0.30
    cy = ny * 0.5
    cz = nz * 0.5
    nu = u_in * L / Re
    tau = 3.0 * nu + 0.5
    dpS = 0.5 * u_in ** 2 * math.pi * D * L
    Cf_ref = 1.328 / math.sqrt(Re)

    tag = f"[SDAA:{device_id} SUBOFF-BBFIX L={L} Re=1000]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} L={L} D={D:.3f} "
          f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} "
          f"dpS={dpS:.6e} Cf_ref={Cf_ref:.6f}", flush=True)

    t0 = time.time()

    # 1. Build geometry
    solid, stats = build_suboff_mask(
        hull_type="bare_hull", nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz, length=L, radius=radius,
        config=config, device=device,
    )
    n_solid = int(solid.sum().item())
    print(f"{tag} solid={n_solid} L/D={stats['L_D_ratio']} ({time.time()-t0:.1f}s)", flush=True)

    # 2. Near-wall mask
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall={n_near}", flush=True)

    # 3. Surface mesh with from_suboff normals (analytical)
    mesh = SurfaceMesh.from_suboff(solid, near, cx, cy, cz, L, radius, config)

    # Normal sign check
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing='ij')
    x_bow = cx - L / 2.0
    xi_field = (xx - x_bow) / L
    bow_mask = near & (xi_field < 0.233)
    stern_mask = near & (xi_field > 0.748)
    if bow_mask.any():
        print(f"{tag} bow nx_n mean={float(mesh.nx_n[bow_mask].mean()):.4f} (expect < 0)", flush=True)
    if stern_mask.any():
        print(f"{tag} stern nx_n mean={float(mesh.nx_n[stern_mask].mean()):.4f} (expect > 0)", flush=True)

    # 4. Initialize flow
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s) mass={im}", flush=True)

    # 5. Main loop using lbm_step_correct (BB fix)
    cd_p_hist = []
    cd_f_hists = {fm: [] for fm in FORMULAS}

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_smagorinsky_mrt3d, tau, solid, u_in, far_field_bc_3d,
            correct_mass_fn=correct_mass3d, target_mass=im,
            step=step, mass_interval=200,
            C_s=cs_smag,
        )

        # 6. Drag computation
        fx_p, _, _ = drag_pressure_integration(f, mesh, dpS, solid=solid)
        cd_p_hist.append(fx_p)
        for fm in FORMULAS:
            fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu, formula=fm)
            cd_f_hists[fm].append(fx_f)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0:
            n_avg = min(500, len(cd_p_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
            parts = " ".join(
                f"Cd_f[{fm}]={sum(cd_f_hists[fm][-n_avg:])/n_avg:.6f}" for fm in FORMULAS)
            print(f"{tag} step={step} Cd_p={cd_p_avg:.6f} {parts} ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_final = min(win, len(cd_p_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / n_final

    result = {
        "case": "suboff_bbfix_grid_conv",
        "device": f"sdaa:{device_id}",
        "Re": Re, "L": L, "D": D, "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in, "nu": nu, "tau": tau, "Cs": cs_smag,
        "n_steps": n_steps, "win": win,
        "n_solid": n_solid, "n_near": n_near, "dpS": dpS,
        "Cf_ref": float(Cf_ref),
        "Cd_pressure": cd_p_final,
        "normal_method": "from_suboff",
        "step_method": "lbm_step_correct",
        "bb_fix": True,
    }
    for fm in FORMULAS:
        cd_f = sum(cd_f_hists[fm][-n_final:]) / n_final
        cd_tot = cd_p_final + cd_f
        result[f"Cd_friction_{fm}"] = cd_f
        result[f"Cd_total_{fm}"] = cd_tot
        result[f"err_pct_{fm}"] = abs(cd_tot - Cf_ref) / Cf_ref * 100

    print(f"\n{tag} === FINAL (Cf_ref={Cf_ref:.6f}) ===", flush=True)
    print(f"{tag} Cd_p = {cd_p_final:.6f}", flush=True)
    for fm in FORMULAS:
        cd_f = result[f"Cd_friction_{fm}"]
        cd_tot = result[f"Cd_total_{fm}"]
        err = result[f"err_pct_{fm}"]
        print(f"{tag} {fm:12s}: Cd_f={cd_f:.6f} Cd_tot={cd_tot:.6f} err={err:.1f}%", flush=True)
    print(f"{tag} time={elapsed:.0f}s", flush=True)

    result["elapsed_s"] = elapsed
    result["finite"] = bool(torch.isfinite(f).all().item())
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


if __name__ == "__main__":
    L = int(sys.argv[1])
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]
    try:
        run_suboff_bbfix(device_id, L, output_path)
    except Exception as e:
        traceback.print_exc()
        Path(output_path).write_text(json.dumps({"error": str(e), "L": L, "device": device_id}))
