#!/usr/bin/env python3
"""Couette grid convergence with BB fix — bounce_back_cells_3d(f_pre).

SDAA:31.  ny=8/16/32, tau=1.0.

Key change from previous (friction_formula_conv_worker.py):
  Previous used bounce_back_cells_3d(f, solid) — NO f_pre → grid divergence
  This uses bounce_back_cells_3d(f, solid, f_pre=f_pre) — BB fix

Previous: 0.00% → 0.11% → 12.3% (diverged at ny=32)
Target:   convergence (all < 1%)
"""
import json, math, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.d3q19 import C, W, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import stream3d, collide_bgk3d
from tensorlbm.drag_pressure import (
    SurfaceMesh, get_near_wall_3d,
    drag_friction_integration,
)

FORMULAS = ['standard', 'lagrange']


def _make_channel_solid(nz, ny, nx, device):
    """Channel walls: top and bottom rows solid."""
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True   # bottom wall
    solid[:, -1, :] = True  # top wall
    return solid


def _near_wall_bottom_only(near, ny):
    """Keep only bottom-wall near cells (for friction measurement)."""
    nb = near.clone()
    nb[:, ny - 2, :] = False  # remove top-wall near cells
    return nb


def _moving_wall_correction(f, top_mask, u_top):
    """Moving wall correction for D3Q19 (standard Ladd formula).

    After bounce-back, add: delta_f_i = 2*w_i*rho*(c_i·u_wall)/cs²
    at the top wall cells.  cs² = 1/3, so 2/cs² = 6.
    """
    device = f.device
    c = C.to(device).float()
    w = W.to(device).float()
    rho = f.sum(dim=0)  # (nz, ny, nx)
    # Correction: 6 * rho * u_top * w_i * c_{i,x}  (u_wall = (u_top, 0, 0))
    correction = (6.0 * u_top * w * c[:, 0]).view(19, 1, 1, 1) * rho.unsqueeze(0)
    tm = top_mask.unsqueeze(0).float()
    return f + correction * tm


def run_couette_bbfix(device_id, ny, output_path):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    nx, nz = 80, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0  # = 1/6
    u_top = 0.05
    n_steps = 4000
    warmup = 500

    H = ny - 2
    Cf_exact = 2.0 * nu / (H * u_top)

    tag = f"[SDAA:{device_id} Couette-BBFIX ny={ny}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} tau={tau} nu={nu:.6f} u_top={u_top} "
          f"H={H} Cf_exact={Cf_exact:.6f}", flush=True)

    t0 = time.time()

    # 1. Geometry
    solid = _make_channel_solid(nz, ny, nx, device)
    top_wall_mask = torch.zeros_like(solid)
    top_wall_mask[:, -1, :] = True
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # 2. Near-wall mask (bottom only for friction measurement)
    near = get_near_wall_3d(solid)
    near_bottom = _near_wall_bottom_only(near, ny)
    mesh_bottom = SurfaceMesh.from_gradient(solid, near_bottom)
    print(f"{tag} near(bottom)={int(near_bottom.sum().item())} ({time.time()-t0:.1f}s)", flush=True)

    # 3. Initialize: quiescent
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    # 4. Main loop with BB fix
    cf_hists = {fm: [] for fm in FORMULAS}

    for step in range(1, n_steps + 1):
        # --- Same pattern as lbm_step_correct, with moving wall ---
        # 1. Save pre-collision
        f_pre = f.clone()

        # 2. Collision (BGK, tau=1.0)
        f = collide_bgk3d(f, tau=tau)

        # 3. NoDynamics: restore solid cells to pre-collision
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (BEFORE streaming) — BB FIX: f_pre
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)

        # 4b. Moving wall correction at top wall (standard Ladd formula)
        f = _moving_wall_correction(f, top_wall_mask, u_top)

        # 5. Streaming (periodic in x, z via torch.roll)
        f = stream3d(f)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        # 6. Friction measurement (after warmup)
        if step > warmup:
            A_wall = nx * nz
            dpS_wall = 0.5 * 1.0 * u_top ** 2 * A_wall
            for fm in FORMULAS:
                ffx, _, _ = drag_friction_integration(f, mesh_bottom, dpS_wall, nu, formula=fm)
                cf_hists[fm].append(ffx)

        if step % 1000 == 0:
            _, ux, _, _ = macroscopic3d(f)
            u_prof = ux.mean(dim=(0, 2))
            parts = " ".join(
                f"Cf[{fm}]={sum(cf_hists[fm])/max(len(cf_hists[fm]),1):.6f}" for fm in FORMULAS)
            print(f"{tag} step={step} {parts} u[1]={float(u_prof[1]):.6f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0

    result = {
        "case": "couette_bbfix_grid_conv",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "tau": tau, "nu": float(nu), "u_top": u_top,
        "H": H, "n_steps": n_steps, "warmup": warmup,
        "Cf_exact": float(Cf_exact),
        "bb_fix": True,
        "step_method": "lbm_step_correct_pattern (BB fix: f_pre)",
    }
    for fm in FORMULAS:
        cf_mean = sum(cf_hists[fm]) / max(len(cf_hists[fm]), 1) if cf_hists[fm] else float("nan")
        cf_err = abs(cf_mean - Cf_exact) / Cf_exact * 100 if Cf_exact > 0 and math.isfinite(cf_mean) else float("nan")
        result[f"Cf_mean_{fm}"] = float(cf_mean)
        result[f"Cf_err_pct_{fm}"] = float(cf_err)

    print(f"\n{tag} === FINAL (Cf_exact={Cf_exact:.6f}) ===", flush=True)
    for fm in FORMULAS:
        cf = result[f"Cf_mean_{fm}"]
        err = result[f"Cf_err_pct_{fm}"]
        print(f"{tag} {fm:12s}: Cf={cf:.6f} err={err:.2f}%", flush=True)
    print(f"{tag} time={elapsed:.0f}s", flush=True)

    result["elapsed_s"] = elapsed
    result["finite"] = bool(torch.isfinite(f).all().item())
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


if __name__ == "__main__":
    ny = int(sys.argv[1])
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]
    try:
        run_couette_bbfix(device_id, ny, output_path)
    except Exception as e:
        traceback.print_exc()
        Path(output_path).write_text(json.dumps({"error": str(e), "ny": ny, "device": device_id}))
