"""Cross-validation: BFL+momentum-exchange vs BB+pressure+friction on cylinder D=100.

Method A (SDAA:0): BFL interpolated bounce-back + momentum exchange (Ladd 1994)
Method B (SDAA:1): Standard half-way bounce-back + pressure integration + friction

Both should give Cd ≈ 1.30 at Re=200, D=100, fine grid.

Usage:
    PYTHONPATH=src python cylinder_drag_crossval_worker.py <device_id> <method> <output_json>
    method: A or B
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from tensorlbm.d3q19 import C, OPPOSITE, W, equilibrium3d, macroscopic3d
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.bfl_d3q19 import compute_q_cylinder_d3q19, bouzidi_bounce_back_d3q19
from tensorlbm.solver3d import stream3d, correct_mass3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_momentum import drag_momentum_exchange, drag_momentum_exchange_vec
from tensorlbm.drag_pressure import (
    SurfaceMesh, drag_total, drag_pressure_integration, drag_friction_integration,
    get_near_wall_2d,
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


def compute_me_bfl(f_post_bfl, f_pre_stream, fluid_boundary_mask, q_field, solid, device):
    """BFL momentum exchange: F = Σ (f_pre[d] + f_bc) * c_d / q.

    Uses the BFL-interpolated f_bc (already applied to f_post_bfl[opp_d])
    and pre-stream f_pre[d] at the fluid boundary cell.

    This is the correct ME formula for BFL (Bouzidi-Firdaouss-Lallemand).
    """
    c = C.to(device).float()
    opp = OPPOSITE.to(device)
    fx = torch.tensor(0.0, device=device)

    for d in range(1, 19):
        opp_d = int(opp[d].item())
        mask = fluid_boundary_mask[d]
        if not mask.any():
            continue

        q_cell = q_field[d][mask].clamp(min=0.01)
        fp_d = f_pre_stream[d][mask]
        f_bc = f_post_bfl[opp_d][mask]

        c_d_x = float(c[d, 0].item())
        contrib = (fp_d + f_bc) * c_d_x / q_cell
        fx = fx + contrib.sum()

    return float(fx.item())


def run_method_a(device_id, output_path):
    """Method A: BFL + Momentum Exchange on cylinder D=100 Re=200."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # Parameters
    nx, ny, nz = 830, 330, 4
    diameter = 100.0
    radius = diameter / 2.0
    u_in = 0.08
    Re = 200.0
    nu = u_in * diameter / Re
    tau = 3.0 * nu + 0.5
    cs_smag = 0.05
    n_steps = 3000
    warmup = 300  # warmup steps, average over steps warmup+1..n_steps

    cx_c = nx * 0.25   # cylinder center x
    cy_c = ny * 0.5    # cylinder center y

    A_frontal = diameter * nz
    dpS = 0.5 * 1.0 * u_in ** 2 * A_frontal

    tag = f"[MethodA BFL+ME SDAA:{device_id}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} D={diameter} u_in={u_in} "
          f"nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} dpS={dpS:.4f}", flush=True)

    t0 = time.time()

    # Build cylinder mask
    solid = build_cylinder_mask(nx, ny, nz, cx_c, cy_c, radius, device)
    n_solid = int(solid.sum().item())
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Near-wall mask
    near = get_near_wall_2d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} solid cells={n_solid} near-wall cells={n_near}", flush=True)

    # BFL q-values
    fbm, qf = compute_q_cylinder_d3q19(nx, ny, nz, cx_c, cy_c, radius, device)
    n_links = int(fbm.sum().item())
    q_min = float(qf[fbm].min().item())
    q_max = float(qf[fbm].max().item())
    q_mean = float(qf[fbm].mean().item())
    print(f"{tag} BFL: {n_links} links, q=[{q_min:.4f},{q_max:.4f}], mean={q_mean:.4f}", flush=True)

    # Initialize
    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())

    print(f"{tag} init done ({time.time()-t0:.1f}s), starting loop...", flush=True)

    # Accumulators
    cd_me_bfl_hist = []    # BFL-specific ME
    cd_me_std_hist = []    # Standard ME formula (for comparison)

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision
        f_pre = f.clone()

        # 2. Collision (MRT + Smagorinsky)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 3. NoDynamics: restore solid cells
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Save post-collision (pre-stream) for BFL
        f_post_coll = f.clone()

        # 5. Stream
        f = stream3d(f)

        # 6. Far-field BC
        f = far_field_bc_3d(f, u_in=u_in)

        # 7. BFL (after stream — correct timing, BFL reconstructs post-stream unknowns)
        f = bouzidi_bounce_back_d3q19(f, f_post_coll, fbm, qf)

        # 8. Momentum exchange (post-BFL, post-stream)
        #    a) BFL-specific ME (correct formula for BFL)
        fx_bfl = compute_me_bfl(f, f_post_coll, fbm, qf, solid, device)
        cd_me_bfl = fx_bfl / dpS

        #    b) Standard ME formula (Ladd 1994, for comparison)
        cd_me_std = drag_momentum_exchange(f, near, solid, dpS)

        # 9. Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # Check divergence
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        # Record post-warmup
        if step > warmup:
            if math.isfinite(cd_me_bfl):
                cd_me_bfl_hist.append(cd_me_bfl)
            if math.isfinite(cd_me_std):
                cd_me_std_hist.append(cd_me_std)

        if step % 200 == 0:
            _, ux, _, _ = macroscopic3d(f)
            ms = float(torch.sqrt(ux * ux).max().item())
            elapsed = time.time() - t0
            cd_avg_bfl = sum(cd_me_bfl_hist) / max(len(cd_me_bfl_hist), 1)
            cd_avg_std = sum(cd_me_std_hist) / max(len(cd_me_std_hist), 1)
            print(f"{tag} step={step} Cd_ME_BFL={cd_avg_bfl:.4f} Cd_ME_std={cd_avg_std:.4f} "
                  f"max|ux|={ms:.4f} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    ref_cd = 1.30

    cd_me_bfl_mean = sum(cd_me_bfl_hist) / max(len(cd_me_bfl_hist), 1) if cd_me_bfl_hist else float("nan")
    cd_me_std_mean = sum(cd_me_std_hist) / max(len(cd_me_std_hist), 1) if cd_me_std_hist else float("nan")

    def _std(vals):
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))

    cd_me_bfl_std = _std(cd_me_bfl_hist)
    cd_me_std_std = _std(cd_me_std_hist)

    err_bfl = abs(cd_me_bfl_mean - ref_cd) / ref_cd * 100 if math.isfinite(cd_me_bfl_mean) else float("nan")
    err_std = abs(cd_me_std_mean - ref_cd) / ref_cd * 100 if math.isfinite(cd_me_std_mean) else float("nan")

    print(f"\n{tag} === FINAL RESULTS ===", flush=True)
    print(f"{tag} Cd_ME_BFL = {cd_me_bfl_mean:.4f} ± {cd_me_bfl_std:.4f}  (err={err_bfl:.2f}%)", flush=True)
    print(f"{tag} Cd_ME_std = {cd_me_std_mean:.4f} ± {cd_me_std_std:.4f}  (err={err_std:.2f}%)", flush=True)
    print(f"{tag} ref Cd = {ref_cd}", flush=True)
    print(f"{tag} time = {elapsed:.0f}s", flush=True)

    result = {
        "method": "A_BFL_momentum_exchange",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "diameter": diameter,
        "Re": Re,
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Cs": cs_smag,
        "n_steps": n_steps,
        "warmup": warmup,
        "Cd_ME_BFL_mean": cd_me_bfl_mean,
        "Cd_ME_BFL_std": cd_me_bfl_std,
        "Cd_ME_BFL_err_pct": err_bfl,
        "Cd_ME_std_mean": cd_me_std_mean,
        "Cd_ME_std_std": cd_me_std_std,
        "Cd_ME_std_err_pct": err_std,
        "Cd_ref": ref_cd,
        "n_samples": len(cd_me_bfl_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    return result


def run_method_b(device_id, output_path):
    """Method B: Standard BB + Pressure + Friction on cylinder D=100 Re=200."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # Parameters
    nx, ny, nz = 830, 330, 4
    diameter = 100.0
    radius = diameter / 2.0
    u_in = 0.08
    Re = 200.0
    nu = u_in * diameter / Re
    tau = 3.0 * nu + 0.5
    cs_smag = 0.05
    n_steps = 3000
    warmup = 300  # warmup steps, average over steps warmup+1..n_steps

    cx_c = nx * 0.25
    cy_c = ny * 0.5

    A_frontal = diameter * nz
    dpS = 0.5 * 1.0 * u_in ** 2 * A_frontal

    tag = f"[MethodB BB+PF SDAA:{device_id}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} D={diameter} u_in={u_in} "
          f"nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} dpS={dpS:.4f}", flush=True)

    t0 = time.time()

    # Build cylinder mask
    solid = build_cylinder_mask(nx, ny, nz, cx_c, cy_c, radius, device)
    n_solid = int(solid.sum().item())
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Near-wall mask
    near = get_near_wall_2d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} solid cells={n_solid} near-wall cells={n_near}", flush=True)

    # Surface mesh for pressure/friction drag
    mesh = SurfaceMesh.from_cylinder(solid, near, cx_c, cy_c, radius)
    print(f"{tag} SurfaceMesh built (analytical cylinder normal)", flush=True)

    # Initialize
    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())

    print(f"{tag} init done ({time.time()-t0:.1f}s), starting loop...", flush=True)

    # Accumulators
    cd_p_hist = []
    cd_f_hist = []
    cd_tot_hist = []

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

        # 6. Far-field BC
        f = far_field_bc_3d(f, u_in=u_in)

        # 7. Drag (post-stream, post-BC — physical state)
        cd_tot, cd_p, cd_f = drag_total(f, mesh, dpS, nu)

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
                cd_p_hist.append(cd_p)
                cd_f_hist.append(cd_f)
                cd_tot_hist.append(cd_tot)

        if step % 200 == 0:
            _, ux, _, _ = macroscopic3d(f)
            ms = float(torch.sqrt(ux * ux).max().item())
            elapsed = time.time() - t0
            cd_avg = sum(cd_tot_hist) / max(len(cd_tot_hist), 1)
            print(f"{tag} step={step} Cd_tot={cd_avg:.4f} "
                  f"(p={sum(cd_p_hist)/max(len(cd_p_hist),1):.4f} "
                  f"f={sum(cd_f_hist)/max(len(cd_f_hist),1):.4f}) "
                  f"max|ux|={ms:.4f} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    ref_cd = 1.30

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

    err_tot = abs(cd_tot_mean - ref_cd) / ref_cd * 100 if math.isfinite(cd_tot_mean) else float("nan")

    print(f"\n{tag} === FINAL RESULTS ===", flush=True)
    print(f"{tag} Cd_p   = {cd_p_mean:.4f} ± {cd_p_std:.4f}", flush=True)
    print(f"{tag} Cd_f   = {cd_f_mean:.4f} ± {cd_f_std:.4f}", flush=True)
    print(f"{tag} Cd_tot = {cd_tot_mean:.4f} ± {cd_tot_std:.4f}  (err={err_tot:.2f}%)", flush=True)
    print(f"{tag} ref Cd = {ref_cd}", flush=True)
    print(f"{tag} time = {elapsed:.0f}s", flush=True)

    result = {
        "method": "B_BB_pressure_friction",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "diameter": diameter,
        "Re": Re,
        "u_in": u_in,
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
        "Cd_tot_err_pct": err_tot,
        "Cd_ref": ref_cd,
        "n_samples": len(cd_tot_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    device_id = int(sys.argv[1])
    method = sys.argv[2].upper()
    output_path = sys.argv[3]

    if method == "A":
        run_method_a(device_id, output_path)
    elif method == "B":
        run_method_b(device_id, output_path)
    else:
        print(f"Unknown method: {method}. Use A or B.")
        sys.exit(1)
