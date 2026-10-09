#!/usr/bin/env python3
"""Friction formula grid convergence study — 4 formulas × 3 geometries.

Tests which friction formula converges with grid refinement:
  a) 'standard'   : τ = 2ν·u_1            (1st-order forward, Δn=0.5)
  b) '2nd_order'  : τ = ν·(3u_1 − u_2)     (task-specified 2nd-order)
  c) 'central'    : τ = ν·u_2              (task-specified central diff)
  d) 'lagrange'   : τ = ν·(3u_1 − u_2/3)   (exact 2nd-order, non-uniform grid)

All 4 formulas are computed from the SAME simulation (post-processing only),
so only 1 simulation per grid size is needed.

TEST 1: SUBOFF  (SDAA:24-25)  L=40/80/160, Re=1000, MRT+Smag
TEST 2: Sphere  (SDAA:26)      D=20/40,    Re=100
TEST 3: Couette (SDAA:27)     ny=8/16/32, tau=1.0

Usage:
  PYTHONPATH=src python friction_formula_conv_worker.py suboff  <L> <device_id> <out>
  PYTHONPATH=src python friction_formula_conv_worker.py sphere  <D> <device_id> <out>
  PYTHONPATH=src python friction_formula_conv_worker.py couette <ny> <device_id> <out>
"""
from __future__ import annotations
import sys, json, math, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import stream3d, correct_mass3d
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d, sphere_mask
from tensorlbm.turbulence import collide_smagorinsky_mrt3d, collide_smagorinsky_bgk3d
from tensorlbm.drag_pressure import (
    SurfaceMesh, get_near_wall_3d, get_near_wall_2d,
    drag_pressure_integration, drag_friction_integration,
)
from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig

FORMULAS = ['standard', '2nd_order', 'central', 'lagrange']


# ---------------------------------------------------------------------------
# TEST 1: SUBOFF grid convergence
# ---------------------------------------------------------------------------
def run_suboff(device_id, L, output_path):
    """SUBOFF bare-hull drag at Re=1000, 4 friction formulas."""
    device = torch.device("sdaa:0")  # SDAA_VISIBLE_DEVICES remaps to 0
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
        n_steps = 5000
    elif L == 80:
        nx, ny, nz = 200, 80, 80
        n_steps = 5000
    elif L == 160:
        nx, ny, nz = 300, 120, 120  # reduced from 400³ to avoid OOM
        n_steps = 2000  # reduced for large grid
    else:
        raise ValueError(f"L must be 40/80/160, got {L}")

    cx = nx * 0.30
    cy = ny * 0.5
    cz = nz * 0.5
    nu = u_in * L / Re
    tau = 3.0 * nu + 0.5
    dpS = 0.5 * u_in ** 2 * math.pi * D * L
    Cf_ref = 1.328 / math.sqrt(Re)

    tag = f"[SDAA:{device_id} SUBOFF L={L} Re=1000]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} L={L} D={D:.3f} "
          f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} "
          f"dpS={dpS:.6e} Cf_ref={Cf_ref:.6f}", flush=True)

    t0 = time.time()
    solid, stats = build_suboff_mask(
        hull_type="bare_hull", nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz, length=L, radius=radius,
        config=config, device=device,
    )
    n_solid = int(solid.sum().item())
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    mesh = SurfaceMesh.from_gradient(solid, near)
    print(f"{tag} solid={n_solid} near={n_near} ({time.time()-t0:.1f}s)", flush=True)

    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    # History: one list per formula
    cd_p_hist = []
    cd_f_hists = {fm: [] for fm in FORMULAS}
    cd_tot_hists = {fm: [] for fm in FORMULAS}

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

        fx_p, fy_p, fz_p = drag_pressure_integration(f, mesh, dpS)
        cd_p_hist.append(fx_p)
        for fm in FORMULAS:
            fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu, formula=fm)
            cd_f_hists[fm].append(fx_f)
            cd_tot_hists[fm].append(fx_p + fx_f)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0:
            n_avg = min(500, len(cd_p_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
            parts = " ".join(
                f"Cd_f[{fm}]={sum(cd_f_hists[fm][-n_avg:])/n_avg:.6f}" for fm in FORMULAS
            )
            print(f"{tag} step={step} Cd_p={cd_p_avg:.6f} {parts} ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_final = min(win, len(cd_p_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / n_final

    results = {
        "case": "suboff_friction_formula_conv",
        "device": f"sdaa:{device_id}",
        "Re": Re, "L": L, "D": D, "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in, "nu": nu, "tau": tau, "Cs": cs_smag,
        "n_steps": n_steps, "win": win,
        "n_solid": n_solid, "n_near": n_near, "dpS": dpS,
        "Cf_ref": Cf_ref,
        "Cd_pressure": cd_p_final,
    }
    for fm in FORMULAS:
        cd_f = sum(cd_f_hists[fm][-n_final:]) / n_final
        cd_tot = sum(cd_tot_hists[fm][-n_final:]) / n_final
        results[f"Cd_friction_{fm}"] = cd_f
        results[f"Cd_total_{fm}"] = cd_tot
        results[f"err_pct_{fm}"] = abs(cd_tot - Cf_ref) / Cf_ref * 100

    print(f"\n{tag} === FINAL (Cf_ref={Cf_ref:.6f}) ===", flush=True)
    print(f"{tag} Cd_p = {cd_p_final:.6f}", flush=True)
    for fm in FORMULAS:
        cd_f = results[f"Cd_friction_{fm}"]
        cd_tot = results[f"Cd_total_{fm}"]
        err = results[f"err_pct_{fm}"]
        print(f"{tag} {fm:12s}: Cd_f={cd_f:.6f} Cd_tot={cd_tot:.6f} err={err:.1f}%", flush=True)
    print(f"{tag} time={elapsed:.0f}s", flush=True)

    results["elapsed_s"] = elapsed
    results["finite"] = bool(torch.isfinite(f).all().item())
    Path(output_path).write_text(json.dumps(results, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return results


# ---------------------------------------------------------------------------
# TEST 2: Sphere grid convergence
# ---------------------------------------------------------------------------
def run_sphere(device_id, D, output_path):
    """Sphere drag at Re=100, 4 friction formulas."""
    device = torch.device("sdaa:0")  # SDAA_VISIBLE_DEVICES remaps to 0
    torch.sdaa.set_device(device)

    Re = 100
    u_in = 0.08
    cs_smag = 0.05
    n_steps = 3000
    win = 500

    grid_map = {20: 120, 40: 180}
    n = grid_map[D]
    nx = ny = nz = n
    R = D / 2.0
    cx = nx * 0.25
    cy = ny * 0.5
    cz = nz * 0.5
    nu = u_in * D / Re
    tau = 3.0 * u_in * D / Re + 0.5
    dpS = 0.5 * u_in ** 2 * math.pi * R ** 2
    Cd_ref = 1.09

    tag = f"[SDAA:{device_id} Sphere D={D} Re=100]"
    print(f"{tag} nx={nx} D={D} R={R} u_in={u_in} nu={nu:.6e} tau={tau:.6f} "
          f"dpS={dpS:.6e} Cd_ref={Cd_ref}", flush=True)

    t0 = time.time()
    solid = sphere_mask(nx, ny, nz, cx, cy, cz, R, device)
    n_solid = int(solid.sum().item())
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    mesh = SurfaceMesh.from_sphere(solid, near, cx, cy, cz, R)
    print(f"{tag} solid={n_solid} near={n_near} ({time.time()-t0:.1f}s)", flush=True)

    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    cd_p_hist = []
    cd_f_hists = {fm: [] for fm in FORMULAS}
    cd_tot_hists = {fm: [] for fm in FORMULAS}

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

        fx_p, fy_p, fz_p = drag_pressure_integration(f, mesh, dpS)
        cd_p_hist.append(fx_p)
        for fm in FORMULAS:
            fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu, formula=fm)
            cd_f_hists[fm].append(fx_f)
            cd_tot_hists[fm].append(fx_p + fx_f)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0:
            n_avg = min(500, len(cd_p_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
            parts = " ".join(
                f"Cd_f[{fm}]={sum(cd_f_hists[fm][-n_avg:])/n_avg:.4f}" for fm in FORMULAS
            )
            print(f"{tag} step={step} Cd_p={cd_p_avg:.4f} {parts} ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_final = min(win, len(cd_p_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / n_final

    results = {
        "case": "sphere_friction_formula_conv",
        "device": f"sdaa:{device_id}",
        "Re": Re, "D": D, "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in, "nu": nu, "tau": tau, "Cs": cs_smag,
        "n_steps": n_steps, "win": win,
        "n_solid": n_solid, "n_near": n_near, "dpS": dpS,
        "Cd_ref": Cd_ref,
        "Cd_pressure": cd_p_final,
    }
    for fm in FORMULAS:
        cd_f = sum(cd_f_hists[fm][-n_final:]) / n_final
        cd_tot = sum(cd_tot_hists[fm][-n_final:]) / n_final
        results[f"Cd_friction_{fm}"] = cd_f
        results[f"Cd_total_{fm}"] = cd_tot
        results[f"err_pct_{fm}"] = abs(cd_tot - Cd_ref) / Cd_ref * 100

    print(f"\n{tag} === FINAL (Cd_ref={Cd_ref}) ===", flush=True)
    print(f"{tag} Cd_p = {cd_p_final:.4f}", flush=True)
    for fm in FORMULAS:
        cd_f = results[f"Cd_friction_{fm}"]
        cd_tot = results[f"Cd_total_{fm}"]
        err = results[f"err_pct_{fm}"]
        print(f"{tag} {fm:12s}: Cd_f={cd_f:.4f} Cd_tot={cd_tot:.4f} err={err:.1f}%", flush=True)
    print(f"{tag} time={elapsed:.0f}s", flush=True)

    results["elapsed_s"] = elapsed
    results["finite"] = bool(torch.isfinite(f).all().item())
    Path(output_path).write_text(json.dumps(results, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return results


# ---------------------------------------------------------------------------
# TEST 3: Couette grid convergence
# ---------------------------------------------------------------------------
def _make_channel_solid(nz, ny, nx, device):
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    return solid


def _near_wall_bottom_only(near, ny):
    near_b = near.clone()
    near_b[:, ny - 2, :] = False
    return near_b


def _moving_wall_bounce_back_3d(f, solid, top_wall_mask, u_top, rho_w=1.0):
    from tensorlbm.d3q19 import C, W, OPPOSITE
    opp = OPPOSITE.to(f.device)
    f = torch.where(solid.unsqueeze(0), f[opp], f)
    c = C.to(f.device).float()
    w = W.to(f.device).float()
    correction = 6.0 * rho_w * u_top * w * c[:, 0]
    top_mask = top_wall_mask.unsqueeze(0).float()
    f = f + correction.view(19, 1, 1, 1) * top_mask
    return f


def run_couette(device_id, ny, output_path):
    """3D Couette: moving top wall. 4 friction formulas on bottom wall."""
    device = torch.device("sdaa:0")  # SDAA_VISIBLE_DEVICES remaps to 0
    torch.sdaa.set_device(device)

    nx, nz = 80, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0  # = 1/6
    u_top = 0.05
    n_steps = 4000
    warmup = 500

    H = ny - 2
    Cf_exact = 2.0 * nu / (H * u_top)

    tag = f"[SDAA:{device_id} Couette ny={ny}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} tau={tau} nu={nu:.6f} u_top={u_top} "
          f"H={H} Cf_exact={Cf_exact:.6f}", flush=True)

    t0 = time.time()
    solid = _make_channel_solid(nz, ny, nx, device)
    top_wall_mask = torch.zeros_like(solid)
    top_wall_mask[:, -1, :] = True

    near = get_near_wall_3d(solid)
    near_bottom = _near_wall_bottom_only(near, ny)
    mesh_bottom = SurfaceMesh.from_gradient(solid, near_bottom)
    print(f"{tag} near(bottom)={int(near_bottom.sum().item())} ({time.time()-t0:.1f}s)", flush=True)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    from tensorlbm.solver3d import collide_bgk3d
    cf_hists = {fm: [] for fm in FORMULAS}

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = _moving_wall_bounce_back_3d(f, solid, top_wall_mask, u_top)
        f = stream3d(f)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

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
                f"Cf[{fm}]={sum(cf_hists[fm])/max(len(cf_hists[fm]),1):.6f}" for fm in FORMULAS
            )
            print(f"{tag} step={step} {parts} u[1]={float(u_prof[1]):.6f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0

    results = {
        "case": "couette_friction_formula_conv",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "tau": tau, "nu": float(nu), "u_top": u_top,
        "H": H, "n_steps": n_steps, "warmup": warmup,
        "Cf_exact": float(Cf_exact),
    }
    for fm in FORMULAS:
        cf_mean = sum(cf_hists[fm]) / max(len(cf_hists[fm]), 1) if cf_hists[fm] else float("nan")
        cf_err = abs(cf_mean - Cf_exact) / Cf_exact * 100 if Cf_exact > 0 and math.isfinite(cf_mean) else float("nan")
        results[f"Cf_mean_{fm}"] = float(cf_mean)
        results[f"Cf_err_pct_{fm}"] = float(cf_err)

    print(f"\n{tag} === FINAL (Cf_exact={Cf_exact:.6f}) ===", flush=True)
    for fm in FORMULAS:
        cf = results[f"Cf_mean_{fm}"]
        err = results[f"Cf_err_pct_{fm}"]
        print(f"{tag} {fm:12s}: Cf={cf:.6f} err={err:.2f}%", flush=True)
    print(f"{tag} time={elapsed:.0f}s", flush=True)

    results["elapsed_s"] = elapsed
    results["finite"] = bool(torch.isfinite(f).all().item())
    Path(output_path).write_text(json.dumps(results, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 5:
        print("Usage: python friction_formula_conv_worker.py <case> <param> <device_id> <output>")
        print("  case: suboff | sphere | couette")
        print("  param: L(40/80/160) | D(20/40) | ny(8/16/32)")
        sys.exit(1)

    case = sys.argv[1]
    param = int(sys.argv[2])
    device_id = int(sys.argv[3])
    output_path = sys.argv[4]

    if case == "suboff":
        run_suboff(device_id, param, output_path)
    elif case == "sphere":
        run_sphere(device_id, param, output_path)
    elif case == "couette":
        run_couette(device_id, param, output_path)
    else:
        print(f"Unknown case: {case}")
        sys.exit(1)


if __name__ == "__main__":
    main()
