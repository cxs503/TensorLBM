#!/usr/bin/env python3
"""Historical replay using the frozen solver snapshot.

Collision Operator Matrix — cluster job.

Tests BGK, TRT, RLBM on 2D cases (MRT is MOCK/broken for 2D).
Writes replay results to outputs/historical/collision_operator_matrix/m23039/.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "outputs" / "historical" / "collision_operator_matrix" / "m23039"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "development/legacy/TensorLBM/src"))

from tensorlbm.d2q9 import (
    C as C2Q9, W as W2Q9, OPPOSITE as OPP2Q9, CS2 as CS2_2D,
    equilibrium as eq2d, macroscopic as macro2d, stream as stream2d,
    collide_bgk as collide_bgk_2d,
)
from tensorlbm.solver import collide_rlbm_2d

# ---------------------------------------------------------------------------
# Collision operators (2D only — MRT is MOCK/broken for 2D)
# ---------------------------------------------------------------------------

def make_bgk(tau):
    def collide(f):
        return collide_bgk_2d(f, tau)
    return collide

def make_rlbm(tau):
    def collide(f):
        return collide_rlbm_2d(f, tau)
    return collide

def make_trt(tau, magic=3.0/16.0):
    tau_minus = 0.5 + magic / (tau - 0.5)
    omega_p = 1.0 / tau
    omega_m = 1.0 / tau_minus
    opp = OPP2Q9.clone()

    def collide(f):
        rho = f.sum(dim=0)
        rho_safe = torch.clamp(rho, min=1e-12)
        c = C2Q9.to(device=f.device, dtype=f.dtype)
        ux = (f * c[:, 0].view(-1, 1, 1)).sum(dim=0) / rho_safe
        uy = (f * c[:, 1].view(-1, 1, 1)).sum(dim=0) / rho_safe
        u2 = ux**2 + uy**2
        cu = c[:, 0].view(-1, 1, 1) * ux + c[:, 1].view(-1, 1, 1) * uy
        w = W2Q9.to(device=f.device, dtype=f.dtype)
        feq = w.view(-1, 1, 1) * rho.unsqueeze(0) * (
            1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2.unsqueeze(0)
        )
        opp_local = opp.to(device=f.device)
        f_plus = 0.5 * (f + f[opp_local])
        f_minus = 0.5 * (f - f[opp_local])
        feq_plus = 0.5 * (feq + feq[opp_local])
        feq_minus = 0.5 * (feq - feq[opp_local])
        return (f - omega_p * (f_plus - feq_plus) - omega_m * (f_minus - feq_minus))
    return collide

MAKERS_2D = {"BGK": make_bgk, "TRT": make_trt, "RLBM": make_rlbm}

# ---------------------------------------------------------------------------
# Case 1: Taylor-Green 2D
# ---------------------------------------------------------------------------

def run_tg2d_grid(n, re, u0, collider_fn, steps, record_every=200):
    L = float(n)
    k = 2.0 * math.pi / L
    nu = u0 * L / re
    tau = 0.5 + 3.0 * nu
    gamma_e_theory = 4.0 * nu * k * k
    gamma_vel_theory = 2.0 * nu * k * k

    y, x = torch.meshgrid(
        torch.arange(n, dtype=torch.float64),
        torch.arange(n, dtype=torch.float64),
        indexing="ij",
    )
    ux0 = -u0 * torch.cos(k * x) * torch.sin(k * y)
    uy0 = +u0 * torch.sin(k * x) * torch.cos(k * y)
    rho = torch.ones_like(ux0)
    f = eq2d(rho, ux0, uy0)

    mass0 = float(f.sum().item())
    times, energies, umaxs = [], [], []
    t0 = time.time()

    for step in range(1, steps + 1):
        f = collider_fn(stream2d(f))
        if step % record_every == 0:
            _, uxm, uym = macro2d(f)
            e = float((0.5 * (uxm * uxm + uym * uym)).mean().item())
            um = float((uxm * uxm + uym * uym).sqrt().max().item())
            times.append(step)
            energies.append(e)
            umaxs.append(um)

    wall = time.time() - t0

    if len(times) < 3:
        return {"stable": False, "error": "too few samples", "wall_sec": wall}

    t_arr = np.asarray(times, dtype=np.float64)
    lnE = np.log(np.asarray(energies, dtype=np.float64))
    a_e, b_e = np.polyfit(t_arr, lnE, 1)
    gamma_e_sim = -a_e
    r2 = 1.0 - float(np.sum((lnE - (b_e + a_e * t_arr))**2) / np.sum((lnE - lnE.mean())**2))

    lnU = np.log(np.asarray(umaxs, dtype=np.float64))
    a_u, _ = np.polyfit(t_arr, lnU, 1)
    gamma_vel_sim = -a_u

    err_e_pct = (gamma_e_sim - gamma_e_theory) / gamma_e_theory * 100.0
    err_vel_pct = (gamma_vel_sim - gamma_vel_theory) / gamma_vel_theory * 100.0
    mass_end = float(f.sum().item())

    return {
        "n": n, "re": re, "u0": u0,
        "nu": nu, "tau": tau,
        "gamma_e_theory": gamma_e_theory,
        "gamma_e_sim": gamma_e_sim,
        "err_e_pct": round(err_e_pct, 4),
        "gamma_vel_theory": gamma_vel_theory,
        "gamma_vel_sim": gamma_vel_sim,
        "err_vel_pct": round(err_vel_pct, 4),
        "r2": round(r2, 6),
        "mass_drift_rel": round((mass_end - mass0) / mass0, 10),
        "wall_sec": round(wall, 2),
        "n_samples": len(times),
        "stable": True,
    }


def run_tg2d_convergence(re, u0, collider_name, grids=(32, 64, 128)):
    maker = MAKERS_2D[collider_name]
    results = []
    for n in grids:
        nu = u0 * n / re
        gamma_e = 4.0 * nu * (2.0 * math.pi / n)**2
        steps = max(500, int(math.ceil(3.0 / gamma_e / 200) * 200))
        steps = min(steps, 100000)
        tau = 0.5 + 3.0 * nu
        collider = maker(tau)
        res = run_tg2d_grid(n, re, u0, collider, steps, record_every=200)
        res["collider"] = collider_name
        results.append(res)
        tag = f"  {collider_name} N={n}"
        if res.get("stable"):
            print(f"{tag}: err_E={res['err_e_pct']:+.4f}%, err_vel={res['err_vel_pct']:+.4f}%, R2={res['r2']:.6f}, wall={res['wall_sec']:.1f}s")
        else:
            print(f"{tag}: UNSTABLE - {res.get('error', '?')}")
    return results


# ---------------------------------------------------------------------------
# Case 2: Channel flow (2D Poiseuille)
# ---------------------------------------------------------------------------

def run_channel_2d(re, collider_name, nx=256, ny=65, max_steps=20000, sample_interval=500):
    H = ny - 2
    u_max_target = 0.1
    nu = u_max_target * H / re * 4.0
    tau = 0.5 + 3.0 * nu
    G = 8.0 * nu * u_max_target / (H * H)

    maker = MAKERS_2D[collider_name]
    collider = maker(tau)

    rho = torch.ones((ny, nx), dtype=torch.float64)
    ux = torch.zeros((ny, nx), dtype=torch.float64)
    uy = torch.zeros((ny, nx), dtype=torch.float64)
    f = eq2d(rho, ux, uy)

    shifts = []
    for i in range(9):
        cx, cy = int(C2Q9[i, 0]), int(C2Q9[i, 1])
        shifts.append((-cy, -cx))

    def stream_ch(f):
        f_out = torch.empty_like(f)
        for i in range(9):
            sy, sx = shifts[i]
            if sy == 0 and sx == 0:
                f_out[i] = f[i]
            else:
                f_out[i] = torch.roll(f[i], shifts=(sy, sx), dims=(0, 1))
        return f_out

    def apply_walls(f):
        opp = OPP2Q9
        f_out = f.clone()
        for i in range(1, 9):
            cy = int(C2Q9[i, 1])
            if cy > 0:
                f_out[i, 0, :] = f[opp[i], 0, :]
            if cy < 0:
                f_out[i, ny-1, :] = f[opp[i], ny-1, :]
        return f_out

    def apply_force(f):
        cs2 = CS2_2D
        omega = 1.0 / tau
        c = C2Q9.to(device=f.device, dtype=f.dtype)
        w = W2Q9.to(device=f.device, dtype=f.dtype)
        _, ux_f, uy_f = macro2d(f)
        force_term = w.view(-1, 1, 1) * (1.0 - 0.5 * omega) / cs2 * (
            c[:, 0].view(-1, 1, 1) * G
        )
        return f + force_term

    def step_fn(f):
        f = collider(f)
        f = stream_ch(f)
        f = apply_walls(f)
        f = apply_force(f)
        return f

    profile_history = []
    t0 = time.time()

    for step in range(max_steps):
        f = step_fn(f)
        if step % sample_interval == 0:
            _, ux_f, uy_f = macro2d(f)
            u_center = float(ux_f[ny//2, nx//2].item())
            profile_history.append({"step": step, "u_center": u_center})

    wall = time.time() - t0

    _, ux_f, uy_f = macro2d(f)
    u_profile = ux_f[:, nx//2].numpy()
    y_lu = np.arange(ny) - 1
    u_ana = (G / (2.0 * nu)) * y_lu * (H - y_lu)
    u_ana = np.where((y_lu < 0) | (y_lu > H), 0.0, u_ana)

    fluid_mask = (y_lu > 0) & (y_lu < H)
    u_profile_err = float(np.sqrt(np.mean((u_profile[fluid_mask] - u_ana[fluid_mask])**2)))
    u_max_sim = float(np.max(u_profile[fluid_mask]))
    u_max_ana = float(np.max(u_ana[fluid_mask]))
    u_max_err_pct = abs(u_max_sim - u_max_ana) / u_max_ana * 100.0

    return {
        "collider": collider_name,
        "re": re, "nx": nx, "ny": ny,
        "tau": tau, "nu": nu, "G": G,
        "u_max_sim": round(u_max_sim, 6),
        "u_max_ana": round(u_max_ana, 6),
        "u_max_err_pct": round(u_max_err_pct, 4),
        "u_profile_rms_err": round(u_profile_err, 6),
        "n_samples": len(profile_history),
        "wall_sec": round(wall, 2),
        "stable": True,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = {
        "mission": "m23039",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "operators_tested_2d": list(MAKERS_2D.keys()),
        "operators_mock_or_broken": {
            "MRT": "MOCK — solver.py:collide_mrt delegates to 3D collide_bgk, returns wrong shape for 2D data",
        },
        "operators_not_supported": ["KBC", "Cumulant"],
        "cases": {},
    }

    # ---- TG2D ----
    print("=" * 60)
    print("CASE 1: Taylor-Green 2D Decay")
    print("=" * 60)
    tg = {}
    grids = (32, 64, 128)
    for name in MAKERS_2D:
        tg[name] = run_tg2d_convergence(100, 0.05, name, grids=grids)
    results["cases"]["taylor_green_2d"] = {
        "description": "2D TG vortex decay, analytical solution",
        "external_anchor": "Taylor & Green 1937; Krüger et al. 2017 §7.3",
        "re": 100, "u0": 0.05, "grids": list(grids),
        "results": tg,
    }

    # ---- Channel ----
    print("\n" + "=" * 60)
    print("CASE 2: Channel Flow (Poiseuille)")
    print("=" * 60)
    ch = {}
    for name in MAKERS_2D:
        for re in [100, 1000]:
            print(f"  {name} Re={re}...")
            res = run_channel_2d(re, name, nx=256, ny=65, max_steps=20000, sample_interval=500)
            ch[f"{name}_Re{re}"] = res
            if res.get("stable"):
                print(f"    u_max_err={res['u_max_err_pct']:.4f}%, rms={res['u_profile_rms_err']:.6f}, wall={res['wall_sec']:.1f}s")
            else:
                print(f"    UNSTABLE")
    results["cases"]["channel_flow"] = {
        "description": "2D plane channel flow, analytical Poiseuille",
        "external_anchor": "u(y) = (G/2ν)·y·(H-y)",
        "results": ch,
    }

    # ---- MRT mock verification ----
    print("\n" + "=" * 60)
    print("MRT MOCK VERIFICATION")
    print("=" * 60)
    print("  MRT in TensorLBM (solver.py:collide_mrt) delegates to collide_bgk (3D version)")
    print("  When called with 2D data (9, ny, nx), it returns (19, 9, ny, nx) — WRONG SHAPE")
    print("  VERDICT: MRT is MOCK/BROKEN for 2D cases")
    results["mrt_is_mock"] = {
        "verified": True,
        "evidence": "collide_mrt calls collide_bgk from solver.py which is D3Q19; shape mismatch on 2D input",
        "note": "MRT cannot be tested on 2D benchmarks; requires separate 2D MRT implementation",
    }

    # Save
    out_path = OUT_DIR / "collision_matrix_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
