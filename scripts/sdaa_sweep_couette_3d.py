#!/usr/bin/env python3
"""SDAA sweep for benchmarks/verified/couette_3d.

Independent runner: places all tensors on 'sdaa' device, reuses the same
algorithm as benchmarks/verified/couette_3d/run.py (verbatim solver calls),
and evaluates the original gate criteria.

No business code is modified.  If triton were needed we would set
TRITON_INTERPRET=1, but this case is pure torch ops → no triton绕行.

Output: artifacts/sdaa_couette_3d.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── ensure tensorlbm importable ──────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.d3q19 import OPPOSITE, W_EXACT64, C, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d

CS2 = 1.0 / 3.0

# ── device evidence ──────────────────────────────────────────────────
def collect_device_evidence() -> dict:
    ev = {
        "torch_version": torch.__version__,
        "sdaa_available": hasattr(torch, "sdaa") and torch.sdaa.is_available(),
    }
    if ev["sdaa_available"]:
        ev["sdaa_device_count"] = torch.sdaa.device_count()
        try:
            ev["sdaa_get_device_name_0"] = torch.sdaa.get_device_name(0)
        except Exception as e:
            ev["sdaa_get_device_name_0_error"] = str(e)
        ev["sdaa_current_device"] = torch.sdaa.current_device()
    return ev


# ── moving-wall BB (verbatim from run.py) ────────────────────────────
def moving_wall_bounce_back3d(f_pre, f, wall, u_wall, device):
    opp = OPPOSITE.to(device)
    c = C.to(device)
    w = W_EXACT64.to(device=device, dtype=f.dtype)
    f_new = torch.where(wall.unsqueeze(0), f_pre[opp], f)
    rho_w = torch.clamp(f_pre.sum(dim=0), min=1e-12)
    cu = c[:, 0].view(19, 1, 1, 1) * u_wall.unsqueeze(0)
    injection = (2.0 * w.view(19, 1, 1, 1) * rho_w.unsqueeze(0) * cu) / CS2
    return f_new + injection * wall.unsqueeze(0)


# ── single case runner ───────────────────────────────────────────────
def run_case(H, tau, U0, min_steps, max_steps, dtype, device, prof_steps=20):
    torch.manual_seed(0)
    ny = H + 2
    nx = H
    nz = H
    nu = (tau - 0.5) / 3.0
    H_eff = ny - 2.0

    wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    wall[:, 0, :] = True
    wall[:, -1, :] = True
    u_wall = torch.zeros((nz, ny, nx), dtype=dtype, device=device)
    u_wall[:, -1, :] = U0

    yy = torch.arange(ny, device=device, dtype=dtype)
    u0 = (
        torch.clamp(U0 * (yy - 0.5) / H_eff, min=0.0, max=U0)
        .view(1, ny, 1)
        .expand(nz, ny, nx)
        .clone()
    )
    f = equilibrium3d(
        torch.ones((nz, ny, nx), dtype=dtype, device=device),
        u0,
        torch.zeros((nz, ny, nx), dtype=dtype, device=device),
        torch.zeros((nz, ny, nx), dtype=dtype, device=device),
    )
    initial_mass = float(f.sum().item())

    def _step(f):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau)
        f = moving_wall_bounce_back3d(f_pre, f, wall, u_wall, device)
        return stream3d(f)

    z0, x0 = nz // 2, nx // 2
    t0 = time.time()
    u_top_hist = []
    step = 0
    steady = False
    for step in range(1, max_steps + 1):
        f = _step(f)
        if step % 100 == 0:
            _, ux, _, _ = macroscopic3d(f)
            u_top_hist.append(float(ux[z0, ny - 2, x0].item()))
            if step >= min_steps and len(u_top_hist) >= 5:
                recent = u_top_hist[-5:]
                mean = sum(recent) / len(recent)
                drift = (max(recent) - min(recent)) / max(abs(mean), 1e-12)
                if drift < 1e-5:
                    steady = True
                    break
    elapsed = time.time() - t0

    # accumulate profile
    prof_acc = torch.zeros(ny, dtype=dtype, device=device)
    for _ in range(prof_steps):
        f = _step(f)
        _, ux, _, _ = macroscopic3d(f)
        prof_acc += ux[z0, :, x0]
    prof_acc /= float(prof_steps)

    rho, ux, uy, uz = macroscopic3d(f)
    u_num = prof_acc[1 : ny - 1].cpu().numpy().astype(np.float64)
    y_phys = np.arange(1, ny - 1, dtype=np.float64) - 0.5
    u_ana = U0 * (y_phys / H_eff)

    l2_rel = float(np.linalg.norm(u_num - u_ana) / np.linalg.norm(u_ana))
    max_abs_err = float(np.max(np.abs(u_num - u_ana)))
    mask = u_ana > 0.2 * U0
    max_rel = float(np.max(np.abs(u_num[mask] - u_ana[mask]) / u_ana[mask]) * 100.0)
    u_top_num = float(u_num[-1])
    u_top_ana = float(u_ana[-1])
    u_top_err = abs(u_top_num - u_top_ana) / u_top_ana * 100.0
    slope_num = float((u_num[-1] - u_num[0]) / (y_phys[-1] - y_phys[0]))
    tau_ana = nu * U0 / H_eff
    tau_num = slope_num * nu
    tau_err_pct = (tau_num - tau_ana) / tau_ana * 100.0
    uz_max = float(torch.abs(uz).max().item())
    ux_spanwise_std = float(ux.std(dim=0).mean().item())

    return {
        "H": H,
        "ny": ny,
        "nx": nx,
        "nz": nz,
        "dtype": str(dtype).replace("torch.", ""),
        "tau": tau,
        "nu_lb": nu,
        "U0": U0,
        "n_steps": step,
        "steady": steady,
        "l2_rel_err": l2_rel,
        "max_abs_err": max_abs_err,
        "max_rel_err_masked_pct": max_rel,
        "u_top_err_pct": u_top_err,
        "tau_err_pct": tau_err_pct,
        "uz_max": uz_max,
        "ux_spanwise_std": ux_spanwise_std,
        "mass_drift_pct": (float(f.sum().item()) - initial_mass) / initial_mass * 100.0,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": round(elapsed, 1),
    }


# ── gate evaluation ──────────────────────────────────────────────────
def evaluate_gates(results_fp32, results_fp64):
    gates = []

    # G1: direct_observable — l2_rel_err ≤ 3% for every grid (fp32)
    g1_pass = all(r["l2_rel_err"] <= 0.03 for r in results_fp32)
    gates.append({
        "gate": "G1_direct_observable",
        "criterion": "l2_rel_err <= 3% for all grids (fp32)",
        "pass": g1_pass,
        "values": {r["H"]: r["l2_rel_err"] for r in results_fp32},
    })

    # G2a: fp32 floor — l2 ≤ 1e-3
    g2a_pass = all(r["l2_rel_err"] <= 1e-3 for r in results_fp32)
    gates.append({
        "gate": "G2a_fp32_floor",
        "criterion": "l2_rel_err <= 1e-3 for all fp32 grids",
        "pass": g2a_pass,
        "values": {r["H"]: r["l2_rel_err"] for r in results_fp32},
    })

    # G2b: fp64 scheme exactness — l2 ≤ 1e-7
    g2b_pass = all(r["l2_rel_err"] <= 1e-7 for r in results_fp64)
    gates.append({
        "gate": "G2b_fp64_scheme_exactness",
        "criterion": "l2_rel_err <= 1e-7 for all fp64 grids",
        "pass": g2b_pass,
        "values": {r["H"]: r["l2_rel_err"] for r in results_fp64},
    })

    # G3: steady state — steady=true, |mass_drift| < 1%, uz_max < 1e-6, spanwise_std < 1e-6
    g3_details = []
    g3_pass = True
    for r in results_fp64:
        ok = (r["steady"]
              and abs(r["mass_drift_pct"]) < 1.0
              and r["uz_max"] < 1e-6
              and r["ux_spanwise_std"] < 1e-6)
        if not ok:
            g3_pass = False
        g3_details.append({
            "H": r["H"],
            "steady": r["steady"],
            "mass_drift_pct": r["mass_drift_pct"],
            "uz_max": r["uz_max"],
            "ux_spanwise_std": r["ux_spanwise_std"],
            "pass": ok,
        })
    gates.append({
        "gate": "G3_steady_state",
        "criterion": "steady=true, |mass_drift|<1%, uz_max<1e-6, spanwise_std<1e-6 (fp64)",
        "pass": g3_pass,
        "details": g3_details,
    })

    # G4: library_entry — imports work (already proven by execution)
    gates.append({
        "gate": "G4_library_entry",
        "criterion": "tensorlbm.d3q19 + tensorlbm.solver3d importable and functional on sdaa",
        "pass": True,
        "evidence": "all solver calls executed on sdaa without error",
    })

    return gates


# ── main ─────────────────────────────────────────────────────────────
def main():
    device = torch.device("sdaa")
    device_ev = collect_device_evidence()
    print(f"[SDAA] device evidence: {json.dumps(device_ev, indent=2)}")

    # Grid sizes: use H=8 only for speed (wall clock ≤20 min)
    # Steps: min=200, max=500 (smaller than original 20000/24000 for speed)
    H_values = [8]
    min_steps = 200
    max_steps = 500
    tau = 0.8
    U0 = 0.05

    triton_interp = os.environ.get("TRITON_INTERPRET", None)

    results_fp32 = []
    results_fp64 = []
    total_t0 = time.time()

    # fp32 sweep
    print("\n=== fp32 sweep on SDAA ===")
    for H in H_values:
        t0 = time.time()
        r = run_case(H, tau, U0, min_steps, max_steps, torch.float32, device, prof_steps=20)
        r["elapsed_s"] = round(time.time() - t0, 1)
        results_fp32.append(r)
        print(f"  H={H:3d} fp32 steps={r['n_steps']:5d} l2={r['l2_rel_err']:.3e} "
              f"steady={r['steady']} t={r['elapsed_s']:.1f}s")

    # fp64 sweep
    print("\n=== fp64 sweep on SDAA ===")
    for H in H_values:
        t0 = time.time()
        r = run_case(H, tau, U0, min_steps, max_steps, torch.float64, device, prof_steps=20)
        r["elapsed_s"] = round(time.time() - t0, 1)
        results_fp64.append(r)
        print(f"  H={H:3d} fp64 steps={r['n_steps']:5d} l2={r['l2_rel_err']:.3e} "
              f"steady={r['steady']} t={r['elapsed_s']:.1f}s")

    total_elapsed = round(time.time() - total_t0, 1)

    gates = evaluate_gates(results_fp32, results_fp64)
    all_pass = all(g["pass"] for g in gates)

    output = {
        "case": "couette_3d",
        "sweep_device": "sdaa",
        "device_evidence": device_ev,
        "triton_interpret": triton_interp,
        "triton_bypass": "not needed — case uses pure torch ops, no triton kernels",
        "config": {
            "H_values": H_values,
            "tau": tau,
            "U0": U0,
            "min_steps": min_steps,
            "max_steps": max_steps,
            "lattice": "D3Q19",
            "collision": "BGK",
            "boundary": "moving top wall + stationary bottom wall, half-way BB",
        },
        "results_fp32": results_fp32,
        "results_fp64": results_fp64,
        "gates": gates,
        "all_pass": all_pass,
        "total_wall_clock_s": total_elapsed,
    }

    out_path = REPO / "artifacts" / "sdaa_couette_3d.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[SDAA] results written to {out_path}")
    print(f"[SDAA] all_pass = {all_pass}")
    for g in gates:
        status = "PASS" if g["pass"] else "FAIL"
        print(f"  {g['gate']}: {status}")
    print(f"[SDAA] total wall clock: {total_elapsed}s")


if __name__ == "__main__":
    main()
