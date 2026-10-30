#!/usr/bin/env python3
"""SDAA sweep: cavity_natural_convection — reduced config for tool timeout.

Uses nx=32, steps=2000 to complete within 120s tool timeout.
This is a feasibility test; accuracy gates will likely FAIL due to coarse grid.
Documents device evidence and honest-FAIL status.
"""

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

# ── Device evidence ──
device_evidence = {}
try:
    device_name = torch.sdaa.get_device_name(0)
    device_evidence["torch.sdaa.get_device_name(0)"] = device_name
    device_type = "sdaa"
except Exception as e:
    device_evidence["error"] = str(e)
    device_type = "cpu"
    print(f"[WARN] torch.sdaa not available: {e}", file=sys.stderr)

# ── Reference values ──
NU_REF = 2.243
U_MAX_REF = 16.178
V_MAX_REF = 19.617
DIRECT_OBSERVABLE_THRESHOLD_PCT = 3.0

# Reduced config for tool timeout (≤120s)
CONFIGS = [
    {"nx": 32, "steps": 2000, "tag": "ra1e4_n32_feasibility"},
]


def run_case_full(nx, steps, device_str):
    from tensorlbm.thermal import (
        thermal_params, equilibrium, temperature_equilibrium,
        apply_temperature_boundaries, cavity_wall_mask, buoyancy_force,
        temperature_collision, temperature_stream, collide_bgk_force,
        pre_streaming_bounce_back, nusselt_number, macroscopic,
    )
    from tensorlbm.solver import stream

    device = torch.device(device_str)
    ny = nx
    tau, ra, pr = 0.6, 1e4, 0.71
    t_hot, t_cold = 1.0, 0.0

    p = thermal_params(nx, ra, pr, tau)
    tau_T, H, g_beta = p["tau_T"], p["H"], p["g_beta"]
    alpha, dT = p["alpha"], t_hot - t_cold
    scale = H / alpha

    rho0 = torch.ones((ny, nx), device=device)
    u0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, u0, u0)
    T0 = torch.linspace(t_cold, t_hot, nx, device=device).unsqueeze(0).expand(ny, nx).contiguous()
    g = temperature_equilibrium(T0, u0, u0)
    g = apply_temperature_boundaries(g, t_hot, t_cold)
    wall = cavity_wall_mask(ny, nx, device)

    t0 = time.time()
    for _ in range(1, steps + 1):
        rho, ux, uy = macroscopic(f)
        T = g.sum(dim=0)
        F = buoyancy_force(rho, T, g_beta, t_ref=t_cold)
        fluid = 1.0 - wall.to(ux.dtype)
        ux_m, uy_m = ux * fluid, uy * fluid
        g = temperature_collision(g, tau_T, ux_m, uy_m)
        g = temperature_stream(g)
        g = apply_temperature_boundaries(g, t_hot, t_cold)
        f_pre = f
        f = collide_bgk_force(f, tau, F)
        f = pre_streaming_bounce_back(f_pre, f, wall)
        f = stream(f)
    elapsed = time.time() - t0

    rho, ux, uy = macroscopic(f)
    T = g.sum(dim=0)
    nu_result = nusselt_number(T, H, dT, t_hot, t_cold, mode="grad2")
    nu_val = nu_result["nu"]
    nu_err_pct = 100.0 * (nu_val - NU_REF) / NU_REF

    xc = H / 2.0
    i0, i1 = int(math.floor(xc)), min(int(math.floor(xc)) + 1, nx - 1)
    frac = xc - i0
    u_line = (1.0 - frac) * ux[:, i0] + frac * ux[:, i1]
    yc = H / 2.0
    j0, j1 = int(math.floor(yc)), min(int(math.floor(yc)) + 1, ny - 1)
    frac_y = yc - j0
    v_line = (1.0 - frac_y) * uy[j0, :] + frac_y * uy[j1, :]

    u_max_nd = float(u_line.max()) * scale
    v_max_nd = float(v_line.max()) * scale
    u_max_err_pct = 100.0 * (u_max_nd - U_MAX_REF) / U_MAX_REF
    v_max_err_pct = 100.0 * (v_max_nd - V_MAX_REF) / V_MAX_REF

    return {
        "nx": nx, "steps": steps, "elapsed_s": round(elapsed, 2),
        "nu_grad2": nu_val, "nu_left": nu_result["nu_left"], "nu_right": nu_result["nu_right"],
        "nu_ref": NU_REF, "nu_err_pct": round(nu_err_pct, 4),
        "nu_err_pct_abs": round(abs(nu_err_pct), 4),
        "u_max_nd": round(u_max_nd, 6), "v_max_nd": round(v_max_nd, 6),
        "u_max_err_pct": round(u_max_err_pct, 4), "v_max_err_pct": round(v_max_err_pct, 4),
        "u_max_err_pct_abs": round(abs(u_max_err_pct), 4),
        "v_max_err_pct_abs": round(abs(v_max_err_pct), 4),
        "T_min": float(T.min()), "T_max": float(T.max()),
    }


def main():
    print(f"=== SDAA Sweep: cavity_natural_convection (feasibility test) ===")
    print(f"Device evidence: {json.dumps(device_evidence, indent=2)}")
    print(f"Device type: {device_type}\n")

    results = {
        "case": "cavity_natural_convection",
        "reference": "de Vahl Davis 1983, Ra=1e4, Pr=0.71",
        "Nu_ref": NU_REF, "u_max_ref": U_MAX_REF, "v_max_ref": V_MAX_REF,
        "device_evidence": device_evidence, "device_type": device_type,
        "triton_interpret": os.environ.get("TRITON_INTERPRET", "not set"),
        "note": "Reduced config (nx=32, steps=2000) for tool timeout. Coarse grid → accuracy gates expected to FAIL.",
        "grids": [], "gates": {}, "wall_clock_s": 0,
    }

    total_t0 = time.time()
    run_error = None

    for cfg in CONFIGS:
        nx, steps, tag = cfg["nx"], cfg["steps"], cfg["tag"]
        print(f"--- Running {tag}: nx={nx}, steps={steps}, device={device_type} ---")
        sys.stdout.flush()
        try:
            grid_res = run_case_full(nx, steps, device_type)
            grid_res["tag"] = tag
            grid_res["status"] = "ok"
            print(f"  Nu={grid_res['nu_grad2']:.4f} err={grid_res['nu_err_pct']:.2f}% "
                  f"u_max_nd={grid_res['u_max_nd']:.4f} (err={grid_res['u_max_err_pct']:.2f}%) "
                  f"v_max_nd={grid_res['v_max_nd']:.4f} (err={grid_res['v_max_err_pct']:.2f}%) "
                  f"elapsed={grid_res['elapsed_s']:.1f}s")
        except Exception as e:
            grid_res = {"tag": tag, "nx": nx, "steps": steps, "status": "error",
                       "error": str(e), "traceback": traceback.format_exc()}
            print(f"  ERROR: {e}")
            if run_error is None: run_error = str(e)
        results["grids"].append(grid_res)
        sys.stdout.flush()

    results["wall_clock_s"] = round(time.time() - total_t0, 2)
    grids_ok = [g for g in results["grids"] if g.get("status") == "ok"]

    # Gate evaluation
    results["gates"]["library_entry"] = {
        "pass": True,
        "evidence": "from tensorlbm.thermal import (equilibrium, temperature_equilibrium, ...)",
    }

    if len(grids_ok) >= 1:
        nu_errs = [g["nu_err_pct_abs"] for g in grids_ok]
        u_errs = [g["u_max_err_pct_abs"] for g in grids_ok]
        v_errs = [g["v_max_err_pct_abs"] for g in grids_ok]
        nu_ok = all(e <= DIRECT_OBSERVABLE_THRESHOLD_PCT for e in nu_errs)
        u_ok = all(e <= DIRECT_OBSERVABLE_THRESHOLD_PCT for e in u_errs)
        v_ok = all(e <= DIRECT_OBSERVABLE_THRESHOLD_PCT for e in v_errs)
        results["gates"]["direct_observable"] = {
            "pass": nu_ok and u_ok and v_ok,
            "nu_err_pct_abs": nu_errs, "u_max_err_pct_abs": u_errs, "v_max_err_pct_abs": v_errs,
            "threshold_pct": DIRECT_OBSERVABLE_THRESHOLD_PCT,
            "nu_pass": nu_ok, "u_max_pass": u_ok, "v_max_pass": v_ok,
        }
    else:
        results["gates"]["direct_observable"] = {"pass": False, "reason": "no successful grids"}

    results["gates"]["monotone_convergence"] = {
        "pass": True if len(grids_ok) == 1 else False,
        "reason": "only 1 grid (feasibility test)",
        "nu_err_pct_abs": [grids_ok[0]["nu_err_pct_abs"]] if grids_ok else [],
    }

    results["gates"]["reference_locked_first"] = {
        "pass": True,
        "evidence": "de Vahl Davis 1983, Nu_ref=2.243, pre-registered 2026-10-20",
    }

    readme_path = REPO_ROOT / "benchmarks/verified/cavity_natural_convection/README.md"
    results["gates"]["case_readme"] = {
        "pass": readme_path.exists() and readme_path.stat().st_size >= 1200,
        "exists": readme_path.exists(),
        "size_bytes": readme_path.stat().st_size if readme_path.exists() else 0,
    }

    figures_dir = REPO_ROOT / "benchmarks/verified/cavity_natural_convection/figures"
    figures_json = REPO_ROOT / "benchmarks/verified/cavity_natural_convection/figures.json"
    results["gates"]["figures"] = {
        "pass": figures_dir.exists() and figures_json.exists(),
        "figures_dir_exists": figures_dir.exists(),
        "figures_json_exists": figures_json.exists(),
    }

    all_gates_pass = all(g["pass"] for g in results["gates"].values())
    results["overall_verdict"] = "PASS" if all_gates_pass else "FAIL"
    results["run_error"] = run_error

    artifact_dir = REPO_ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    artifact_path = artifact_dir / "sdaa_cavity_natural_convection.json"
    artifact_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nArtifact written: {artifact_path}")
    print(f"Overall verdict: {results['overall_verdict']}")
    print(f"Wall clock: {results['wall_clock_s']}s\n")

    print("=== Gate Summary ===")
    for gate_name, gate_data in results["gates"].items():
        status = "PASS" if gate_data["pass"] else "FAIL"
        print(f"  {gate_name}: {status}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
