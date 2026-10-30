#!/usr/bin/env python3
"""SDAA sweep for Taylor-Green 3D benchmark (B17-3D).

Runs the verified taylor_green_3d case on SDAA device, records device
evidence, criterion values, and per-gate PASS/FAIL.

Output: artifacts/sdaa_taylor_green_3d.json
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

# Ensure tensorlbm is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

# Import torch_sdaa to register SDAA device
import torch_sdaa  # noqa: F401

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d


def auto_steps(n: int, re: float, u0: float, record_every: int,
               target_efolds: float = 10.0, cap: int = 10000,
               min_steps: int = 2000) -> int:
    nu = u0 * n / re
    k = 2.0 * math.pi / n
    gamma_e = 6.0 * nu * k * k
    steps = int(math.ceil(target_efolds / gamma_e / record_every) * record_every)
    return min(max(steps, min_steps), cap)


def run_case(n: int, re: float, u0: float, steps: int, record_every: int,
             device: str) -> dict:
    dt = torch.float32
    k = 2.0 * math.pi / float(n)
    nu = u0 * n / re
    tau = 0.5 + 3.0 * nu
    gamma_e_theory = 6.0 * nu * k * k
    gamma_vel_theory = 3.0 * nu * k * k

    # Initial field
    z, y, x = torch.meshgrid(
        torch.arange(n, device=device, dtype=dt),
        torch.arange(n, device=device, dtype=dt),
        torch.arange(n, device=device, dtype=dt),
        indexing="ij",
    )
    ux0 = u0 * torch.sin(k * x) * torch.cos(k * y) * torch.cos(k * z)
    uy0 = -u0 * torch.cos(k * x) * torch.sin(k * y) * torch.cos(k * z)
    uz0 = torch.zeros_like(ux0)
    rho = torch.ones_like(ux0)
    f = equilibrium3d(rho, ux0, uy0, uz0)

    mass0 = float(f.sum().item())
    e0_theory = u0 * u0 / 8.0

    # Main loop (eager, no torch.compile — SDAA teco_inductor incomplete)
    times = [0]
    energies = [0.0]
    exs = [0.0]
    eys = [0.0]
    ezs = [0.0]
    wmaxs = [0.0]

    wall0 = time.time()
    for step in range(1, steps + 1):
        f = collide_bgk3d(stream3d(f), tau)
        if step % record_every == 0:
            _, uxm, uym, uzm = macroscopic3d(f)
            ex = float((0.5 * (uxm * uxm)).mean().item())
            ey = float((0.5 * (uym * uym)).mean().item())
            ez = float((0.5 * (uzm * uzm)).mean().item())
            wm = float(uzm.abs().max().item())
            times.append(step)
            energies.append(ex + ey + ez)
            exs.append(ex)
            eys.append(ey)
            ezs.append(ez)
            wmaxs.append(wm)
    wall = time.time() - wall0

    # t=0 initial values
    ex0 = float((0.5 * (ux0 * ux0)).mean().item())
    ey0 = float((0.5 * (uy0 * uy0)).mean().item())
    energies[0] = ex0 + ey0
    exs[0] = ex0
    eys[0] = ey0
    ezs[0] = 0.0
    wmaxs[0] = 0.0

    # Fit ln(E) vs t
    t = np.asarray(times[1:], dtype=np.float64)
    lnE = np.log(np.asarray(energies[1:], dtype=np.float64))
    a_e, b_e = np.polyfit(t, lnE, 1)
    gamma_e_sim = -a_e
    resid = lnE - (b_e + a_e * t)
    r2 = 1.0 - float(np.sum(resid**2) / np.sum((lnE - lnE.mean())**2))

    # Velocity decay
    urms = np.sqrt(2.0 * np.asarray(energies, dtype=np.float64))
    lnU = np.log(urms[1:])
    a_u, _ = np.polyfit(t, lnU, 1)
    gamma_vel_sim = -a_u

    err_e_pct = (gamma_e_sim - gamma_e_theory) / gamma_e_theory * 100.0
    err_vel_pct = (gamma_vel_sim - gamma_vel_theory) / gamma_vel_theory * 100.0
    mass_end = float(f.sum().item())

    return {
        "n": n,
        "re": re,
        "u0": u0,
        "nu": nu,
        "tau": tau,
        "k": k,
        "steps": steps,
        "record_every": record_every,
        "device": device,
        "gamma_e_theory": gamma_e_theory,
        "gamma_e_sim": gamma_e_sim,
        "err_e_pct": err_e_pct,
        "gamma_vel_theory": gamma_vel_theory,
        "gamma_vel_sim": gamma_vel_sim,
        "err_vel_pct": err_vel_pct,
        "r2": r2,
        "e0_theory": e0_theory,
        "e0_meas": energies[0],
        "ez_max": max(ezs),
        "w_max": max(wmaxs),
        "mass_drift_rel": (mass_end - mass0) / mass0,
        "wall_sec": wall,
    }


def main():
    device = "sdaa"
    u0 = 0.05
    re = 24.0
    record_every = 50

    # Device evidence
    device_name = torch.sdaa.get_device_name(0)
    device_count = torch.sdaa.device_count()

    print(f"=== SDAA Taylor-Green 3D Sweep ===")
    print(f"Device: {device}")
    print(f"Device name: {device_name}")
    print(f"Device count: {device_count}")
    print(f"torch_sdaa version: {torch_sdaa.__version__ if hasattr(torch_sdaa, '__version__') else 'unknown'}")
    print()

    # Run all three grids
    grids = [64, 96, 128]
    results = []
    total_wall = 0.0

    for n in grids:
        steps = auto_steps(n, re, u0, record_every)
        print(f"Running N={n}, steps={steps} on {device}...")
        t0 = time.time()
        res = run_case(n, re, u0, steps, record_every, device)
        wall = time.time() - t0
        total_wall += wall
        res["wall_sec"] = wall
        results.append(res)
        print(f"  err_E = {res['err_e_pct']:+.4f}%, R² = {res['r2']:.6f}, "
              f"wall = {wall:.1f}s")

    print(f"\nTotal wall clock: {total_wall:.1f}s")

    # Judgment
    errs = [r["err_e_pct"] for r in results]
    r2s = [r["r2"] for r in results]
    ns = [r["n"] for r in results]

    within_tol = all(abs(e) <= 3.0 for e in errs)
    converged = all(abs(errs[i+1]) < abs(errs[i]) for i in range(len(errs)-1))
    exponential = all(r2 >= 0.999 for r2 in r2s)
    verified = within_tol and converged and exponential and len(ns) >= 2

    # Build output
    output = {
        "case": "taylor_green_3d",
        "benchmark": "B17-3D",
        "device_evidence": {
            "device": device,
            "device_name": device_name,
            "device_count": device_count,
            "torch_sdaa_version": torch_sdaa.__version__ if hasattr(torch_sdaa, '__version__') else "unknown",
            "torch_version": torch.__version__,
        },
        "config": {
            "u0": u0,
            "re": re,
            "record_every": record_every,
            "compile_mode": "eager",
            "compile_note": "SDAA teco_inductor incomplete; run eager (no torch.compile)",
            "triton_bypass": "TRITON_INTERPRET not needed — d3q19/solver3d modules are pure PyTorch",
        },
        "grids": results,
        "judgment": {
            "criterion_err_e_within_3pct": {
                "values": {f"N{n}": f"{e:+.4f}%" for n, e in zip(ns, errs)},
                "threshold": "≤3%",
                "pass": within_tol,
            },
            "criterion_monotone_convergence": {
                "values": {f"N{n}": f"{e:+.4f}%" for n, e in zip(ns, errs)},
                "description": "|err_E| must decrease with N",
                "pass": converged,
            },
            "criterion_exponential_r2": {
                "values": {f"N{n}": f"{r2:.6f}" for n, r2 in zip(ns, r2s)},
                "threshold": "≥0.999",
                "pass": exponential,
            },
            "overall_pass": verified,
        },
        "total_wall_sec": total_wall,
    }

    out_path = REPO_ROOT / "artifacts" / "sdaa_taylor_green_3d.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    print(f"Overall: {'PASS' if verified else 'FAIL'}")


if __name__ == "__main__":
    main()
