#!/usr/bin/env python
"""SDAA verification sweep for collision_kernels_3d_tg benchmark.

Runs the 3D Taylor-Green vortex decay protocol on SDAA device for all
pre-registered collision kernels, reproducing the NVIDIA-side判据 and
producing artifacts/sdaa_collision_kernels_3d_tg.json with device evidence.

Protocol (from README, verbatim):
  Re=24, U0=0.05, N=64/96/128, nu=U0*N/Re, tau=0.5+3*nu
  gamma_E_theory = 6*nu*k^2, k=2*pi/N
  gamma_E_sim = -d(lnE)/dt (LSQ over recorded samples, t>=record_every)
  Criteria: |err_E|<=3%, monotonic decrease with N, R^2>=0.999
"""

from __future__ import annotations

import datetime
import json
import math
import os
import sys
import time

import numpy as np
import torch

# Use the local tensorlbm from the worktree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tensorlbm

from tensorlbm.cascaded_collision import collide_cascaded_d3q27
from tensorlbm.cumulant import collide_cumulant_d3q27
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import (
    collide_bgk27,
    collide_mrt27,
    collide_rlbm27,
    collide_trt27,
    equilibrium27,
    macroscopic27,
    stream27_roll,
)
from tensorlbm.entropic_kbc import collide_kbc_d3q19
from tensorlbm.solver3d import stream3d

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
ARTIFACTS = os.path.join(ROOT, "artifacts")

DEVICE = "sdaa:0"
CANONICAL = dict(re=24.0, u0=0.05, dtype="float32")
LADDER = (64, 96, 128)

KERNELS = {
    "cumulant_d3q27": {
        "lattice": "d3q27",
        "op": lambda f, tau: collide_cumulant_d3q27(f, tau),
        "params": "tau; omega_b=1.0, omega_odd=1.0, omega_even=1.0 (defaults)",
        "module": "tensorlbm.cumulant.collide_cumulant_d3q27",
    },
    "cascaded_d3q27": {
        "lattice": "d3q27",
        "op": lambda f, tau: collide_cascaded_d3q27(f, tau),
        "params": "tau; s_bulk=1.0, s_3=1.0, s_4=1.0 (defaults)",
        "module": "tensorlbm.cascaded_collision.collide_cascaded_d3q27",
    },
    "mrt27": {
        "lattice": "d3q27",
        "op": lambda f, tau: collide_mrt27(f, tau),
        "params": "tau; s_e=1.19, s_eps=1.4, s_q=1.2 (defaults)",
        "module": "tensorlbm.d3q27.collide_mrt27",
    },
    "kbc_d3q19": {
        "lattice": "d3q19",
        "op": lambda f, tau: collide_kbc_d3q19(f, tau),
        "params": "tau; max_iter=28, tol=1e-8 (defaults)",
        "module": "tensorlbm.entropic_kbc.collide_kbc_d3q19",
    },
    "trt27": {
        "lattice": "d3q27",
        "op": lambda f, tau: collide_trt27(f, tau),
        "params": "tau_plus=tau; lambda_trt=3/16 (default, Ginzburg 2008)",
        "module": "tensorlbm.d3q27.collide_trt27",
    },
    "rlbm27": {
        "lattice": "d3q27",
        "op": lambda f, tau: collide_rlbm27(f, tau),
        "params": "tau (no extra parameters)",
        "module": "tensorlbm.d3q27.collide_rlbm27",
    },
    "bgk27": {
        "lattice": "d3q27",
        "op": lambda f, tau: collide_bgk27(f, tau),
        "params": "tau (diagnostic control)",
        "module": "tensorlbm.d3q27.collide_bgk27",
    },
}

LATTICE_FUNCS = {
    "d3q19": dict(equilibrium=equilibrium3d, macroscopic=macroscopic3d, streaming=stream3d),
    "d3q27": dict(equilibrium=equilibrium27, macroscopic=macroscopic27, streaming=stream27_roll),
}


def k_of(n: int) -> float:
    return 2.0 * math.pi / float(n)


def auto_steps(n: int, re: float, u0: float, record_every: int,
               target_efolds: float = 10.0, cap: int = 10000, min_steps: int = 2000) -> int:
    nu = u0 * n / re
    k = 2.0 * math.pi / n
    gamma_e = 6.0 * nu * k * k
    steps = int(math.ceil(target_efolds / gamma_e / record_every) * record_every)
    return min(max(steps, min_steps), cap)


def build_initial_field(n: int, u0: float, dt, device, lattice: str):
    z, y, x = torch.meshgrid(
        torch.arange(n, device=device, dtype=dt),
        torch.arange(n, device=device, dtype=dt),
        torch.arange(n, device=device, dtype=dt),
        indexing="ij",
    )
    k = k_of(n)
    ux0 = u0 * torch.sin(k * x) * torch.cos(k * y) * torch.cos(k * z)
    uy0 = -u0 * torch.cos(k * x) * torch.sin(k * y) * torch.cos(k * z)
    rho = torch.ones_like(ux0)
    f = LATTICE_FUNCS[lattice]["equilibrium"](rho, ux0, uy0, torch.zeros_like(ux0))
    return f, ux0, uy0


def run_case(kernel: str, n: int, re: float, u0: float, steps: int,
             record_every: int = 50, device: str = DEVICE, dtype: str = "float32") -> dict:
    info = KERNELS[kernel]
    lattice = info["lattice"]
    funcs = LATTICE_FUNCS[lattice]
    dt = torch.float64 if dtype == "float64" else torch.float32

    k = k_of(n)
    nu = u0 * n / re
    tau = 0.5 + 3.0 * nu
    gamma_e_theory = 6.0 * nu * k * k
    gamma_vel_theory = 3.0 * nu * k * k

    f, ux0, uy0 = build_initial_field(n, u0, dt, device, lattice)
    mass0 = float(f.sum().item())
    e0_theory = u0 * u0 / 8.0

    op = info["op"]

    times: list[int] = [0]
    energies: list[float] = [0.0]
    exs: list[float] = [0.0]
    eys: list[float] = [0.0]
    ezs: list[float] = [0.0]
    umaxs: list[float] = [0.0]
    wmaxs: list[float] = [0.0]

    wall0 = time.time()
    for step in range(1, steps + 1):
        f = op(funcs["streaming"](f), tau)
        if step % record_every == 0:
            _, uxm, uym, uzm = funcs["macroscopic"](f)
            ex = float((0.5 * (uxm * uxm)).mean().item())
            ey = float((0.5 * (uym * uym)).mean().item())
            ez = float((0.5 * (uzm * uzm)).mean().item())
            um = float((uxm * uxm + uym * uym + uzm * uzm).sqrt().max().item())
            wm = float(uzm.abs().max().item())
            times.append(step)
            energies.append(ex + ey + ez)
            exs.append(ex)
            eys.append(ey)
            ezs.append(ez)
            umaxs.append(um)
            wmaxs.append(wm)
    wall = time.time() - wall0

    ex0 = float((0.5 * (ux0 * ux0)).mean().item())
    ey0 = float((0.5 * (uy0 * uy0)).mean().item())
    energies[0] = ex0 + ey0
    exs[0] = ex0
    eys[0] = ey0
    ezs[0] = 0.0
    umaxs[0] = float((ux0 * ux0 + uy0 * uy0).sqrt().max().item())
    wmaxs[0] = 0.0

    t = np.asarray(times[1:], dtype=np.float64)
    lnE = np.log(np.asarray(energies[1:], dtype=np.float64))
    a_e, b_e = np.polyfit(t, lnE, 1)
    gamma_e_sim = -a_e
    resid = lnE - (b_e + a_e * t)
    r2 = 1.0 - float(np.sum(resid**2) / np.sum((lnE - lnE.mean()) ** 2))

    urms = np.sqrt(2.0 * np.asarray(energies, dtype=np.float64))
    lnU = np.log(urms[1:])
    a_u, _ = np.polyfit(t, lnU, 1)
    gamma_vel_sim = -a_u

    half = len(t) // 2
    g_h1 = -float(np.polyfit(t[:half], lnE[:half], 1)[0])
    g_h2 = -float(np.polyfit(t[half:], lnE[half:], 1)[0])

    finite = bool(np.isfinite(energies).all())
    mass_end = float(f.sum().item())

    return {
        "kernel": kernel,
        "lattice": lattice,
        "n": n,
        "re": re,
        "u0": u0,
        "nu": nu,
        "tau": tau,
        "k": k,
        "steps": steps,
        "record_every": record_every,
        "dtype": dtype,
        "device": device,
        "finite": finite,
        "gamma_e_theory": gamma_e_theory,
        "gamma_e_sim": gamma_e_sim,
        "err_e_pct": (gamma_e_sim - gamma_e_theory) / gamma_e_theory * 100.0,
        "gamma_vel_theory": gamma_vel_theory,
        "gamma_vel_sim": gamma_vel_sim,
        "err_vel_pct": (gamma_vel_sim - gamma_vel_theory) / gamma_vel_theory * 100.0,
        "r2": r2,
        "gamma_e_half1": g_h1,
        "gamma_e_half2": g_h2,
        "e0_theory": e0_theory,
        "e0_meas": energies[0],
        "mass_drift_rel": (mass_end - mass0) / mass0,
        "n_samples": len(times) - 1,
        "wall_sec": wall,
    }


def judge_kernel(kernel: str, cases: list[dict], tol_pct: float = 3.0) -> dict:
    usable = [c for c in cases if c.get("finite", True)]
    ns = sorted(c["n"] for c in usable)
    by_n = {c["n"]: c for c in usable}
    if not ns:
        return {"kernel": kernel, "verified": False, "judgment": "FAIL: no finite tier"}
    errs = [by_n[n]["err_e_pct"] for n in ns]
    errs_v = [by_n[n]["err_vel_pct"] for n in ns]
    r2s = [by_n[n]["r2"] for n in ns]
    converged = all(abs(errs[i + 1]) < abs(errs[i]) for i in range(len(errs) - 1))
    within = all(abs(e) <= tol_pct for e in errs)
    within_v = all(abs(e) <= tol_pct for e in errs_v)
    expo = all(r2 >= 0.999 for r2 in r2s)
    verified = bool(within and within_v and converged and expo and len(ns) >= 2)
    return {
        "kernel": kernel,
        "grids": ns,
        "err_e_pct_per_grid": {str(n): by_n[n]["err_e_pct"] for n in ns},
        "err_vel_pct_per_grid": {str(n): by_n[n]["err_vel_pct"] for n in ns},
        "r2_per_grid": {str(n): by_n[n]["r2"] for n in ns},
        "converged_monotone": converged,
        "within_tol": within,
        "within_tol_vel": within_v,
        "exponential_r2_ok": expo,
        "tol_pct": tol_pct,
        "verified": verified,
        "judgment": "PASS" if verified else "FAIL: see per-tier err_e_pct / monotonicity / R^2",
    }


def main():
    print("=" * 70)
    print("SDAA Sweep: collision_kernels_3d_tg")
    print("=" * 70)

    # Device evidence
    device_name = torch.sdaa.get_device_name(0)
    print(f"SDAA device: {device_name}")
    print(f"SDAA available: {torch.sdaa.is_available()}")

    os.makedirs(ARTIFACTS, exist_ok=True)

    # Load baseline (NVIDIA-side) for comparison
    baseline_path = os.path.join(ROOT, "benchmarks", "pending", "collision_kernels_3d_tg", "result.json")
    baseline = None
    if os.path.exists(baseline_path):
        with open(baseline_path) as fh:
            baseline = json.load(fh)
        print(f"Loaded baseline from {baseline_path}")

    # Run all kernels on all grid levels
    results = {}
    all_cases = []

    # Pre-register kernels (excluding bgk27 diagnostic)
    pre_registered = ["cumulant_d3q27", "cascaded_d3q27", "mrt27", "kbc_d3q19", "trt27", "rlbm27"]
    # Also run bgk27 for diagnostic comparison
    all_kernels = pre_registered + ["bgk27"]

    for kernel in all_kernels:
        print(f"\n--- Kernel: {kernel} ---")
        cases = []
        for n in LADDER:
            steps = auto_steps(n, CANONICAL["re"], CANONICAL["u0"], record_every=50)
            print(f"  N={n}: tau={0.5+3.0*CANONICAL['u0']*n/CANONICAL['re']:.4f}, steps={steps}...", end=" ", flush=True)
            t0 = time.time()
            res = run_case(kernel, n, CANONICAL["re"], CANONICAL["u0"], steps, device=DEVICE)
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s) err={res['err_e_pct']:+.4f}% R²={res['r2']:.6f}")
            cases.append(res)
            all_cases.append(res)

        verdict = judge_kernel(kernel, cases)
        results[kernel] = {
            "lattice": KERNELS[kernel]["lattice"],
            "module": KERNELS[kernel]["module"],
            "params": KERNELS[kernel]["params"],
            "cases": cases,
            "convergence": verdict,
        }
        print(f"  -> verdict: {verdict['judgment']}")

    # Build artifact
    artifact = {
        "benchmark": "collision_kernels_3d_tg",
        "sdaa_device": device_name,
        "sdaa_available": torch.sdaa.is_available(),
        "protocol": {
            "re": CANONICAL["re"],
            "u0": CANONICAL["u0"],
            "dtype": CANONICAL["dtype"],
            "ladder": list(LADDER),
            "record_every": 50,
            "target_efolds": 10.0,
        },
        "kernels": results,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tensorlbm_file": tensorlbm.__file__,
    }

    # Add baseline comparison
    if baseline:
        comparisons = {}
        for kernel in pre_registered:
            if kernel in baseline.get("kernels", {}) and kernel in results:
                bl_cases = baseline["kernels"][kernel].get("cases", [])
                sd_cases = results[kernel]["cases"]
                comp = []
                for bl_c, sd_c in zip(bl_cases, sd_cases):
                    comp.append({
                        "n": bl_c["n"],
                        "gamma_e_sim_baseline": bl_c["gamma_e_sim"],
                        "gamma_e_sim_sdaa": sd_c["gamma_e_sim"],
                        "err_e_pct_baseline": bl_c["err_e_pct"],
                        "err_e_pct_sdaa": sd_c["err_e_pct"],
                        "r2_baseline": bl_c["r2"],
                        "r2_sdaa": sd_c["r2"],
                        "gamma_e_sim_diff_pct": (sd_c["gamma_e_sim"] - bl_c["gamma_e_sim"]) / bl_c["gamma_e_sim"] * 100.0 if bl_c["gamma_e_sim"] != 0 else None,
                    })
                comparisons[kernel] = comp
        artifact["baseline_comparison"] = comparisons

    # Write artifact
    out_path = os.path.join(ARTIFACTS, "sdaa_collision_kernels_3d_tg.json")
    with open(out_path, "w") as fh:
        json.dump(artifact, fh, indent=2, ensure_ascii=False)
    print(f"\nArtifact written to {out_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Kernel':<20s} {'N':>4s} {'γ_sim(SDAA)':>14s} {'γ_sim(NV)':>14s} {'err%(SDAA)':>10s} {'err%(NV)':>10s} {'R²(SDAA)':>9s} {'Δγ%':>8s}")
    print("-" * 100)
    for kernel in pre_registered:
        if kernel in results and baseline and kernel in baseline.get("kernels", {}):
            bl_cases = baseline["kernels"][kernel]["cases"]
            sd_cases = results[kernel]["cases"]
            for bl_c, sd_c in zip(bl_cases, sd_cases):
                delta = (sd_c["gamma_e_sim"] - bl_c["gamma_e_sim"]) / bl_c["gamma_e_sim"] * 100.0 if bl_c["gamma_e_sim"] != 0 else 0.0
                print(f"{kernel:<20s} {sd_c['n']:4d} {sd_c['gamma_e_sim']:14.6e} {bl_c['gamma_e_sim']:14.6e} {sd_c['err_e_pct']:>+10.4f} {bl_c['err_e_pct']:>+10.4f} {sd_c['r2']:9.6f} {delta:>+8.4f}")
        elif kernel in results:
            for sd_c in results[kernel]["cases"]:
                print(f"{kernel:<20s} {sd_c['n']:4d} {sd_c['gamma_e_sim']:14.6e} {'N/A':>14s} {sd_c['err_e_pct']:>+10.4f} {'N/A':>10s} {sd_c['r2']:9.6f} {'N/A':>8s}")

    # Verdict summary
    print("\n" + "=" * 70)
    print("VERDICT SUMMARY (SDAA runs)")
    print("=" * 70)
    for kernel in pre_registered:
        if kernel in results:
            v = results[kernel]["convergence"]
            status = "PASS" if v["verified"] else "FAIL"
            reasons = []
            if not v["within_tol"]:
                reasons.append("tolerance")
            if not v["converged_monotone"]:
                reasons.append("monotonicity")
            if not v["exponential_r2_ok"]:
                reasons.append("R²")
            reason_str = ", ".join(reasons) if reasons else "all gates passed"
            print(f"  {kernel}: {status} ({reason_str})")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
