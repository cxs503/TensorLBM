#!/usr/bin/env python3
"""M12 · Phase-field Laplace benchmark: 2-grid recut + historical comparison.

Runs 2 grids (R=20, R=32 on 128^3) to confirm Δp·R/σ vs analytical ≤ 3%.
Compares with m24/m25/m52 historical values.

Output: artifacts/sdaa_laplace_recut.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from tensorlbm import cac_lbm as m

# Route-B calibration constants (frozen from verified benchmark)
SIGMA_TARGET = 0.01
BAR_EFF_W4 = 0.9238958795895513
SIGMA_ROUTE_B = SIGMA_TARGET / BAR_EFF_W4

# Historical values from m24/m25/m52 (from verified result.json)
HISTORICAL = {
    "m24_b1_R20": {
        "sigma_rec": 0.009993760665757663,
        "err_pct": 0.0623933424233683,
        "dP": 0.0009993760665757663,
        "dp_r_sigma": None,  # will compute
    },
    "m25_b1_R32": {
        "sigma_rec": 0.009997902817535833,
        "err_pct": 0.0209718246416704,
        "dP": 0.0006248626760959896,
        "dp_r_sigma": None,
    },
    "m52_b1_R48": {
        "sigma_rec": 0.010045977965927124,
        "err_pct": 0.45977965927124,
        "dP": 0.0004185824152469635,
        "dp_r_sigma": None,
    },
}

# Compute dp_r_sigma for historical values
# dp_r_sigma = dP * R / sigma_target (should be ~1.0 for Laplace law: dP = 2*sigma/R => dP*R/sigma = 2)
# Wait - the task says Δp·R/σ. For 3D Laplace: Δp = 2σ/R, so Δp·R/σ = 2.
# But the criterion says ≤1.03 (i.e., error ≤3%). This suggests they're using
# Δp·R/(2σ) which should be 1.0, or σ_rec/σ_target which should be 1.0.
# Looking at the run.py: sigma_rec = (p_in - p_out) * radius / 2.0
# So sigma_rec/sigma_target = dP*R/(2*sigma_target) should be ~1.0
# The criterion dp_r_sigma ≤ 1.03 means |dP*R/(2σ) - 1| ≤ 0.03, i.e., ≤3% error.
for k, v in HISTORICAL.items():
    R = float(k.split("_R")[1])
    v["dp_r_sigma"] = v["sigma_rec"] / SIGMA_TARGET  # = dP*R/(2*sigma_target)
    v["radius"] = R

N_STEPS = 12000
MOB = 0.05
NU = 0.1
W_INT = 4.0


def run_laplace_arm(radius, shape, device_str, n_steps=N_STEPS):
    """Run a single Laplace arm and return results."""
    device = torch.device(device_str)
    dtype = torch.float64

    rho_l, rho_g = 1.0, 0.01
    mu_l, mu_g = NU * rho_l, NU * rho_g

    sim = m.CACSim(
        shape,
        rho_l=rho_l, rho_g=rho_g,
        mu_l=mu_l, mu_g=mu_g,
        sigma=SIGMA_ROUTE_B,
        width=W_INT,
        mobility=MOB,
        p0=0.01,
        axes="ppp",
        mu_law="harmonic",
        dtype=dtype,
        device=device,
        gravity=(0.0, 0.0, 0.0),
        gravity_mode="rho",
    )
    center = tuple(s / 2.0 for s in shape)
    sim.initialize(
        m.init_phi_droplet(shape, radius=radius, width=W_INT, center=center, dtype=dtype)
    )

    t0 = time.time()
    series = []
    phi0_int = None
    for it in range(1, n_steps + 1):
        sim.step()
        if it == 1 or it % 500 == 0:
            phi, rho, grad, mu_phi, F, u, p, u_star = m._macro_fp64(sim)
            if phi0_int is None:
                phi0_int = phi.sum().item()
            inside = phi > 0.9
            outside = phi < 0.1
            p_in = p[inside].mean().item() if inside.any() else float("nan")
            p_out = p[outside].mean().item() if outside.any() else float("nan")
            umax = (u[0] ** 2 + u[1] ** 2 + u[2] ** 2).sqrt().max().item()
            dP = p_in - p_out
            sigma_rec = dP * radius / 2.0
            dp_r_sigma = sigma_rec / SIGMA_TARGET  # = dP*R/(2*sigma_target)
            series.append({
                "step": it,
                "p_in": p_in,
                "p_out": p_out,
                "dP": dP,
                "sigma_rec": sigma_rec,
                "dp_r_sigma": dp_r_sigma,
                "u_max": umax,
                "phi_drift_rel": abs(phi.sum().item() - phi0_int) / phi0_int if phi0_int else 0.0,
            })
            if it % 2000 == 0:
                s = series[-1]
                print(f"  [R={radius}] step {it}  dp_r_sigma={s['dp_r_sigma']:.6f}  "
                      f"u_max={s['u_max']:.2e}  drift={s['phi_drift_rel']:.2e}", flush=True)

    elapsed = time.time() - t0
    win = [s for s in series if s["step"] > n_steps - 2000]
    dp_r_sigma_mean = sum(s["dp_r_sigma"] for s in win) / len(win)
    sigma_rec_mean = sum(s["sigma_rec"] for s in win) / len(win)
    dP_mean = sum(s["dP"] for s in win) / len(win)

    return {
        "radius": radius,
        "shape": list(shape),
        "device": device_str,
        "n_steps": n_steps,
        "sigma_target": SIGMA_TARGET,
        "sigma_model_input": SIGMA_ROUTE_B,
        "sigma_rec_window_mean": sigma_rec_mean,
        "dP_window_mean": dP_mean,
        "dp_r_sigma": dp_r_sigma_mean,
        "dp_r_sigma_err_pct": abs(dp_r_sigma_mean - 1.0) * 100.0,
        "u_max_window": max(s["u_max"] for s in win),
        "phi_drift_rel_10k": next(s["phi_drift_rel"] for s in series if s["step"] == 10000),
        "elapsed_s": elapsed,
        "tail_dp_r_sigma": [s["dp_r_sigma"] for s in win],
        "tail_sigma_rec": [s["sigma_rec"] for s in win],
    }


def main():
    print("=" * 60)
    print("M12 Laplace Phase-field Recut: 2 grids")
    print("=" * 60)

    # Determine device
    try:
        t = torch.zeros(1, device="sdaa:0")
        device_str = "sdaa:0"
        print("Using SDAA device")
    except Exception:
        device_str = "cpu"
        print("SDAA not available, using CPU")

    # Run 2 grids: R=20 and R=32 on 128^3
    arms = {}
    for radius in [20.0, 32.0]:
        arm_name = f"R{int(radius)}"
        print(f"\n===== {arm_name} on {device_str} =====")
        arms[arm_name] = run_laplace_arm(radius, (128, 128, 128), device_str)

    # Build comparison with historical values
    historical_comparison = []
    hist_map = {
        "R20": "m24_b1_R20",
        "R32": "m25_b1_R32",
    }
    for arm_name, result in arms.items():
        hist_key = hist_map.get(arm_name)
        if hist_key and hist_key in HISTORICAL:
            hist = HISTORICAL[hist_key]
            historical_comparison.append({
                "arm": arm_name,
                "historical_ref": hist_key,
                "current_dp_r_sigma": result["dp_r_sigma"],
                "historical_dp_r_sigma": hist["dp_r_sigma"],
                "delta_dp_r_sigma": result["dp_r_sigma"] - hist["dp_r_sigma"],
                "current_err_pct": result["dp_r_sigma_err_pct"],
                "historical_err_pct": hist["err_pct"],
            })

    # Also add m52 R48 historical for reference (not re-run)
    m52 = HISTORICAL["m52_b1_R48"]
    historical_comparison.append({
        "arm": "R48",
        "historical_ref": "m52_b1_R48",
        "current_dp_r_sigma": None,
        "historical_dp_r_sigma": m52["dp_r_sigma"],
        "delta_dp_r_sigma": None,
        "current_err_pct": None,
        "historical_err_pct": m52["err_pct"],
        "note": "Not re-run in this campaign; historical value from m52.",
    })

    # Determine pass/fail
    all_dp_r_sigma = [a["dp_r_sigma"] for a in arms.values()]
    max_err = max(abs(v - 1.0) for v in all_dp_r_sigma)
    pass_laplace = max_err <= 0.03  # ≤ 3%

    output = {
        "case": "laplace_cac_phasefield_recut",
        "mission": "M12",
        "status": "PASS" if pass_laplace else "FAIL",
        "model": "CAC conservative Allen-Cahn two-phase LBM, D3Q19, fp64, "
                 "harmonic mu(phi), W=4, tensorlbm.cac_lbm",
        "calibration": {
            "route": "B",
            "sigma_target": SIGMA_TARGET,
            "bar_eff_W4": BAR_EFF_W4,
            "sigma_model_input": SIGMA_ROUTE_B,
        },
        "config": {
            "W": W_INT,
            "mobility": MOB,
            "nu": NU,
            "n_steps": N_STEPS,
            "dtype": "float64",
            "device": device_str,
        },
        "arms": arms,
        "dp_r_sigma": {k: v["dp_r_sigma"] for k, v in arms.items()},
        "max_dp_r_sigma_err_pct": max(a["dp_r_sigma_err_pct"] for a in arms.values()),
        "criteria": {
            "dp_r_sigma_threshold": 1.03,
            "err_threshold_pct": 3.0,
        },
        "historical_comparison": historical_comparison,
        "historical_values": HISTORICAL,
    }

    # Save
    outdir = Path("artifacts")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "sdaa_laplace_recut.json"
    with open(outpath, "w") as fp:
        json.dump(output, fp, indent=2)
    print(f"\nSaved: {outpath}")
    print(f"Status: {output['status']}")
    for arm_name, result in arms.items():
        print(f"  {arm_name}: dp_r_sigma={result['dp_r_sigma']:.6f} "
              f"err={result['dp_r_sigma_err_pct']:.4f}%")


if __name__ == "__main__":
    main()
