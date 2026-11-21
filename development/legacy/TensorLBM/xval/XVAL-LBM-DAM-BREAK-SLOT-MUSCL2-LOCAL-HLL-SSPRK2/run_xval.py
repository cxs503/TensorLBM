#!/usr/bin/env python3
"""XVAL for anchor XVAL-LBM-DAM-BREAK-SLOT-MUSCL2-LOCAL-HLL-SSPRK2.

Runs the MUSCL2+HLL+SSPRK2 shallow-water dam-break solver at multiple
resolutions and compares against the Ritter (1892) analytical solution.

Acceptance criteria (<15% error per owner standard):
  C1: Front position error (mass-based) < 15%
  C2: Profile L2 error vs Ritter < 15%
  C3: Mass conservation drift < 1%
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorlbm.shallow_water import (
    ShallowWaterConfig,
    init_dam_break,
    ritter_depth,
    ritter_front,
    swe_step,
    front_position,
    front_from_mass,
)

X_MIN, X_MAX = -4.0, 12.0
H0, G, T_END = 1.0, 1.0, 5.0


def run_arm(nx: int, cfl: float, scheme: str = "muscl2") -> dict:
    """Run a single arm and return results."""
    cfg = ShallowWaterConfig(
        scheme=scheme, nx=nx, x_min=X_MIN, x_max=X_MAX, dam_x=0.0,
        h0=H0, g=G, cfl=cfl, t_end=T_END, sample_every=max(1, nx // 8),
    )
    cfg.validate()

    U = init_dam_break(cfg)
    mass0 = float(U[0].sum().item() * cfg.dx)
    dt = cfg.cfl * cfg.dx / (2.0 * cfg.c0)
    n_steps = int(round(T_END / dt))

    x = np.linspace(X_MIN, X_MAX, nx, dtype=np.float64)
    down = x > 0.0

    t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        U = swe_step(U, cfg, dt)
    elapsed = time.perf_counter() - t0

    t_final = n_steps * dt
    h = np.asarray(U[0].tolist(), dtype=np.float64)
    mass_final = float(h.sum() * cfg.dx)
    mass_drift_pct = abs(mass_final - mass0) / mass0 * 100.0

    # Front position (threshold-based)
    h_torch = torch.from_numpy(h)
    x_f_sim = front_position(h_torch, cfg)
    x_f_ref = ritter_front(t_final, H0, G)
    front_err_pct = abs(x_f_sim - x_f_ref) / x_f_ref * 100.0

    # Front position (mass-based) — physically robust estimator
    v_down = float((h * down).sum() * cfg.dx)
    x_f_mass = front_from_mass(v_down, H0)
    front_mass_err_pct = abs(x_f_mass - x_f_ref) / x_f_ref * 100.0

    # Profile L2 error vs Ritter
    mask = (x >= 0.0) & (x <= x_f_ref)
    h_ref = np.array([ritter_depth(float(xi), t_final, H0, G) for xi in x[mask]])
    h_sim = h[mask]
    if len(h_ref) > 0:
        l2_num = np.sqrt(np.sum((h_sim - h_ref) ** 2))
        l2_den = np.sqrt(np.sum(h_ref ** 2))
        profile_l2_pct = l2_num / l2_den * 100.0 if l2_den > 0 else 0.0
    else:
        profile_l2_pct = 0.0

    return {
        "scheme": scheme,
        "nx": nx,
        "cfl": cfl,
        "t_final": round(t_final, 6),
        "n_steps": n_steps,
        "dt": round(dt, 8),
        "elapsed_s": round(elapsed, 4),
        "x_f_sim": round(x_f_sim, 6),
        "x_f_ref": round(x_f_ref, 6),
        "x_f_mass": round(x_f_mass, 6),
        "front_err_pct": round(front_err_pct, 4),
        "front_mass_err_pct": round(front_mass_err_pct, 4),
        "profile_l2_pct": round(profile_l2_pct, 4),
        "mass_initial": round(mass0, 8),
        "mass_final": round(mass_final, 8),
        "mass_drift_pct": round(mass_drift_pct, 8),
        "h_profile": h.tolist(),
        "x_grid": x.tolist(),
    }


def main():
    resolutions = [100, 200, 400, 800]
    cfl = 0.4
    results = {}

    print(f"{'nx':>6} {'front_err%':>12} {'mass_front%':>12} {'profile_L2%':>12} {'mass_drift%':>12}")
    print("-" * 52)

    for nx in resolutions:
        r = run_arm(nx, cfl, scheme="muscl2")
        results[f"nx{nx}"] = r
        print(f"{nx:6d} {r['front_err_pct']:12.4f} {r['front_mass_err_pct']:12.4f} {r['profile_l2_pct']:12.4f} {r['mass_drift_pct']:12.8f}")

    # Save results
    out = HERE / "artifacts" / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # Check acceptance — use mass-based front (physically robust)
    print("\n=== Acceptance Check (mass-based front) ===")
    all_pass = True
    for nx in resolutions:
        r = results[f"nx{nx}"]
        checks = {
            "front_mass_err < 15%": r["front_mass_err_pct"] < 15.0,
            "profile_L2 < 15%": r["profile_l2_pct"] < 15.0,
            "mass_drift < 1%": r["mass_drift_pct"] < 1.0,
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        if not all(checks.values()):
            all_pass = False
        print(f"  nx={nx}: {status} | front_mass={r['front_mass_err_pct']:.4f}% L2={r['profile_l2_pct']:.4f}% mass_drift={r['mass_drift_pct']:.8f}%")

    verdict = "PASS" if all_pass else "FAIL"
    print(f"\nOverall verdict: {verdict}")

    # Save verdict
    verdict_out = HERE / "artifacts" / "verdict.json"
    with open(verdict_out, "w") as f:
        json.dump({
            "verdict": verdict,
            "all_pass": all_pass,
            "criteria": {
                "C1_front_mass_err_pct": {"threshold": 15.0, "unit": "%"},
                "C2_profile_L2_pct": {"threshold": 15.0, "unit": "%"},
                "C3_mass_drift_pct": {"threshold": 1.0, "unit": "%"},
            },
            "results": {nx: {k: v for k, v in r.items() if k not in ("h_profile", "x_grid")}
                        for nx, r in results.items()},
        }, f, indent=2)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
