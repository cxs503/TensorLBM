#!/usr/bin/env python3
"""Convergence analysis for XVAL-LBM-FREE-SURFACE-FILLLEVEL-RECONSTRUCT-REFIL (round 2).

Verifies the round-1 (m17768) extrapolation claim:
  "a≈36–48 才可能把逐点最大误差压进 15%"  (was UNVERIFIED at a>=32)

Reads all pg2_*_a*.json + the round-1 xval_results.json, builds the
resolution-convergence table for the max pointwise front-position error
and the rel-RMSE, and emits a verdict + convergence figure.

Usage:
  python development/experiments/xval_filllevel_refill_2d/analyze_convergence.py
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"

# round-1 baseline (m17768) — repo_average, rho_g=1, tau=0.6
R1 = {
    16: {"rel_rmse_pct": 20.889593092766066, "max_rel_err_pct": 30.177111716621255,
         "mass_drift_pct": 0.010038806546118951},
    24: {"rel_rmse_pct": 14.570153485825477, "max_rel_err_pct": 22.797456857402356,
         "mass_drift_pct": 0.01635767595289373},
}

def load_pg2(a):
    p = ART / f"pg2_repo_average_a{a}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    r = d["results"][0]
    return {"rel_rmse_pct": r["rel_rmse_pct"], "max_rel_err_pct": r["max_rel_err_pct"],
            "mass_drift_pct": r["mass_drift_pct"], "T_max": r["T_max"],
            "Z_end": r["Z_end"], "steps": r["steps"], "wall_seconds": r["wall_seconds"],
            "n_cmp": r["n_cmp"], "compare": r["compare"]}

RES = [16, 24, 32, 36, 40, 48, 64]
rows = []
for a in sorted(RES):
    r = R1.get(a) or load_pg2(a)
    if r is None:
        print(f"[missing] a={a} — not run yet")
        continue
    rows.append({"a": a, **r})

print("\n=== resolution-convergence (repo_average, rho_g=1, tau=0.6) ===")
print(f"{'a':>4} {'rel_RMSE%':>10} {'max|err|%':>10} {'drift%':>10} {'pass_rmse':>9} {'pass_max':>8}")
for r in rows:
    print(f"{r['a']:>4} {r['rel_rmse_pct']:>10.2f} {r['max_rel_err_pct']:>10.2f} "
          f"{r['mass_drift_pct']:>10.3e} {str(r['rel_rmse_pct']<=15):>9} {str(r['max_rel_err_pct']<=15):>8}")

# --- extrapolation check: fit max|err| vs 1/a (linear in 1/a) ---
if len(rows) >= 3:
    xs = [1.0 / r["a"] for r in rows]
    ys = [r["max_rel_err_pct"] for r in rows]
    n = len(xs)
    sx = sum(xs); sy = sum(ys); sxx = sum(x*x for x in xs); sxy = sum(x*y for x, y in zip(xs, ys))
    syy = sum(y*y for y in ys)
    denom = n * sxx - sx * sx
    m = (n * sxy - sx * sy) / denom
    b = (sy - m * sx) / n
    r2 = (n * sxy - sx * sy) ** 2 / (denom * (n * syy - sy * sy)) if denom * (n * syy - sy * sy) else float("nan")
    a15 = 1.0 / ((15.0 - b) / m) if m > 0 else float("inf")
    print(f"\n[fit] max|err| = {m:.2f}/a + {b:.2f}  (R2={r2:.3f})")
    print(f"[fit] predicted a for max|err|=15% : {a15:.1f}")
    ys2 = [r["rel_rmse_pct"] for r in rows]
    sxy2 = sum(x*y for x, y in zip(xs, ys2)); syy2 = sum(y*y for y in ys2)
    m2 = (n * sxy2 - sx * sy) / denom
    b2 = (sy - m2 * sx) / n
    r22 = (n * sxy2 - sx * sy) ** 2 / (denom * (n * syy2 - sy * sy)) if denom * (n * syy2 - sy * sy) else float("nan")
    a15r = 1.0 / ((15.0 - b2) / m2) if m2 > 0 else float("inf")
    print(f"[fit] rel_RMSE = {m2:.2f}/a + {b2:.2f}  (R2={r22:.3f}) ; predicted a for rel_RMSE=15% : {a15r:.1f}")

# --- verdict ---
max_err = [r["max_rel_err_pct"] for r in rows]
rmse = [r["rel_rmse_pct"] for r in rows]
print("\n=== VERDICT ===")
print("max|err| at highest resolved a:", max_err[-1] if max_err else "n/a")
print("rel_RMSE at highest resolved a:", rmse[-1] if rmse else "n/a")
if max_err and max_err[-1] <= 15:
    print(f"PASS: max pointwise error < 15% at a={rows[-1]['a']} — round-1 extrapolation CONFIRMED")
elif max_err:
    print(f"FAIL: max pointwise error still {max_err[-1]:.2f}% at a={rows[-1]['a']} — round-1 extrapolation NOT yet confirmed")
else:
    print("INSUFFICIENT DATA")

# --- write summary csv ---
import csv
with open(ART / "xval_convergence_r2.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["a", "rel_rmse_pct", "max_rel_err_pct", "mass_drift_pct", "pass_rmse", "pass_max"])
    for r in rows:
        w.writerow([r["a"], r["rel_rmse_pct"], r["max_rel_err_pct"], r["mass_drift_pct"],
                    r["rel_rmse_pct"] <= 15, r["max_rel_err_pct"] <= 15])
print("csv ->", ART / "xval_convergence_r2.csv")
