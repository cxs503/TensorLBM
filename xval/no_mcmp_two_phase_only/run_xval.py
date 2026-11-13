#!/usr/bin/env python3
"""XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY (m18067).

Cross-validation of TensorLBM's no-MCMP two-phase (fill-level free-surface)
capability against the lssfau/walberla 'no-mcmp-two-phase-only' approach
(oss_solutions id=274) on the Martin & Moyce (1952) planar dam break —
the same showcase walberla ships for its fill-level two-phase model.

Arms
----
A (primary): TensorLBM fill-level Korner free surface, D3Q19 slab
   (benchmarks/dam_break/fslbm_driver.py), two resolutions, Re fixed.
B (secondary): TensorLBM color-gradient pseudo-potential dam break
   (tensorlbm.verification.dam_break_2d_validation), two resolutions.

Reference: src/artifacts/martin_moyce_1952_dam_break.json (digitized, sha256).

Zero hand-copy: every number in results.json is computed here from the
simulations and the reference file.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "dam_break"))

from fslbm_driver import FSConfig, run as run_fill_level  # noqa: E402
from tensorlbm.verification.dam_break_2d_validation import (  # noqa: E402
    DamBreak2DConfig,
    interpolate_mm_data,
    load_martin_moyce_data,
    run_dam_break_2d,
)

OUT = Path(__file__).resolve().parent
MM_PATH = ROOT / "src" / "artifacts" / "martin_moyce_1952_dam_break.json"
CHECKPOINTS = (1.0, 2.0, 3.0)
FRONT_DEPTH_THRESHOLD = 0.5  # cells; predeclared


def front_from_depth(depth_x: list[float], L: float) -> float:
    """X* = x_front/L; front = last x index with depth >= threshold."""
    d = np.asarray(depth_x, dtype=float)
    idx = np.where(d >= FRONT_DEPTH_THRESHOLD)[0]
    if idx.size == 0:
        return float("nan")
    return (float(idx.max()) + 0.5) / L


def arm_fill_level(h0: int, L: int) -> dict:
    cfg = FSConfig(h0=h0, column_width=L, c0=0.16, reynolds_target=2000.0)
    t0 = time.time()
    res = run_fill_level(cfg, checkpoints=CHECKPOINTS)
    wall = time.time() - t0
    samples = []
    for cp in CHECKPOINTS:
        key = f"{cp:.3f}"
        prof = res["profiles"].get(key)
        if prof is None:
            continue
        X = front_from_depth(prof["depth_x"], L)
        samples.append({"t_star": cp, "step": prof["step"], "X_star_sim": X})
    return {
        "h0": h0,
        "L": L,
        "grid": res["grid"],
        "c0": res["c0"],
        "g": res["g"],
        "nu": res["nu"],
        "reynolds": res["reynolds"],
        "tau": res["tau"],
        "n_steps": res["n_steps"],
        "wall_s": wall,
        "nan_step": res["nan_step"],
        "mass_drift_max_abs": res["mass_drift_max_abs"],
        "inventory_drift_max_abs": res["inventory_drift_max_abs"],
        "samples": samples,
        "final_depth_xz": res["snapshots"].get("depth_xz"),
    }


def main() -> None:
    mm_data, mm_sha = load_martin_moyce_data(MM_PATH)
    mm_ref = {round(p["t_star"], 3): p["X_star"] for p in mm_data}

    results: dict = {
        "anchor": "XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY",
        "mission_id": 18067,
        "reference": {
            "file": str(MM_PATH),
            "sha256": mm_sha,
            "doi": "10.1098/rsta.1952.0006",
            "points": mm_data,
        },
        "arms": {},
    }

    # ---- Arm A: fill-level (direct walberla approach counterpart) ----
    armA = {}
    for h0, L in ((25, 25), (50, 50)):
        print(f"[armA] fill-level h0={h0} L={L} ...", flush=True)
        armA[f"h0{h0}"] = arm_fill_level(h0, L)
        print(f"[armA] h0={h0} done in {armA[f'h0{h0}']['wall_s']:.1f}s "
              f"nan_step={armA[f'h0{h0}']['nan_step']}", flush=True)
    results["arms"]["A_fill_level"] = armA

    # ---- Arm B: color-gradient (different two-phase family) ----
    armB = {}
    for nx in (48, 64):
        print(f"[armB] color-gradient nx={nx} ...", flush=True)
        cfg = DamBreak2DConfig(nx=nx)
        res = run_dam_break_2d(cfg, mm_data=mm_data)
        armB[f"nx{nx}"] = {
            "nx": nx,
            "ny": res["resolution"]["ny"],
            "dam_width": res["resolution"]["dam_width"],
            "metrics": res["metrics"],
            "criteria": res["criteria"],
            "samples": res["samples"],
        }
        print(f"[armB] nx={nx} done, max_front_err={res['metrics']['max_front_error_pct_early']}%",
              flush=True)
    results["arms"]["B_color_gradient"] = armB

    # ---- M1: front error vs MM at matched t* ----
    for arm_name, arm in results["arms"].items():
        for res_key, res in arm.items():
            for s in res["samples"]:
                t = s["t_star"]
                X_mm = interpolate_mm_data(mm_data, t)
                s["X_star_mm"] = X_mm
                s["front_error_pct"] = 100.0 * abs(s["X_star_sim"] - X_mm) / max(X_mm, 1e-12)
            early = [s for s in res["samples"] if s["t_star"] <= 3.0]
            if early:
                res["max_front_error_pct"] = max(s["front_error_pct"] for s in early)
                res["mean_front_error_pct"] = float(np.mean([s["front_error_pct"] for s in early]))
            else:
                res["max_front_error_pct"] = float("nan")
                res["mean_front_error_pct"] = float("nan")

    # ---- verdicts (predeclared thresholds) ----
    verdicts = {}
    for arm_name, arm in results["arms"].items():
        for res_key, res in arm.items():
            m1 = res.get("max_front_error_pct", float("nan"))
            m2 = res.get("mass_drift_max_abs", float("nan"))
            if arm_name == "B_color_gradient":
                m2 = res["metrics"].get("max_mass_drift_pct", float("nan"))
            verdicts[f"{arm_name}/{res_key}"] = {
                "M1_max_front_error_pct": m1,
                "M1_pass_lt_15pct": (not math.isnan(m1)) and m1 < 15.0,
                "M1_repo_gate_lt_10pct": (not math.isnan(m1)) and m1 < 10.0,
                "M2_max_mass_drift_pct": m2,
                "M2_pass_lt_0_5pct": (not math.isnan(m2)) and m2 < 0.5,
            }
    results["verdicts"] = verdicts

    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(verdicts, indent=2))
    print(f"results -> {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
