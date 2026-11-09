#!/usr/bin/env python3
"""Zero-pressure-gradient flat-plate benchmark (criterion c).

Runs the wall-law-driven momentum-integral boundary layer over
``Re_theta = 1e4 ... 5e4`` (the mission's band) and compares the local skin
friction with the ITTC-1957 line (differentiated to its local form) and with
Prandtl-Schlichting.

Design (predeclared, see the report)
------------------------------------
``baseline``     ``kappa = 0.41``, ``B = 5.2``, ``Pi = 0.45``  -- Coles' value
                 for a zero-pressure-gradient boundary layer.
``criterion``    max relative error of ``c_f`` inside the band <= 5 %.
``secondary``    a one-factor-at-a-time sweep of ``Pi`` and ``kappa`` *after*
                 the baseline has been recorded, purely as a diagnosis of the
                 model-form sensitivity.  The baseline verdict is not changed by
                 the sweep (no threshold tuning).

Usage
-----
    python benchmarks/wall_model_les/flat_plate.py --out artifacts/wall_model_les
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.wall_model_les.flat_plate import (  # noqa: E402
    ittc_1957_local,
    prandtl_schlichting_local,
    run_flat_plate,
)

BASELINE = {"kappa": 0.41, "b": 5.2, "pi_wake": 0.45}
BAND = (1.0e4, 5.0e4)
CRITERION = 0.05
# 80 stations over the Re_theta band and 401 wall-normal quadrature points give
# <0.1 % change in c_f versus the 200/801 resolution (checked once, recorded in
# the report); the coarser setting keeps the sensitivity sweep affordable.
N_STEPS = 80
N_POINTS = 401


def _classical_local_cf(re_theta: np.ndarray) -> np.ndarray:
    """``c_f = 0.024 Re_theta^-0.25`` (classical ZPG correlation, diagnostic)."""
    return 0.024 * re_theta ** -0.25


def run(out_dir: Path) -> dict:
    result: dict = {"band": BAND, "baseline": BASELINE, "criterion_max_rel_error": CRITERION}

    base = run_flat_plate(n_steps=N_STEPS, n_profile_points=N_POINTS, **BASELINE)
    metrics = base.metrics()
    cf_classical = _classical_local_cf(base.re_theta)
    rel_classical = np.abs(base.c_f - cf_classical) / cf_classical
    metrics["max_rel_error_vs_classical_024_retheta"] = float(rel_classical.max())
    metrics["rms_rel_error_vs_classical_024_retheta"] = float(
        np.sqrt(np.mean(rel_classical**2))
    )
    result["baseline_metrics"] = metrics
    result["profile_samples"] = {
        "re_theta": base.re_theta[:: max(1, len(base.re_theta) // 12)].tolist(),
        "re_x": base.re_x[:: max(1, len(base.re_x) // 12)].tolist(),
        "c_f": base.c_f[:: max(1, len(base.c_f) // 12)].tolist(),
        "shape_factor": base.shape_factor[:: max(1, len(base.shape_factor) // 12)].tolist(),
        "delta_plus": base.delta_plus[:: max(1, len(base.delta_plus) // 12)].tolist(),
    }
    result["verdict_baseline_pass"] = bool(
        metrics["max_rel_error_vs_ittc_local"] <= CRITERION
    )

    # ---- secondary sensitivity (diagnosis only, after the verdict) ------
    sweep = {}
    for pi_wake in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55):
        res = run_flat_plate(kappa=BASELINE["kappa"], b=BASELINE["b"], pi_wake=pi_wake,
                             n_steps=N_STEPS, n_profile_points=N_POINTS)
        m = res.metrics()
        sweep[f"pi_{pi_wake:.2f}"] = {
            "max_rel_error_vs_ittc_local": m["max_rel_error_vs_ittc_local"],
            "shape_factor_start": m["shape_factor_start"],
            "shape_factor_end": m["shape_factor_end"],
            "c_f_start": m["c_f_start"],
            "c_f_end": m["c_f_end"],
        }
    for kappa in (0.38, 0.41, 0.44):
        res = run_flat_plate(kappa=kappa, b=BASELINE["b"], pi_wake=BASELINE["pi_wake"],
                             n_steps=N_STEPS, n_profile_points=N_POINTS)
        m = res.metrics()
        sweep[f"kappa_{kappa:.2f}"] = {
            "max_rel_error_vs_ittc_local": m["max_rel_error_vs_ittc_local"],
            "shape_factor_start": m["shape_factor_start"],
            "c_f_start": m["c_f_start"],
        }
    result["sensitivity_sweep"] = sweep

    # ---- inverse check: what Pi reproduces ITTC at this Re_theta band? --
    # Bisection on Pi is a *diagnostic* only (it quantifies how strongly the
    # verdict is controlled by the assumed wake strength).
    lo, hi = 0.20, 0.60
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        res = run_flat_plate(kappa=BASELINE["kappa"], b=BASELINE["b"], pi_wake=mid,
                             n_steps=N_STEPS, n_profile_points=N_POINTS)
        if res.metrics()["max_rel_error_vs_ittc_local"] > CRITERION:
            lo = mid
        else:
            hi = mid
    result["pi_wake_matching_ittc_within_criterion"] = 0.5 * (lo + hi)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "flat_plate_profiles.npz",
        re_theta=base.re_theta,
        re_x=base.re_x,
        c_f=base.c_f,
        ue_plus=base.ue_plus,
        delta_plus=base.delta_plus,
        shape_factor=base.shape_factor,
        cf_ittc=ittc_1957_local((base.re_theta / 0.0365) ** 1.25),
        cf_prandtl_schlichting=prandtl_schlichting_local((base.re_theta / 0.0365) ** 1.25),
        cf_classical=cf_classical,
    )
    (out_dir / "flat_plate.json").write_text(json.dumps(result, indent=2, default=float))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/wall_model_les")
    args = parser.parse_args()
    result = run(Path(args.out))
    print(json.dumps(result["baseline_metrics"], indent=2, default=float))
    print("verdict baseline pass:", result["verdict_baseline_pass"])


if __name__ == "__main__":
    main()