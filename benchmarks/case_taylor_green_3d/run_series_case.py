#!/usr/bin/env python
"""CASE-TAYLOR-GREEN-3D round 3: classic (z-dependent) 3-D Taylor–Green vortex
against an exact time-Taylor-series reference.

One command reproduces every number in
``docs/CASE_TAYLOR_GREEN_3D_round3_series_exact.md``::

    python benchmarks/case_taylor_green_3d/run_series_case.py --all

Individual legs::

    --mode prereg        write the predeclared metrics/thresholds (run first)
    --mode exactness     series vs spectral DNS + extruded-limit collapse + truncation
    --mode main          primary run (N=32) incl. field comparison
    --mode mach          amplitude (Mach) sweep -> O(Ma^2) attribution
    --mode grid          grid ladder at fixed (A, Re) -> observed spatial order

Everything lands in ``runs/case_taylor_green_3d_series/<leg>.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.d3q27 import C, W  # noqa: E402
from tensorlbm.tg3d_lbm_check import (  # noqa: E402
    compare_to_series,
    run_tg3d_lbm,
    series_reference,
    tg3d_parameters,
)
from tensorlbm.tg3d_series import (  # noqa: E402
    TG3DSeries,
    convective_acceleration,
    spectral_dns_reference,
    taylor_green_classic,
    taylor_green_extruded,
)

OUT = ROOT / "runs" / "case_taylor_green_3d_series"

DESIGN = {
    "amplitude": 0.05,
    "reynolds": 160.0,
    "n_main": 32,
    "eddy_times": 1.0,
    "series_order": 12,
    "series_order_field": 14,
    "tolerance_pct": 8.0,
    "stretch_tolerance_pct": 3.0,
    "mach_sweep": [0.025, 0.05, 0.10],
    "grid_sizes": [24, 32],
    "grid_amplitude": 0.05,
    "min_spatial_order": 1.8,
}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _dump(name: str, payload) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    print(f"[write] {path} ({path.stat().st_size} B)")
    return path


def _pure_viscous_ratio(n: int, *, amplitude: float, reynolds: float, steps: float) -> float:
    """Ratio E(t)/E(0) if the flow were a single decaying Fourier mode."""
    p = tg3d_parameters(n, amplitude=amplitude, reynolds=reynolds)
    return math.exp(-2.0 * p["nu"] * 3.0 * p["k"] ** 2 * steps)


# ---------------------------------------------------------------------------
# leg 1: prereg (written before any measurement)
# ---------------------------------------------------------------------------


def leg_prereg() -> dict:
    payload = {
        "case": "CASE-TAYLOR-GREEN-3D",
        "round": 3,
        "question": (
            "Does the D3Q27 BGK path reproduce the *nonlinear* evolution of the classic "
            "z-dependent 3-D Taylor–Green vortex, measured against an exact reference?"
        ),
        "reference": {
            "kind": "exact time-Taylor series of the projected Navier–Stokes equations",
            "construction": "u^{n+1} = (nu*Lap u^{n} - P sum_j (u^j . grad) u^{n-j}) / (n+1)",
            "provenance": [
                "Taylor & Green 1937, Proc. R. Soc. A 158:499 (initial field)",
                "Morf, Orszag & Frisch 1980, Phys. Rev. A 23:267 (early-time series)",
                "Brachet et al. 1983, J. Fluid Mech. 130:411 (same flow, series/DNS)",
            ],
            "independent_crosscheck": "dealiased Fourier pseudo-spectral DNS, integrating-factor RK4",
        },
        "design": DESIGN,
        "metrics": {
            "M1": "|E(T=1)/E(0) LBM vs series, rel| < 8 %",
            "M2": "|Omega(T=1)/Omega(0) LBM vs series, rel| < 8 %",
            "M3": "velocity L2 error at T=1, rel to exact rms < 8 %",
            "M4": "vorticity L2 error at T=1, rel to exact rms < 8 %",
            "M5": "series truncation at T=1 (last-order / value) < 2e-3 (reference is exact enough)",
            "M6": "series vs independent spectral DNS agreement at T=1 < 1e-2 (reference is not a code artefact)",
            "M7": "series collapses to exp(-2 nu k^2 t) for the z-invariant IC to < 1e-12",
            "M8": "observed spatial order from the grid ladder >= 1.8",
            "M9": "compressibility bias scales like Ma^2: slope of (E error) vs Ma^2 within [0.3, 3]",
        },
        "falsifiers": [
            "a constant (grid- and time-independent) error offset would indicate a parameter/bias bug, not discretisation",
            "series vs DNS disagreement > 1e-2 would invalidate the reference and the case verdict",
            "series failing to collapse to the closed form for the z-invariant IC would invalidate the recursion",
        ],
        "declared_in_advance": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _dump("prereg", payload)
    return payload


# ---------------------------------------------------------------------------
# leg 2: reference exactness
# ---------------------------------------------------------------------------


def leg_exactness() -> dict:
    n, a, re = DESIGN["n_main"], DESIGN["amplitude"], DESIGN["reynolds"]
    p = tg3d_parameters(n, amplitude=a, reynolds=re)
    order = DESIGN["series_order"]
    ser = series_reference(n, amplitude=a, reynolds=re, order=order)
    steps = int(round(DESIGN["eddy_times"] / p["eddy_time_per_step"]))
    e0, o0 = ser.energy(0.0), ser.enstrophy(0.0)

    # (a) truncation study: what does the tail look like vs time and order?
    trunc = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        step = t / p["eddy_time_per_step"]
        trunc.append({"eddy_time": t, "steps": step, **ser.relative_tail(step)})

    # (b) order-by-order convergence of E and Omega at the primary sample time
    conv = []
    for k in (0, 1, 2, 4, 6, 8, 10, 12, 14, 16):
        conv.append(
            {
                "order": k,
                "energy": ser.energy(steps, n=k),
                "enstrophy": ser.enstrophy(steps, n=k),
                "energy_ratio": ser.energy(steps, n=k) / e0,
                "enstrophy_ratio": ser.enstrophy(steps, n=k) / o0,
            }
        )

    # (c) independent cross-check: pseudo-spectral DNS on the same problem
    t0 = time.time()
    dns = spectral_dns_reference(
        taylor_green_classic(a), box=float(n), nu=p["nu"], steps=steps, dt=1.0, n=n,
        sample_every=max(1, steps // 10),
    )
    dns_wall = time.time() - t0
    dns_rows = []
    for h in dns["history"]:
        t = h["time"]
        dns_rows.append(
            {
                "step": h["step"],
                "eddy_time": t * p["eddy_time_per_step"],
                "energy_dns": h["energy"],
                "energy_series": ser.energy(t),
                "energy_rel": ser.energy(t) / h["energy"] - 1.0 if h["energy"] else float("nan"),
                "enstrophy_dns": h["enstrophy"],
                "enstrophy_series": ser.enstrophy(t),
                "enstrophy_rel": ser.enstrophy(t) / h["enstrophy"] - 1.0 if h["enstrophy"] else float("nan"),
            }
        )

    # (d) z-invariant limit must collapse to the closed form exp(-2 nu k^2 t)
    ser2 = TG3DSeries(taylor_green_extruded(a), box=float(n), nu=p["nu"], order=8)
    k2 = 2.0 * p["k"] ** 2
    extruded = []
    for step in (10.0, 50.0, 100.0, 200.0):
        exact = math.exp(-2.0 * p["nu"] * k2 * step)
        extruded.append(
            {
                "steps": step,
                "energy_ratio_series": ser2.energy(step) / ser2.energy(0.0),
                "energy_ratio_exact": exact,
                "abs_diff": abs(ser2.energy(step) / ser2.energy(0.0) - exact),
                "enstrophy_ratio_series": ser2.enstrophy(step) / ser2.enstrophy(0.0),
                "enstrophy_ratio_exact": exact,
            }
        )

    # (e) why the z-invariant mode is *linear*: its convective acceleration is a
    # gradient, so the projector annihilates it.  The classic mode's is not.
    rot = {
        "extruded": convective_acceleration(taylor_green_extruded(a), box=float(n), n=n),
        "classic": convective_acceleration(taylor_green_classic(a), box=float(n), n=n),
    }

    payload = {
        "n": n,
        "amplitude": a,
        "reynolds": re,
        "parameters": p,
        "steps": steps,
        "series_order": order,
        "series_nfft": ser.nfft,
        "dealias_margin": ser.dealias_margin(),
        "energy_initial": e0,
        "enstrophy_initial": o0,
        "energy_initial_analytic": a * a / 8.0,
        "energy_initial_rel_err": e0 / (a * a / 8.0) - 1.0,
        "truncation": trunc,
        "order_convergence": conv,
        "dns_crosscheck": dns_rows,
        "dns_wall_seconds": dns_wall,
        "extruded_collapse": extruded,
        "nonlinearity_witness": rot,
        "max_energy_rel_vs_dns": max(abs(r["energy_rel"]) for r in dns_rows),
        "max_enstrophy_rel_vs_dns": max(abs(r["enstrophy_rel"]) for r in dns_rows),
        "extruded_collapse_max_abs_diff": max(r["abs_diff"] for r in extruded),
        "truncation_at_main": ser.relative_tail(steps),
    }
    _dump("exactness", payload)
    return payload


# ---------------------------------------------------------------------------
# leg 3: main run
# ---------------------------------------------------------------------------


def leg_main() -> dict:
    n, a, re = DESIGN["n_main"], DESIGN["amplitude"], DESIGN["reynolds"]
    p = tg3d_parameters(n, amplitude=a, reynolds=re)
    ser = series_reference(n, amplitude=a, reynolds=re, order=DESIGN["series_order"])
    t0 = time.time()
    run = run_tg3d_lbm(n, amplitude=a, reynolds=re, eddy_times=DESIGN["eddy_times"])
    wall = time.time() - t0
    rows = compare_to_series(run, ser, order=DESIGN["series_order_field"])
    last = rows[-1]

    # reference-sensitivity ruler: how much would a *linear* (single-mode) code miss?
    viscous_e = _pure_viscous_ratio(n, amplitude=a, reynolds=re, steps=run["steps"])
    e0, o0 = ser.energy(0.0), ser.enstrophy(0.0)
    discrimination = {
        "energy_ratio_series": last["energy_ratio_series"],
        "energy_ratio_pure_viscous": viscous_e,
        "nonlinear_energy_signature_pct": 100.0 * (last["energy_ratio_series"] / viscous_e - 1.0),
        "enstrophy_ratio_series": last["enstrophy_ratio_series"],
        "enstrophy_ratio_pure_viscous": viscous_e,
        "nonlinear_enstrophy_signature_pct": 100.0 * (last["enstrophy_ratio_series"] / viscous_e - 1.0),
        "enstrophy_ratio_lbm": last["enstrophy_ratio_lbm"],
        "lbm_recovers_fraction_of_signature": (
            (last["enstrophy_ratio_lbm"] - viscous_e) / (last["enstrophy_ratio_series"] - viscous_e)
            if abs(last["enstrophy_ratio_series"] - viscous_e) > 0
            else float("nan")
        ),
    }

    # enstrophy production rate dOmega/dT (the 3-D vortex-stretching fingerprint)
    tt = np.array([r["time"] for r in rows], dtype=float)
    T = np.array([r["eddy_time"] for r in rows], dtype=float)
    o_lbm = np.array([r["enstrophy_ratio_lbm"] for r in rows], dtype=float)
    o_ser = np.array([r["enstrophy_ratio_series"] for r in rows], dtype=float)
    prod = {
        "dOmega_dT_lbm": np.gradient(o_lbm, T).tolist(),
        "dOmega_dT_series": np.gradient(o_ser, T).tolist(),
        "eddy_time": T.tolist(),
        "times": tt.tolist(),
    }

    payload = {
        "n": n,
        "parameters": p,
        "steps": run["steps"],
        "tau": run["tau"],
        "series_order": ser.order,
        "series_nfft": ser.nfft,
        "field_order": DESIGN["series_order_field"],
        "runtime_seconds": wall,
        "rows": rows,
        "final": {
            k: last[k]
            for k in (
                "energy_rel_err_pct",
                "enstrophy_rel_err_pct",
                "u_l2_rel_err_pct",
                "vorticity_l2_rel_err_pct",
                "energy_ratio_lbm",
                "energy_ratio_series",
                "enstrophy_ratio_lbm",
                "enstrophy_ratio_series",
            )
        },
        "rho_range": [run["history"][-1]["rho_min"], run["history"][-1]["rho_max"]],
        "mass_drift": run["history"][-1]["rho_max"] - run["history"][-1]["rho_min"],
        "truncation_at_main": ser.relative_tail(run["steps"]),
        "discrimination": discrimination,
        "enstrophy_production": prod,
        "grid": {
            "c": C.tolist(),
            "w": W.tolist(),
        },
    }
    _dump("main", payload)
    return payload


# ---------------------------------------------------------------------------
# leg 4: amplitude (Mach) sweep -> O(Ma^2) attribution
# ---------------------------------------------------------------------------


def leg_mach() -> dict:
    n, re = DESIGN["n_main"], DESIGN["reynolds"]
    rows = []
    for a in DESIGN["mach_sweep"]:
        p = tg3d_parameters(n, amplitude=a, reynolds=re)
        ser = series_reference(n, amplitude=a, reynolds=re, order=DESIGN["series_order"])
        run = run_tg3d_lbm(n, amplitude=a, reynolds=re, eddy_times=DESIGN["eddy_times"])
        cmp_rows = compare_to_series(run, ser, order=DESIGN["series_order_field"])
        last = cmp_rows[-1]
        rows.append(
            {
                "amplitude": a,
                "ma": p["ma"],
                "ma2": p["ma"] ** 2,
                "nu": p["nu"],
                "tau": p["tau"],
                "steps": run["steps"],
                "energy_rel_err_pct": last["energy_rel_err_pct"],
                "enstrophy_rel_err_pct": last["enstrophy_rel_err_pct"],
                "u_l2_rel_err_pct": last["u_l2_rel_err_pct"],
                "vorticity_l2_rel_err_pct": last["vorticity_l2_rel_err_pct"],
                "rows": cmp_rows,
            }
        )
    ma2 = np.array([r["ma2"] for r in rows])
    e = np.array([r["energy_rel_err_pct"] for r in rows])
    slope, intercept = np.polyfit(ma2, e, 1)
    # quality of the linear-in-Ma^2 model
    pred = slope * ma2 + intercept
    ss_res = float(np.sum((e - pred) ** 2))
    ss_tot = float(np.sum((e - e.mean()) ** 2))
    payload = {
        "n": n,
        "reynolds": re,
        "eddy_times": DESIGN["eddy_times"],
        "rows": rows,
        "fit_energy_err_pct_vs_ma2": {
            "slope_pct_per_ma2": float(slope),
            "intercept_pct": float(intercept),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        },
        "interpretation": (
            "a constant offset that scales with Ma^2 is the lattice compressibility bias; "
            "the Ma->0 intercept is the residual (spatial) error at this grid"
        ),
    }
    _dump("mach_sweep", payload)
    return payload


# ---------------------------------------------------------------------------
# leg 5: grid ladder -> observed spatial order
# ---------------------------------------------------------------------------


def leg_grid() -> dict:
    re = DESIGN["reynolds"]
    a = DESIGN["grid_amplitude"]
    out = []
    for n in DESIGN["grid_sizes"]:
        p = tg3d_parameters(n, amplitude=a, reynolds=re)
        ser = series_reference(n, amplitude=a, reynolds=re, order=DESIGN["series_order"])
        run = run_tg3d_lbm(n, amplitude=a, reynolds=re, eddy_times=DESIGN["eddy_times"])
        cmp_rows = compare_to_series(
            run, ser, order=min(DESIGN["series_order_field"], n // 2 - 2)
        )
        last = cmp_rows[-1]
        out.append(
            {
                "n": n,
                "k": p["k"],
                "nu": p["nu"],
                "tau": p["tau"],
                "steps": run["steps"],
                "ma": p["ma"],
                "energy_rel_err_pct": last["energy_rel_err_pct"],
                "enstrophy_rel_err_pct": last["enstrophy_rel_err_pct"],
                "u_l2_rel_err_pct": last["u_l2_rel_err_pct"],
                "vorticity_l2_rel_err_pct": last["vorticity_l2_rel_err_pct"],
                "field_order": last["field_order"],
                "series_truncation": ser.relative_tail(run["steps"]),
                "rows": cmp_rows,
            }
        )
    sizes = np.array([r["n"] for r in out], dtype=float)
    err = np.array([abs(r["u_l2_rel_err_pct"]) for r in out], dtype=float)
    order_fit = float("nan")
    if len(out) >= 2:
        slope = np.polyfit(np.log(sizes), np.log(np.maximum(err, 1e-12)), 1)[0]
        order_fit = -slope
    payload = {
        "amplitude": a,
        "reynolds": re,
        "eddy_times": DESIGN["eddy_times"],
        "rows": out,
        "observed_order_velocity_l2": order_fit,
        "note": (
            "fixed (A, Re, T_end): the continuum problem is identical on every grid, so the "
            "error sequence isolates discretisation. The Ma^2 floor is common to all grids "
            "and therefore biases the order fit *downwards*, not upwards."
        ),
    }
    _dump("grid_ladder", payload)
    return payload


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------


def leg_verdict() -> dict:
    def load(name):
        path = OUT / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    ex, main, mach, grid = load("exactness"), load("main"), load("mach_sweep"), load("grid_ladder")
    checks = []
    if main:
        f = main["final"]
        checks += [
            ("M1 energy rel err < 8 %", abs(f["energy_rel_err_pct"]), DESIGN["tolerance_pct"]),
            ("M2 enstrophy rel err < 8 %", abs(f["enstrophy_rel_err_pct"]), DESIGN["tolerance_pct"]),
            ("M3 velocity L2 < 8 %", abs(f["u_l2_rel_err_pct"]), DESIGN["tolerance_pct"]),
            ("M4 vorticity L2 < 8 %", abs(f["vorticity_l2_rel_err_pct"]), DESIGN["tolerance_pct"]),
        ]
    if ex:
        checks += [
            ("M5 series truncation at T=1 < 2e-3", abs(ex["truncation_at_main"]["enstrophy_step_rel"]), 2e-3),
            ("M6 series vs DNS < 1e-2", max(ex["max_energy_rel_vs_dns"], ex["max_enstrophy_rel_vs_dns"]), 1e-2),
            ("M7 extruded collapse < 1e-12", ex["extruded_collapse_max_abs_diff"], 1e-12),
        ]
    if grid:
        checks.append(("M8 observed order >= 1.8", grid["observed_order_velocity_l2"], None))
    if mach:
        checks.append(
            (
                "M9 compressibility slope in [0.3, 3] %/Ma^2",
                abs(mach["fit_energy_err_pct_vs_ma2"]["slope_pct_per_ma2"]),
                None,
            )
        )
    verdict_rows = []
    for name, val, thr in checks:
        if "M8" in name:
            ok = val >= DESIGN["min_spatial_order"]
        elif "M9" in name:
            ok = 0.3 <= val <= 3.0
        else:
            ok = val < thr
        verdict_rows.append({"check": name, "value": val, "threshold": thr, "pass": bool(ok)})
    payload = {
        "checks": verdict_rows,
        "pass": all(r["pass"] for r in verdict_rows) if verdict_rows else False,
        "n_checks": len(verdict_rows),
    }
    _dump("verdict", payload)
    return payload


LEGS = {
    "prereg": leg_prereg,
    "exactness": leg_exactness,
    "main": leg_main,
    "mach": leg_mach,
    "grid": leg_grid,
    "verdict": leg_verdict,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=sorted(LEGS), help="run one leg")
    ap.add_argument("--all", action="store_true", help="run every leg in order")
    args = ap.parse_args(argv)
    if args.all:
        order = ["prereg", "exactness", "main", "mach", "grid", "verdict"]
        for name in order:
            t0 = time.time()
            print(f"=== leg {name} ===")
            LEGS[name]()
            print(f"--- {name} done in {time.time() - t0:.1f} s")
        return 0
    if not args.mode:
        ap.error("give --mode or --all")
    LEGS[args.mode]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())