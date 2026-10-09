#!/usr/bin/env python
"""W5-C independent verification: recompute gamma/err/verdicts from raw E(t) series.

For every case JSON + energy-history CSV pair: closed-form LSQ (no np.polyfit) of
ln(E) vs step over recorded samples (step >= record_every), gamma_E_sim = -slope,
independent recomputation of gamma_E_theory = 6*nu*k^2 from (n, re, u0), err,
R^2, velocity-rate err, and the precedent-semantics verdict. Cross-checks every
number against result.json / case JSONs. Exits nonzero on any mismatch.

Usage: verify.py [--dir STAGING] [--kernels k1,k2,...] [--include-diagnostics]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PRE_REGISTERED = [
    "cumulant_d3q27",
    "cascaded_d3q27",
    "mrt27",
    "kbc_d3q19",
    "trt27",
    "rlbm27",
]
LADDER = (64, 96, 128)
TOL_GAMMA_REL = 1e-9
TOL_ERR_ABS = 1e-7  # percentage points


def lsq_slope_intercept(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return slope, intercept, 1.0 - ss_res / ss_tot


def load_series(dirpath, kernel, n):
    csv_path = os.path.join(dirpath, f"energy_history_{kernel}_N{n}.csv")
    steps, energies = [], []
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            steps.append(int(float(row["step"])))
            energies.append(float(row["energy"]))
    return steps, energies


def recompute_case(case):
    kernel, n = case["kernel"], case["n"]
    steps, energies = load_series(case["_dir"], kernel, n)
    rec = case["record_every"]
    xs = [float(s) for s, e in zip(steps, energies) if s >= rec]
    ys = [math.log(e) for s, e in zip(steps, energies) if s >= rec]
    slope, _, r2 = lsq_slope_intercept(xs, ys)
    gamma_e_sim = -slope

    ln_u = [0.5 * y for y in ys]
    slope_u, _, _ = lsq_slope_intercept(xs, ln_u)
    gamma_vel_sim = -slope_u

    # independent analytic reference from raw protocol numbers
    nu = case["u0"] * n / case["re"]
    k = 2.0 * math.pi / n
    gamma_theory = 6.0 * nu * k * k
    gamma_vel_theory = 3.0 * nu * k * k

    return {
        "n_samples_fit": len(xs),
        "gamma_e_sim": gamma_e_sim,
        "gamma_e_theory": gamma_theory,
        "err_e_pct": (gamma_e_sim - gamma_theory) / gamma_theory * 100.0,
        "gamma_vel_sim": gamma_vel_sim,
        "gamma_vel_theory": gamma_vel_theory,
        "err_vel_pct": (gamma_vel_sim - gamma_vel_theory) / gamma_vel_theory * 100.0,
        "r2": r2,
        "tau_expected": 0.5 + 3.0 * nu,
        "e0_theory": case["u0"] ** 2 / 8.0,
    }


def verdict_semantics(cases, tol_pct=3.0):
    ns = sorted(c["n"] for c in cases)
    by_n = {c["n"]: c for c in cases}
    errs = [by_n[n]["err_e_pct"] for n in ns]
    errs_v = [by_n[n]["err_vel_pct"] for n in ns]
    r2s = [by_n[n]["r2"] for n in ns]
    converged = all(abs(errs[i + 1]) < abs(errs[i]) for i in range(len(errs) - 1))
    within = all(abs(e) <= tol_pct for e in errs)
    within_v = all(abs(e) <= tol_pct for e in errs_v)
    expo = all(r2 >= 0.999 for r2 in r2s)
    verified = bool(within and within_v and converged and expo and len(ns) >= 2)
    return converged, within, within_v, expo, verified


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--include-diagnostics", action="store_true")
    args = ap.parse_args()

    kernels = list(PRE_REGISTERED) + (["bgk27"] if args.include_diagnostics else [])
    failures = []

    with open(os.path.join(args.dir, "result.json")) as fh:
        result = json.load(fh)

    print(
        f"{'kernel':16s} {'N':>4s} {'g_sim(re)':>12s} {'g_sim(run)':>12s} "
        f"{'err% (re)':>10s} {'err% (run)':>10s} {'R2(re)':>9s} {'ok':>3s}"
    )
    for kernel in kernels:
        rk = result["kernels"].get(kernel)
        if rk is None:
            failures.append(f"{kernel}: missing in result.json")
            continue
        recomputed = []
        for case in rk["cases"]:
            case = dict(case)
            case["_dir"] = args.dir
            path = os.path.join(args.dir, f"case_{kernel}_N{case['n']}.json")
            with open(path) as fh:
                case_disk = json.load(fh)
            for key in ("gamma_e_sim", "err_e_pct", "gamma_vel_sim", "err_vel_pct", "r2"):
                if abs(case_disk[key] - case[key]) > 1e-15:
                    failures.append(
                        f"{kernel} N{case['n']}: result.json vs case JSON disagree on {key}: "
                        f"{case[key]!r} vs {case_disk[key]!r}"
                    )
            rec = recompute_case(case)

            ok = True
            if abs(rec["gamma_e_sim"] - case["gamma_e_sim"]) / case["gamma_e_sim"] > TOL_GAMMA_REL:
                ok = False
                failures.append(
                    f"{kernel} N{case['n']}: gamma_e_sim recompute {rec['gamma_e_sim']:.9e} "
                    f"!= stored {case['gamma_e_sim']:.9e}"
                )
            if abs(rec["err_e_pct"] - case["err_e_pct"]) > TOL_ERR_ABS:
                ok = False
                failures.append(
                    f"{kernel} N{case['n']}: err_e_pct recompute {rec['err_e_pct']:.6f} "
                    f"!= stored {case['err_e_pct']:.6f}"
                )
            if abs(rec["err_vel_pct"] - case["err_vel_pct"]) > TOL_ERR_ABS:
                ok = False
                failures.append(f"{kernel} N{case['n']}: err_vel_pct mismatch")
            if abs(rec["r2"] - case["r2"]) > 1e-9:
                ok = False
                failures.append(f"{kernel} N{case['n']}: r2 mismatch")
            if abs(rec["gamma_e_theory"] - case["gamma_e_theory"]) / case["gamma_e_theory"] > 1e-12:
                ok = False
                failures.append(f"{kernel} N{case['n']}: gamma_e_theory mismatch")
            if abs(rec["tau_expected"] - case["tau"]) > 1e-12:
                ok = False
                failures.append(f"{kernel} N{case['n']}: tau != 0.5+3*nu")
            if abs(case["e0_meas"] - rec["e0_theory"]) / rec["e0_theory"] > 1e-3:
                failures.append(
                    f"{kernel} N{case['n']}: e0_meas {case['e0_meas']:.6e} vs u0^2/8 "
                    f"{rec['e0_theory']:.6e} off by >0.1%"
                )
            if rec["n_samples_fit"] != case["n_samples"]:
                failures.append(f"{kernel} N{case['n']}: sample count mismatch")

            print(
                f"{kernel:16s} {case['n']:4d} {rec['gamma_e_sim']:12.6e} "
                f"{case['gamma_e_sim']:12.6e} {rec['err_e_pct']:+10.4f} "
                f"{case['err_e_pct']:+10.4f} {rec['r2']:9.6f} {'ok' if ok else 'X':>3s}"
            )
            recomputed.append(dict(case, **rec))

        conv, within, within_v, expo, verified = verdict_semantics(recomputed)
        stored = rk["convergence"]
        for key, val in (
            ("converged_monotone", conv),
            ("within_tol", within),
            ("within_tol_vel", within_v),
            ("exponential_r2_ok", expo),
            ("verified", verified),
        ):
            if stored[key] != val:
                failures.append(
                    f"{kernel}: verdict component {key}: stored {stored[key]} != recomputed {val}"
                )
        label = "PASS" if stored["verified"] else "FAIL"
        print(
            f"  -> {kernel}: verified={stored['verified']} ({label}) "
            f"[recomputed verified={verified}]\n"
        )

    if failures:
        print("VERIFY FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("VERIFY OK: all stored gamma/err/r2/verdicts independently reproduced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
