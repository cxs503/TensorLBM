#!/usr/bin/env python3
"""Recompute Martin & Moyce (1952) dam-break errors on the CORRECT reference.

Background
----------
Martin & Moyce (1952), Part IV, Phil. Trans. R. Soc. A 244, 312-324
(doi:10.1098/rsta.1952.0006) define the dimensionless time

        T = t * sqrt(2 g / a)

where `a` is the initial column WIDTH (the column is a wide x 2a tall) and
the dimensionless front position is

        X = x_front / a          (X(0) = 1).

This is confirmed verbatim by the Lethe post-processing script
(examples/multiphysics/dam-break/dam-break-2d.py):

        time_list = [x * ((2 * g / L1) ** 0.5) for x in time_list]
        x_list    = [x / L1 for x in x_list]

The repo's benchmark scripts (bench_fs_2d.py / dam_break_sc/run.py /
dam_break_3d_mm/run.py) instead used T_code = t * sqrt(g / a), i.e. a factor
sqrt(2) too small.  Therefore every existing dam-break error number is on the
wrong axis and must be recomputed:

        T_MM = sqrt(2) * T_code

Reference table (Lethe digitisation, x_exp/y_exp in dam-break-2d.py), which
matches the K&O-family digitisation used by the other sources:

    T = [0.00, 0.41, 0.84, 1.19, 1.43, 1.63, 1.82, 1.97, 2.20, 2.32, 2.50, 2.64, 2.82, 2.96]
    Z = [1.00, 1.11, 1.23, 1.44, 1.67, 1.89, 2.11, 2.33, 2.56, 2.78, 3.00, 3.22, 3.44, 3.67]

This script reloads the stored benchmark series (no re-simulation) and reports
the error under four schemes:

  [old/old]  old axis  + repo's (wrong) table        -> what was reported
  [old/new]  old axis  + correct table
  [newA/new] new axis  + correct table, method A: the physical sample formerly
             labelled T_code = T* is relabelled T_MM = sqrt(2) T* and compared
             against X_ref(sqrt(2) T*)
  [newB/new] new axis  + correct table, method B: re-interpolate the series at
             the correct checkpoint T_MM = T* (1, 2, 2.96/3)

Usage:  PYTHONPATH=src python scripts/recompute_mm_errors.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# Authoritative digitisation (Lethe x_exp / y_exp; T = t sqrt(2g/a), X=x/a)
# --------------------------------------------------------------------------
LETHE_T = [0.00, 0.41, 0.84, 1.19, 1.43, 1.63, 1.82, 1.97, 2.20,
           2.32, 2.50, 2.64, 2.82, 2.96]
LETHE_Z = [1.00, 1.11, 1.23, 1.44, 1.67, 1.89, 2.11, 2.33, 2.56,
           2.78, 3.00, 3.22, 3.44, 3.67]

# The repo's old (inconsistent) table, still used by the three benchmark files.
OLD_T = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
OLD_Z = [1.0, 1.1, 1.4, 1.8, 2.2, 2.7, 3.1, 3.5, 3.8, 4.1]

SQRT2 = math.sqrt(2.0)


def interp(x: float, xs: list[float], ys: list[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        # linear extrapolation off the last segment (flagged by caller)
        x0, x1, y0, y1 = xs[-2], xs[-1], ys[-2], ys[-1]
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            f = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + f * (ys[i + 1] - ys[i])
    return ys[-1]


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------
def load_3d(name: str) -> tuple[list[float], list[float], dict]:
    p = (ROOT / "benchmarks/pending/dam_break_3d_mm/outputs_remeasure"
         / f"{name}_result.json")
    r = json.loads(p.read_text())
    T = [s["T"] for s in r["series"]]      # T = step*sqrt(g/a)  [code axis]
    X = [s["X"] for s in r["series"]]
    return T, X, r


def load_sc(name: str) -> tuple[list[float], list[float], dict]:
    p = ROOT / "benchmarks/pending/dam_break_sc" / f"hist_{name}.csv"
    T, X = [], []
    with p.open() as fh:
        for row in csv.DictReader(fh):
            T.append(float(row["T_raw"]))   # T_raw = step*sqrt(g/a) [code axis]
            X.append(float(row["X_toe"]))
    case = json.loads((ROOT / "benchmarks/pending/dam_break_sc"
                       / f"case_{name}.json").read_text())
    return T, X, case


def scheme(Tcode, X, Tstar):
    """Return dict with the 4 error schemes at checkpoint Tstar (MM axis)."""
    z_old = interp(Tstar, OLD_T, OLD_Z)
    z_new = interp(Tstar, LETHE_T, LETHE_Z)
    # [old/old] and [old/new]: evaluate the series on the CODE axis at Tstar
    x_code = interp(Tstar, Tcode, X)
    # method A: physical point formerly at code-Tstar, MM time sqrt2*Tstar
    Ta = SQRT2 * Tstar
    x_a = interp(Tstar, Tcode, X)
    z_a = interp(Ta, LETHE_T, LETHE_Z)
    # method B: re-interpolate on the MM axis at the checkpoint itself
    Tmm = [SQRT2 * t for t in Tcode]
    x_b = interp(Tstar, Tmm, X)
    z_b = interp(Tstar, LETHE_T, LETHE_Z)
    return {
        "T": Tstar,
        "T_code_at_checkpoint": Tstar / SQRT2,
        "old_old": {"X_sim": x_code, "X_ref": z_old,
                    "err_pct": 100 * (x_code - z_old) / z_old},
        "old_new": {"X_sim": x_code, "X_ref": z_new,
                    "err_pct": 100 * (x_code - z_new) / z_new},
        "newA_new": {"T_MM": Ta, "X_sim": x_a, "X_ref": z_a,
                     "err_pct": 100 * (x_a - z_a) / z_a},
        "newB_new": {"T_MM": Tstar, "X_sim": x_b, "X_ref": z_b,
                     "err_pct": 100 * (x_b - z_b) / z_b},
    }


CASES = {
    "dam_break_3d_mm a=16 (g=1e-4)": load_3d("a16_g1e-04_rg1.0"),
    "dam_break_3d_mm a=32 (g=1e-4)": load_3d("a32_g1e-04_rg1.0"),
    "dam_break_sc a=40 (g=2e-4)": load_sc("a40"),
    "dam_break_sc a=80 (g=2e-4)": load_sc("a80"),
}


def main() -> None:
    print("#" * 78)
    print("# Martin & Moyce reference recomputation  (T_MM = sqrt(2)*T_code)")
    print("#" * 78)
    print("\nReference checkpoints from the authoritative Lethe digitisation:")
    for T in (1.0, 2.0, 2.96, 3.0):
        tag = " (extrapolated)" if T > LETHE_T[-1] else ""
        print(f"   X_ref(T_MM={T:4.2f}) = {interp(T, LETHE_T, LETHE_Z):.4f}{tag}"
              f"   [old table gave {interp(T, OLD_T, OLD_Z):.4f}]")

    out = {}
    for name, (Tcode, X, _case) in CASES.items():
        Tmax_mm = SQRT2 * Tcode[-1]
        print("\n" + "=" * 78)
        print(f"  {name}   code-axis T_max={Tcode[-1]:.3f}  -> MM-axis T_max={Tmax_mm:.3f}")
        print("=" * 78)
        rows = []
        for Tstar in (1.0, 2.0, 2.96, 3.0):
            if Tstar > Tmax_mm + 1e-9:
                print(f"  T_MM={Tstar:4.2f}: beyond simulated range "
                      f"(T_max={Tmax_mm:.3f}) -> not evaluated")
                continue
            rows.append(scheme(Tcode, X, Tstar))
            s = rows[-1]
            print(f"  T_MM={Tstar:4.2f}  (code axis T={s['T_code_at_checkpoint']:.3f})")
            print(f"     [old axis / old table]  X_sim={s['old_old']['X_sim']:7.4f} "
                  f"X_ref={s['old_old']['X_ref']:7.4f}  err={s['old_old']['err_pct']:+7.2f}%")
            print(f"     [old axis / NEW table]  X_sim={s['old_new']['X_sim']:7.4f} "
                  f"X_ref={s['old_new']['X_ref']:7.4f}  err={s['old_new']['err_pct']:+7.2f}%")
            print(f"     [NEW axis / NEW table, method A]  at T_MM={s['newA_new']['T_MM']:.3f}: "
                  f"X_sim={s['newA_new']['X_sim']:7.4f} X_ref={s['newA_new']['X_ref']:7.4f} "
                  f"err={s['newA_new']['err_pct']:+7.2f}%")
            print(f"     [NEW axis / NEW table, method B]  at T_MM={s['newB_new']['T_MM']:.3f}: "
                  f"X_sim={s['newB_new']['X_sim']:7.4f} X_ref={s['newB_new']['X_ref']:7.4f} "
                  f"err={s['newB_new']['err_pct']:+7.2f}%")
        out[name] = {"T_max_mm": Tmax_mm, "rows": rows}

    dest = ROOT / "benchmarks/pending/dam_break_recompute_mm.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nSaved -> {dest}")


if __name__ == "__main__":
    main()