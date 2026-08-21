#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""Analyze the 3D cylinder Re=40 7-formula comparison JSONs (D20/D40/D60).

Prints per-grid per-formula Cd_f / Cd / err%, then the decision:
  - which formulas are within +-3% at BOTH grids
  - grid convergence |Cd(D60)-Cd(D40)|/ref <= 3%
  - the best formula per the task rule (principled > scaled)
"""

from __future__ import annotations

import json
import sys

REF = 1.54
FORMULAS = ["standard", "lagrange", "bfl_smooth", "bfl_lag_exact", "faces", "dA_scale", "u05"]


def load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        print("usage: analyze_compare.py <d20.json> <d40.json> <d60.json> ...")
        return 1
    grids = []
    for p in paths:
        try:
            r = load(p)
        except FileNotFoundError:
            print(f"[missing] {p}")
            continue
        grids.append(r)

    print(f"{'formula':14s} | " + " | ".join(f"D{r['D_cells']}: cd_f   cd    err%" for r in grids))
    best = {}
    for k in FORMULAS:
        cells = []
        for r in grids:
            cf = r["cd_friction"][k]
            ct = r["cd_total"][k]
            err = r["err_cd_pct"][k]
            cells.append((cf, ct, err))
            best.setdefault(k, []).append(err)
        print(f"{k:14s} | " + " | ".join(f"{cf:.3f} {ct:.3f} {err:+.2f}" for cf, ct, err in cells))

    print("\n--- per-grid details ---")
    for r in grids:
        D = r["D_cells"]
        print(
            f"D={D}: near={r['n_near_cells']} faces={r['n_wall_faces']} "
            f"ratio={r['face_cell_ratio']:.4f} q_smooth_mean={r['q_smooth_mean']:.4f} "
            f"Cd_p={r['cd_pressure']:.4f} drift={r['drift_cd_pct_std']:+.3f}% "
            f"steps={r['n_finished']} mass_drift={r['mass_drift_pct']:+.4f}%"
        )
        if "per_cell_diag" in r:
            d = r["per_cell_diag"]
            print(
                f"    diag: yface={d['n_y_face_cells']} xonly={d['n_x_only_cells']} "
                f"twoface={d['n_two_face_cells']} | sum_std={d['sum_std_x']:.3f} "
                f"sum_faces={d['sum_faces_x']:.3f} | mean_ux_yface={d['mean_ux_yface']:.5f} "
                f"mean_ux_twface={d['mean_ux_twface']:.5f} mean_ux_xonly={d['mean_ux_xonly']:.5f} "
                f"mean_utx_xonly={d['mean_utx_xonly']:.5f}"
            )

    # decision across the two largest grids present
    big = [r for r in grids if r["D_cells"] >= 40]
    if len(big) >= 2:
        big.sort(key=lambda r: r["D_cells"])
        d40, d60 = big[-2], big[-1]
        print(f"\n--- decision gate (D={d40['D_cells']} vs D={d60['D_cells']}) ---")
        print("gate: |err|<=3% at both grids AND |Cd60-Cd40|/ref<=3% (convergence)")
        for k in FORMULAS:
            e40 = d40["err_cd_pct"][k]
            e60 = d60["err_cd_pct"][k]
            conv = abs(d60["cd_total"][k] - d40["cd_total"][k]) / REF * 100.0
            ok = abs(e40) <= 3.0 and abs(e60) <= 3.0 and conv <= 3.0
            print(
                f"  {k:14s} err40={e40:+.2f}% err60={e60:+.2f}% "
                f"|d60-d40|={conv:.2f}% -> {'PASS' if ok else 'fail'}"
            )
    else:
        print("\n(need >=2 grids with D>=40 for the two-grid gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
