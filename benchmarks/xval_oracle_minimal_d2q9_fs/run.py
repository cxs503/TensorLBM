#!/usr/bin/env python3
"""Repo-side entry point for anchor XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS.

This is a *caller*: it drives the research harness
``reference_study/resolution_study.py`` (which itself drives the
production solver ``tensorlbm.free_surface_lbm_2d.free_surface_step_2d``) and
writes a machine-readable verdict artifact, so the anchor is reachable from a
normal repo command instead of only from a mission worktree.

    python benchmarks/xval_oracle_minimal_d2q9_fs/run.py            # CI tier (a=16, T<=1)
    python benchmarks/xval_oracle_minimal_d2q9_fs/run.py --full     # a=16/32, T<=2 (slower)

Artifact: benchmarks/xval_oracle_minimal_d2q9_fs/result.json
Exit code 0 = gate met (max |rel err| vs Martin & Moyce 1952 <= 15 %), 1 = gate failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
XV = HERE / "reference_study"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(XV))

import resolution_study as R  # noqa: E402  (module path is the integration contract)

TIERS = {"ci": [(16, 1.0)], "full": [(16, 2.0), (32, 2.0)]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="a=16/32 up to T=2 (slower)")
    ap.add_argument("--out", default=str(HERE / "result.json"))
    args = ap.parse_args()
    levels = TIERS["full" if args.full else "ci"]
    rows, worst, ok = [], 0.0, True
    for a, t_max in levels:
        res = R.run_task({"kind": "arm", "arm": "tensorlbm_koerner", "a": a,
                          "tau": 1.0, "g": R.DESIGN["case"]["g"], "t_max": t_max,
                          "sample_dt": R.DESIGN["sample_dT"]})
        if res.get("status") != "ok":
            rows.append({"grid": f"a={a}", "t_max": t_max, "status": res.get("status"),
                         "metrics": {}})
            ok = False
            continue
        ev = R.eval_run(res, "mm")
        m = {k: ev["vs_mm"][k]["max_abs_rel_err_pct"]
             for k in ("Z_cell", "Z_cell_flag", "Z_sub0.02", "Z_sub0.05", "Z_sub0.1")}
        z = m["Z_sub0.05"]
        worst = max(worst, float(z))
        passed = bool(z is not None and z <= R.DESIGN["thresholds_pct"]["z_max_rel_err"])
        ok = ok and passed
        rows.append({"grid": f"a={a}", "t_max": t_max, "steps": res["steps"],
                     "u_max": round(res["u_max"], 5),
                     "inventory_drift_max_pct": ev.get("inventory_drift_max_pct"),
                     "metrics": m, "z_sub05_pass_15pct": passed,
                     "series": [R.series_row(s) for s in ev["series"]]})
    art = {
        "anchor": R.DESIGN["anchor"],
        "case": "Körner/Thürey D2Q9 free-surface dam break vs Martin & Moyce 1952",
        "reference": R.DESIGN["experiment_reference"],
        "gate": {"metric": "Z_sub0.05", "tol_pct": R.DESIGN["thresholds_pct"]["z_max_rel_err"]},
        "tier": "full" if args.full else "ci",
        "levels": rows,
        "worst_z_sub05_err_pct": worst,
        "ref_vs_got": {"ref": "Martin & Moyce 1952 Fig. 4 (digitised)",
                       "got": "Körner free-surface D2Q9, sub-cell toe f=0.05",
                       "max_abs_rel_err_pct": worst},
        "pass": ok,
        "repro_cmd": "python benchmarks/xval_oracle_minimal_d2q9_fs/run.py --full",
        "series_layout": ["T"] + list(R.SER_KEYS),
    }
    Path(args.out).write_text(json.dumps(art, indent=1))
    print(f"[xval-oracle-d2q9] tier={art['tier']} worst Z_sub.05 err {worst:.2f}% "
          f"gate 15% -> {'PASS' if ok else 'FAIL'}; wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
