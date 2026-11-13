#!/usr/bin/env python3
"""m18218 audit: independent re-run of m18073 XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS.

Audit questions (independent of the author's claims):
  A. does the shipped headline point (tensorlbm a=32, tmax=1.5) reproduce 13.806 %?
  B. is the pre-declared window T in [0.5, 4.0] honoured, i.e. does the run with
     the script default (--tmax = window_T[1] = 4.0) also stay inside 15 %?
  C. does the OSS-port quirk ablation (rho_prev_alias=False) reproduce 15.961 %?

Writes docs/audit_m18218/repro_results.json (programmatic, no hand copy).
"""
import json
import sys
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "xval" / "oracle_minimal_d2q9_fs"))

import run_xval as X  # noqa: E402

TASKS = [
    {"arm": "tensorlbm_koerner", "a": 32, "tau": 1.0, "g": 5e-5, "t_max": 1.5,
     "sample_dt": 0.25},
    {"arm": "tensorlbm_koerner", "a": 32, "tau": 1.0, "g": 5e-5, "t_max": 4.0,
     "sample_dt": 0.25},
    {"arm": "oss_port", "a": 32, "tau": 1.0, "g": 5e-5, "t_max": 1.5,
     "sample_dt": 0.25, "rho_prev_alias": False},
]


def main() -> int:
    with Pool(processes=3) as pool:
        res = [X.evaluate(r) for r in pool.map(X.run_task, TASKS)]
    out = []
    for r in res:
        slim = {k: v for k, v in r.items() if k not in ("final_fields", "series", "inventory")}
        slim["series_tail"] = (r.get("series") or [])[-3:]
        out.append(slim)
        m = r.get("vs_mm", {})
        print(json.dumps({"label": r.get("label"), "status": r.get("status"),
                          "steps": r.get("steps"), "T_last":
                          (r.get("series") or [{}])[-1].get("T"),
                          "Z_err_max": m.get("max_abs_rel_err_pct"),
                          "pass15": m.get("pass_15pct"),
                          "drift": r.get("inventory_drift_max_pct"),
                          "u_max": r.get("u_max")}, ensure_ascii=False))
    (Path(__file__).parent / "repro_results.json").write_text(
        json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())