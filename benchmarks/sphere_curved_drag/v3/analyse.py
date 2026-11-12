#!/usr/bin/env python3
"""Three-instrument force comparison for the converged sphere control case.

Reads every record in ``benchmarks/sphere_curved_drag/out/*.json`` and emits
``v3/three_instrument.json`` + the markdown tables used by
``docs/GAP-LBM-SPHERE_v3_converged_control_verdict.md``.  No number in the
report is typed by hand: the doc is rendered from the JSON by ``render.py``.
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"

KEYS = [
    "tag", "controller", "L", "R", "lambda", "Re_target", "Re_measured", "tau",
    "u_target", "u_measured", "steps", "steady", "u_tail_drift", "g_tail_drift",
    "g_tail", "F_momentum_balance", "F_bfl", "F_analytic", "bfl_rel_diff_pct",
    "analytic_vs_mb_pct", "analytic_vs_bfl_pct", "cd_mb_fluid", "cd_mb_sup",
    "cd_stokes_confined", "mfp_over_R", "hydrodynamic_regime", "diverged",
    "wall_s", "n_bfl_links",
]


def load() -> list[dict]:
    rows = []
    for p in sorted(glob.glob(str(OUT / "*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, dict) or "tag" not in d:
            continue
        d["_file"] = Path(p).name
        rows.append(d)
    return rows


def summarise(rows: list[dict]) -> dict:
    steady = [r for r in rows if r.get("steady") and not r.get("diverged")]
    stokes = [r for r in steady if r.get("hydrodynamic_regime")
              and abs(r.get("Re_target", 0) - 0.1) < 1e-9]
    verdict = {}
    for r in stokes:
        k = r["hasimoto_K"] if "hasimoto_K" in r else None
        verdict[r["tag"]] = {
            "lambda": r["lambda"],
            "Re_measured": r["Re_measured"],
            "cd_mb_fluid": r["cd_mb_fluid"],
            "cd_stokes_confined": r["cd_stokes_confined"],
            "s1_err_pct_mb": 100.0 * (r["cd_mb_fluid"] / r["cd_stokes_confined"] - 1.0),
            "s1_pass_3pct": bool(abs(r["cd_mb_fluid"] / r["cd_stokes_confined"] - 1.0) < 0.03),
            "s2_bfl_vs_mb_pct": r["bfl_rel_diff_pct"],
            "s2_pass_2pct": bool(abs(r["bfl_rel_diff_pct"]) < 2.0),
            "analytic_vs_mb_pct": r["analytic_vs_mb_pct"],
            "analytic_vs_bfl_pct": r["analytic_vs_bfl_pct"],
        }
    return {
        "n_records": len(rows),
        "n_steady": len(steady),
        "steady_tags": [r["tag"] for r in steady],
        "stokes_steady": verdict,
        "rows": [{k: r.get(k) for k in KEYS} for r in rows],
    }


def main() -> int:
    rows = load()
    s = summarise(rows)
    (HERE / "three_instrument.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: s[k] for k in ("n_records", "n_steady", "steady_tags", "stokes_steady")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
