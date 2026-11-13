#!/usr/bin/env python3
"""m17853 board audit — independent re-run comparison for m17726.

Compares, point by point, the shipped ``runs/<tag>/result.json`` of mission
m17726 (snapshot in ``refs/xval``) against the auditor's own re-run of the very
same deliverable code (``refs/rerun``, driven by ``rerun_all.sh``).

Nothing is typed by hand: every number in the audit table comes from those two
sets of JSON files.  Writes ``artifacts/rerun_compare.json``.

Run: python3 docs/board_audit/m17853/compare_rerun.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIPPED = HERE / "refs/xval/runs"
RERUN = HERE / "refs/rerun/runs"
ART = HERE / "artifacts"
TOL_PCT = 15.0          # predeclared accept_pct in refs/xval/predeclared.json


def rows(base: Path) -> dict:
    out = {}
    for p in sorted(base.glob("*/result.json")):
        d = json.loads(p.read_text())
        if d.get("arm") == "port":
            out[p.parent.name] = d
    return out


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    ship, rerun = rows(SHIPPED), rows(RERUN)
    missing = sorted(set(ship) - set(rerun))
    table = []
    for tag in sorted(ship):
        s = ship[tag]
        r = rerun.get(tag)
        if r is None:
            table.append({"tag": tag, "status": "not_rerun"})
            continue
        d_theta = r["theta_deg"] - s["theta_deg"]
        table.append({
            "tag": tag,
            "psi_wall": s["psi_wall"],
            "xi": s["interface_width"],
            "theta_shipped_deg": s["theta_deg"],
            "theta_rerun_deg": r["theta_deg"],
            "delta_deg": d_theta,
            "rel_delta_pct": 100.0 * d_theta / s["theta_deg"] if s["theta_deg"] else None,
            "theta_analytic_deg": s["theta_analytic_deg"],
            "err_vs_analytic_shipped_pct": s["err_vs_analytic_pct"],
            "err_vs_analytic_rerun_pct": r["err_vs_analytic_pct"],
            "within_15pct_shipped": abs(s["err_vs_analytic_pct"]) <= TOL_PCT,
            "within_15pct_rerun": abs(r["err_vs_analytic_pct"]) <= TOL_PCT,
            "converged": r["converged"],
            "iterations": r["iterations"],
            "drift": r["drift"],
        })
    done = [t for t in table if t.get("status") != "not_rerun"]
    errs = sorted(abs(t["err_vs_analytic_rerun_pct"]) for t in done)
    summary = {
        "n_points_shipped": len(ship),
        "n_points_rerun": len(done),
        "missing_reruns": missing,
        "max_abs_delta_deg": max(abs(t["delta_deg"]) for t in done) if done else None,
        "bitwise_identical_points": sum(abs(t["delta_deg"]) < 1e-9 for t in done),
        "n_in_15pct_rerun": sum(t["within_15pct_rerun"] for t in done),
        "median_abs_err_pct_rerun": (errs[len(errs) // 2] if len(errs) % 2 else
                                     (errs[len(errs) // 2 - 1] + errs[len(errs) // 2]) / 2.0),
        "max_abs_err_pct_rerun": errs[-1] if errs else None,
        "all_converged_rerun": all(t["converged"] for t in done),
        "tol_pct": TOL_PCT,
    }
    out = ART / "rerun_compare.json"
    out.write_text(json.dumps({"summary": summary, "rows": table}, indent=2))
    print(out, "written")
    print(json.dumps(summary, indent=1))
    for t in table:
        if t.get("status") == "not_rerun":
            print(f"{t['tag']:<28} NOT RERUN")
            continue
        print(f"{t['tag']:<28} shipped {t['theta_shipped_deg']:8.3f}  rerun {t['theta_rerun_deg']:8.3f}  "
              f"d={t['delta_deg']:+.2e}  err {t['err_vs_analytic_rerun_pct']:+7.2f}%  "
              f"{'PASS' if t['within_15pct_rerun'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())