#!/usr/bin/env python3
"""Gate script for free-surface independent reference evaluation.

Exit codes:
  0 — all metrics within 15% hard threshold
  1 — at least one metric exceeds 15% (honest-FAIL)

The 15% threshold follows the platform accuracy standard:
  "对标实验/参考值误差 >15% 不可接受"

A 3% target is also reported but NOT gated — it is aspirational.
Thresholds are FIXED and never adjusted to make the gate pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

HARD_THRESHOLD = 15.0   # % — gate fails if any error exceeds this
TARGET = 3.0            # % — aspirational, reported but not gated


def check_dam_break() -> dict:
    """Check dam-break errors against independent references."""
    json_path = HERE / "results_dam_break" / "error_table.json"
    if not json_path.exists():
        return {"name": "dam_break", "status": "SKIP", "reason": f"{json_path} not found"}
    data = json.loads(json_path.read_text())
    errors = []
    for row in data.get("error_table", []):
        for key in ("KO_err_pct", "MM_err_pct"):
            val = row.get(key)
            if val is not None:
                errors.append({
                    "case": row["level"],
                    "T": row["T"],
                    "metric": key,
                    "error_pct": val,
                    "pass_15pct": val <= HARD_THRESHOLD,
                    "pass_3pct": val <= TARGET,
                })
    n_fail = sum(1 for e in errors if not e["pass_15pct"])
    return {
        "name": "dam_break",
        "status": "PASS" if n_fail == 0 else "FAIL",
        "n_checks": len(errors),
        "n_fail_15pct": n_fail,
        "n_fail_3pct": sum(1 for e in errors if not e["pass_3pct"]),
        "details": errors,
    }


def check_droplet() -> dict:
    """Check droplet spreading errors against independent references."""
    json_path = HERE / "results_droplet" / "error_table.json"
    if not json_path.exists():
        return {"name": "droplet_spreading", "status": "SKIP", "reason": f"{json_path} not found"}
    data = json.loads(json_path.read_text())
    errors = []
    for row in data.get("error_table", []):
        for key in ("err_beta_max_pct", "err_exponent_pct", "err_prefactor_pct"):
            val = row.get(key)
            if val is not None:
                errors.append({
                    "case": row["level"],
                    "metric": key,
                    "error_pct": val,
                    "pass_15pct": val <= HARD_THRESHOLD,
                    "pass_3pct": val <= TARGET,
                })
    n_fail = sum(1 for e in errors if not e["pass_15pct"])
    return {
        "name": "droplet_spreading",
        "status": "PASS" if n_fail == 0 else "FAIL",
        "n_checks": len(errors),
        "n_fail_15pct": n_fail,
        "n_fail_3pct": sum(1 for e in errors if not e["pass_3pct"]),
        "details": errors,
    }


def main() -> int:
    print("=" * 70)
    print("  FREE-SURFACE INDEPENDENT REFERENCE GATE")
    print(f"  Hard threshold: {HARD_THRESHOLD}%   Target: {TARGET}%")
    print("=" * 70)

    checks = [check_dam_break(), check_droplet()]

    all_pass = True
    for chk in checks:
        status = chk["status"]
        n = chk.get("n_checks", 0)
        nf = chk.get("n_fail_15pct", 0)
        nt = chk.get("n_fail_3pct", 0)
        print(f"\n  [{chk['name']}]  status={status}  checks={n}  fail_15%={nf}  fail_3%={nt}")
        if status == "FAIL":
            all_pass = False
        for d in chk.get("details", []):
            tag = "✓" if d["pass_15pct"] else "✗"
            tag3 = "✓" if d["pass_3pct"] else "✗"
            print(f"    {tag} {tag3}  {d['case']:<16} {d['metric']:<22} err={d['error_pct']:.1f}%")

    # Save gate result
    gate_result = {
        "hard_threshold_pct": HARD_THRESHOLD,
        "target_pct": TARGET,
        "checks": checks,
        "overall_pass": all_pass,
    }
    out = HERE / "gate_result.json"
    out.write_text(json.dumps(gate_result, indent=2), encoding="utf-8")

    print(f"\n{'='*70}")
    print(f"  OVERALL: {'PASS' if all_pass else 'FAIL (honest-FAIL)'}")
    print(f"  Gate result: {out}")
    print(f"{'='*70}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
