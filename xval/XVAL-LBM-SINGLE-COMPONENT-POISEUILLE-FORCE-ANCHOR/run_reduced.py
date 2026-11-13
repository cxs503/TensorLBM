#!/usr/bin/env python3
"""Reduced-but-predeclared execution of the anchor matrix (wall-clock budget).

Selection (documented deviation from the full predeclared matrix, see
``REPORT.md`` §5): the case list below is a subset of :func:`run_xval.build_cases`
chosen BEFORE any of these cases was run, by dropping the cells whose only role
is redundancy (the middle tau values 0.60/1.00, the two TRT cells outside the
base case, and ny=64 for the port cell AR_OS).  Every predeclared axis keeps at
least two levels: grid {16,32,64}, tau {0.55,0.80,1.50}, u_target {0.01,0.02,0.05}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_xval as R  # noqa: E402

WANT = [
    # factorial base (all six arms)
    "AO_OS|ny32|tau0.8|ut0.02|factorial_base",
    "AO_RS|ny32|tau0.8|ut0.02|factorial_base",
    "AR_RS|ny32|tau0.8|ut0.02|factorial_base",
    "AR_OS|ny32|tau0.8|ut0.02|factorial_base",
    "AR_RS|ny32|tau0.8|ut0.02|factorial_base",  # TRT cells added below by arm name
    # grid sweep
    "AO_OS|ny16|tau0.8|ut0.02|grid_sweep",
    "AR_RS|ny16|tau0.8|ut0.02|grid_sweep",
    "AR_OS|ny16|tau0.8|ut0.02|grid_sweep",
    "AO_OS|ny64|tau0.8|ut0.02|grid_sweep",
    "AR_RS|ny64|tau0.8|ut0.02|grid_sweep",
    # relaxation-rate sensitivity (two extreme levels + base)
    "AO_OS|ny32|tau0.55|ut0.02|tau_sweep",
    "AR_RS|ny32|tau0.55|ut0.02|tau_sweep",
    "AO_OS|ny32|tau1.5|ut0.02|tau_sweep",
    "AR_RS|ny32|tau1.5|ut0.02|tau_sweep",
    "AR_OS|ny32|tau1.5|ut0.02|tau_sweep",
    # force / Mach sensitivity
    "AO_OS|ny32|tau0.8|ut0.01|force_sweep",
    "AO_RS|ny32|tau0.8|ut0.01|force_sweep",
    "AR_RS|ny32|tau0.8|ut0.01|force_sweep",
    "AR_OS|ny32|tau0.8|ut0.01|force_sweep",
    "AO_OS|ny32|tau0.8|ut0.05|force_sweep",
    "AO_RS|ny32|tau0.8|ut0.05|force_sweep",
    "AR_RS|ny32|tau0.8|ut0.05|force_sweep",
    "AR_OS|ny32|tau0.8|ut0.05|force_sweep",
]


def selected() -> list[dict]:
    cases = R.build_cases()
    by_id = {c["case_id"]: c for c in cases}
    out, seen = [], set()
    for cid in WANT:
        if cid in seen:
            continue
        seen.add(cid)
        out.append(by_id[cid])
    # the two TRT cells of the factorial base (ids carry the arm suffix)
    for c in cases:
        if c["tag"] == "factorial_base" and c["arm"].endswith("TRT") and c["case_id"] not in seen:
            seen.add(c["case_id"])
            out.append(c)
    return out


def main() -> int:
    cases = selected()
    rows = []
    for case in cases:
        print(f"[run] {case['case_id']}", flush=True)
        rec = R.run_case(case)
        print(f"      err_u_max={rec['metrics']['rel_err_u_max_pct']:.4f}%  "
              f"nu_eff/nu={rec['metrics']['nu_eff_over_nu']:.5f}  steps={rec['metrics']['steps']}"
              f"  {rec['wall_clock_s']}s", flush=True)
        rows.append(rec)
        (HERE / "results_shard_0.json").write_text(
            json.dumps({"shard": 0, "nshards": 1, "cases": rows}, indent=1)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())