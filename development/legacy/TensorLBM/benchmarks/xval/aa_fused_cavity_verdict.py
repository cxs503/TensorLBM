"""Acceptance verdict for the AA-pattern fused stream-collide cavity sweep.

Anchor: ``XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE``.  Consumes the per-task
artifacts written by ``aa_fused_cavity_acceptance.py``
(``artifacts/xval_aa_pattern/cavity_accept_<idx>.json``) and produces

* a single machine-readable verdict (``cavity_verdict.json``), and
* a markdown table block (``cavity_verdict.md``) that the report generator
  pastes verbatim, so no number is ever hand-copied,

under one rule: **the acceptance is the worst gate over the grids that actually
ran, and a grid that did not run is reported as ``not_run``, never as a pass.**

Observed grid order
-------------------
``L2_interior(L)`` is fitted with ``log L2 = log C - p log L`` (least squares on
the grids present, need >= 3).  ``p`` is only quoted when the fit is
monotone-decreasing; otherwise it is ``None`` with the reason recorded, because
a non-monotone sequence has no meaningful order.

Usage
-----
``python benchmarks/xval/aa_fused_cavity_verdict.py``
``python benchmarks/xval/aa_fused_cavity_verdict.py --gate 0.03``
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.abspath(os.path.join(_HERE, "..", "..", "artifacts", "xval_aa_pattern"))

GRIDS = (32, 64, 128, 192)


def load_tasks() -> dict:
    """``{task_index: artifact}`` for every acceptance artifact on disk."""
    out = {}
    for path in sorted(glob.glob(os.path.join(ART, "cavity_accept_*.json"))):
        name = os.path.basename(path)
        if not name[len("cavity_accept_"):-len(".json")].isdigit():
            continue
        idx = int(name[len("cavity_accept_"):-len(".json")])
        with open(path) as fh:
            out[idx] = {"artifact": os.path.relpath(path, os.path.join(ART, "..", "..")),
                        **json.load(fh)}
    return out


def _find(tasks: dict, task: str):
    """First artifact whose ``task`` field matches (index-independent)."""
    for t in tasks.values():
        if t.get("task") == task:
            return t
    return None


def _couette(tasks: dict) -> dict:
    t = _find(tasks, "couette")
    if t is None or "rel_err_max" not in t:
        return {"status": "not_run"}
    return {"status": "ran", "rel_err_max": t["rel_err_max"],
            "fluid_mass_drift_rel": t["fluid_mass_drift_rel"],
            "gate_1pct": bool(t["gate_1pct"])}


def _equality(tasks: dict) -> dict:
    t = _find(tasks, "equality_cavity")
    if t is None:
        return {"status": "not_run"}
    return {"status": "ran", "L": t["L"], "steps": t["steps"],
            "max_abs_diff": t["max_abs_diff"], "bit_equal": bool(t["bit_equal"])}


def _cavity_rows(tasks: dict) -> list:
    rows = []
    seen = set()
    for idx, t in sorted(tasks.items()):
        if t.get("task") != "cavity" or t["L"] in seen:
            continue
        seen.add(t["L"])
        rows.append({
            "task_index": idx,
            "L": t["L"],
            "tau": t["tau"],
            "steps_run": t["steps_run"],
            "converged_at": t.get("converged_at"),
            "L2_u_full": t.get("L2_u_full"),
            "L2_v_full": t.get("L2_v_full"),
            "L2_u_interior": t.get("L2_u_interior"),
            "L2_v_interior": t.get("L2_v_interior"),
            "floor_u": t.get("metric_floor", {}).get("floor_u"),
            "fluid_mass_rel_drift": t.get("fluid_mass_rel_drift"),
            "elapsed_s": t.get("elapsed_s"),
            "artifact": t["artifact"],
        })
    return sorted(rows, key=lambda r: r["L"])


def _order(rows: list, key: str) -> dict:
    pts = [(r["L"], r[key]) for r in rows if r.get(key) is not None and r[key] > 0]
    if len(pts) < 3:
        return {"p": None, "reason": f"need >= 3 grids, have {len(pts)}", "points": pts}
    L = np.log([p[0] for p in pts])
    e = np.log([p[1] for p in pts])
    slope, intercept = np.polyfit(L, e, 1)
    mono = all(pts[i][1] > pts[i + 1][1] for i in range(len(pts) - 1))
    return {"p": float(-slope) if mono else None,
            "reason": None if mono else "error not monotonically decreasing in L",
            "C": float(np.exp(intercept)), "points": pts}


def build(tasks: dict, gate: float = 0.15) -> dict:
    rows = _cavity_rows(tasks)
    worst = None
    if rows:
        scored = [max(r["L2_u_full"], r["L2_v_full"]) for r in rows
                  if r.get("L2_u_full") is not None and r.get("L2_v_full") is not None]
        worst = max(scored) if scored else None
    missing = [L for L in GRIDS if L not in {r["L"] for r in rows}]
    verdict = {
        "anchor": "XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE",
        "acceptance_rule": f"max(L2_u, L2_v) < {gate:g} vs Ghia et al. (1982) Re=100",
        "metric": "L2 of u/U against Ghia centre-line tables; normalisation before subtraction",
        "couette_gate": _couette(tasks),
        "ordering_equality": _equality(tasks),
        "cavity": rows,
        "grids_run": [r["L"] for r in rows],
        "grids_not_run": missing,
        "worst_cavity_error": worst,
        "gate": gate,
        "gate_pass_all_run_grids": (None if worst is None else bool(worst < gate)),
        "observed_order": {"u_interior": _order(rows, "L2_u_interior"),
                           "v_interior": _order(rows, "L2_v_interior")},
        "honest_note": ("a grid in grids_not_run carries no evidence; the pass "
                        "statement is restricted to grids_run"),
    }
    return verdict


def to_markdown(v: dict) -> str:
    lines = ["| L | steps | converged at | L2 u (full) | L2 v (full) | L2 u (interior) "
             "| L2 v (interior) | metric floor u | mass drift |",
             "|---|-------|--------------|-------------|-------------|-----------------"
             "|-----------------|----------------|------------|"]
    for r in v["cavity"]:
        def f(x, fmt="{:.4f}"):
            return "n/a" if x is None else fmt.format(x)
        lines.append(
            f"| {r['L']} | {r['steps_run']} | {f(r['converged_at'], '{}')} | "
            f"{f(r['L2_u_full'])} | {f(r['L2_v_full'])} | {f(r['L2_u_interior'])} | "
            f"{f(r['L2_v_interior'])} | {f(r['floor_u'])} | "
            f"{f(r['fluid_mass_rel_drift'], '{:.1e}')} |")
    return "\n".join(lines)


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=float, default=0.15)
    a = ap.parse_args()
    v = build(load_tasks(), gate=a.gate)
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "cavity_verdict.json"), "w") as fh:
        json.dump(v, fh, indent=2)
    md = to_markdown(v)
    with open(os.path.join(ART, "cavity_verdict.md"), "w") as fh:
        fh.write(md + "\n")
    print(md)
    print("\nverdict:", json.dumps({k: v[k] for k in (
        "grids_run", "grids_not_run", "worst_cavity_error", "gate_pass_all_run_grids")}))
    return v


if __name__ == "__main__":
    main()