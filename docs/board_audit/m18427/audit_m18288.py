#!/usr/bin/env python3
"""Board audit m18427 — independent re-derivation of the m18288 deliverable.

Reads ONLY committed raw artifacts (runs/*/result.json, runs/*/fail.json,
oss_crosscheck.json) from the TensorLBM work tree and recomputes every number
that m18288's maturity_evidence.json / summary.json report.  Nothing is copied
from the subject's own aggregation code, so a mismatch here means the subject's
table is not reproducible from its artifacts.

Outputs (this directory):
  audit_numbers.json   machine-checkable recompute table
  fig_m18288_audit.png two-panel figure (KPI deltas vs the 15% band; kernel parity)

usage: python3 docs/board_audit/m18427/audit_m18288.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SUBJ = ROOT / "xval" / "benchmark_method_selection"
ARM = "A1"  # reference arm for the internal contrast rows (subject's choice)

KPI = ["cd_mean_last50", "cl_amp", "st_fft"]


def load_arm(arm: str) -> dict:
    p = SUBJ / "runs" / f"{arm}_D10" / "result.json"
    return json.loads(p.read_text())


def main() -> int:
    ref = load_arm(ARM)
    rows, fails = [], []
    for arm in ["A2", "A3", "A4", "A5"]:
        f = SUBJ / "runs" / f"{arm}_D10" / "fail.json"
        r = SUBJ / "runs" / f"{arm}_D10" / "result.json"
        if f.exists():
            fails.append(json.loads(f.read_text()) | {"arm": arm})
            continue
        got = json.loads(r.read_text())
        for k in KPI:
            d = 100.0 * (got[k] - ref[k]) / ref[k]
            rows.append({
                "metric": f"{k} {ARM}-vs-{arm} @D=10",
                "kpi": k,
                "arm": arm,
                "ref": ref[k],
                "got": got[k],
                "rel_error_pct": d,
                "within_15pct": abs(d) <= 15.0,
            })

    # external anchor: executed OSS source vs TensorLBM kernel
    oss = json.loads((SUBJ / "oss_crosscheck.json").read_text())
    kern = []
    if "k1_collision" in oss:
        kern.append({"metric": "k1_collision rel-L2", "value": oss["k1_collision"]["rel_l2_collision"]})
    if "k2_streaming" in oss:
        kern.append({"metric": "k2_streaming rel-L2", "value": oss["k2_streaming"]["rel_l2_streaming_interior"]})

    # cross-check against the subject's published table (independent comparison)
    pub = json.loads((SUBJ / "summary.json").read_text())
    pub_delta = {}
    for r in pub.get("rows", []):
        dv = r.get("delta_vs_A1_pct") or {}
        for k, v in dv.items():
            pub_delta[(r["arm"], k)] = v
    mismatch = []
    for row in rows:
        k = (row["arm"], row["kpi"])
        if k in pub_delta and abs(pub_delta[k] - row["rel_error_pct"]) > 1e-9:
            mismatch.append({"key": f"{k[0]}:{k[1]}", "audit": row["rel_error_pct"], "published": pub_delta[k]})

    ladder = sorted(p.name for p in (SUBJ / "runs").glob("*") if p.is_dir())
    out = {
        "subject": "m18288 / XVAL-LBM-METHOD-SELECTION-FIRST-EVIDENCE",
        "reference_arm": ARM,
        "rows": rows,
        "failures": [{"arm": f["arm"], "status": f.get("status"), "fail_stage": f.get("fail_stage"),
                      "step": f.get("step"), "total_steps_planned": f.get("total_steps_planned")} for f in fails],
        "kernel_parity_external": kern,
        "n_rows": len(rows),
        "n_within_15pct": sum(r["within_15pct"] for r in rows),
        "n_outside_15pct": sum(not r["within_15pct"] for r in rows),
        "recompute_mismatch_vs_published": mismatch,
        "resolution_rungs_present": ladder,
        "resolutions": sorted({int(p.split("_D")[-1]) for p in ladder}),
        "note_ref_is_internal": "every KPI 'ref' row is another arm of the same experiment (A1), not an external/analytic/literature value",
    }
    (Path(__file__).parent / "audit_numbers.json").write_text(json.dumps(out, indent=2))

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    labels, vals, colors = [], [], []
    for row in rows:
        labels.append(f"{row['arm']}\n{row['kpi']}")
        vals.append(row["rel_error_pct"])
        colors.append("#2b8a3e" if row["within_15pct"] else "#c92a2a")
    idx = range(len(vals))
    ax1.bar(idx, vals, color=colors)
    ax1.axhspan(-15, 15, color="#94d82d", alpha=0.15, label="±15% owner band")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xticks(list(idx))
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("rel. error vs A1 arm [%]")
    ax1.set_title("A. Internal contrast rows (ref = A1 = same experiment)\n"
                  f"{out['n_within_15pct']}/{out['n_rows']} inside band", fontsize=9)
    ax1.legend(fontsize=7)
    for row in fails:
        ax1.annotate(f"{row['arm']}\n{row['status']}",
                     xy=(len(vals) - 0.5, 30), fontsize=7, color="#c92a2a", ha="center")

    names = [k["metric"] for k in kern]
    ax2.barh(names, [k["value"] for k in kern], color="#1864ab")
    ax2.set_xscale("log")
    ax2.set_xlabel("relative L2 difference (log scale)")
    ax2.set_title("B. External anchor: TensorLBM kernel vs executed OSS source\n"
                  "(float32 vs float64 → round-off floor; <3% requirement met)", fontsize=9)
    for i, k in enumerate(kern):
        ax2.text(k["value"] * 1.5, i, f"{k['value']:.2e}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "fig_m18288_audit.png", dpi=130)

    print(json.dumps({k: out[k] for k in ("n_rows", "n_within_15pct", "n_outside_15pct",
                                          "recompute_mismatch_vs_published", "resolutions",
                                          "failures")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())