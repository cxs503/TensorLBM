#!/usr/bin/env python3
"""Figure + numbers for the m17741 audit of XVAL-LBM-CHAPMAN-ENSKOG-AUTO (m17672).

Reads the shipped artefact of the audited worktree (default
/root/tp-agent/work/m17672/benchmarks/xval_chapman_enskog_auto/result.json) plus my
own fresh re-run (/tmp/xval_fresh/result.json) and renders
docs/audit_m17741/audit_m17741.png.  Every number in the audit note is printed by
this script (no hand-typed values).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TARGET = Path(os.environ.get("TARGET_REPO", "/root/tp-agent/work/m17672"))
HERE = Path(__file__).resolve().parent
shipped = json.loads((TARGET / "benchmarks/xval_chapman_enskog_auto/result.json").read_text())
fresh = json.loads(Path(os.environ.get("FRESH_JSON", "/tmp/xval_fresh/result.json")).read_text())

checks = shipped["checks"]
ids = [c["id"] for c in checks]
err = np.array([abs(c.get("err_pct", float("nan"))) for c in checks])
gate = float(shipped["gate_pct"])

stats = {
    "n_checks": len(checks),
    "n_passed_shipped": int(sum(bool(c["pass"]) for c in checks)),
    "n_passed_fresh": int(sum(bool(c["pass"]) for c in fresh["checks"])),
    "gate_pct": gate,
    "worst_id": max(checks, key=lambda c: abs(c.get("err_pct", 0)))["id"],
    "worst_err_pct": float(max(err)),
    "n_below_3pct": int((err < 3).sum()),
    "n_below_0p4pct": int((err < 0.4).sum()),
    "median_err_pct": float(np.median(err)),
    "n_zero_ref_rows": int(sum(1 for c in checks if c.get("ref") == 0)),
    "n_distinct_measurement_runs": len({(c["quantity"], c.get("ref"), round(float(c.get("got", 0)), 12))
                                        for c in checks}),
    "checks_in_arm": {},
}
for pref, name in (("A", "A shear 2D BGK"), ("B", "B shear 2D MRT"), ("C", "C shear 3D BGK"),
                   ("D", "D sound 2D BGK/MRT"), ("E", "E sound 3D BGK")):
    stats["checks_in_arm"][name] = int(sum(1 for i in ids if i.startswith(pref)))
stats["distinct_got_values"] = len({round(float(c.get("got", 0)), 12) for c in checks})

order = np.argsort(-err)
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])

# (a) per-check relative error against the gate
ax = fig.add_subplot(gs[0, :])
col = ["#1b7f3b" if c["pass"] else "#b00020" for c in checks]
ax.bar(range(len(checks)), err[order], color=[col[i] for i in order], width=0.75)
ax.set_xticks(range(len(checks)))
ax.set_xticklabels([ids[i] for i in order], rotation=72, ha="right", fontsize=6.5)
ax.axhline(gate, color="#b00020", ls="--", lw=1.2, label=f"owner gate {gate:g}%")
ax.axhline(3.0, color="#444", ls=":", lw=1.0, label="3% (paper-grade threshold)")
ax.set_yscale("log")
ax.set_ylabel("|relative error| [%] (log)")
ax.set_title(f"XVAL-LBM-CHAPMAN-ENSKOG-AUTO (m17672) — {stats['n_passed_shipped']}/{stats['n_checks']} checks, "
             f"worst {stats['worst_id']} = {stats['worst_err_pct']:.3f}%", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.25, axis="y")

# (b) parity ref vs got
ax2 = fig.add_subplot(gs[1, 0])
pos = [c for c in checks if c.get("ref")]
ref = np.array([c["ref"] for c in pos], dtype=float)
got = np.array([c["got"] for c in pos], dtype=float)
zr = [c for c in checks if c.get("ref") == 0]
ax2.loglog(ref, got, "o", ms=4, mfc="none", mec="#14507a")
lim = [min(ref.min(), got.min()) * 0.8, max(ref.max(), got.max()) * 1.2]
ax2.loglog(lim, lim, "k-", lw=0.8)
ax2.fill_between(lim, [v * 0.85 for v in lim], [v * 1.15 for v in lim], color="#14507a", alpha=0.12,
                 label="±15%")
ax2.set_xlabel("reference (symbolic Chapman-Enskog)")
ax2.set_ylabel("measured (solver)")
ax2.set_title(f"parity, {len(pos)} non-zero refs; {len(zr)} zero-ref row(s) excluded", fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.25, which="both")

# (c) two-grid convergence for the arms that were refined
ax3 = fig.add_subplot(gs[1, 1])
for arm, color, tau in (("A", "#14507a", "0.6"), ("A", "#14507a", "0.8"), ("A", "#14507a", "1.2"),
                        ("C", "#c46a00", "0.6"), ("C", "#c46a00", "0.8"), ("C", "#c46a00", "1.2")):
    e32 = next((abs(c["err_pct"]) for c in checks if c["id"] == f"{arm}-shear2d-tau{tau}-n32"
                 or c["id"] == f"{arm}-shear3d-tau{tau}-n32"), None)
    e64 = next((abs(c["err_pct"]) for c in checks if c["id"] == f"{arm}-shear2d-tau{tau}-n64"
                 or c["id"] == f"{arm}-shear3d-tau{tau}-n64"), None)
    if e32 is not None and e64 is not None:
        ax3.plot([32, 64], [e32, e64], "o-", color=color, ls="-" if arm == "A" else "--",
                 label=f"arm {arm} tau={tau}")
for cid, lbl in (("E1-sound-atten-3d-n32", "E1 3D sound n32"), ("E1-sound-atten-3d-n64", "E1 3D sound n64")):
    c = next(c for c in checks if c["id"] == cid)
    ax3.plot([32 if "n32" in cid else 64], [abs(c["err_pct"])], "s", color="#b00020", ms=7)
ax3.annotate("E1 3D sound: 0.010% (n32) -> 1.155% (n64)\nnon-monotone — not a converged series",
             xy=(64, abs(next(c for c in checks if c["id"] == "E1-sound-atten-3d-n64")["err_pct"])),
             xytext=(38, 1.0), fontsize=7.5, color="#b00020",
             arrowprops=dict(arrowstyle="->", color="#b00020", lw=0.8))
ax3.set_xlabel("grid (n32 -> n64)")
ax3.set_ylabel("|relative error| [%]")
ax3.set_yscale("log")
ax3.set_title("mesh refinement actually run (shear arms + the 3D sound outlier)", fontsize=9)
ax3.legend(fontsize=7, ncol=2)
ax3.grid(alpha=0.25, which="both")

fig.tight_layout()
out_png = HERE / "audit_m17741.png"
fig.savefig(out_png, dpi=115)
stats["figure"] = str(out_png)
print(json.dumps(stats, indent=1))
(HERE / "audit_stats.json").write_text(json.dumps(stats, indent=1))