#!/usr/bin/env python3
"""Freeze the m17955 delivery evidence (byte copy + sha256) and draw the audit figure."""
from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = Path("/root/tp-agent/work/m17955/TensorLBM")
HERE = Path(__file__).resolve().parent
SNAP = HERE / "evidence_snapshot"; SNAP.mkdir(exist_ok=True)
FILES = [
    "xval/XVAL-LBM-LIST-BASED-SPARSE-LBM-PLUS-PERMEABILITY/run_xval.py",
    "xval/XVAL-LBM-LIST-BASED-SPARSE-LBM-PLUS-PERMEABILITY/oss_ref.py",
    "xval/XVAL-LBM-LIST-BASED-SPARSE-LBM-PLUS-PERMEABILITY/make_report.py",
    "xval/XVAL-LBM-LIST-BASED-SPARSE-LBM-PLUS-PERMEABILITY/results.json",
    "xval/XVAL-LBM-LIST-BASED-SPARSE-LBM-PLUS-PERMEABILITY/kpi.csv",
    "xval/XVAL-LBM-LIST-BASED-SPARSE-LBM-PLUS-PERMEABILITY/run.log",
    "src/tensorlbm/porous/sparse_index_lbm.py",
    "src/tensorlbm/porous/__init__.py",
    "docs/XVAL-LBM-LIST-BASED-SPARSE-LBM-PLUS-PERMEABILITY.md",
]
lines = []
for rel in FILES:
    s = SRC / rel
    dst = SNAP / Path(rel).name
    shutil.copy2(s, dst)
    h = hashlib.sha256(dst.read_bytes()).hexdigest()
    lines.append(f"{h}  {dst.stat().st_size:>8d}  {rel}")
(SNAP / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ---------------------------------------------------------------- figure
res = json.load(open(HERE / "verify_m17955.json"))
perc = json.load(open(HERE / "verify_perc_oss.json"))["percolation"]
c1 = json.load(open(HERE / "verify_case1_oss.json"))
rep = json.load(open(SNAP / "results.json"))
Hs = [r["H"] for r in res["channel"]]
aud = [abs(r["err_int"]) for r in res["channel"]]
deld = [abs(r["err_vs_analytic"]) for r in rep["case2_channel"]["ladder"]]

fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))
a = ax[0]
a.loglog(Hs, deld, "o-", lw=2, label="m17955 reported |err| (kpi.csv)")
a.loglog(Hs, aud, "s-", lw=2, label="auditor recomputed |err| (k_int = ν⟨u⟩_fluid/g)")
ref = np.array(Hs, float)
a.loglog(ref, 0.25 * ref ** -1, "k--", lw=1, label="slope −1 (report's O(1/H) claim)")
a.loglog(ref, 0.28 * ref ** -2, "k:", lw=1, label="slope −2 (half-way bounce-back, 2nd order)")
a.axhline(3, color="crimson", ls="--", lw=1.2, label="3 % gate")
a.set_xlabel("channel height H [lattice]"); a.set_ylabel("|k_int error| vs k = H²/12  [%]")
a.set_title("Channel permeability: reported vs recomputed")
a.grid(True, which="both", alpha=.3); a.legend(fontsize=8, loc="lower left")

b = ax[1]
labels = ["case1 sparse vs OSS\nkernel max|Δf|", "case1 sparse vs dense\nmax|Δf|", "percolation 24² sparse\nvs dense Δk/k"]
x = np.arange(len(labels)); w = .36
reported = [rep["case1_kernel"]["sparse_vs_oss_max_abs"], rep["case1_kernel"]["sparse_vs_dense_max_abs"],
            rep["case3_percolation"]["cases"][0]["rel_diff_super_vs_dense"]]
recomputed = [c1["max_sparse_vs_oss_postcollision"], c1["max_sparse_vs_dense"], perc["rel_diff_intrinsic"]]
b.bar(x - w / 2, reported, w, label="m17955 reported (results.json)")
b.bar(x + w / 2, recomputed, w, label="auditor recomputed (shipped code)")
b.set_yscale("log"); b.set_ylabel("difference (log)"); b.set_xticks(x); b.set_xticklabels(labels, fontsize=8)
b.axhline(1e-4, color="crimson", ls="--", lw=1.2, label="report's own c3 gate 1e-4")
b.set_title("Kernel / representation differences: reported vs recomputed")
b.grid(True, axis="y", alpha=.3); b.legend(fontsize=8)
for i, (r_, c_) in enumerate(zip(reported, recomputed)):
    b.annotate(f"{r_:.2e}", (i - w / 2, r_), ha="center", va="bottom", fontsize=7, rotation=90)
    b.annotate(f"{c_:.2e}", (i + w / 2, c_), ha="center", va="bottom", fontsize=7, rotation=90)
plt.tight_layout()
out = HERE / "fig_audit_m17955.png"
plt.savefig(out, dpi=140)
print("figure ->", out, out.stat().st_size)
