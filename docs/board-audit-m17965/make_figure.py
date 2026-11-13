#!/usr/bin/env python3
"""Figure for the m17772 board audit: claimed vs re-measured kernel equivalence."""
import json, base64, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
d = json.loads((HERE / "verify_m17772.json").read_text())
f64 = d["internal_equivalence"]["float64"]
ea = d["external_anchor"]

claimed = [2.66e-15, 4.81e-08, 4.44e-15, 1.90e-06]      # mission REPORT.md table
measured = [f64["ref_vs_matmul_1step_maxabs"], f64["ref_vs_fused_1step_maxabs"],
            f64["ref_vs_matmul_50step_maxabs"], f64["ref_vs_fused_50step_maxabs"]]
labels = ["bgk↔matmul\n1 step", "bgk↔fused\n1 step", "bgk↔matmul\n50 steps", "bgk↔fused\n50 steps"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
x = range(len(labels)); w = 0.38
ax1.bar([i - w / 2 for i in x], claimed, w, label="m17772 claim (pre-fix state)", color="#c44e52")
ax1.bar([i + w / 2 for i in x], measured, w, label="auditor re-run on main 8cb29b59", color="#4c72b0")
ax1.set_yscale("log"); ax1.set_xticks(list(x)); ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("max |Δf|  (float64, 9×32×32, tau=0.8)")
ax1.axhline(1e-12, ls=":", c="k", lw=1); ax1.text(2.6, 1.6e-12, "machine precision band", fontsize=8)
ax1.set_title("① fused/BGK kernel equivalence: claim vs re-measurement"); ax1.legend(fontsize=8)
ax1.grid(axis="y", alpha=.3)

vals = [100 * ea["lbmpy_vs_tl_collide_bgk_rel"], 100 * ea["lbmpy_vs_tl_collide_bgk_fused_rel"]]
names = ["lbmpy SRT collide\nvs collide_bgk", "lbmpy SRT collide\nvs collide_bgk_fused"]
ax2.bar(names, vals, 0.5, color="#55a868")
ax2.axhline(15.0, ls="--", c="#c44e52", label="15% gate")
for i, v in enumerate(vals):
    ax2.text(i, v * 1.4, f"{v:.3f}%", ha="center", fontsize=9)
ax2.set_yscale("log"); ax2.set_ylabel("relative difference (%)")
ax2.set_ylim(1e-2, 100)
ax2.set_title("② external anchor actually executed: lbmpy 2.0 D2Q9 SRT (ω=1.25)\nMa=0.086, moment-verified channel mapping")
ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=.3)
fig.suptitle("m17772 XVAL-LBM-FUSED-KERNEL-TYPE-SELECTOR — board audit (m17965): L1, 3/3 artifacts missing", fontsize=10)
fig.tight_layout()
out = HERE / "fig_m17772_audit.png"
fig.savefig(out, dpi=150)
print("saved", out, out.stat().st_size)
(HERE / "fig_m17772_audit.b64").write_text(base64.b64encode(out.read_bytes()).decode())
