#!/usr/bin/env python3
"""m18148 审计图：独立复算 vs 报告声称（误差臂）+ A1 判据灵敏度（变异对照）。

数据源全部程序化注入：independent_check.json（审计者自算）与
/root/tp-agent/work/m18112/artifacts/xval_ppermute_periodic_halo/summary.json（被审工件）。
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
CHK = json.loads((HERE / "independent_check.json").read_text())
ORIG = json.loads(pathlib.Path(
    "/root/tp-agent/work/m18112/artifacts/xval_ppermute_periodic_halo/summary.json").read_text())


def find_rel_err(obj, out):
    """在工件 JSON 里递归收 'rel_err'（百分数）标量。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("rel_err_pct", "rel_err_percent") and isinstance(v, (int, float)):
                out.append((obj.get("grid"), obj.get("steps"), float(v)))
            else:
                find_rel_err(v, out)
    elif isinstance(obj, list):
        for v in obj:
            find_rel_err(v, out)


claimed = []
find_rel_err(ORIG, claimed)
claimed_sorted = sorted(claimed, key=lambda r: (r[0] or 0))
mine = CHK["independent_shear_wave"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
labels = [f"L={m['grid']}\n({m['steps']} steps)" for m in mine]
mine_err = [m["rel_err_pct"] for m in mine]
by_grid = {g: e for g, _s, e in claimed_sorted}
claim_err = [by_grid.get(m["grid"], float("nan")) for m in mine]
x = range(len(labels))
ax.bar([i - 0.2 for i in x], mine_err, width=0.4, label="auditor independent rerun (numpy f64)", color="#1f77b4")
ax.bar([i + 0.2 for i in x], claim_err, width=0.4, label="mission artifact claim (torch)", color="#ff7f0e")
ax.axhline(-15, color="r", ls="--", lw=1, label="15% gate (owner std)")
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylabel("shear-wave terminal amplitude rel err [%]")
ax.set_title("A3 analytic arm: independent rerun == claim (<0.001 pp)")
ax.legend(fontsize=7)
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
mut = CHK["mutation_control"]
vals = [mut["max_abs_df_correct_perm"], mut["max_abs_df_swapped_perm"]]
ax.bar(["correct perms", "swapped perms\n(wrong-neighbour mutation)"], vals, color=["#2ca02c", "#d62728"])
ax.set_yscale("symlog", linthresh=1e-6)
ax.set_ylabel("max|df| sharded vs monolithic")
ax.set_title(f"A1 criterion sensitivity (mutation, N={mut['n_ranks']}, {mut['steps']} steps)")
for i, v in enumerate(vals):
    ax.text(i, v if v > 0 else 1e-9, f"{v:.3e}", ha="center", va="bottom", fontsize=9)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
out = HERE / "fig_m18148_audit.png"
fig.savefig(out, dpi=130)
print(out, out.stat().st_size)