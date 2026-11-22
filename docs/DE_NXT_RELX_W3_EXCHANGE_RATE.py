"""DE-NXT-RELX-W3 · 深化-时刻兑换率图（程序化生成，数字全部来自定谳卡）

数据出处（每个数字带卡号，脚本内不做手抄推导，只算派生量）：
  ctrl       1.435220 D @ abs 11.090571 ms / pc 10.543339 ms  ← DE-X-COL2-REG §3 ctrl_bgk
  rr(global) 1.555257 D @ pc 12.306643 ms                     ← DE-X-COL2-REG §3 rr_base / CMB3 cj6070
  phasetau   1.671619 D @ abs 12.635 ms                       ← RELX1-PHASETAU-R2 §3 pt_p20 (pc 由 ctrl 偏移量换算)
  hydro      1.636628 D @ pc 13.534875 ms                     ← DE-X-CMB3-REGSMAG §4 hydro_only cj6080
  combo      1.750541 D @ pc 14.313161 ms                     ← DE-X-CMB3-REGSMAG §4 combo_hydro cj6073
  dca        1.355520 D @ pc 10.531000 ms                     ← DE-X-CMB1-HYDRODCA §7 引用 MMC6-DYNANGLE-B
  experiment 1.857434 D @ pc 10.394460 ms                     ← DE_A1_FACTS 实验锚
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- raw measured inputs (D, pc_ms, abs_ms or None) -------------------------------
ARMS = [
    # name,        peak_D,   pc_ms,     abs_ms,   source
    ("ctrl BGK",   1.435220, 10.543339, 11.090571, "DE-X-COL2-REG §3 ctrl_bgk"),
    ("RR global",  1.555257, 12.306643, 12.841700, "DE-X-COL2-REG §3 rr_base"),
    ("PHASE-TAU",  1.671619, None,     12.635000, "RELX1-PHASETAU-R2 §3 pt_p20"),
    ("HYDRO",      1.636628, 13.534875, None,     "DE-X-CMB3-REGSMAG §4 cj6080"),
    ("RR+SMAG+HYDRO", 1.750541, 14.313161, None,  "DE-X-CMB3-REGSMAG §4 cj6073"),
    ("DCA slip",   1.355520, 10.531000, None,     "MMC6-DYNANGLE-B (DE-X-CMB1-HYDRODCA §7 引用)"),
]
EXP_D, EXP_PC = 1.857434, 10.394460

# ctrl 的 (abs - pc) 偏移 = 触水瞬间到计时原点的固定差，用于把只报绝对时刻的臂换算到 pc 口径
OFFSET = next(t for t in ARMS if t[0] == "ctrl BGK")
_pc_offset = OFFSET[3] - OFFSET[2]

rows = []
for name, d, pc, ab, src in ARMS:
    pc_eff = pc if pc is not None else ab - _pc_offset
    rows.append(dict(name=name, peak=d, pc=pc_eff, src=src))

ctrl = next(r for r in rows if r["name"] == "ctrl BGK")
for r in rows:
    r["dD"] = r["peak"] - ctrl["peak"]
    r["dt"] = r["pc"] - ctrl["pc"]
    r["rate"] = (r["dt"] / r["dD"]) if abs(r["dD"]) > 1e-9 else np.nan

# --- campaign gates ----------------------------------------------------------------
gate_d = 0.03 * EXP_D
gate_t = 0.03 * EXP_PC
need_dD = EXP_D - ctrl["peak"]
allowed_dt = (EXP_PC * 1.03) - ctrl["pc"]
max_rate = allowed_dt / need_dD

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)

# Panel A: peak depth vs peak time, with the joint 3% gate box
ax1.add_patch(
    plt.Rectangle(
        (EXP_PC - gate_t, EXP_D - gate_d),
        2 * gate_t,
        2 * gate_d,
        fill=False,
        edgecolor="crimson",
        lw=1.6,
        ls="--",
        label="3% joint gate (depth x time)",
    )
)
for r in rows:
    ax1.plot(r["pc"], r["peak"], "o", ms=7, color="#1f77b4")
    ax1.annotate(
        r["name"],
        (r["pc"], r["peak"]),
        textcoords="offset points",
        xytext=(6, 5),
        fontsize=8,
    )
ax1.plot(EXP_PC, EXP_D, "*", ms=15, color="black", label="experiment 1.857434 D @ 10.39446 ms")
ax1.plot(ctrl["pc"], ctrl["peak"], "s", ms=7, color="gray", label="ctrl baseline")
ax1.set_xlabel("peak time (post-contact ms)")
ax1.set_ylabel("cavity peak depth (D)")
ax1.set_title("A) joint reachability of depth vs peak time (cells=8 full window)")
ax1.grid(alpha=0.3)
ax1.legend(fontsize=7, loc="lower left")

# Panel B: exchange rate
deep = [r for r in rows if r["dD"] > 0.01]
labels = [r["name"] for r in deep]
vals = [r["rate"] for r in deep]
cols = ["#2ca02c" if v <= 8.0 else "#d62728" for v in vals]
ax2.bar(range(len(vals)), vals, color=cols)
ax2.axhline(max_rate, color="crimson", ls="--", lw=1.6,
            label=f"gate-allowed max rate {max_rate:.3f} ms/D")
ax2.set_yscale("log")
ax2.set_xticks(range(len(vals)))
ax2.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
ax2.set_ylabel("exchange rate  dt/ddepth (ms/D)")
ax2.set_title("B) timing cost of deepening knobs (lower is better; need <= %.3f)" % max_rate)
for i, v in enumerate(vals):
    ax2.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
ax2.grid(alpha=0.3, which="both", axis="y")
ax2.legend(fontsize=8)

fig.suptitle(
    "DE-NXT-RELX-W3 exchange-rate verdict frame"
    f" (depth gate needs +{need_dD:.6f} D, timing budget only +{allowed_dt:.6f} ms)",
    fontsize=11,
)
out = "docs/DE_NXT_RELX_W3_exchange_rate.png"
fig.savefig(out, dpi=150)
print("saved", out)
print(f"need_dD={need_dD:.6f} allowed_dt={allowed_dt:.6f} max_rate={max_rate:.4f}")
for r in rows:
    print(f"{r['name']:<16} peak={r['peak']:.6f} pc={r['pc']:.6f} dD={r['dD']:+.6f} "
          f"dt={r['dt']:+.6f} rate={r['rate']:.3f}  [{r['src']}]")
