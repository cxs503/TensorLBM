#!/usr/bin/env python3
"""Figures for the m18159 board audit — every value read from evidence.json.

fig1_evidence_matrix.png : (a) pass/fail of the 10 predeclared evidence checks,
                           (b) the L0-L4 maturity ladder with prerequisite status.
fig2_budget_utilization.png : budget actually consumed vs the narrative claim of
                           "budget exhausted" (the claim is the audit finding).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EV = json.loads((ROOT / "docs/board_audit/m18159/evidence.json").read_text())
OUT = ROOT / "docs/board_audit/m18159/figures"
OUT.mkdir(parents=True, exist_ok=True)

GREEN, RED, GREY, BLUE = "#2e7d32", "#c62828", "#9e9e9e", "#1565c0"

checks = EV["checks"]
keys = [c["key"] for c in checks][::-1]
vals = [c["satisfied"] for c in checks][::-1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.55, 1]})

colors = [GREEN if v else RED for v in vals]
ax1.barh(keys, [1] * len(keys), color=colors, height=0.62)
for i, (k, v) in enumerate(zip(keys, vals)):
    ax1.text(0.02, i, "MET" if v else "MISSING", va="center", ha="left",
             color="white", fontsize=9, fontweight="bold")
ax1.set_xlim(0, 1)
ax1.set_xticks([])
ax1.set_title("(a) Predeclared evidence checks — m18068 vs anchor\n"
              f"{sum(vals)}/{len(vals)} satisfied", fontsize=11)
ax1.tick_params(axis="y", labelsize=9)
for s in ("top", "right", "bottom"):
    ax1.spines[s].set_visible(False)

v = EV["verdict"]
ladder = [("L2 real code", v["l2_prereq_met"]), ("L3 integration", v["l3_prereq_met"]),
          ("L4 benchmark", v["l4_prereq_met"])]
names = [n for n, _ in ladder]
ok = [o for _, o in ladder]
ax2.bar(names, [1 if o else 0 for o in ok], color=[GREEN if o else GREY for o in ok], width=0.55)
for i, o in enumerate(ok):
    ax2.text(i, (1 if o else 0) + 0.03, "prereq met" if o else "prereq unmet",
             ha="center", fontsize=9, color=GREEN if o else RED)
ax2.set_ylim(0, 1.35)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(["no", "yes"])
ax2.set_title(f"(b) Maturity ladder — awarded level = L{v['level']}\n"
              f"gate = {v['expected_gate_state']}", fontsize=11)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

fig.suptitle("m18068 XVAL-LBM-NO-MULTIPHASE-LAYOUT-PATTERN-ONLY — board audit (m18159)",
             fontsize=12, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT / "fig1_evidence_matrix.png", dpi=150)
plt.close(fig)

# ---- fig2: budget utilisation vs the "exhausted" claim ----------------------
m = EV["mission"]
frac = m["budget_used_frac"]
labels = ["tool calls\n(used / allowed)", "wall clock\n(used / allowed)"]
used = [frac["tool_calls"], frac["wall_seconds"]]
raw = [f'{m["tool_calls"]} / {m["budget"]["max_tool_calls"]}',
       f'{m["wall_seconds_used"] / 60:.1f} min / {m["budget"]["max_wall_seconds"] / 60:.0f} min']

fig, ax = plt.subplots(figsize=(7.6, 4.4))
bars = ax.bar(labels, [100] * 2, color="#e0e0e0", width=0.5, label="budget ceiling")
ax.bar(labels, [u * 100 for u in used], color=BLUE, width=0.5, label="actually consumed")
for b, u, r in zip(bars, used, raw):
    ax.text(b.get_x() + b.get_width() / 2, u * 100 + 2,
            f"{u * 100:.1f}%  ({r})", ha="center", fontsize=10, color=BLUE, fontweight="bold")
ax.axhline(100, color=RED, ls="--", lw=1.2)
ax.set_ylim(0, 125)
ax.set_ylabel("share of budget (%)")
ax.set_title('Contradiction: narrative claims "budget exhausted"\n'
             f'(approval #{m["approval"]["id"]} via {m["decided_by"]}; '
             f'fail_stage={m["fail_stage"]})', fontsize=11)
ax.legend(loc="upper center", frameon=False, ncol=2, fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig2_budget_utilization.png", dpi=150)
plt.close(fig)
print("wrote", *(p.name for p in sorted(OUT.glob("*.png"))))