#!/usr/bin/env python3
"""Audit figure for m15894: ref-vs-got contact angle and the measured wall law.

Every number comes from the mission's artifact JSON (no hand-typed values):
  /root/tp-agent/work/m15470/artifacts/CASE-MCMP-CONTACT-ANGLE/contact_angle_sweep.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC = Path("/root/tp-agent/work/m15470/artifacts/CASE-MCMP-CONTACT-ANGLE/contact_angle_sweep.json")
OUT = Path(__file__).resolve().parent / "audit_m15894"
OUT.mkdir(parents=True, exist_ok=True)
d = json.loads(SRC.read_text())
acc = d["meta"]["accept_deg"]
psil = d["meta"]["coexistence"]["psi_l"]
psiv = d["meta"]["coexistence"]["psi_v"]

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))

# ---- left: prescribed (Young-law inversion) vs measured, two grids ----
ax = axes[0]
cases = d["stage_a_young"]["cases"]
grids = ["coarse", "fine"]
w = 0.36
for i, g in enumerate(grids):
    sub = [c for c in cases if c["grid"] == g]
    xs = list(range(len(sub)))
    ref = [c["theta_young"] if c["theta_young"] is not None else 0.0 for c in sub]
    got = [c["theta_measured"] if c.get("theta_measured") is not None else 0.0 for c in sub]
    ax.bar([x + (i - 0.5) * w for x in xs], ref, width=w, alpha=0.35, color="0.5",
           label="ref = Young set-point (library inversion)" if i == 0 else None)
    ax.bar([x + (i - 0.5) * w for x in xs], got, width=w, color=["C0", "C3"][i],
           label=f"got = measured ({g})")
    for x, c, v in zip(xs, sub, got):
        if c.get("theta_measured") is None:
            ax.text(x + (i - 0.5) * w, 6, "OUT_OF\nRANGE", ha="center", va="bottom",
                    fontsize=6.5, color="darkred", rotation=90)
        else:
            ax.text(x + (i - 0.5) * w, v + 2, f"{v - c['theta_young']:+.1f}", ha="center", fontsize=6.5)
ax.axhline(180, color="0.75", ls=":", lw=0.8)
ax.set_xticks(range(3))
ax.set_xticklabels([("n/a\n(psi_w<0)" if c["theta_young"] is None else f"{c['theta_young']:.0f} deg")
                    for c in cases if c["grid"] == "coarse"])
ax.set_xlabel("prescribed angle via cos(theta)=(psi_w-psi_v)/(psi_l-psi_v)")
ax.set_ylabel("contact angle (deg)")
ax.set_ylim(0, 205)
v = d["verdicts"]["young"]
ax.set_title(f"Young-law set-point vs measured sessile drop\n"
             f"code tol = {acc:.0f} deg: {v['n_pass']}/{v['n_cases']} pass, "
             f"max |delta| = {v['max_abs_delta']:.2f} deg")
ax.legend(fontsize=7, loc="upper left")

# ---- right: measured wall law vs the library's linear cos(theta) assumption ----
ax = axes[1]
c = d["stage_b_calibration"]["curve"]
fw = c["wall_fraction"]
th = c["theta_deg"]
ax.plot(fw, th, "o-", color="C0", label="measured theta(f_w), coarse 96x64 (B1 scan)")
xs = [0.30 + 0.0025 * i for i in range(201)]
ax.plot(xs, [math.degrees(math.acos(max(-1.0, min(1.0, ((f * psil - psiv) / (psil - psiv))))))
             for f in xs], "--", color="0.4",
        label="library assumption cos(theta)=(psi_w-psi_v)/(psi_l-psi_v)")
if len(fw) >= 2:
    n = len(fw)
    X = fw
    Y = [math.cos(math.radians(t)) for t in th]
    mx, my = sum(X) / n, sum(Y) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(X, Y)) / sum((x - mx) ** 2 for x in X)
    inter = my - slope * mx
    slope_lib = 1.0 / (psil - psiv)
    inter_lib = -psiv / (psil - psiv)
    ax.plot(xs, [math.degrees(math.acos(max(-1.0, min(1.0, slope * f + inter)))) for f in xs],
            "-", color="C3", lw=1.2,
            label=f"fit: cos(theta)={slope:.3f} f_w {inter:+.3f} ({slope / slope_lib:.2f}x library slope)")
    ax.set_title("measured wall law vs library's linear cosine assumption\n"
                 f"neutral wall f_w* = {-inter / slope:.3f} (library says {-inter_lib / slope_lib:.3f})")
else:
    ax.set_title("measured wall law vs library's linear cosine assumption")
ax.set_xlabel("wall fraction f_w = psi_w / psi_l")
ax.set_ylabel("equilibrium contact angle (deg)")
ax.legend(fontsize=6.5, loc="lower left")
ax.grid(alpha=0.3)

fig.tight_layout()
p = OUT / "fig_m15894_contact_angle_audit.png"
fig.savefig(p, dpi=150)
print("wrote", p, p.stat().st_size)