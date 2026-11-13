#!/usr/bin/env python3
"""Audit figure for m18149 (verification of m18096 / XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY).

Panel A: the mission's E3 a(phi) (24 orientations, from its own run artifacts)
         against the closed form sqrt(2)/9 and the +-15% acceptance band.
Panel B: auditor-independent sweep (numpy, no tensorlbm import) of
         Q/(rho tau |S|) = sqrt(2)/3 versus |S|.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "fig_audit_m18096.png"
KPI = Path("/root/tp-agent/work/m18096/xval/XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY/out/kpi_prim.json")

W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
              [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=float)
EEK = np.abs(C).sum(axis=1) ** 2


def q_ratio(Sxx, Sxy, rho=1.0, tau=1.0):
    A = ((C[:, 0] ** 2 - 1 / 3) * Sxx + (C[:, 1] ** 2 - 1 / 3) * (-Sxx)
         + 2.0 * C[:, 0] * C[:, 1] * Sxy)
    fneq = -3.0 * W * rho * tau * A
    Q = np.sqrt(2.0 * (fneq ** 2 * EEK).sum())
    S_mag = np.sqrt(2.0 * (Sxx ** 2 + Sxx ** 2 + 2.0 * Sxy ** 2))
    return Q / (rho * tau * S_mag), Q, S_mag


fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

# ---- Panel A -------------------------------------------------------------
d = json.loads(KPI.read_text())
rows = d["E3"]["rows"]
phi = np.array([r["phi_deg"] for r in rows])
a_meas = np.array([r["a_Q_over_3rhotauS"] for r in rows])
closed = np.sqrt(2.0) / 9.0
ax[0].plot(phi, a_meas, "o-", color="#1f4e79", ms=5, lw=1.2,
           label=f"mission E3 a(φ), 24 φ (spread {d['E3']['a_spread_pct_about_mean']:.1e} %)")
ax[0].axhline(closed, color="k", ls="--", lw=1.4,
              label=r"closed form $\sqrt{2}/9$ = 0.15713484")
ax[0].fill_between([0, 180], closed * 0.85, closed * 1.15, color="0.85", alpha=0.6,
                   label="±15 % acceptance band")
ax[0].set_xlabel(r"strain orientation $\phi$ [deg]")
ax[0].set_ylabel(r"$a = \bar{Q}/(3\rho\tau|S|)$")
ax[0].set_xlim(0, 180)
ax[0].set_ylim(0.12, 0.19)
ax[0].set_title("A  |cc|-proxy isotropy (measured vs closed form)")
ax[0].legend(fontsize=7.5, loc="lower right")
ax[0].grid(alpha=0.3)

# ---- Panel B -------------------------------------------------------------
mag = np.array([0.005, 0.01, 0.02, 0.04, 0.08])
ratio, qvals = [], []
for m in mag:
    r, Q, S = q_ratio(m / 2, m / 2)
    ratio.append(r)
    qvals.append(Q)
ratio = np.array(ratio)
ax2 = ax[1]
ax2.plot(mag, ratio, "s-", color="#a33", ms=6,
         label=r"independent numpy check  $Q/(\rho\tau|S|)$")
ax2.axhline(np.sqrt(2) / 3, color="k", ls="--", lw=1.4,
            label=r"$\sqrt{2}/3$ = 0.47140452")
ax2.set_xlabel(r"imposed $|S| = \sqrt{2 S:S}$")
ax2.set_ylabel(r"$Q/(\rho\tau|S|)$")
ax2.set_title("B  Auditor re-derivation (no tensorlbm import)")
ax2.set_ylim(0.0, 0.75)
ax2.legend(fontsize=8, loc="lower right")
ax2.grid(alpha=0.3)
ax2.text(0.02, 0.14,
         f"max rel dev vs $\\sqrt{{2}}/3$ = {np.max(np.abs(ratio - np.sqrt(2)/3)/(np.sqrt(2)/3)):.1e}\n"
         "⇒ $a=\\sqrt{2}/9$ is an algebraic identity, not a resolution effect",
         fontsize=8, transform=ax2.transAxes,
         bbox=dict(fc="white", ec="0.6", alpha=0.9))

fig.suptitle("m18096 audit · XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY (TensorLBM, board-audit m18149)",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT, dpi=140)
print("wrote", OUT, OUT.stat().st_size, "B")
print("panel A spread:", a_meas.max() - a_meas.min(), "closed:", closed)
print("panel B ratio:", ratio.tolist())