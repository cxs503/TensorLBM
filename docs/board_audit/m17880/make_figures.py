#!/usr/bin/env python3
"""Board audit m17880 — audit figure, generated only from artifacts/ref_vs_got.json."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DATA = json.loads((HERE / "artifacts/ref_vs_got.json").read_text())
rows = DATA["dispersion_rows"]
kappa, A = 0.0375, 0.075
tgm = 1.0 / (3.0 - np.sqrt(3.0)) - 0.5
Gamma = 1.0

k = np.array([r["k"] for r in rows])
ana = np.array([r["ref_analytic_omega"] for r in rows])
oss = np.array([r["got_oss_omega"] for r in rows])
tl = np.array([r["got_tl_lee_lin_omega"] for r in rows])
err_tl = 100.0 * np.array([r["err_tl_vs_analytic_rel"] for r in rows])
err_oss = 100.0 * np.array([r["err_oss_vs_analytic_rel"] for r in rows])
kc = float(np.sqrt(A / kappa))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
kk = np.linspace(0.05, 1.45, 400)
ax[0].plot(kk, Gamma * tgm * kk**2 * (A - kappa * kk**2), "k-", lw=1.6,
           label=r"analytic CH: $\Gamma(\tau_g-1/2)k^2(A-\kappa k^2)$")
swift_model = [r["model_spurious"] for r in DATA["swift_spurious_decay_scan"]]
ax[0].plot(k, swift_model, "s", mfc="none", ms=7, color="tab:red",
           label=r"swift arm model: $-c_s^2(\tau_g-1/2)k^2$")
ax[0].plot(k, tl, "o", ms=6, color="tab:blue", label="TensorLBM lee_lin (got)")
ax[0].plot(k, oss, "x", ms=7, color="tab:green", label="OSS ebinan92@150a5739 (got)")
ax[0].axvline(kc, color="0.6", ls=":", lw=1)
ax[0].text(kc, 0.0105, r" $k_c=\sqrt{A/\kappa}$", fontsize=8, color="0.4")
ax[0].axhline(0, color="0.8", lw=0.8)
ax[0].set_xlabel("wavenumber k")
ax[0].set_ylabel(r"growth rate $\omega$")
ax[0].set_title("(a) Cahn-Hilliard single-mode dispersion, N=64, " r"$\Gamma$=1")
ax[0].legend(fontsize=7.5, loc="upper left")
ax[0].grid(alpha=0.3)

w = 0.06
m = np.array([r["m"] for r in rows])
ax[1].bar(m - w, err_oss, width=2 * w, color="tab:green", alpha=0.75, label="OSS vs analytic")
ax[1].bar(m + w, err_tl, width=2 * w, color="tab:blue", alpha=0.75, label="TL lee_lin vs analytic")
ax[1].axhline(15, color="tab:red", ls="--", lw=1.2, label="owner bar 15%")
ax[1].axhline(-15, color="tab:red", ls="--", lw=1.2)
for xi, yi in zip(m, err_tl):
    if abs(yi) > 15:
        ax[1].annotate(f"{yi:.0f}%", (xi + w, yi), fontsize=7, ha="center",
                       va="bottom" if yi > 0 else "top", color="tab:red")
ax[1].set_xlabel("mode m (k = 2 pi m / N)")
ax[1].set_ylabel("relative error vs analytic [%]")
ax[1].set_title("(b) referee check: 3/5 modes inside the 15% bar")
ax[1].legend(fontsize=7.5)
ax[1].grid(alpha=0.3, axis="y")

fig.suptitle("Audit m17880 — m17762 XVAL-LBM-FREE-ENERGY-CAHN-HILLIARD-CORE", fontsize=10)
fig.tight_layout()
out = HERE / "artifacts/fig_audit_dispersion.png"
fig.savefig(out, dpi=140)
print("wrote", out, out.stat().st_size)