#!/usr/bin/env python3
"""m16256 audit figure: independent rerun of m16237 vs analytic erfc (Stokes first problem).

Inputs are the AUDITOR's own re-executions (repro_main.json, repro_fine_dref50.json) plus
one shipped config for the collision-scheme comparison (U1e-3_H200_trt.json).
Zero manual numbers: everything is read from the JSON artifacts.
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
SHIP = pathlib.Path("/root/tp-agent/work/m16237/benchmarks/verified/stokes_first_problem/scan_tstar1")


def prof(p):
    return json.loads(pathlib.Path(p).read_text())["profiles"][0]


coarse = prof(HERE / "repro_main.json")
fine = prof(HERE / "repro_fine_dref50.json")
trt = prof(SHIP / "U1e-3_H200_trt.json")

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))

# panel 1 — profile vs analytic similarity solution
ax[0].plot(coarse["eta_native"], coarse["u_over_U_ana_native"], "k-", lw=2.2,
           label=r"analytic $u/U=\mathrm{erfc}(\eta)$")
ax[0].plot(coarse["eta_native"], coarse["u_over_U_native"], "o", ms=3, mfc="none",
           color="tab:blue", label=rf"LBM $\delta=25$ (re-run, max {coarse['max_rel_err_pct_eta0.1_2']:.4f}%)")
ax[0].plot(fine["eta_native"], fine["u_over_U_native"], "s", ms=3, mfc="none",
           color="tab:orange", label=rf"LBM $\delta=50$ (re-run, max {fine['max_rel_err_pct_eta0.1_2']:.4f}%)")
ax[0].set_yscale("log")
ax[0].set_xlim(0, 3.0)
ax[0].set_ylim(1e-7, 1.2)
ax[0].set_xlabel(r"$\eta = y/(2\sqrt{\nu t})$")
ax[0].set_ylabel(r"$u/U$")
ax[0].set_title("Stokes 1st problem, $t^*=1$: profile incl. $10^{-3}$ tail")
ax[0].legend(fontsize=8, loc="lower left")
ax[0].grid(alpha=0.3, which="both")

# panel 2 — pointwise error, criterion line 2 %
for p, lab, c in ((coarse, r"BGK $\delta=25$", "tab:blue"),
                  (fine, r"BGK $\delta=50$", "tab:orange"),
                  (trt, r"TRT $\Lambda=3/16$, $\delta=25$ (shipped)", "tab:green")):
    ax[1].semilogy(p["eta_grid"], np.maximum(p["rel_err_pct_at_eta"], 1e-6), "-o", ms=2.5,
                   color=c, label=lab)
ax[1].axhline(2.0, color="crimson", ls="--", lw=1.5, label="criterion 2% (case pool)")
ax[1].set_xlabel(r"$\eta$")
ax[1].set_ylabel("pointwise relative error [%]")
ax[1].set_title("Pointwise error vs erfc, $\\eta\\in[0,2]$")
ax[1].set_xlim(0, 2.05)
ax[1].grid(alpha=0.3, which="both")
ax[1].legend(fontsize=8)
ax[1].annotate("$\\eta=0$: no node (wall at $y=0.5$)\nclamped $\\to 1-\\mathrm{erfc}(1/4\\delta)$",
               xy=(0.0, coarse["rel_err_pct_at_eta"][0]), xytext=(0.35, 0.05),
               fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))

fig.tight_layout()
out = HERE / "fig_audit_profile_error.png"
fig.savefig(out, dpi=150)
print("wrote", out)

# machine-readable summary of what the figure shows
print(json.dumps({
    "coarse_max_pct_eta01_2": coarse["max_rel_err_pct_eta0.1_2"],
    "coarse_full_window_pct": coarse["max_rel_err_pct_eta0_2"],
    "fine_max_pct_eta01_2": fine["max_rel_err_pct_eta0.1_2"],
    "fine_full_window_pct": fine["max_rel_err_pct_eta0_2"],
    "trt_max_pct_eta01_2": trt["max_rel_err_pct_eta0.1_2"],
}, indent=1))