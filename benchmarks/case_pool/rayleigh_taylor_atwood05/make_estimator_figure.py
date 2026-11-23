"""Figure for the front-estimator calibration (reads artifacts_estimator/calibration.json)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE.parents[1] / "figures"
payload = json.loads((HERE / "artifacts_estimator" / "calibration.json").read_text())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
sw = payload["K1"]["sweep"]
a = [r["a0_over_lam"] for r in sw]
ax1.axhline(5.0, color="0.4", ls="--", lw=1.0, label="calibration criterion (5 %)")
ax1.plot(
    a,
    [r["column_err_pct"] for r in sw],
    "o-",
    color="#c0392b",
    label="column_height (legacy, conserved moment)",
)
ax1.plot(
    a,
    [r["gradient_err_pct"] for r in sw],
    "s-",
    color="#1f6fb4",
    label="gradient_front_state (new)",
)
ax1.axhspan(-5, 5, color="#1f6fb4", alpha=0.08)
ax1.set_xscale("log")
ax1.set_xlabel(r"imposed amplitude $a_0/\lambda$")
ax1.set_ylabel("amplitude readback error [%]")
ax1.set_title("K1 seed readback on a relaxed SCMP interface")
ax1.set_ylim(-115, 40)
ax1.grid(alpha=0.3)
ax1.legend(fontsize=8, loc="lower right")

g = payload["relax_geometry"]
prof = g["profile_every_8_rows"]
rows = [8 * i for i in range(len(prof))]
ax2.plot(prof, rows, "-", color="#2c3e50", label="horizontal-mean $\\rho(y)$, relaxed")
ax2.axhspan(
    g["transition_row"] - g["ten_ninety_rows"] / 2,
    g["transition_row"] + g["ten_ninety_rows"] / 2,
    color="#e67e22",
    alpha=0.25,
    label=f"10–90 % interface band ({g['ten_ninety_rows']} rows)",
)
ax2.axhline(g["transition_row"], color="#e67e22", lw=1.0, ls=":")
ax2.axvline(
    g["bulk_low_target"],
    color="0.5",
    ls="--",
    lw=1.0,
    label=f"coexistence $\\rho_l$ = {g['bulk_low_target']:.3f}",
)
ax2.axvline(
    g["bulk_high_target"],
    color="0.5",
    ls=":",
    lw=1.0,
    label=f"coexistence $\\rho_h$ = {g['bulk_high_target']:.3f}",
)
ax2.set_xlabel(r"$\rho$ [lattice]")
ax2.set_ylabel("row index $y$  (heavy phase on top)")
ax2.set_title(
    f"Relaxed interface geometry: $k\\delta$ = {g['k_delta']:.2f}  "
    f"(sharp-interface reference needs $\\ll$ 0.2)"
)
ax2.grid(alpha=0.3)
ax2.legend(fontsize=8)
fig.suptitle(
    "CASE-RT-INSTABILITY-ATWOOD · finite-density-ratio arm: instrument calibration", fontsize=10
)
fig.tight_layout()
FIG.mkdir(parents=True, exist_ok=True)
out = FIG / "fig_rt_estimator_calibration.png"
fig.savefig(out, dpi=150)
print("wrote", out)
