"""Fig 4: n=12 prescribed run — global vs liquid mach + phase of the max-mach cell."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "t1_figs"
r = json.loads((ROOT / "runs" / "t1_cac_w6_s03.json").read_text())
h = r["history"]
t = [row["time_ms"] for row in h]
mach = [row["mach_max"] for row in h]
mach_liq = [row["mach_liquid_max"] for row in h]
phi_at = [row["phi_at_mach_max"] for row in h]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
a1.plot(t, mach, color="#d62728", marker=".", label="mach_max (global)")
a1.plot(t, mach_liq, color="#1f77b4", marker=".", label="mach_liquid_max (phi>=0.5)")
a1.axhline(0.3, color="k", ls="--", lw=1, label="gate 0.3")
a1.set_xlabel("time / ms")
a1.set_ylabel("lattice Mach")
a1.set_title("n=12, w=6, s=0.03: global mach dies at 0.166 ms")
a1.legend(fontsize=8)
a2.semilogy(t, phi_at, color="#2ca02c", marker=".")
a2.axhline(0.5, color="gray", ls=":", lw=1)
a2.set_xlabel("time / ms")
a2.set_ylabel(r"$\phi$ at global max-Mach cell")
a2.set_title("max-Mach cell is in the GAS (phi~1e-3)")
a2.legend([r"$\phi<0.5$ = gas side"], loc="upper right", fontsize=8)
fig.suptitle("The spurious current lives in the gas phase; the width lever cannot smooth it out")
fig.tight_layout()
fig.savefig(OUT / "fig4_n12_gas_phase_mach.png", dpi=150)
plt.close(fig)
print("fig4 written")
