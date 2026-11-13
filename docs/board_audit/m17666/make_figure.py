import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AUD = Path(__file__).resolve().parents[3]
d = json.loads((AUD / "runs/m17666_audit/verify_m17637.json").read_text())
rel = d["C2_equivalence_reproduced"]["rel_l2_pct_by_rank"]
gam = [float(x) for x in d["C2_equivalence_reproduced"]["gamma_err_pct"]]
mut = d["C5_discrimination"]

fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))

k1 = list(rel) + ["f32_P2 noop-halo\n(mutation probe)"]
v1 = [float(x) if float(x) > 0 else 1e-12 for x in rel.values()] + [mut["noop_halo_rel_l2_pct"]]
ax[0].bar(range(len(k1)), v1, color=["#3b7dd8"] * 3 + ["#d8863b"] * 3 + ["#c0392b"])
ax[0].set_yscale("log")
ax[0].axhline(15, color="k", ls="--", lw=1)
ax[0].text(0.02, 18, "15% platform gate", fontsize=7)
ax[0].set_xticks(range(len(k1)))
ax[0].set_xticklabels([k.replace("float", "f") for k in k1], fontsize=7, rotation=20)
ax[0].set_ylabel("field rel-L2 vs serial [%]")
ax[0].set_title("decomposition equivalence (0.0 plotted as 1e-12)", fontsize=9)

ax[1].bar([0, 1], gam, color="#3b7dd8")
ax[1].axhline(3, color="#c0392b", ls="--", lw=1)
ax[1].axhline(15, color="k", ls="--", lw=1)
ax[1].text(1.55, 3.1, "3%", fontsize=7, ha="right")
ax[1].text(1.55, 15.2, "15%", fontsize=7, ha="right")
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(["f64", "f32"], fontsize=8)
ax[1].set_ylim(0, 18)
ax[1].set_ylabel(r"|$\gamma_K$ err| vs analytic [%]")
ax[1].set_title(r"TGV decay rate vs 4$\nu k^2$ (24$^3$, 100 steps)", fontsize=9)

cov = d["C3_pylbm_decomposition_rows"]
ax[2].bar([0, 1], [cov["total"], cov["total"]], color="#dddddd", label="rows")
ax[2].bar([0, 1], [cov["total"], cov["with_any_local_size"]], color=["#3b7dd8", "#c0392b"])
ax[2].set_xticks([0, 1])
ax[2].set_xticklabels(["TensorLBM arm", "pylbm arm"], fontsize=8)
ax[2].set_ylabel("rows with numbers")
ax[2].set_title("arm-A coverage in table_decomposition.csv", fontsize=9)
ax[2].text(0, cov["total"] * 0.5, f"{cov['total']}/{cov['total']}", ha="center", color="w", fontsize=10)
ax[2].text(1, 1.0, f"{cov['with_any_local_size']}/{cov['total']}\nall IndexError", ha="center",
           color="#c0392b", fontsize=9)

fig.suptitle("m17637 XVAL-LBM-CARTESIAN-MPI board audit (m17666): what reproduces, what does not", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = AUD / "runs/m17666_audit/m17666_audit_figure.png"
fig.savefig(out, dpi=130)
print("wrote", out, out.stat().st_size)
