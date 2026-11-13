"""Figures for the ib3d-diffuse-interface-MDF XVAL report (matplotlib, no hand typing)."""
from __future__ import annotations
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)

op = json.loads((ART / "operator_xval.json").read_text())
cases = {c["kernel"]: c for c in op["cases"]}

# 1 — slip residual vs MDF iteration
fig, ax = plt.subplots(figsize=(6, 4.2))
for kernel, case in cases.items():
    r = case["residual_series_tl"]
    ax.semilogy(range(1, len(r) + 1), r, "o-", label=f"TensorLBM MDF ({kernel})")
r_oss = cases["peskin4pt"]["residual_series_oss_recipe_via_tl"]
ax.semilogy(range(1, len(r_oss) + 1), r_oss, "kx--", label="vivsim MDF (executed)")
r_acc = cases["peskin4pt"]["residual_series_accumulated_recipe"]
ax.semilogy(range(1, len(r_acc) + 1), r_acc, "s:", label="accumulated variant (diagnostic)")
ax.set_xlabel("MDF iteration"); ax.set_ylabel(r"$\|u_t-u_m\|_\infty / \|u_m(0)\|_\infty$")
ax.set_title("Slip residual vs. multi-direct-forcing iterations"); ax.grid(True, which="both", alpha=.3)
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / "fig1_mdf_convergence.png", dpi=160); plt.close(fig)

# 2 — marker density sensitivity
fig, ax = plt.subplots(figsize=(6, 4.2))
d = op["marker_density"]
ax.plot([x["n_markers"] for x in d], [x["slip_residual_n_iter_1"] for x in d], "o-", label="n_iter=1")
ax.plot([x["n_markers"] for x in d], [x["slip_residual_n_iter_5"] for x in d], "s-", label="n_iter=5")
ax.set_xlabel("number of Lagrangian markers"); ax.set_ylabel("slip residual")
ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, which="both", alpha=.3)
ax.set_title("Marker density controls MDF convergence (r=4 lattice cells)")
ax.legend(); fig.tight_layout(); fig.savefig(FIG / "fig2_marker_density.png", dpi=160); plt.close(fig)

# 3 — physics: Cd history per implementation
hist_files = sorted(ART.glob("physics_*.json"))
if hist_files:
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ref = None
    for path in hist_files:
        data = json.loads(path.read_text())
        h = data["history"]
        ax.plot([s["step"] for s in h], [s["cd_drive"] for s in h], "-", lw=1.2,
                label=f"{data['tag']} (Cd from drive closure)")
        ref = data["results"]["cd_reference_schiller_naumann"]
    if ref:
        ax.axhline(ref, color="k", ls="--", lw=1, label=f"Schiller–Naumann Re=100: {ref:.3f}")
    ax.set_xlabel("lattice step"); ax.set_ylabel(r"$C_d$")
    ax.set_title("Sphere at Re=100: drag history (IBM-agnostic drive closure)")
    ax.grid(alpha=.3); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(FIG / "fig3_cd_history.png", dpi=160); plt.close(fig)

    # 4 — bar chart of final Cd
    fig, ax = plt.subplots(figsize=(7, 4.0))
    tags = [json.loads(p.read_text())["tag"] for p in hist_files]
    drv = [json.loads(p.read_text())["results"]["cd_drive_mean"] for p in hist_files]
    mk = [json.loads(p.read_text())["results"]["cd_marker_mean"] for p in hist_files]
    import numpy as np
    x = np.arange(len(tags)); w = 0.38
    ax.bar(x - w/2, drv, w, label="Cd (drive closure)")
    ax.bar(x + w/2, mk, w, label="Cd (marker reaction)")
    if ref:
        ax.axhline(ref, color="k", ls="--", lw=1, label=f"S–N ref {ref:.3f}")
        ax.axhline(1.15*ref, color="r", ls=":", lw=1, label="+15% tolerance")
        ax.axhline(0.85*ref, color="r", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel(r"$C_d$"); ax.set_title("Sphere Re=100 (D=8, 3000 steps): Cd by implementation")
    ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(FIG / "fig4_cd_bars.png", dpi=160); plt.close(fig)

print("figures:", sorted(p.name for p in FIG.glob("*.png")))
