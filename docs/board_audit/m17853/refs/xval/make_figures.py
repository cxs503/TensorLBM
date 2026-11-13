#!/usr/bin/env python3
"""Figures for XVAL-LBM-DIFFUSE-INTERFACE-WETTING-VIA-WALL-ORDER-PARAMETER.

Every number is read back from runs/*/result.json and runs/*/psi.npz.
"""
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
summ = json.loads((HERE / "summary.json").read_text())
rows = [r for r in summ["port_rows"] if r["xi"] == 3.0]
rows.sort(key=lambda r: r["psi_wall"])
pw = [r["psi_wall"] for r in rows]
th = [r["theta_deg"] for r in rows]
pw_c = np.linspace(-0.75, 0.75, 401)
th_c = np.degrees(np.arccos((3 * pw_c - pw_c**3) / 2))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(pw_c, th_c, "k-", lw=1.5, label=r"Young law $\cos\theta=(3\psi_w-\psi_w^3)/2$")
ax[0].plot(pw, th, "o", ms=7, color="tab:red", label="TensorLBM wall-order-parameter port")
for r in rows:
    ax[0].annotate(f"{r['err_vs_analytic_pct']:+.1f}%", (r["psi_wall"], r["theta_deg"]),
                   textcoords="offset points", xytext=(4, -12), fontsize=7)
ax[0].set_xlabel(r"wall order parameter $\psi_w$"); ax[0].set_ylabel(r"contact angle $\theta$ [deg]")
ax[0].set_title("static contact-angle law (xi=3, 15% gate: 9/10 rows pass)")
ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)

xi = sorted({r["xi"] for r in summ["port_rows"]})
for pwc, mk in ((0.0, "o"), (0.35, "s")):
    y = []
    for x in xi:
        c = [r for r in summ["port_rows"] if r["xi"] == x and abs(r["psi_wall"] - pwc) < 1e-9]
        y.append(c[0]["theta_deg"] if c else np.nan)
    ax[1].plot(xi, y, mk + "-", label=fr"$\psi_w={pwc:+.2f}$")
    ax[1].plot(xi, [math.degrees(math.acos((3 * pwc - pwc**3) / 2))] * len(xi), "k--", lw=.8)
ax[1].set_xlabel(r"interface width $\xi$ [lattice units]"); ax[1].set_ylabel(r"$\theta$ [deg]")
ax[1].set_title(r"resolution sensitivity (black dashed = analytic)")
ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig1_contact_angle_law.png", dpi=150); plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
for axx, tag in zip(axes, ["port_psi-0.50_xi3.0", "port_psi+0.00_xi3.0",
                           "port_psi+0.35_xi3.0", "port_psi+0.70_xi3.0"]):
    f = HERE / "runs" / tag / "psi.npz"
    z = np.load(f)["psi"]
    pwv = float(tag.split("psi")[1].split("_")[0])
    im = axx.imshow(z, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axx.contour(z, levels=[0.0], colors="k", linewidths=.7)
    axx.set_title(fr"$\psi_w={pwv:+.2f}$", fontsize=9); axx.set_xlabel("x"); axx.set_ylabel("y")
fig.colorbar(im, ax=axes, label=r"$\psi$", shrink=.8)
fig.suptitle("equilibrium caps (Allen-Cahn descent of the wall-order-parameter free energy)", fontsize=9)
fig.savefig(FIG / "fig2_cap_fields.png", dpi=150, bbox_inches="tight"); plt.close(fig)

lbm = [r for r in summ["lbm_rows"] if r["run"] == "lbm_psi+0.00_m0.1_isotropic"]
if lbm:
    d = json.loads((HERE / "runs" / lbm[0]["run"] / "result.json").read_text())
    st = [s["step"] for s in d["series"] if s["psi_max"] is not None]
    pm = [s["psi_max"] for s in d["series"] if s["psi_max"] is not None]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(st, pm, "o-", label=r"faithful LBM port ($\Gamma=0.1$)")
    ax.axhline(0.0, color="k", lw=.8, ls=":")
    ax.set_xlabel("lattice step"); ax.set_ylabel(r"$\max\psi$ (cap peak)")
    ax.set_title("reference order-parameter closure: cap dissolves\n(spurious $\\propto\\nabla^2\\psi$ diffusion)", fontsize=9)
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig3_lbm_dissociation.png", dpi=150); plt.close(fig)
print("figures written:", sorted(p.name for p in FIG.glob("*.png")))
