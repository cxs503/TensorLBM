"""P40W3B-N02 figures (all numbers read from artifacts/p40w3b_n02/*.json)."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tensorlbm.cavitation_airfoil import CavitationFlowConfig, MixtureCavitationSolver
from tensorlbm.cavitation_numerics import cavity_length_isobaric, integrate_closure_growth, pressure_anchor, rayleigh_plesset_time
from tensorlbm.cavitation_free_streamline import flow_solve, panel_geometry

ART = ROOT / "artifacts" / "p40w3b_n02"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---- fig 1: L/c vs sigma ---------------------------------------------------
sheet = json.loads((ART / "sheet.json").read_text())
ref = sheet["reference"]
sig = np.array([r["sigma"] for r in sheet["rows"]])
lc = np.array([r["l_cav"] for r in sheet["rows"]])
lcu = np.array([r["l_cav_un_anchored"] for r in sheet["rows"]])
fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.plot(sig, ref["l_cav_exp"], "ks-", label="Brennen 1994/95 experiment", lw=1.6)
ax.plot(sig, ref["l_cav_merkle"], "b^--", label="Merkle et al. 1998 (numerical)", lw=1.4)
ax.plot(sig, lc, "ro-", label="this work: anchored isobaric criterion")
ax.plot(sig, lcu, "o", mfc="none", color="gray", label="this work: un-anchored (P41-N02)")
ax.set_xlabel(r"cavitation number $\sigma$"); ax.set_ylabel(r"$L_{cav}/c$")
ax.set_title("Sheet cavity length, NACA0015 $\\alpha=5^\\circ$, 50 cells/chord\n(transient, t = 2 c/U; run diverges at t = 2.28 c/U)")
ax.invert_xaxis(); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "p40w3b_n02_lcav_vs_sigma.png", dpi=150); plt.close(fig)

# ---- fig 2: closure time scales and integrated growth ----------------------
clo = json.loads((ART / "closure.json").read_text())
rows = [r for r in clo["rows"] if r["r_b_floor"] == 1e-4]
al = np.array([r["alpha"] for r in rows])
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8))
ax[0].loglog(al, [r["ss_growth_time"] for r in rows], "o-", label="Schnerr-Sauer e-folding")
ax[0].loglog(al, [r["rp_time"] for r in rows], "s--", label="Rayleigh-Plesset $1/\\omega$")
ax[0].set_xlabel("void fraction $\\alpha_v$"); ax[0].set_ylabel("time [$c/U$]")
ax[0].set_title("phase-change time scales ($\\sigma=0.8$)"); ax[0].grid(alpha=0.3, which="both"); ax[0].legend(fontsize=8)
g = integrate_closure_growth(0.4, n_steps=20000, alpha0=1e-4, r_b_floor=1e-4)
t = np.array(g["time"]); a = np.array(g["alpha"])
ax[1].loglog(t, a, "k-", label="closure ODE integration")
ax[1].loglog(t, np.clip(a, 1e-30, 1) * 0 + (t / t[-1]) ** 3 * a[-1], "r--", label="$\\alpha\\propto t^3$ (RP inertia)")
ax[1].set_xlabel("time [$c/U$]"); ax[1].set_ylabel("$\\alpha_v$")
ax[1].set_title(f"closure growth, exponent={g['growth_exponent']:.2f}, $R/R_{{RP}}$={g['ratio_r_rp']:.2f}")
ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "p40w3b_n02_closure_timescales.png", dpi=150); plt.close(fig)

# ---- fig 3: stability / divergence -----------------------------------------
stab = json.loads((ART / "stability.json").read_text())
h = np.array(stab["max_u_history"])
fig, ax = plt.subplots(figsize=(6.0, 3.8))
ax.semilogy(h[:, 0], h[:, 1], "ko-", ms=3)
ax.axvline(stab["diverged_at_step"], color="r", ls="--", label=f"blow-up at step {stab['diverged_at_step']}")
ax.set_xlabel("time step"); ax.set_ylabel("$\\max|u|$ [$U_\\infty$]")
ax.set_title("wake instability of the mixture engine ($\\sigma=0.8$, 50 cells/chord)")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "p40w3b_n02_divergence.png", dpi=150); plt.close(fig)

# ---- fig 4: field map + isobaric cavity contour ----------------------------
cfg = CavitationFlowConfig(airfoil="NACA0015", alpha_deg=5.0, sigma=0.8, cells_per_chord=50,
    domain_x_chords=6.0, domain_y_chords=3.0, le_x_chords=1.5, le_y_chords=1.5,
    cfl=0.2, smagorinsky=0.1, c_dest=1.0, c_prod=1.0, spinup_steps=300, ramp_steps=200,
    sample_interval=10, n_steps=500)
s = MixtureCavitationSolver(cfg)
s.u = s.u.masked_fill(s.damp_u > 1.5, 0.0); s.v = s.v.masked_fill(s.damp_v > 1.5, 0.0)
for _ in range(500):
    s.step()
off = pressure_anchor(s.p)
pp = (s.p + off).numpy()
h_ = cfg.h
x = (np.arange(cfg.nx) + 0.5) * h_ - cfg.le_x_chords
y = (np.arange(cfg.ny) + 0.5) * h_ - cfg.le_y_chords
m = cavity_length_isobaric(s.p, cfg, s.geometry)
fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.6))
cs = ax[0].contourf(x, y, pp.T, levels=24, cmap="viridis")
ax[0].contour(x, y, pp.T, levels=[cfg.p_v], colors="w", linewidths=1.4)
ax[0].set_xlim(-0.4, 2.0); ax[0].set_ylim(-0.6, 0.6)
ax[0].set_title(f"$p/\\rho U^2$ (anchored), $\\sigma=0.8$: white = $p=p_v$, $L/c$={m['l_cav']:.3f}")
ax[0].set_xlabel("x/c"); ax[0].set_ylabel("y/c")
fig.colorbar(cs, ax=ax[0], shrink=0.85)
ax[1].plot(np.arange(len(s.cl_history)) * cfg.sample_interval, s.cl_history, "b-")
ax[1].set_xlabel("time step"); ax[1].set_ylabel("$C_L$")
ax[1].set_title(f"$C_L(t)$ (Brinkman reaction; $C_D$={s.cd():.2f} vs 0.02 expected)")
ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "p40w3b_n02_field_map.png", dpi=150); plt.close(fig)

# ---- fig 5: panel engine verification --------------------------------------
n = 160; a = 0.5
th = np.linspace(0.0, 2 * math.pi, n + 1)
nodes = np.stack([a * np.cos(th), a * np.sin(th)], axis=1)
st = flow_solve(panel_geometry(nodes), 0.0, kutta="none")
thm = th[:-1] + np.diff(th) / 2
fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.plot(np.degrees(thm), st.qt, "k-", label="panel method")
ax.plot(np.degrees(thm), -2 * np.sin(thm), "r--", label="analytic $2U\\sin\\theta$")
ax.set_xlabel("$\\theta$ [deg]"); ax.set_ylabel("$q_\\theta/U$")
ax.set_title(f"flow-engine verification: max err {abs(st.qt + 2*np.sin(thm)).max()/2:.1e}, $C_{{p,min}}$={st.cp.min():.3f}")
ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "p40w3b_n02_panel_verification.png", dpi=150); plt.close(fig)
print("figures written:", sorted(p.name for p in FIG.glob("p40w3b_n02_*.png")))
