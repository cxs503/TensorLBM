"""Figures for XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION (analytic data only)."""
import json, sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from tensorlbm.multiphase.laplace import sc_maxwell_coexistence, sc_pressure
from tensorlbm.multiphase import RHO_L_DEFAULT, RHO_V_DEFAULT, SIGMA_DEFAULT

ART = Path(__file__).resolve().parent / "artifacts"; ART.mkdir(exist_ok=True)
mw5, mw47 = sc_maxwell_coexistence(5.0), sc_maxwell_coexistence(4.7)
rho = np.linspace(1e-3, 3.0, 400)
psi = 1 - np.exp(-rho)

fig, ax = plt.subplots(figsize=(6.4, 4.2))
for G, c, mw in ((5.0, "C0", mw5), (4.7, "C1", mw47)):
    p = rho / 3 - (G / 6) * psi ** 2
    ax.plot(rho, p, c, label=f"SC EOS  p=rho/3-G psi^2/6, G={G}")
    ax.axhline(mw["p"], ls=":", c=c, lw=1)
    ax.plot([mw["rho_v"], mw["rho_l"]], [mw["p"]] * 2, "o", c=c, ms=7,
            label=f"Maxwell G={G}: {mw['rho_v']:.3f}/{mw['rho_l']:.3f}")
ax.set_xlabel("rho"); ax.set_ylabel("p"); ax.grid(alpha=.3); ax.legend(fontsize=7)
ax.set_title("Shan-Chen EOS + Maxwell construction (attractive, p-int<0)")
fig.tight_layout(); fig.savefig(ART / "fig1_eos_maxwell.png", dpi=145); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.6, 4.0))
labels = ["library\nconstants", "Maxwell\nG=5", "Maxwell\nG=4.7", "reference C++\nrun (g=-4.7)"]
rl = [RHO_L_DEFAULT, mw5["rho_l"], mw47["rho_l"], 1.6707]
rv = [RHO_V_DEFAULT, mw5["rho_v"], mw47["rho_v"], 0.18473]
x = np.arange(4)
ax.bar(x - .2, rl, .4, label="rho_l"); ax.bar(x + .2, rv, .4, label="rho_v")
for i, (a, b) in enumerate(zip(rl, rv)):
    ax.text(i - .2, a + .03, f"{a:.3f}", ha="center", fontsize=7)
    ax.text(i + .2, b + .03, f"{b:.3f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
ax.set_title("Coexistence: Maxwell (analytic) vs documented / measured")
fig.tight_layout(); fig.savefig(ART / "fig2_coexistence.png", dpi=145); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.6, 4.0))
dp_ok = float(sc_pressure(torch.tensor(RHO_L_DEFAULT), 5.0) - sc_pressure(torch.tensor(RHO_V_DEFAULT), 5.0))
pl_wrong = RHO_L_DEFAULT / 3 + 5 / 6 * (1 - np.exp(-RHO_L_DEFAULT)) ** 2
pv_wrong = RHO_V_DEFAULT / 3 + 5 / 6 * (1 - np.exp(-RHO_V_DEFAULT)) ** 2
dp_wrong = float(pl_wrong - pv_wrong)
ax.bar(["correct\n(-G psi^2/6)", "as in\nwetting.measure_young_laplace\n(+G psi^2/6)"],
       [dp_ok, dp_wrong], color=["C2", "C3"])
ax.set_yscale("log"); ax.set_ylabel("Delta p = p_l - p_v")
for i, v in enumerate([dp_ok, dp_wrong]):
    ax.text(i, v * 1.4, f"{v:.4g}", ha="center", fontsize=9)
ax.set_title(f"Pressure-term sign: wrong branch x{dp_wrong/dp_ok:.0f}")
ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(ART / "fig3_pressure_sign.png", dpi=145); plt.close(fig)

summary = {
    "maxwell_G5": {k: float(v) for k, v in mw5.items()},
    "maxwell_G4.7": {k: float(v) for k, v in mw47.items()},
    "library_constants": {"rho_l": RHO_L_DEFAULT, "rho_v": RHO_V_DEFAULT, "sigma": SIGMA_DEFAULT},
    "reference_cpp_run": {"g": -4.7, "rho_l": 1.6707, "rho_v": 0.18473, "source": "OSS inventory §9"},
    "dp_correct_sign": dp_ok, "dp_plus_sign": dp_wrong, "ratio": dp_wrong / dp_ok,
    "solver_arms": {
        "library_current_force_CS2k_G5_ramp800_R10_n48": {"status": "uniform (no phase separation)", "rho": 0.4173},
        "force_without_CS2_k_Gdiv3_G5_R10_12_14_n48": {"status": "diverged", "steps": [24, 23, 21]},
        "run_young_laplace_radius16_n64_G5_and_G2": {"status": "diverged", "steps": "<1500"},
    },
}
(ART / "laplace_calibration_analytics.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=1))
