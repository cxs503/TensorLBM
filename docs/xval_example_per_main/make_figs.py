#!/usr/bin/env python3
"""Figures for XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN (numbers read from artifacts, never typed)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

XV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                  "xval", "XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN", "run_artifacts")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)


def load(n):
    with open(os.path.join(XV, n), encoding="utf-8") as fh:
        return json.load(fh)


# ---- fig1: channel profile, got vs closed form, 3 resolutions -------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for ny, color in ((16, "C0"), (32, "C2"), (48, "C3")):
    rec = load(f"arm_a1_channel_ny{ny}.json")
    nu, g = rec["spec"]["tau"], rec["spec"]["body_force"]
    nu = (rec["spec"]["tau"] - 0.5) / 3.0
    y = list(range(1, ny - 1))
    ref = [g / (2 * nu) * (j - 0.5) * (ny - 1.5 - j) for j in y]
    ax[0].plot(ref, y, "--", color=color, lw=1.0, label=f"analytic ny={ny}")
    ax[0].plot(rec["profile"], y, "o", color=color, ms=3.5, label=f"LBM ny={ny}")
ax[0].set_xlabel(r"$u_x$ (lattice units)"); ax[0].set_ylabel("y (lattice row)")
ax[0].set_title("force-driven Poiseuille: profile vs $gH^2/(8\\nu)$")
ax[0].legend(fontsize=7)

ers = [load(f"arm_a1_channel_ny{ny}.json")["kpis"]["relative_error_vs_analytic"] for ny in (16, 32, 48)]
l2s = [load(f"arm_a1_channel_ny{ny}.json")["kpis"]["profile_rel_l2"] for ny in (16, 32, 48)]
ax[1].loglog([16, 32, 48], [100 * e for e in ers], "o-", label=r"$u_{max}$ rel. err [%]")
ax[1].loglog([16, 32, 48], [100 * e for e in l2s], "s-", label="profile $L_2$ rel. err [%]")
ax[1].axhline(3.0, color="r", ls=":", label="3% target")
ax[1].axhline(15.0, color="k", ls="--", label="15% owner band")
ax[1].set_xlabel("ny (channel rows)"); ax[1].set_ylabel("relative error [%]")
ax[1].set_title("resolution convergence"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_channel_profile_convergence.png"), dpi=140)
plt.close(fig)

# ---- fig2: Taylor-Green decay + antipattern audit ------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
tg = load("arm_a3_taylor_green_32x32.json")
t = [h["step"] for h in tg["history"]]; e = [h["kinetic_energy"] for h in tg["history"]]
ax[0].semilogy(t, e, "o", ms=3.5, label="LBM energy")
rate = tg["kpis"]["decay_rate_measured"]; e0 = e[0]
ax[0].semilogy(t, [e0 * pow(2.718281828459045, -rate * s) for s in t], "-",
               label=f"fit $\\lambda$={rate:.5f}")
ax[0].semilogy(t, [e0 * pow(2.718281828459045, -tg["analytic"]["decay_rate"] * s) for s in t], "--",
               label=f"analytic 4$\\nu k^2$={tg['analytic']['decay_rate']:.5f}")
ax[0].set_xlabel("step"); ax[0].set_ylabel("kinetic energy density")
ax[0].set_title(f"Taylor-Green decay, rel. err {100*tg['kpis']['relative_error_vs_analytic']:.3f}%")
ax[0].legend(fontsize=7)

summ = load("summary.json")
a5 = summ["arms"]["A5_examples_audit"]["M2"]; a6 = summ["arms"]["A6_detector_control"]
labels = ["repo examples/\n(161 files)", "planted\nanti-pattern", "planted\nconsolidated"]
dup = [100 * a5["dup_loc_ratio"], 100 * a6["antipattern_dup_loc_ratio"], 100 * a6["consolidated_dup_loc_ratio"]]
drv = [100 * a5["dup_driver_loc_ratio"], 100 * a6["antipattern_driver_loc_ratio"], 100 * a6["consolidated_driver_loc_ratio"]]
x = range(len(labels)); w = 0.38
ax[1].bar([i - w / 2 for i in x], dup, w, label="all duplicated lines [%]")
ax[1].bar([i + w / 2 for i in x], drv, w, label="duplicated driver lines [%]")
ax[1].set_xticks(list(x)); ax[1].set_xticklabels(labels, fontsize=8)
ax[1].set_ylabel("share of source lines [%]")
ax[1].set_title("example-per-main anti-pattern detector")
ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_tg_decay_and_antipattern.png"), dpi=140)
plt.close(fig)
print("wrote", sorted(os.listdir(OUT)))
