#!/usr/bin/env python3
"""Generate figures for AG27W2-N08 report."""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorlbm.ice_fullship.floe_field import FloeFieldConfig, generate_floe_field

RUNS_DIR = os.path.join("runs", "ag27w2_n08")
os.makedirs(RUNS_DIR, exist_ok=True)

# Load criteria
with open(os.path.join(RUNS_DIR, "criteria.json")) as f:
    report = json.load(f)

with open(os.path.join(RUNS_DIR, "lbm_study.json")) as f:
    lbm = json.load(f)

with open(os.path.join(RUNS_DIR, "dem_base.json")) as f:
    dem = json.load(f)

figs = []

# Fig 1: Floe field
print("Generating Fig. 1: Floe field ...")
cfg_c1 = FloeFieldConfig(length=120.0, width=60.0, concentration=0.60,
                         thickness=0.15, d50=1.0, sigma_g=1.35, seed=20261108)
field = generate_floe_field(cfg_c1)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].scatter(field.x, field.y, s=(field.d * 6) ** 2 / 40, c="#7fb2d9", alpha=0.7, linewidths=0)
ax[0].set_aspect("equal")
ax[0].set_xlabel("x [m]")
ax[0].set_ylabel("y [m]")
c_meas = report["criteria"]["C1"]["audit"]["concentration_measured"]
ax[0].set_title(f"Broken-ice field: C_i={c_meas:.3f} (target 0.60), N={field.n}")
ax[1].hist(field.d, bins=40, color="#4c72b0")
ax[1].axvline(cfg_c1.d50, color="k", ls="--", label=f"target d50={cfg_c1.d50} m")
d50_meas = report["criteria"]["C1"]["audit"]["d50_measured_number"]
ax[1].axvline(d50_meas, color="r", ls=":", label=f"measured d50={d50_meas:.3f} m")
ax[1].set_xlabel("floe diameter [m]")
ax[1].set_ylabel("count")
ax[1].legend(fontsize=8)
fig.tight_layout()
p = os.path.join(RUNS_DIR, "fig1_floe_field.png")
fig.savefig(p, dpi=150)
plt.close(fig)
figs.append(("fig1_floe_field.png", "Broken-ice floe field: concentration and size distribution (C1)"))

# Fig 2: LBM convergence and GCI
print("Generating Fig. 2: LBM convergence and GCI ...")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for key, lab in [("h20_ice2", "n_hull=20"), ("h28_ice2", "n_hull=28"), ("h36_ice2", "n_hull=36")]:
    r = lbm["runs"][key]
    h = r["residual_history"]
    ax[0].semilogy(np.arange(1, len(h) + 1) * 100, h, label=lab)
ax[0].axhline(1e-3, color="k", ls=":")
ax[0].set_xlabel("LBM step")
ax[0].set_ylabel("velocity residual (relative)")
ax[0].set_title("C2: Residual decay (3 grid levels)")
ax[0].legend(fontsize=8)

sizes = [lbm["runs"][k]["grid"]["n_hull"] for k in ("h20_ice0", "h28_ice0", "h36_ice0")]
tot = lbm["R_total_fine_med_coarse"]
inc = lbm["R_ice_hydro_fine_med_coarse"]
ax[1].plot(sizes, tot, "o-", label="R_total (with ice)")
ax[1].plot(sizes, [t - i for t, i in zip(tot, [lbm["runs"][k]["R_x_total"] for k in ("h20_ice0", "h28_ice0", "h36_ice0")])], "s-", label="R_ice (porous increment)")
ax[1].set_xlabel("hull length in lattice cells")
ax[1].set_ylabel("hull force (lattice units)")
ax[1].set_title("C6: Grid study of water-phase resistance")
ax[1].legend(fontsize=8)
fig.tight_layout()
p = os.path.join(RUNS_DIR, "fig2_lbm_gci.png")
fig.savefig(p, dpi=150)
plt.close(fig)
figs.append(("fig2_lbm_gci.png", "Water-phase residual decay and three-grid study (C2/C6)"))

# Fig 3: Resistance comparison
print("Generating Fig. 3: Resistance comparison ...")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
labels = ["DEM chain", "Riska baseline", "Lindqvist\nsubmersion", "Clearing\nmomentum"]
vals = [dem["R_ice_mean"],
        report["criteria"]["C5"]["R_riska_baseline"]["R_ice"],
        report["criteria"]["C5"]["lindqvist_submersion"],
        report["criteria"]["C5"]["clearing_momentum"]]
cols = ["#4c72b0", "#c44e52", "#8172b2", "#55a868"]
ax[0].bar(labels, vals, color=cols)
ax[0].set_yscale("log")
ax[0].set_ylabel("ice resistance [N]")
ax[0].set_title("C5: Chain prediction vs reference models")
ax[0].tick_params(axis='x', labelsize=8)

ea = dem["energy_audit"]
keys = ["P_resistance", "P_balance", "D_contacts", "P_water", "P_hull_slip", "P_hull_perp"]
colors_bar = ["#4c72b0", "#c44e52", "#8172b2", "#55a868", "#ddb845", "#6b8fa3"]
ax[1].bar(range(len(keys)), [abs(ea.get(k, 0)) for k in keys], color=colors_bar)
ax[1].set_xticks(range(len(keys)))
ax[1].set_xticklabels(keys, rotation=45, ha="right", fontsize=7)
ax[1].set_ylabel("power [W] (absolute value)")
ax[1].set_title("C4: Floe-phase power terms")
fig.tight_layout()
p = os.path.join(RUNS_DIR, "fig3_resistance_energy.png")
fig.savefig(p, dpi=150)
plt.close(fig)
figs.append(("fig3_resistance_energy.png", "Ice resistance vs references and floe-phase power terms (C4/C5)"))

# Fig 4: DEM force time series
print("Generating Fig. 4: DEM force time series ...")
ts = dem.get("R_series", [])
if ts:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(np.arange(len(ts)) * dem.get("dt", 1e-3), ts, lw=0.6, color="#4c72b0")
    ax.axhline(dem["R_ice_mean"], color="r", ls="--", label=f"mean={dem['R_ice_mean']:.2f} N")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("hull force x [N]")
    ax.set_title("DEM instantaneous ice resistance on the hull")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(RUNS_DIR, "fig4_dem_series.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append(("fig4_dem_series.png", "Instantaneous DEM ice resistance on the hull"))

# Update criteria.json with figure list
report["figures"] = [name for name, _ in figs]
with open(os.path.join(RUNS_DIR, "criteria.json"), "w") as f:
    json.dump(report, f, indent=2, default=float)

print(f"\nGenerated {len(figs)} figures in {RUNS_DIR}/")
for name, caption in figs:
    print(f"  {name}: {caption}")
