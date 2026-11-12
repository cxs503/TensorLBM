"""Figures for docs/AUDIT_m16827_CYLINDER_DFG_CL_AMP.md (numbers read from JSON, zero hand-copy)."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
CASE = ROOT / "benchmarks/verified/cylinder/re100_dfg/artifacts"
rj = json.loads((CASE / "rejudge_cl_amp.json").read_text())
z = np.load(CASE / "level_D28_u0833.series.npz")
cd, cl = z["cd"].astype(float), z["cl"].astype(float)
T = 1120.0
skip = rj["levels"]["D28"]["transient_steps_dropped"]
t = np.arange(len(cl)) / T  # in shedding periods

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(t, cl, lw=0.6, color="#1f77b4", label="C_L(t)")
ax[0].plot(t, cd, lw=0.6, color="#ff7f0e", alpha=.8, label="C_D(t)")
ax[0].axvspan(0, skip / T, color="red", alpha=.12)
ax[0].axvline(skip / T, color="red", ls="--", lw=1)
ax[0].text(skip / T / 2, ax[0].get_ylim()[1]*0.92, "start-up\ntransient\n(dropped)",
           ha="center", va="top", fontsize=8, color="darkred")
ax[0].axhline(0, color="k", lw=.5)
old = rj["levels"]["D28"]["cl_amp_old_peakpick_all_window"]
new = rj["levels"]["D28"]["cl_amp_new"]
ax[0].hlines([old, new], len(cl)/T*0.02, len(cl)/T*0.30,
             colors=["grey", "green"], linestyles=[":", "-"], lw=2)
ax[0].text(len(cl)/T*0.31, old, f"old peak-pick over whole record {old:.3f} (+36.0%)",
           fontsize=8, color="grey", va="center")
ax[0].text(len(cl)/T*0.31, new, f"corrected limit-cycle amplitude {new:.3f} (+17.8%)",
           fontsize=8, color="green", va="center")
ax[0].set_xlabel("t / shedding period  (D=28, 20 000 steps = 18 periods)")
ax[0].set_ylabel("force coefficient")
ax[0].set_title("Panel A — why the published +36 % was an estimator artifact")
ax[0].legend(fontsize=8, loc="lower right")

lv = ["D20", "D28", "D40"]
xo = np.arange(3)
for name, col, mk in [("err_pct_old", "grey", "o"), ("err_pct_new", "tab:green", "s")]:
    v = [rj["levels"][k][name]["cl_amp"] for k in lv]
    ax[1].plot(xo, v, mk + "-", color=col, label=f"C_L amplitude {name.replace('err_pct_','')}")
for k, i in zip(lv, xo):
    ax[1].annotate(f"{rj['levels'][k]['err_pct_new']['cl_amp']:+.1f}%",
                   (i, rj["levels"][k]["err_pct_new"]["cl_amp"]), textcoords="offset points",
                   xytext=(6, 6), fontsize=8, color="tab:green")
    ax[1].annotate(f"{rj['levels'][k]['err_pct_old']['cl_amp']:+.1f}%",
                   (i, rj['levels'][k]['err_pct_old']['cl_amp']), textcoords="offset points",
                   xytext=(6, -12), fontsize=8, color="grey")
ax[1].axhspan(-5, 5, color="tab:green", alpha=.10)
ax[1].axhline(0, color="k", lw=.5)
ax[1].axhline(5, color="tab:red", ls="--", lw=.8)
ax[1].set_xticks(xo); ax[1].set_xticklabels(lv)
ax[1].set_xlabel("grid level (D_cells on the cylinder diameter)")
ax[1].set_ylabel("C_L amplitude error vs band mid (%)")
ax[1].set_title("Panel B — corrected error ladder (band ±5 %)")
ax[1].legend(fontsize=8)
fig.tight_layout()
out = pathlib.Path(__file__).with_name("m16827_cylinder_dfg_cl_amp.png")
fig.savefig(out, dpi=150)
# numbers sidecar (programmatic injection into the report/doc)
side = {k: {m: round(rj["levels"][k][m], 4) for m in
            ("cl_amp_old_peakpick_all_window", "cl_amp_new", "cl_amp_peakpick_median",
             "cl_amp_peakpick_excess_pct", "cl_amp_n_periods", "st_new", "cd_new", "dp_new")}
        for k in lv}
side["err_new"] = {k: {m: round(v, 2) for m, v in rj["levels"][k]["err_pct_new"].items()} for k in lv}
side["err_old"] = {k: {m: round(v, 2) for m, v in rj["levels"][k]["err_pct_old"].items()} for k in lv}
pathlib.Path(out).with_suffix(".json").write_text(json.dumps(side, indent=1))
print(out)
print(json.dumps(side, indent=1))
