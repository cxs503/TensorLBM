#!/usr/bin/env python3
"""Figures for CASE-LBM-SQUARE-CYL-NORBERG: Cl/Cd traces, Cl spectrum, ref-vs-got."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/m16321/sqcyl_norberg"
FIG = RUN / "figs"; FIG.mkdir(parents=True, exist_ok=True)
refs = json.loads((ROOT / "cases/square_cylinder_norberg/refs/norberg1993.json").read_text())
st_ref, cd_ref = refs["metrics"]["st"]["value"], refs["metrics"]["cd"]["value"]

res, series = {}, {}
for D in (8, 10):
    rp, sp = RUN / f"D{D}" / f"result_D{D}.json", RUN / f"D{D}" / f"series_D{D}.npz"
    if rp.exists() and sp.exists():
        res[D] = json.loads(rp.read_text()); series[D] = np.load(sp)

# fig 1: traces
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for D, s in series.items():
    n = s["cd"].size; t = np.arange(n)
    ax[0].plot(t, s["cd"], lw=0.6, label=f"D={D} (mean {res[D]['cd']:.3f})")
    ax[1].plot(t, s["cl"], lw=0.6, label=f"D={D} (St {res[D]['st']:.4f})")
    ax[0].axvspan(0, int(s["warmup_from"]), color="0.85", alpha=0.6)
ax[0].set_ylabel("$C_d$"); ax[0].legend(loc="lower right", fontsize=8)
ax[0].axhline(cd_ref, color="k", ls="--", lw=0.8, label="ref")
ax[1].set_ylabel("$C_l$"); ax[1].set_xlabel("lattice step (sampled)")
ax[1].legend(loc="upper right", fontsize=8); ax[0].set_title("square cylinder Re=200, blockage 1/20 - force histories (grey = warm-up)")
fig.tight_layout(); fig.savefig(FIG / "fig1_force_traces.png", dpi=130); plt.close(fig)

# fig 2: Cl spectrum
fig, ax = plt.subplots(figsize=(7, 4))
for D, s in series.items():
    cl = s["cl"][int(s["warmup_from"]):]; cl = cl - cl.mean()
    spec = np.abs(np.fft.rfft(cl * np.hanning(cl.size), n=8 * cl.size))
    f = np.fft.rfftfreq(8 * cl.size) * D / 0.05   # -> St
    m = f > 0.05
    ax.semilogy(f[m], spec[m], lw=0.8, label=f"D={D}: peak St={res[D]['st']:.4f}")
ax.axvline(st_ref, color="k", ls="--", lw=0.8); ax.text(st_ref, ax.get_ylim()[1]*0.5, " ref 0.13", fontsize=8)
ax.set_xlabel("St = f D / U"); ax.set_ylabel("|FFT($C_l$)|"); ax.legend(fontsize=8)
ax.set_title("lift spectrum (Hann window, 8x zero-padding)"); fig.tight_layout()
fig.savefig(FIG / "fig2_cl_spectrum.png", dpi=130); plt.close(fig)

# fig 3: ref vs got
if res:
    Ds = sorted(res); x = np.arange(len(Ds)); w = 0.35
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(x - w/2, [res[d]["st"] for d in Ds], w, label="got")
    ax[0].axhline(st_ref, color="r", ls="--", label=f"ref {st_ref}")
    ax[0].fill_between([-0.5, len(Ds)-0.5], st_ref*0.92, st_ref*1.08, color="r", alpha=0.12, label="8% tol")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"D={d}" for d in Ds]); ax[0].set_ylabel("St"); ax[0].legend(fontsize=8)
    ax[0].set_title("Strouhal number vs Norberg-1993 anchor value")
    ax[1].bar(x - w/2, [res[d]["cd"] for d in Ds], w, label="got")
    ax[1].axhline(cd_ref, color="r", ls="--", label=f"ref {cd_ref}")
    ax[1].fill_between([-0.5, len(Ds)-0.5], cd_ref*0.9, cd_ref*1.1, color="r", alpha=0.12, label="10% tol")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"D={d}" for d in Ds]); ax[1].set_ylabel("$C_d$"); ax[1].legend(fontsize=8)
    ax[1].set_title("mean drag coefficient vs anchor value")
    fig.tight_layout(); fig.savefig(FIG / "fig3_ref_vs_got.png", dpi=130); plt.close(fig)
print("figs:", [p.name for p in FIG.glob("*.png")], "res D:", list(res))
for d in res:
    print(d, res[d]["cd"], res[d]["cd_err_pct"], res[d]["st"], res[d]["st_err_pct"], res[d]["samples"])
