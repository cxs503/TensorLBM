#!/usr/bin/env python3
"""DE-X-CAVSWEEP figures: (1) u_limit sensitivity curve, (2) depth-time vs experiment,
(3) drift trajectories of dead arms. All numbers read from local json copies."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "runs", "cavsweep")
DATA = os.path.abspath(DATA)
OUT = DATA
EXP = "fig5_experiment.csv"

def load(arm):
    p = os.path.join(OUT, arm + ".json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))

def hist(d):
    h = d.get("history") or []
    t = np.array([x["time_ms"] for x in h])
    dr = np.array([x["volume_drift"] for x in h])
    dep = np.array([x.get("depth_over_D", np.nan) for x in h])
    return t, dr, dep

# ---- fig 1: u_limit sensitivity ----
fig, ax = plt.subplots(figsize=(6, 4))
ul_arms = {"ul006": 0.06, "ul010": 0.10, "ctrl": 0.15, "ul020": 0.20, "ul024": 0.24}
xs, ys, marks, labels = [], [], [], []
for a, ul in ul_arms.items():
    d = load(a)
    if d is None:
        continue
    t, dr, dep = hist(d)
    dep = np.where(np.isnan(dep), np.nan, dep)
    pk = np.nanmax(dep) if np.any(~np.isnan(dep)) else np.nan
    died = d.get("stop_reason") == "health_limit"
    xs.append(ul); ys.append(pk)
    marks.append("x" if died else "o")
    labels.append(f"{a} ({d.get('stop_reason')})" if died else f"{a}")
    ax.plot(t, dep, lw=0.8, alpha=0.5)
    ax.plot(ul, pk, marks, ms=9, mew=2)
    ax.annotate(f"{pk:.3f}D", (ul, pk), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.axhline(1.8574, color="r", ls="--", lw=1)
ax.text(0.065, 1.87, "exp peak 1.8574D", color="r", fontsize=8)
ax.set_xlabel("u_limit"); ax.set_ylabel("peak depth / D")
ax.set_title("u_limit sensitivity (cells=8, base config)")
ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_ulimit.png"), dpi=140)

# ---- fig 2: depth-time vs experiment ----
fig, ax = plt.subplots(figsize=(6, 4))
exp = np.genfromtxt(os.path.join(OUT, EXP), delimiter=",", names=True)
ax.plot(exp["time_ms"], exp["depth_over_D"], "k^", ms=5, label="experiment (fig5)")
for a, c in [("ctrl", "tab:blue"), ("ul006", "tab:orange"), ("ul020", "tab:green")]:
    d = load(a)
    if d is None:
        continue
    t, dr, dep = hist(d)
    ax.plot(t, dep, color=c, lw=1.2, label=a)
ax.set_xlabel("t / ms"); ax.set_ylabel("depth / D")
ax.set_title("cavity depth vs time (Santini case-3)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_depth_time.png"), dpi=140)

# ---- fig 3: drift of dead arms ----
fig, ax = plt.subplots(figsize=(6, 4))
for a in ["gap12", "pd20", "rs175", "spd007", "ul012pd20", "caponly", "simonly", "ul006"]:
    d = load(a)
    if d is None:
        continue
    t, dr, dep = hist(d)
    ax.semilogy(t, np.maximum(dr, 1e-9), label=a)
ax.axhline(1e-4, color="k", ls="--", lw=1)
ax.text(0.05, 1.1e-4, "gate 1e-4", fontsize=8)
ax.set_xlabel("t / ms"); ax.set_ylabel("volume_drift")
ax.set_title("volume_drift trajectories (all deaths at the 1e-4 gate)")
ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_drift.png"), dpi=140)
print("figures done")
