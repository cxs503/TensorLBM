#!/usr/bin/env python3
"""m28104 — DFG 2D-2 Cl-amplitude deviation: convergence + estimator comparison figure."""
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from tensorlbm.validation import cylinder_dfg as cdfg

ART = REPO / "benchmarks/verified/cylinder/re100_dfg/artifacts"
BANDS = cdfg.REFERENCE_BANDS["2D-2"]

def rejudge(D):
    sp = ART / f"level_D{D}_u0833.series.npz"
    jp = ART / f"level_D{D}_u0833.json"
    if not (sp.exists() and jp.exists()): return None
    old = json.loads(jp.read_text()); z = np.load(sp)
    T = cdfg.CaseConfig(D_cells=D, u_mean_lb=0.0833, n_steps=10**9,
                        transient_periods=5.0, avg_periods=5.0).period_steps
    skip = int(round(5.0*T))
    cl = z["cl"].astype("float64")[skip:]
    amp = cdfg.limit_cycle_amplitude(__import__("torch").tensor(cl), T, d_cells=D, u_mean_lb=0.0833)
    return {"D": D, "tau": old["tau"], "legacy": old["cl_amp"], "fixed": amp["amp"]}

rows = [r for D in (20,28,40,48,56) if (r := rejudge(D)) is not None]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
Ds = [r["D"] for r in rows]
leg = [r["legacy"] for r in rows]; fix = [r["fixed"] for r in rows]
mid = cdfg.band_mid(BANDS["cl_amp"])
ax[0].plot(Ds, [ (v-mid)/mid*100 for v in leg ], "o--", color="tab:red", label="legacy (peak-pick, whole record)")
ax[0].plot(Ds, [ (v-mid)/mid*100 for v in fix ], "s-", color="tab:green", label="fixed (fundamental, post-transient)")
ax[0].axhspan(-5, 5, color="green", alpha=0.15, label="±5% tol")
ax[0].axhline(15, color="gray", ls=":", label="+15% (task criterion)")
ax[0].set_xlabel("D [cells]"); ax[0].set_ylabel("C_L amplitude error vs DFG mid [%]")
ax[0].set_title("(a) Cl-amplitude error: estimator + resolution")
ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(Ds, [r["tau"] for r in rows], "o-", color="tab:blue")
ax[1].set_xlabel("D [cells]"); ax[1].set_ylabel("τ (MRT)")
ax[1].set_title("(b) τ vs resolution (Re=100 fixed, u_mean=0.0833)")
ax[1].grid(alpha=0.3)
fig.tight_layout()
out = pathlib.Path(__file__).parent / "m28104_dfg_cl_amp.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
