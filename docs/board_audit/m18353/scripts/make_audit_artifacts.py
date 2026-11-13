"""m18353 audit: NaN-onset probe + figure for the m18236 verdict (board-audit seat).

Reads the delivered artefacts of m18236 and re-measures the one thing the report
does not: where the plug dies relative to the imposed pressure.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

W = "/root/tp-agent/work/m18236"
sys.path.insert(0, W + "/src")
from tensorlbm import cac_lbm2d as cac          # noqa: E402
from tensorlbm import pc_ladder as P            # noqa: E402

torch.set_num_threads(1)
OUT = "/root/tp-agent/work/m18353/docs/board_audit/m18353"
os.makedirs(OUT, exist_ok=True)

spec = importlib.util.spec_from_file_location("rx", W + "/TensorLBM/xval/"
    "XVAL-LBM-QUASI-STATIC-PC-LADDER-WITH-CHECKPOINT/run_xval.py")
rx = importlib.util.module_from_spec(spec)
exec(compile(open(spec.origin).read().split("def main(")[0], spec.origin, "exec"), rx.__dict__)

res = json.load(open(W + "/TensorLBM/xval/XVAL-LBM-QUASI-STATIC-PC-LADDER-WITH-CHECKPOINT/"
                     "artifacts/results.json"))
e0 = res["cases"]["E0_measurement"]


def trace(g: float, total: int = 600, every: int = 10):
    """u_max / phi range every `every` steps at a fixed body force g."""
    sim = rx.build()
    sim.gravity = (g, 0.0)
    out = {"g": g, "steps": [], "u_max": [], "phi_min": [], "finite": []}
    done = 0
    while done < total:
        cac.run(sim, every, sample_every=0)
        done += every
        phi = sim.f.sum(dim=0)
        ok = bool(torch.isfinite(sim.f).all().item() and torch.isfinite(sim.g).all().item())
        out["steps"].append(done)
        out["u_max"].append(float("nan") if not ok else P._u_rate(sim)[1])
        out["phi_min"].append(float("nan") if not ok else float(phi.min()))
        out["finite"].append(ok)
        if not ok:
            break
    out["first_nonfinite_step"] = next((s for s, f in zip(out["steps"], out["finite"]) if not f), None)
    return out


t0 = trace(0.0)                       # rung-0 situation: NO imposed pressure
t1 = trace(1.929012345679013e-05)     # rung-1 body force g = dP/(drho L)
meas = {}
for k in ("rungs4", "conv1e-2"):
    arm = res["cases"]["E4_sensitivity"][k]
    meas[k] = [(r["rung"], r["meas"].get("ok"), r["laplace"]["relerr"]) for r in arm["rows"]]

payload = {"e0": {k: e0[k] for k in ("kappa_band", "kappa_from_Req", "kappa_rel_err_vs_Req",
                                     "sigma_kappa", "dp_field", "laplace_relerr_field",
                                     "sigma_effective_from_dp")},
           "trace_g0": t0, "trace_g1": t1, "ladder_rows_ok": meas,
           "ratio_g1_g0_nan_step": [t1["first_nonfinite_step"], t0["first_nonfinite_step"]]}
json.dump(payload, open(OUT + "/probe_nan_onset.json", "w"), indent=1, default=str)

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
names = ["curvature\n$\\kappa$ vs $1/R_{eq}$", "Laplace\n$\\Delta p$ vs $\\sigma\\kappa$"]
vals = [abs(e0["kappa_rel_err_vs_Req"]) * 100, e0["laplace_relerr_field"] * 100]
bars = ax[0].bar(names, vals, color=["#2a7", "#c44"], width=0.55)
ax[0].axhline(15, ls="--", c="k", lw=1, label="owner gate 15%")
ax[0].axhline(3, ls=":", c="grey", lw=1, label="ecosystem target 3%")
ax[0].set_yscale("log")
ax[0].set_ylabel("relative error  [%]")
ax[0].set_title("E0 measurement chain (m18236 delivered numbers)")
for b, v in zip(bars, vals):
    ax[0].text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.2f}%", ha="center", fontsize=9)
ax[0].legend(fontsize=8)

for t, lab, st in ((t0, "g = 0 (rung 0, no imposed $\\Delta p$)", "-o"),
                   (t1, "g = $\\Delta p/(\\Delta\\rho L)$ (rung 1)", "-s")):
    ax[1].plot(t["steps"], t["u_max"], st, ms=3, label=lab)
    if t["first_nonfinite_step"]:
        ax[1].axvline(t["first_nonfinite_step"], ls=":", c="grey", lw=1)
        ax[1].annotate(f"NaN at step {t['first_nonfinite_step']}",
                       xy=(t["first_nonfinite_step"], 1e-2), fontsize=8,
                       xytext=(5, -25), textcoords="offset points")
ax[1].set_yscale("log")
ax[1].set_xlabel("LBM step")
ax[1].set_ylabel("$u_{max}$  [lu]")
ax[1].set_title("plug on the quantised taper: blow-up without any pressure ramp")
ax[1].legend(fontsize=8)
fig.suptitle("m18353 audit of m18236 — XVAL-LBM-QUASI-STATIC-PC-LADDER-WITH-CHECKPOINT", fontsize=10)
fig.tight_layout()
fig.savefig(OUT + "/fig_m18353_m18236_audit.png", dpi=140)
print(json.dumps({"g0_nan_step": t0["first_nonfinite_step"], "g1_nan_step": t1["first_nonfinite_step"],
                  "e0_kappa_pct": vals[0], "e0_laplace_pct": vals[1]}, indent=1))