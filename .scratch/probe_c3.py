"""Focused probe: resolve the Kn=0.1 DVM/DSMC discrepancy (C3 worst point)."""
import json
import time

from tensorlbm.rarefied import knudsen as K
from tensorlbm.rarefied.bgk_slab import LinearizedBGKSlab
from tensorlbm.rarefied.dsmc_slab import run_dsmc_slab
from tensorlbm.rarefied.ladder import richardson

out = {"dvm": [], "dsmc": []}
g = 0.05
kn = 0.1
for ny in (96, 192, 384):
    t0 = time.time()
    sol = LinearizedBGKSlab(kn, sigma=1.0, ny=ny, nv=32).solve_poiseuille(g)
    out["dvm"].append({"ny": ny, "dy_over_lambda": 1.0 / (ny * kn), "q": sol.q,
                       "sec": round(time.time() - t0, 1)})
    print("dvm", out["dvm"][-1], flush=True)
q2 = richardson(out["dvm"][-2]["q"], out["dvm"][-1]["q"], out["dvm"][-2]["ny"], out["dvm"][-1]["ny"])
q3 = richardson(out["dvm"][0]["q"], out["dvm"][2]["q"], out["dvm"][0]["ny"], out["dvm"][2]["ny"])
out["dvm_richardson_fine"] = q2
out["dvm_richardson_span"] = q3
print("richardson fine", q2, "span", q3, flush=True)

for seed, ncells, steps in ((101, 80, 60000), (202, 80, 60000)):
    t0 = time.time()
    r = run_dsmc_slab(kn, sigma=1.0, g=g, ncells=ncells, ppc=60, steps=steps, seed=seed)
    out["dsmc"].append({"seed": seed, "ncells": ncells, "steps": steps, "q": r.q,
                        "q_err": r.q_err, "cells_per_mfp": r.extra["cells_per_mfp"],
                        "sec": round(time.time() - t0, 1)})
    print("dsmc", out["dsmc"][-1], flush=True)
out["q_ns_slip"] = K.ns_slip_flow_rate(1.0, kn * K.tau_from_kn(1.0) / K.tau_from_kn(1.0) * 0.7978845608, g,
                                      K.maxwell_slip_length(1.0, kn))
out["q_ns"] = K.ns_slip_flow_rate(1.0, kn * 0.7978845608, g, 0.0)
with open("/tmp/probe_c3.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("done", out["q_ns"], out["q_ns_slip"])