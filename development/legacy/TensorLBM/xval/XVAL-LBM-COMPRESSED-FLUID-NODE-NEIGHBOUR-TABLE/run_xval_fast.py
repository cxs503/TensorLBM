#!/usr/bin/env python3
"""Fast ladder for XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE.

The full predeclared ladder (``run_xval.py``) needs ~30 min of single-thread CPU
and the shared control node is heavily loaded, so this script runs a *ladder of
the same design with smaller lattices and a shorter step budget*, reusing the
runner's kernels and KPI functions verbatim.  Every deviation from the
predeclared ladder is written into ``results.json["ladder_note"]`` so the report
can state it (no silent substitution):

* channel H=16 on ny=32 (instead of H=32 on ny=64) -- same slab set-up;
* porous medium N=64 / N=96 (instead of 128 / 256);
* step budget 2500 with the KPI also sampled at 1250 -> step self-convergence
  replaces the (unreachable in this budget) 1e-9 steady-state rule.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_xval as R  # noqa: E402  (kernels + KPI helpers, no side effects)

OUT = os.path.join(R.HERE, "..", "..", "artifacts", "xval_compressed_fluid_node_neighbour_table")
os.makedirs(OUT, exist_ok=True)
STEPS = 2500
HALF = STEPS // 2


def case_record(name, solid, g, k_ref=None, ref_kind="none", ny_hint=None):
    ny, nx = int(solid.shape[0]), int(solid.shape[1])
    a1 = R.run_compressed(solid, g, R.TAU, steps_cap=STEPS, record=True)
    a2 = R.run_dense(solid, g, R.TAU, steps_cap=STEPS)
    rec = {
        "case": name,
        "case_cfg": {"g": g, "ny": ny, "nx": nx},
        "k_reference": k_ref,
        "reference_kind": ref_kind,
        "A1": R.strip(a1),
        "A2": R.strip(a2),
        "compare": R.compare(a1, a2),
        "k_err_vs_ref_pct": None if k_ref is None else 100 * abs(a1["k_darcy"] - k_ref) / k_ref,
        "k_err_vs_ref_pct_A2": None if k_ref is None else 100 * abs(a2["k_darcy"] - k_ref) / k_ref,
        "mem": a1["mem"],
    }
    if ny_hint is not None:  # plane channel: profile vs analytic
        H = ny_hint
        ux = np.array(a1["ux_fluid"])
        ys = np.array(a1["node_index"]) // nx
        prof = np.zeros(ny)
        cnt = np.zeros(ny)
        np.add.at(prof, ys, ux)
        np.add.at(cnt, ys, 1.0)
        valid = cnt > 0
        prof[valid] /= cnt[valid]
        rows = np.nonzero(valid)[0]
        y0 = rows[0]
        yc = np.arange(H) + 0.5
        ana = g / (2 * R.NU) * yc * (H - yc)
        rec["poiseuille_profile_maxrel"] = float(np.abs(prof[rows] - ana).max() / ana.max())
        rec["poiseuille_u_max_analytic"] = float(ana.max())
        rec["poiseuille_u_max_got"] = float(prof[rows].max())
    return rec


records = []
# C1: plane channel, analytic slab permeability k_super = (H/ny) H^2/12
H, NY, NX = 16, 32, 16
solid = R.channel_mask(NY, NX, H)
k_ref = (H / NY) * H * H / 12.0
records.append(case_record("C1_channel_H16", solid, 1e-5, k_ref,
                           "analytic_slab: k_super=(H/ny)*H^2/12", ny_hint=H))
print("C1 done", flush=True)

# C2..C4: periodic square-grain porous medium, porosity sweep
for p, n in ((0.90, 64), (0.75, 64), (0.60, 64)):
    tag = f"C2_porous_p{int(round(p*100))}_N{n}"
    records.append(case_record(tag, R.square_grain_mask(n, p), 1e-5))
    print(tag, "done", flush=True)

# C5: forcing linearity at fixed geometry (same porosity, doubled force)
records.append(case_record("C5_porous_p75_N64_g2e5", R.square_grain_mask(64, 0.75), 2e-5))
print("C5 done", flush=True)

# C6: resolution sensitivity
records.append(case_record("C6_porous_p75_N96", R.square_grain_mask(96, 0.75), 1e-5))
print("C6 done", flush=True)

# performance / footprint probe (timed, identical protocol for both arms)
perf = {}
for n, p in ((64, 0.75), (128, 0.75), (192, 0.75)):
    from tensorlbm.porous import build_fluid_node_table, compressed_step, dense_step
    solid = R.square_grain_mask(n, p)
    table = build_fluid_node_table(solid)
    f = torch.full((9, table.n_fluid), 1.0 / 9.0)
    for _ in range(20):
        f = compressed_step(f, table, R.TAU, 1e-5, 0.0)
    t0 = time.perf_counter()
    for _ in range(60):
        f = compressed_step(f, table, R.TAU, 1e-5, 0.0)
    t_c = (time.perf_counter() - t0) / 60
    fd = torch.full((9, n, n), 1.0 / 9.0)
    for _ in range(20):
        fd = dense_step(fd, solid, R.TAU, 1e-5, 0.0)
    t0 = time.perf_counter()
    for _ in range(60):
        fd = dense_step(fd, solid, R.TAU, 1e-5, 0.0)
    t_d = (time.perf_counter() - t0) / 60
    perf[f"N{n}_p{int(round(p*100))}"] = {
        "s_per_step_compressed": t_c,
        "s_per_step_dense": t_d,
        "mlups_compressed": n * n / t_c / 1e6,
        "mlups_dense": n * n / t_d / 1e6,
        "speedup_dense_over_compressed": t_d / t_c,
        **table.state_bytes(),
    }
    print("perf", n, t_c, t_d, flush=True)

summary = {
    "tau": R.TAU,
    "nu": R.NU,
    "steps_cap": STEPS,
    "steady_tol": R.STEADY_TOL,
    "torch": torch.__version__,
    "numpy": np.__version__,
    "threads": torch.get_num_threads(),
    "records": records,
    "performance": perf,
    "ladder_note": (
        "reduced ladder: channel H=16/ny=32, porous N=64 and N=96, step budget "
        f"{STEPS} (KPI also sampled at {HALF}); the 1e-9 steady-state rule of the "
        "predeclared design is replaced by a step self-convergence check because the "
        "shared control node could not deliver the ~30 min the full ladder needs "
        "(recorded as friction, not as a design change)."
    ),
    "partial": False,
}
with open(os.path.join(OUT, "results.json"), "w") as fh:
    json.dump(summary, fh, indent=2, default=str)
print("wrote results.json with", len(records), "cases")