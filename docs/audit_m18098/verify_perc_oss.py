#!/usr/bin/env python3
"""Second audit probe: (a) percolation sparse-vs-dense under ONE Darcy convention;
(b) OSS (lbmpy/pystencils) kernel arm vs the sparse arm on the delivery's case-1 box."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
W = Path("/root/tp-agent/work/m17955/TensorLBM")
sys.path.insert(0, str(W / "src")); sys.path.insert(0, str(W / "xval" / "XVAL-LBM-LIST-BASED-SPARSE-LBM-PLUS-PERMEABILITY"))
import numpy as np, torch
from tensorlbm.porous import build_index_table, macroscopic_sparse, sparse_step, equilibrium_sparse, to_dense
from tensorlbm.porous.fluid_node_table import macroscopic_with_forcing
import run_xval as RX

out = {}
# (a) percolation, same geometry/params as case3 seed 11
g, tau, steps = 2e-5, 0.8, 6000
solid = RX.random_matrix(24, 24, phi=0.70, seed=11)
st = torch.as_tensor(solid)
tab = build_index_table(st, periodic=(True, True))
f = equilibrium_sparse(tab)
for _ in range(steps):
    f = sparse_step(f, tab, tau=tau, force_x=g)
_, ux_s, _ = macroscopic_sparse(f, tab, g, 0.0)
fd = RX.dense_equilibrium(24, 24)
for _ in range(steps):
    fd = RX.dense_step(fd, st, tau=tau, force_x=g)
_, ux_d, _ = macroscopic_with_forcing(fd, g, 0.0)
nu = (tau - 0.5) / 3
phi = tab.porosity
u_s, u_d = float(ux_s.mean()), float(ux_d[~st].mean())
out["percolation"] = {
    "phi": phi, "u_fluid_sparse": u_s, "u_fluid_dense": u_d,
    "k_int_sparse": nu * u_s / g, "k_int_dense": nu * u_d / g,
    "k_super_sparse": phi * nu * u_s / g, "k_super_dense": phi * nu * u_d / g,
    "rel_diff_intrinsic": abs(u_s - u_d) / abs(u_d),
    "max_abs_pdf_diff": float((to_dense(f, tab)[:, ~st] - fd[:, ~st]).abs().max()),
}
print(json.dumps(out["percolation"], indent=1), flush=True)

# (b) OSS arm on case-1 geometry, independent invocation
try:
    import oss_ref
    box = RX.case1_geometry() if hasattr(RX, "case1_geometry") else None
    out["oss_import"] = "ok"
except Exception as e:
    out["oss_import"] = f"FAILED {type(e).__name__}: {e}"
print(out["oss_import"])
Path("/root/tp-agent/work/m18098/docs/audit_m18098/verify_perc_oss.json").write_text(json.dumps(out, indent=1))
