#!/usr/bin/env python3
"""Third audit probe: which arm-pairing is physically comparable in the delivery's
case-1 kernel-equivalence test (c3 'OSS kernel agreement < 1e-4' = FAIL at 5.9e-3)."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
W = Path("/root/tp-agent/work/m17955/TensorLBM")
sys.path.insert(0, str(W / "src")); sys.path.insert(0, str(W / "xval" / "XVAL-LBM-LIST-BASED-SPARSE-LBM-PLUS-PERMEABILITY"))
import numpy as np, torch
from tensorlbm.porous import build_index_table, sparse_step, equilibrium_sparse, to_dense
import oss_ref as O
import run_xval as RX

ny, nx = 24, 32
solid = np.zeros((ny, nx), dtype=bool)
solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
solid[8:12, 10:12] = True
solid[14:16, 20:24] = True
st = torch.as_tensor(solid)
tab = build_index_table(st, periodic=(False, False))
g, tau, steps = 1e-4, 0.8, 100
f = equilibrium_sparse(tab)
fd = RX.dense_equilibrium(ny, nx)
stepper = O.OssSparseStepper(solid, tau=tau, force=(g, 0.0))
stepper.set_pdfs_from_tensorlbm(f.numpy())
post_post, post_pre, dense_post, dense_pre = [], [], [], []
t0 = time.time()
for _ in range(steps):
    f = sparse_step(f, tab, tau=tau, force_x=g)
    fd = RX.dense_step(fd, st, tau=tau, force_x=g)
    stepper.step(1)
    post = torch.as_tensor(stepper.tensorlbm_state())
    pre = None
    post_post.append(float((post - f).abs().max()))
    post_pre.append(float("nan"))
    dense_post.append(float((to_dense(f, tab)[:, ~st] - fd[:, ~st]).abs().max()))
out = {"steps": steps, "scale_max_pdf": float(f.abs().max()), "wall_seconds": time.time() - t0,
       "max_sparse_vs_oss_postcollision": max(post_post),
       "max_sparse_vs_oss_precollision": max(post_pre),
       "max_sparse_vs_dense": max(dense_post),
       "last5_postcollision": post_post[-5:], "first5_precollision": post_pre[:5]}
print(json.dumps(out, indent=1))
Path("/root/tp-agent/work/m18098/docs/audit_m18098/verify_case1_oss.json").write_text(json.dumps(out, indent=1))
