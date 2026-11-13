#!/usr/bin/env python3
"""m18098 board-audit: independent re-computation of the m17955 numbers.

Reads the audited delivery from /root/tp-agent/work/m17955/TensorLBM (read-only).
Recomputes Darcy permeability with ONE explicit convention (intrinsic = ν·<u>_fluid/g,
superficial = φ·intrinsic) and compares against the analytic channel k=H²/12,
instead of trusting the delivery's stored keys (which make_report.py mutates).
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path

W = Path("/root/tp-agent/work/m17955/TensorLBM")
sys.path.insert(0, str(W / "src"))
sys.path.insert(0, str(W / "xval" / "XVAL-LBM-LIST-BASED-SPARSE-LBM-PLUS-PERMEABILITY"))
import numpy as np, torch
from tensorlbm.porous import build_index_table, macroscopic_sparse, sparse_step
from tensorlbm.porous.fluid_node_table import macroscopic_with_forcing

def channel(H, tau=0.8, g=1e-5, nx=8, max_steps=20000, tol=1e-10):
    ny = H + 1
    solid = np.zeros((ny, nx), dtype=bool); solid[0, :] = True
    st = torch.as_tensor(solid)
    tab = build_index_table(st, periodic=(True, True))
    nu = (tau - 0.5) / 3.0
    from tensorlbm.porous import equilibrium_sparse
    f = equilibrium_sparse(tab); up = None; steps = 0
    for steps in range(1, max_steps + 1):
        f = sparse_step(f, tab, tau=tau, force_x=g)
        _, ux, _ = macroscopic_sparse(f, tab, g, 0.0)
        u = float(ux.mean())
        if up is not None and abs(u - up) <= tol * max(abs(u), 1e-30):
            break
        up = u
    _, ux_s, _ = macroscopic_sparse(f, tab, g, 0.0)
    u_int = float(ux_s.mean()); phi = tab.porosity
    k_int = darcy(u_int, nu, g); k_sup = phi * k_int
    return dict(H=H, tau=tau, steps=steps, phi=phi, nu=nu, u_int=u_int,
                k_intrinsic=k_int, k_superficial=k_sup, k_analytic=H * H / 12.0,
                err_int=(k_int - H * H / 12.0) / (H * H / 12.0))

def darcy(u, nu, g):
    return nu * u / g

# dense control with identical geometry, for cross-arm parity
def channel_dense(H, tau=0.8, g=1e-5, nx=8, steps=0):
    from run_xval import dense_step, dense_equilibrium  # delivery's own control arm
    ny = H + 1
    solid = np.zeros((ny, nx), dtype=bool); solid[0, :] = True
    st = torch.as_tensor(solid)
    f = dense_equilibrium(ny, nx)
    for _ in range(steps):
        f = dense_step(f, st, tau=tau, force_x=g)
    _, ux, _ = macroscopic_with_forcing(f, g, 0.0)
    return float(ux[~st].mean())

out = {"channel": [], "notes": []}
t0 = time.time()
for H in (8, 16, 32):
    r = channel(H)
    r["dense_u_int"] = channel_dense(H, steps=r["steps"])
    r["dense_k_int"] = darcy(r["dense_u_int"], r["nu"], 1e-5)
    r["err_dense_int"] = (r["dense_k_int"] - H * H / 12.0) / (H * H / 12.0)
    r["sparse_vs_dense_u_rel"] = (r["u_int"] - r["dense_u_int"]) / r["dense_u_int"]
    out["channel"].append(r); print(json.dumps(r, indent=1), flush=True)
out["wall_seconds"] = time.time() - t0
Path("/root/tp-agent/work/m18098/docs/audit_m18098/verify_m17955.json").write_text(json.dumps(out, indent=1))
print("done", out["wall_seconds"])
