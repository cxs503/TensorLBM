#!/usr/bin/env python3
"""P6: single-line kernel swap inside the *delivered* benchmark loop.

The delivered loop (delivered_run_sha_94cb63e6ff82.py :: run_case) collides on all
nodes, streams (pull, periodic), then overwrites the two wall rows.  We keep every
other line identical and only swap the wall kernel:

  delivered   -> D.apply_abb_inamuro_wetnode(g_pre, g, wall, C_w)   (as shipped)
  equilibrium -> g[q](wall) = w_q * C_w                            (classic wet node)
  oss_abb     -> g[2](bottom wall) = -g_pre[4](first fluid row) + 2 w_2 C_b
                 g[4](top wall)    = -g_pre[2](last fluid row)  + 2 w_4 C_t
                 (lbm-principles-practice/code chapter8/film_antibb.cpp:152-205,
                  written at the wall node from the adjacent fluid node's
                  post-collision population)

Float32 throughout, exactly like the delivered script.
"""
from __future__ import annotations

import json
from pathlib import Path

import importlib.util

import numpy as np
import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("delivered", HERE / "delivered_run_sha_94cb63e6ff82.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)

W = torch.tensor(D.W_D2Q5, dtype=torch.float32)


def run_swapped(H, kernel, tau=0.8, U0=0.05, cb=0.0, ct=1.0, steps=1000):
    ny, nx = H + 2, max(H, 8)
    wall = torch.zeros((ny, nx), dtype=torch.bool)
    wall[0, :] = True
    wall[-1, :] = True
    bot = torch.zeros((ny, nx), dtype=torch.bool); bot[0, :] = True
    top = torch.zeros((ny, nx), dtype=torch.bool); top[-1, :] = True

    ux = torch.full((ny, nx), U0); uy = torch.zeros((ny, nx))
    ux[wall] = 0.0
    yy = torch.arange(ny, dtype=torch.float32)
    c_init = (cb + (ct - cb) * (yy - 0.5) / H).view(ny, 1).expand(ny, nx).clone()
    c_init[0, :] = cb
    c_init[-1, :] = ct
    g = D.equilibrium_scalar_d2q5(c_init, ux, uy)

    for _ in range(steps):
        c = D.scalar_field(g)
        g = D.collide_scalar_bgk(g, c, ux, uy, tau)
        g_pre = g.clone()
        if kernel == "oss_abb":
            # OSS order: collide -> reconstruct the wall populations (post-collision
            # array f2 in film_antibb.cpp) -> stream.  Pull streaming pushes
            # g_pre[2] of the bottom wall row into the first fluid row.
            g_pre[2, 0, :] = -g_pre[4, 1, :] + 2.0 * D.W_D2Q5[2] * cb
            g_pre[4, -1, :] = -g_pre[2, -2, :] + 2.0 * D.W_D2Q5[4] * ct
        g = D.stream_scalar(g_pre)
        if kernel == "delivered":
            g = D.apply_abb_inamuro_wetnode(g_pre, g, bot, cb)
            g = D.apply_abb_inamuro_wetnode(g_pre, g, top, ct)
        elif kernel == "equilibrium":
            g[:, 0, :] = (W * cb).view(5, 1)
            g[:, -1, :] = (W * ct).view(5, 1)
        elif kernel == "oss_abb":
            pass  # applied on g_pre before streaming, above
        else:
            raise ValueError(kernel)

    c = D.scalar_field(g)
    return c[1:ny - 1, nx // 2].numpy().astype(np.float64)


def main():
    out = {"kernel_swap_in_delivered_loop": {}}
    prof = {}
    for H in (20, 40, 80):
        Y = np.arange(1, H + 1, dtype=np.float64)
        ref_half = (Y - 0.5) / H          # half-way ABB reference (what the report used)
        ref_node = Y / (H + 1)            # node Dirichlet reference (wet node)
        row = {}
        for k in ("delivered", "equilibrium", "oss_abb"):
            p = run_swapped(H, k, steps=1000)
            prof[f"{H}_{k}"] = p.tolist()
            row[k] = {
                "l2_vs_halfway_ref_pct": 100 * float(np.linalg.norm(p - ref_half) / np.linalg.norm(ref_half)),
                "l2_vs_noderef_pct": 100 * float(np.linalg.norm(p - ref_node) / np.linalg.norm(ref_node)),
                "max_abs_vs_halfway_ref": float(np.max(np.abs(p - ref_half))),
                "wall_value_C0": float(p[0]), "wall_value_Ctop": float(p[-1]),
            }
        row["max_abs_diff_delivered_vs_equilibrium"] = float(
            np.max(np.abs(np.array(prof[f"{H}_delivered"]) - np.array(prof[f"{H}_equilibrium"]))))
        row["max_abs_diff_delivered_vs_oss_abb"] = float(
            np.max(np.abs(np.array(prof[f"{H}_delivered"]) - np.array(prof[f"{H}_oss_abb"]))))
        out["kernel_swap_in_delivered_loop"][str(H)] = row
    (HERE / "audit_p6_kernel_swap.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()