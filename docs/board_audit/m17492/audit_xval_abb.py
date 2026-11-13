#!/usr/bin/env python3
"""m17492 board audit probes for XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE (m17450).

Independent re-derivation of the deliverables of m17450:
  (P1) byte-faithful replay of the delivered benchmark (vendored run.py)  -> is the
       committed got-number reproducible?
  (P2) which population feeds the wall reconstruction?  Compare the delivered
       'inamuro_wetnode' kernel against (a) an equilibrium wet node w_i*C_w and
       (b) the OSS ABB of lbm-principles-practice/code chapter8/film_antibb.cpp:152-205
       (g_in(x_w) = -g_out(x_f) + 2 w_i C_w, sampled at the *adjacent fluid node*).
  (P3) reference alignment: the delivered report compares the wet-node profile with
       C=(Y-0.5)/H although the Dirichlet values sit *on* the wall nodes (gap H+1).
       Recompute every error against both references, and check whether the claimed
       grid-convergence order is a real discretisation order or a reference artefact.

All runs are plain numpy/torch CPU, <= a few seconds each, deterministic.
Output: audit_xval_abb_results.json
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
DELIVERED = HERE / "delivered_run_sha_94cb63e6ff82.py"
OUT = HERE / "audit_xval_abb_results.json"

spec = importlib.util.spec_from_file_location("delivered", DELIVERED)
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)  # type: ignore[attr-defined]

W = D.W_D2Q5
CX, CY = D.CX_D2Q5, D.CY_D2Q5
DTYPE = torch.float64  # audit runs in double precision to expose reference offsets


def eq(c, ux, uy):
    w = torch.tensor(W, dtype=DTYPE).view(5, 1, 1)
    cx = torch.tensor(CX, dtype=DTYPE).view(5, 1, 1)
    cy = torch.tensor(CY, dtype=DTYPE).view(5, 1, 1)
    return w * c * (1.0 + 3.0 * (cx * ux + cy * uy))


def solve(H, tau=0.8, U0=0.05, cb=0.0, ct=1.0, variant="delivered_inamuro", steps=1000):
    """Clean D2Q5 scalar Couette solver, one Dirichlet-wall kernel per variant.

    Convention: wall nodes are lattice rows y=0 (C=cb) and y=H+1 (C=ct); fluid rows
    are y=1..H.  Collision is applied on fluid nodes only; the wall rows are pure
    boundary nodes (their populations are reconstructed, never collided), which is
    the wet-node convention used by the OSS reference.
    """
    ny, nx = H + 2, max(H, 8)
    wall = torch.zeros((ny, nx), dtype=torch.bool)
    wall[0, :] = True
    wall[-1, :] = True
    fluid = ~wall

    ux = torch.full((ny, nx), U0, dtype=DTYPE)
    uy = torch.zeros((ny, nx), dtype=DTYPE)
    ux[wall] = 0.0

    yy = torch.arange(ny, dtype=DTYPE)
    c_init = (cb + (ct - cb) * (yy - 0.5) / H).view(ny, 1).expand(ny, nx).clone()
    c_init[0, :] = cb
    c_init[-1, :] = ct
    g = eq(c_init, ux, uy)

    w = torch.tensor(W, dtype=DTYPE).view(5, 1, 1)
    n_fluid = int(fluid.sum())

    for _ in range(steps):
        c = g.sum(dim=0)
        geq = eq(c, ux, uy)
        g_post = torch.where(fluid.unsqueeze(0), g - (g - geq) / tau, g)

        if variant == "equilibrium_wetnode":
            # classic equilibrium wet node: wall populations = w_i * C_w
            g_post[:, 0, :] = (w[:, 0, 0] * cb).view(5, 1)
            g_post[:, -1, :] = (w[:, 0, 0] * ct).view(5, 1)
        elif variant == "abb_oss":
            # OSS ABB (film_antibb.cpp:152-205): population sent into the fluid is
            # rebuilt from the *adjacent fluid node's* post-collision counterpart.
            g_post[2, 0, :] = -g_post[4, 1, :] + 2.0 * W[2] * cb
            g_post[4, -1, :] = -g_post[2, -2, :] + 2.0 * W[4] * ct
        elif variant == "abb_oss_wallself":
            # ABB, but sampling the wall node's own post-collision counterpart
            # (this is what the delivered kernel effectively does)
            g_post[2, 0, :] = -g_post[4, 0, :] + 2.0 * W[2] * cb
            g_post[4, -1, :] = -g_post[2, -1, :] + 2.0 * W[4] * ct
        elif variant == "delivered_inamuro":
            pass  # handled by the delivered path below
        else:
            raise ValueError(variant)

        g = D.stream_scalar(g_post.to(torch.float32)).to(DTYPE)

        if variant == "delivered_inamuro":
            # replay of delivered loop tail: g_pre == post-collision, g == post-stream
            g = D.apply_abb_inamuro_wetnode(g_post.to(torch.float32), g.to(torch.float32),
                                            wall[0:1].expand(ny, nx) & (torch.arange(ny).view(ny, 1) == 0),
                                            cb).to(DTYPE)
            g = D.apply_abb_inamuro_wetnode(g_post.to(torch.float32), g.to(torch.float32),
                                            wall & (torch.arange(ny).view(ny, 1) == ny - 1),
                                            ct).to(DTYPE)

    c = g.sum(dim=0)
    # wall-node concentration actually realised by the kernel (Dirichlet audit)
    c_wall = [float(c[0, nx // 2]), float(c[-1, nx // 2])]
    return c[1:ny - 1, nx // 2].cpu().numpy(), c_wall, n_fluid


def refs(H, cb=0.0, ct=1.0):
    Y = np.arange(1, H + 1, dtype=np.float64)          # distance from bottom wall node
    r_report = cb + (ct - cb) * (Y - 0.5) / H           # what the report used
    r_wetnode = cb + (ct - cb) * Y / (H + 1)            # walls sit on nodes 0 and H+1
    return {"report_halfcell": r_report, "wetnode_nodegap": r_wetnode}


def l2(num, ana):
    return float(np.linalg.norm(num - ana) / np.linalg.norm(ana))


def maxabs(num, ana):
    return float(np.max(np.abs(num - ana)))


def order(e_h, e_2h):
    return float(math.log(e_h / e_2h) / math.log(2.0))


def main():
    res = {"anchor": "XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE",
           "audited_mission": 17450, "audit_mission": 17492,
           "delivered_run_sha256": hashlib.sha256(DELIVERED.read_bytes()).hexdigest()}

    # ---------------- P1: replay delivered benchmark ------------------------
    repro = []
    for H in (20, 40):
        for variant in ("standard", "inamuro_wetnode"):
            committed = json.loads(
                (Path("/root/tp-agent/work/m17450/benchmarks/xval_abb_inamuro_wetnode/out")
                 / f"case_H{H}_{variant}.json").read_text())
            fresh = D.run_case(H, 0.8, 0.05, 0.0, 1.0, variant, 1000, "/tmp/_replay.json")
            repro.append({
                "H": H, "variant": variant,
                "committed_l2": committed["l2_rel_err"],
                "replayed_l2": fresh["l2_rel_err"],
                "abs_diff_l2": abs(committed["l2_rel_err"] - fresh["l2_rel_err"]),
                "committed_max_abs": committed["max_abs_err"],
                "replayed_max_abs": fresh["max_abs_err"],
                "profile_max_diff": float(np.max(np.abs(
                    np.array(committed["c_profile_num"]) - np.array(fresh["c_profile_num"])))),
            })
    res["P1_replay"] = repro

    # ---------------- P2 / P3: kernel identity + reference alignment -------
    variants = ["delivered_inamuro", "equilibrium_wetnode", "abb_oss", "abb_oss_wallself"]
    table = {}
    profiles = {}
    for H in (20, 40, 80):
        rr = refs(H)
        table[H] = {}
        for v in variants:
            num, c_wall, _ = solve(H, variant=v, steps=1000)
            profiles[(H, v)] = num.tolist()
            row = {"c_wall_realised": [round(c_wall[0], 8), round(c_wall[1], 8)]}
            for rname, r in rr.items():
                row[f"l2_vs_{rname}"] = l2(num, r)
                row[f"maxabs_vs_{rname}"] = maxabs(num, r)
            # effective gap from a linear fit of the numerical profile
            Y = np.arange(1, H + 1, dtype=np.float64)
            A = np.vstack([Y, np.ones_like(Y)]).T
            slope, intercept = np.linalg.lstsq(A, num, rcond=None)[0]
            row["linear_fit_slope"] = float(slope)
            row["implied_gap"] = float(1.0 / slope) if abs(slope) > 1e-12 else None
            row["linear_fit_residual_max"] = float(np.max(np.abs(num - (A @ [slope, intercept]))))
            table[H][v] = row
    res["P2P3_variants"] = table

    # kernel identity: delivered kernel vs equilibrium wet node (same steady state?)
    ident = {}
    for H in (20, 40, 80):
        a = np.array(profiles[(H, "delivered_inamuro")])
        b = np.array(profiles[(H, "equilibrium_wetnode")])
        ident[str(H)] = {"max_abs_profile_diff": float(np.max(np.abs(a - b))),
                         "l2_rel_diff": float(np.linalg.norm(a - b) / np.linalg.norm(b))}
    res["P2_kernel_identity_delivered_vs_equilibrium"] = ident

    # grid convergence, honest reference vs report reference, per kernel
    conv = {}
    for v in variants:
        e1 = table[20][v]["l2_vs_wetnode_nodegap"]
        e2 = table[40][v]["l2_vs_wetnode_nodegap"]
        e3 = table[80][v]["l2_vs_wetnode_nodegap"]
        f1 = table[20][v]["l2_vs_report_halfcell"]
        f2 = table[40][v]["l2_vs_report_halfcell"]
        conv[v] = {
            "l2_wetnode_ref": [e1, e2, e3],
            "order_wetnode_ref": [order(e1, e2), order(e2, e3)],
            "l2_report_ref": [f1, f2],
            "order_report_ref": order(f1, f2),
        }
    res["P4_convergence"] = conv

    # steady state reached in 1000 steps?  compare 1000 vs 3000 steps (abb_oss)
    ss = {}
    for H in (20, 40):
        a, _, _ = solve(H, variant="abb_oss", steps=1000)
        b, _, _ = solve(H, variant="abb_oss", steps=3000)
        ss[str(H)] = {"max_abs_diff_1000_vs_3000": float(np.max(np.abs(a - b)))}
    res["P5_steady_state"] = ss

    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("P1_replay", "P2_kernel_identity_delivered_vs_equilibrium",
                                          "P5_steady_state")}, indent=2))
    print("\n== per-variant errors ==")
    for H in (20, 40, 80):
        for v in variants:
            r = table[H][v]
            print(f"H={H:3d} {v:22s} l2(report_ref)={r['l2_vs_report_halfcell']*100:7.3f}%  "
                  f"l2(nodegap_ref)={r['l2_vs_wetnode_nodegap']*100:7.3f}%  "
                  f"maxabs={r['maxabs_vs_wetnode_nodegap']:.4f}  "
                  f"gap_fit={r['implied_gap']:.4f}  C_wall={r['c_wall_realised']}")
    print("\n== orders ==")
    for v in variants:
        print(v, "wetnode_ref:", [round(x, 3) for x in conv[v]["order_wetnode_ref"]],
              " report_ref:", round(conv[v]["order_report_ref"], 3))


if __name__ == "__main__":
    sys.exit(main())