#!/usr/bin/env python3
"""CPU audit: analytic-mask vs inside_fn-mask BFL geometry + q impact on force.

Replicates the integrated-run geometry (nx=96, ny=64, nz=64, R=6, bl=3,
D3Q27) on CPU and compares:

  1. leaf sets / BFL masks / direction distributions / q distributions
  2. q-field consistency between the two paths
  3. uniform-equilibrium initial BFL impulse under q ablations
  4. per-branch (lin/quad) reconstruction weights and their amplification

No GPU, static + CPU numerics only.
"""
from __future__ import annotations

import json
import math
import sys

import torch

from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.geometry_adapters import sphere_inside_fn
from tensorlbm.octree_boundary.qfield import compute_q_sphere_at_points
from tensorlbm.octree_boundary.bfl import bfl_apply_gather, leaf_force_weights
from tensorlbm.octree_boundary.stepping import build_ghost_plan
from tensorlbm.d3q27 import equilibrium27, C, OPPOSITE

SHAPE = (64, 64, 96)      # (nz, ny, nx)  -- integrated run
CENTER = (48.0, 32.0, 32.0)
RADIUS = 6.0
BL = 3.0                  # max(2, round(R/2))
U_IN = 0.06
Q = 27


def q_stats(mask, q_field, name):
    idx = torch.nonzero(mask, as_tuple=False)
    qq = q_field[mask].to(torch.float64)
    n = qq.numel()
    if n == 0:
        return {name: {"n": 0}}
    return {
        name: {
            "n": int(n),
            "q_min": float(qq.min().item()),
            "q_max": float(qq.max().item()),
            "q_mean": float(qq.mean().item()),
            "n_q_eq_0.5": int((qq == 0.5).sum().item()),
            "frac_q_eq_0.5": float((qq == 0.5).float().mean().item()),
            "n_q_lt_0.5": int((qq < 0.5).sum().item()),
            "n_q_gt_0.5": int((qq > 0.5).sum().item()),
            "n_q_le_0.01": int((qq <= 0.01).sum().item()),
            "n_q_ge_0.99": int((qq >= 0.99).sum().item()),
            "q_hist": [int(((qq >= lo) & (qq < hi)).sum().item())
                       for lo, hi in
                       [(0.0, 0.05), (0.05, 0.15), (0.15, 0.25), (0.25, 0.35),
                        (0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.75),
                        (0.75, 0.85), (0.85, 1.01)]],
        }
    }


def mask_consistency(grid, name):
    """For every masked link, does the analytic ray-sphere boundary agree?"""
    dx = 2.0 ** (-grid.leaf_level.to(torch.float64))
    mask_a, q_a = compute_q_sphere_at_points(
        grid.leaf_center, dx, CENTER, RADIUS,
        device=grid.leaf_center.device, lattice="D3Q27",
    )
    m = grid.bfl_mask
    agree = int((m & mask_a).sum().item())
    only_nt = int((m & ~mask_a).sum().item())     # mask True, analytic False
    only_an = int((~m & mask_a).sum().item())     # mask False, analytic True
    # q values on the nt-only links (analytic boundary False -> default 0.5?)
    q_on_only_nt = grid.q_field[m & ~mask_a].to(torch.float64)
    return {
        f"{name}_mask_vs_analytic": {
            "mask_links": int(m.sum().item()),
            "analytic_links": int(mask_a.sum().item()),
            "agree": int(agree),
            "only_nt_SOLID": int(only_nt),
            "only_analytic": int(only_an),
            "q_on_only_nt_min": float(q_on_only_nt.min().item())
            if q_on_only_nt.numel() else None,
            "q_on_only_nt_max": float(q_on_only_nt.max().item())
            if q_on_only_nt.numel() else None,
            "q_on_only_nt_eq0.5_frac": float(
                (q_on_only_nt == 0.5).float().mean().item()
            ) if q_on_only_nt.numel() else None,
        }
    }


def build(inside_fn, d_max):
    dev = torch.device("cpu")
    g = build_octree_shell(
        SHAPE, center=CENTER, radius=RADIUS,
        bl_thickness_cells=BL, d_max=d_max, lattice="D3Q27", device=dev,
        inside_fn=inside_fn,
    )
    # uniform equilibrium populations
    rho = torch.ones(g.n_leaf)
    ux = torch.full((g.n_leaf,), U_IN)
    uy = torch.zeros(g.n_leaf)
    uz = torch.zeros(g.n_leaf)
    g.f_leaf = equilibrium27(rho, ux, uy, uz).view(Q, -1)
    return g


def uniform_ghost_vals(g):
    """Uniform equilibrium ghost values (exact for a uniform field)."""
    plan = build_ghost_plan(g, SHAPE)
    n_g = plan.n_ghost
    rho = torch.ones(n_g)
    ux = torch.full((n_g,), U_IN)
    uy = torch.zeros(n_g)
    uz = torch.zeros(n_g)
    return plan, equilibrium27(rho, ux, uy, uz).view(Q, -1)


def bfl_force(g, q_override=None, mask_override=None, q_min=None):
    """One substep BFL MEM force (leaf units, force_weights applied)."""
    plan, gvals = uniform_ghost_vals(g)
    g2 = g
    if q_override is not None or mask_override is not None:
        import copy
        g2 = copy.copy(g)
        g2.q_field = (q_override if q_override is not None
                      else g.q_field.clone())
        g2.bfl_mask = (mask_override if mask_override is not None
                       else g.bfl_mask.clone())
        # ghost plan references octree internals (leaf enums are the same)
    f = g.f_leaf.clone()
    out, force = bfl_apply_gather(
        g2, f, g.f_leaf.clone(),
        ghost_plan=plan, ghost_vals=gvals,
        force_weights=leaf_force_weights(g), return_force=True,
        q_min=q_min,
    )
    return force


def direction_report(g, name):
    m = g.bfl_mask
    c = C.to(torch.float64)
    per_dir = {f"d{d}({int(c[d,0])},{int(c[d,1])},{int(c[d,2])})": int(m[d].sum().item())
               for d in range(1, Q)}
    csum = c[1:].unsqueeze(1) * m[1:].unsqueeze(-1).to(torch.float64)  # (Q-1, n, 3)
    net_c = csum.sum(dim=(0, 1))
    # link-centre symmetry check: sum of leaf centres of masked links
    idx = torch.nonzero(m, as_tuple=False)  # (d, i)
    leafs = idx[:, 1]
    cen_sum = g.leaf_center[leafs].to(torch.float64).sum(dim=0)
    # net x-direction mask imbalance (drag-relevant)
    cx_sign = torch.sign(c[:, 0])
    net_x_links = int((m[1:] * cx_sign[1:].unsqueeze(1)).sum().item())
    return {
        f"{name}_links": int(m.sum().item()),
        f"{name}_per_dir": per_dir,
        f"{name}_net_c_vec": net_c.tolist(),
        f"{name}_leaf_center_sum": cen_sum.tolist(),
        f"{name}_net_x_links": net_x_links,
        f"{name}_n_leaf": int(g.n_leaf),
    }


def main():
    out = {}
    dev = torch.device("cpu")
    for d_max in (1, 2):
        tag = f"dmax{d_max}"
        A = build(None, d_max)
        B = build(sphere_inside_fn(CENTER, RADIUS), d_max)
        out[f"stats_{tag}"] = {
            "A_analytic": {"n_leaf": int(A.n_leaf),
                           "solid_cells": int(A._solid.sum().item()),
                           "shell_cells": int(A._shell_mask.sum().item())},
            "B_inside_fn": {"n_leaf": int(B.n_leaf),
                            "solid_cells": int(B._solid.sum().item()),
                            "shell_cells": int(B._shell_mask.sum().item())},
        }
        # leaf sets: which leaves are in A but not B (by Morton code)
        mA = set(A.leaf_morton.tolist())
        mB = set(B.leaf_morton.tolist())
        out[f"leafsets_{tag}"] = {
            "A_only": len(mA - mB), "B_only": len(mB - mA),
            "common": len(mA & mB),
        }
        out.update(direction_report(A, f"A_{tag}"))
        out.update(direction_report(B, f"B_{tag}"))
        out.update(q_stats(A.bfl_mask, A.q_field, f"A_{tag}_q"))
        out.update(q_stats(B.bfl_mask, B.q_field, f"B_{tag}_q"))
        out.update(mask_consistency(A, f"A_{tag}"))
        out.update(mask_consistency(B, f"B_{tag}"))

        # ---- uniform-equilibrium BFL impulse (per substep, weights on) ----
        fA = bfl_force(A)
        fB = bfl_force(B)
        # q=0.5 ablation on B (pure reflection everywhere)
        q05 = torch.full_like(B.q_field, 0.5)
        fB_q05 = bfl_force(B, q_override=q05)
        # B leaves with the *analytic* mask (same q convention as A)
        dx = 2.0 ** (-B.leaf_level.to(torch.float64))
        maskA_on_B, _ = compute_q_sphere_at_points(
            B.leaf_center, dx, CENTER, RADIUS, device=dev, lattice="D3Q27",
        )
        fB_maskA = bfl_force(B, mask_override=maskA_on_B)
        # B with analytic-consistent q on its own mask links (q from ray test
        # only where the analytic boundary holds; 0.5 elsewhere -> as-is)
        fB_qmin = bfl_force(B, q_min=0.25)
        out[f"force_{tag}"] = {
            "F_A_analytic": fA.tolist(),
            "F_B_inside": fB.tolist(),
            "F_B_q05": fB_q05.tolist(),
            "F_B_maskA": fB_maskA.tolist(),
            "F_B_qmin025": fB_qmin.tolist(),
            "dynamic_area_leaf": float(
                0.5 * U_IN ** 2 * math.pi * (RADIUS / 2.0 ** (-d_max)) ** 2),
        }
        # sanity: per-direction force contribution on B (which directions
        # dominate the 22x?)
        plan, gvals = uniform_ghost_vals(B)
        f = B.f_leaf.clone()
        per_d = {}
        for d in range(1, Q):
            m = B.bfl_mask[d]
            if not bool(m.any()):
                continue
            g2 = B
            od = int(OPPOSITE[d].item())
            idx = torch.nonzero(m, as_tuple=False).squeeze(1)
            qq = B.q_field[d, idx].to(torch.float64)
            fp_d = B.f_leaf[d, idx].to(torch.float64)
            fp_opp = B.f_leaf[od, idx].to(torch.float64)
            up = B.neighbor_table[od, idx]
            fp_up = torch.zeros_like(fp_d)
            valid = up >= 0
            if bool(valid.any()):
                fp_up[valid] = B.f_leaf[d, up[valid]].to(torch.float64)
            ghost = up == -1
            if bool(ghost.any()):
                slots = plan.slot[d, idx[ghost]]
                fp_up[ghost] = gvals[d, slots].to(torch.float64)
            lin = qq < 0.5
            fbc = torch.where(
                lin,
                2.0 * qq * fp_d + (1.0 - 2.0 * qq) * fp_up,
                fp_d / (2.0 * torch.where(lin, torch.ones_like(qq), qq))
                + (2.0 * torch.where(lin, torch.ones_like(qq), qq) - 1.0)
                / (2.0 * torch.where(lin, torch.ones_like(qq), qq)) * fp_opp,
            )
            exch = fp_d + fbc
            per_d[f"d{d}"] = {
                "n": int(m.sum().item()),
                "q_mean": float(qq.mean().item()),
                "n_lin": int(lin.sum().item()),
                "n_quad": int((~lin).sum().item()),
                "mean_exch": float(exch.mean().item()),
                "link_x": float((exch * C[d, 0]).sum().item()),
                "link_y": float((exch * C[d, 1]).sum().item()),
                "link_z": float((exch * C[d, 2]).sum().item()),
            }
        out[f"perdir_{tag}"] = per_d

    print(json.dumps(out, indent=1))
    with open("/root/bfl_q_audit_out.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
