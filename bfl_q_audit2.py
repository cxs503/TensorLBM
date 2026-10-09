#!/usr/bin/env python3
"""Diagnostic 2: verify the effective body location of the inside_fn path.

Hypothesis: build_shell_cell_mask and topology._classify_targets pass
(z,y,x)-ordered centres to sphere_inside_fn (which expects (x,y,z) matching
the (cx,cy,cz) tuple), so with centre=(48,32,32) and shape (nz,ny,nx)=
(64,64,96) the effective body is a sphere of radius 6 at (x=32, y=32, z=48)
instead of (48,32,32).  This dumps leaf/mask distributions to confirm, and
quantifies the pure-q=0.5 vs analytic-q effect on the healthy (analytic)
path.
"""
import json
import torch

from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.geometry_adapters import sphere_inside_fn
from tensorlbm.octree_boundary.bfl import bfl_apply_gather, leaf_force_weights
from tensorlbm.octree_boundary.stepping import build_ghost_plan
from tensorlbm.d3q27 import equilibrium27, C, OPPOSITE

SHAPE = (64, 64, 96)
CENTER = (48.0, 32.0, 32.0)
RADIUS = 6.0
BL = 3.0
U_IN = 0.06
Q = 27


def build(inside_fn, d_max=1):
    g = build_octree_shell(
        SHAPE, center=CENTER, radius=RADIUS,
        bl_thickness_cells=BL, d_max=d_max, lattice="D3Q27",
        device=torch.device("cpu"), inside_fn=inside_fn,
    )
    rho = torch.ones(g.n_leaf)
    ux = torch.full((g.n_leaf,), U_IN)
    uy = torch.zeros(g.n_leaf)
    uz = torch.zeros(g.n_leaf)
    g.f_leaf = equilibrium27(rho, ux, uy, uz).view(Q, -1)
    return g


def bfl_force(g, q_override=None, q_min=None):
    import copy
    plan = build_ghost_plan(g, SHAPE)
    n_g = plan.n_ghost
    rho = torch.ones(n_g)
    ux = torch.full((n_g,), U_IN)
    uy = torch.zeros(n_g)
    uz = torch.zeros(n_g)
    gvals = equilibrium27(rho, ux, uy, uz).view(Q, -1)
    g2 = g
    if q_override is not None:
        g2 = copy.copy(g)
        g2.q_field = q_override
    f = g.f_leaf.clone()
    out, force = bfl_apply_gather(
        g2, f, g.f_leaf.clone(),
        ghost_plan=plan, ghost_vals=gvals,
        force_weights=leaf_force_weights(g), return_force=True,
        q_min=q_min,
    )
    return force


def report(g, name):
    m = g.bfl_mask
    idx = torch.nonzero(m, as_tuple=False)          # (d, i)
    leafs = idx[:, 1]
    lc = g.leaf_center.to(torch.float64)            # (n, 3) (x, y, z)
    ml = lc[leafs]
    full = lc
    out = {
        f"{name}_n_leaf": int(g.n_leaf),
        f"{name}_full_leaf_centroid": full.mean(dim=0).tolist(),
        f"{name}_full_leaf_min": full.min(dim=0).values.tolist(),
        f"{name}_full_leaf_max": full.max(dim=0).values.tolist(),
        f"{name}_masked_centroid": ml.mean(dim=0).tolist(),
        f"{name}_masked_min": ml.min(dim=0).values.tolist(),
        f"{name}_masked_max": ml.max(dim=0).values.tolist(),
    }
    # per-axis-direction masked-link centroids (d1..d6)
    for d in (1, 2, 3, 4, 5, 6):
        dm = m[d]
        if bool(dm.any()):
            out[f"{name}_d{d}_centroid"] = lc[dm].mean(dim=0).tolist()
            out[f"{name}_d{d}_n"] = int(dm.sum().item())
    # distance of each masked leaf centre to the true vs permuted centre
    c_true = torch.tensor([48.0, 32.0, 32.0])
    c_perm = torch.tensor([32.0, 32.0, 48.0])
    d_true = (ml - c_true).norm(dim=1)
    d_perm = (ml - c_perm).norm(dim=1)
    out[f"{name}_masked_dist_to_true_mean"] = float(d_true.mean().item())
    out[f"{name}_masked_dist_to_perm_mean"] = float(d_perm.mean().item())
    out[f"{name}_masked_dist_to_true_hist"] = [
        int(((d_true >= lo) & (d_true < hi)).sum().item())
        for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10),
                       (10, 12), (12, 14), (14, 16), (16, 18), (18, 30)]]
    out[f"{name}_masked_dist_to_perm_hist"] = [
        int(((d_perm >= lo) & (d_perm < hi)).sum().item())
        for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10),
                       (10, 12), (12, 14), (14, 16), (16, 18), (18, 30)]]
    return out


def main():
    out = {}
    A = build(None)
    B = build(sphere_inside_fn(CENTER, RADIUS))
    out.update(report(A, "A"))
    out.update(report(B, "B"))

    # ---- pure q ablation on the healthy analytic path (dmax=1) ----
    fA = bfl_force(A)
    q05 = torch.full_like(A.q_field, 0.5)
    fA_q05 = bfl_force(A, q_override=q05)
    # analytic q but clamped to >= 0.5 (forces quad branch with 1/(2q))
    q_ge05 = A.q_field.clone()
    q_ge05[q_ge05 < 0.5] = 0.5
    fA_qge05 = bfl_force(A, q_override=q_ge05)
    # analytic q but clamped to <= 0.5 (forces lin branch)
    q_le05 = A.q_field.clone()
    q_le05[q_le05 > 0.5] = 0.5
    fA_qle05 = bfl_force(A, q_override=q_le05)
    out["q_ablation_A"] = {
        "F_analytic_q": fA.tolist(),
        "F_q05_pure_reflection": fA_q05.tolist(),
        "F_q_clamped_ge05": fA_qge05.tolist(),
        "F_q_clamped_le05": fA_qle05.tolist(),
    }
    # q-min clamps on B (all-0.5) do nothing; on A check q_min effect
    out["q_min_A"] = {
        "F_qmin0": bfl_force(A, q_min=0.0).tolist(),
        "F_qmin0.2": bfl_force(A, q_min=0.2).tolist(),
    }
    print(json.dumps(out, indent=1))
    with open("/root/bfl_q_audit_out2.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
