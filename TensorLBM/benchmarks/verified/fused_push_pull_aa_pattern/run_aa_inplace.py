#!/usr/bin/env python3
"""Driver for the strict A-A (single-field, in-place) fused push-pull verification.

Anchor: ``XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN``.

Produces ``result_aa_inplace.json`` (numbers quoted in
``docs/xval_fused_push_pull_aa_strict_inplace.md``) and the figures under
``figures/``.  Everything quoted in the write-up is read back from this file --
no number is transcribed by hand.

    python run_aa_inplace.py            # ~2 min single core, D2Q9 float64
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parents[2] / "tests"))

import tensorlbm.aa_fused_push_pull as AB          # noqa: E402
import tensorlbm.aa_fused_push_pull_inplace as AA  # noqa: E402

NU = 0.05
G = 1e-6
TAU = 3.0 * NU + 0.5


def rest(nx, ny, perturb=False):
    return AA.rest_equilibrium(ny, nx, perturb=perturb)


def equivalence_series(nx, ny, steps, solid=None, wall_u=None, force=None, tau=0.9,
                       perturb=True):
    """Step the fused A-B and the in-place A-A kernel side by side."""
    f_ab = rest(nx, ny, perturb=perturb)
    f_aa = f_ab.clone()
    diffs = []
    for _ in range(steps):
        f_ab = AB.fused_push_pull_step(f_ab, tau, solid=solid, wall_u=wall_u, force=force)
        AA.aa_fused_push_pull_step(f_aa, tau, solid=solid, wall_u=wall_u, force=force)
        diffs.append(float((f_ab - f_aa).abs().max()))
    return f_ab, f_aa, diffs


def poiseuille(n_fluid, steps):
    nx = 4
    ny = n_fluid + 2
    solid = AA.channel_masks(nx, ny, 1)
    fluid = ~solid
    force = torch.zeros(2, ny, nx, dtype=torch.float64)
    force[0] = G
    f = rest(nx, ny)
    t0 = time.perf_counter()
    for _ in range(steps):
        AA.aa_fused_push_pull_step(f, TAU, solid=solid, force=force)
    wall = time.perf_counter() - t0
    _, ux, uy = AA.macroscopic(f, force)
    prof = ux[fluid].reshape(n_fluid, nx).mean(dim=1)
    analytic, u_max = AA.analytic_poiseuille(n_fluid, G, NU)
    return {
        "n_fluid": n_fluid,
        "steps": steps,
        "u_max_rel_err": float(abs(prof.max() - u_max) / u_max),
        "profile_rel_l2": float((prof - analytic).norm() / analytic.norm()),
        "max_uy_over_umax": float(uy[fluid].abs().max() / u_max),
        "u_max_numeric": float(prof.max()),
        "u_max_analytic": float(u_max),
        "sec_per_step": wall / steps,
        "profile": [float(v) for v in prof],
        "analytic": [float(v) for v in analytic],
    }


def couette(n_fluid=8, steps=4000, u_wall=0.02):
    nx = 4
    ny = n_fluid + 2
    solid = AA.channel_masks(nx, ny, 1)
    fluid = ~solid
    wall = torch.zeros(2, ny, nx, dtype=torch.float64)
    wall[0, -1, :] = u_wall
    f = rest(nx, ny)
    for _ in range(steps):
        AA.aa_fused_push_pull_step(f, TAU, solid=solid, wall_u=wall)
    _, ux, _ = AA.macroscopic(f)
    prof = ux[fluid].reshape(n_fluid, nx).mean(dim=1)
    analytic = AA.analytic_couette(n_fluid, u_wall)
    return {
        "n_fluid": n_fluid,
        "steps": steps,
        "u_wall": u_wall,
        "max_rel_err": float((prof - analytic).abs().max() / u_wall),
        "profile": [float(v) for v in prof],
        "analytic": [float(v) for v in analytic],
    }


def mass_closed_box(n=16, steps=500):
    solid = torch.zeros(n, n, dtype=torch.bool)
    solid[0, :] = solid[-1, :] = True
    solid[:, 0] = solid[:, -1] = True
    fluid = ~solid
    f = rest(n, n, perturb=True)
    m0 = float(f[:, fluid].sum())
    drifts = []
    for _ in range(steps):
        AA.aa_fused_push_pull_step(f, 0.9, solid=solid)
        drifts.append(abs(float(f[:, fluid].sum()) - m0) / m0)
    return {"grid": n, "steps": steps, "max_rel_drift": max(drifts), "final_rel_drift": drifts[-1]}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=("all", "equiv", "poiseuille", "couette", "mass"))
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0)
    args = ap.parse_args()
    path = HERE / "result_aa_inplace.json"
    res = json.loads(path.read_text()) if (args.part != "all" and path.exists()) else {}
    res.setdefault("anchor", "XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN")
    res.setdefault("kernel", "AA in-place fused push-pull")

    if args.part in ("all", "couette"):
        res["couette"] = couette(n_fluid=8, steps=args.steps or 2500)
    if args.part in ("all", "mass"):
        res["mass"] = mass_closed_box()
    if args.part in ("all", "poiseuille"):
        ladder = ((8, 2500), (16, 5000), (24, 6000)) if args.part == "all" else ()
        if args.n:
            ladder = ((args.n, args.steps or {8: 2500, 16: 5000, 24: 6000}.get(args.n, 5000)),)
        cur = {r["n_fluid"]: r for r in res.get("poiseuille", [])}
        for n, steps in ladder:
            cur[n] = poiseuille(n, steps)
        res["poiseuille"] = [cur[k] for k in sorted(cur)]
    if args.part == "couette" or args.part == "mass" or args.part == "poiseuille":
        path.write_text(json.dumps(res, indent=2) + "\n")
        print(f"partial write: {args.part} ({len(res.get('poiseuille', []))} poiseuille points)")
        return 0

    # ---- storage discipline: never allocate a second full field --------------
    nx, ny = 12, 10
    solid = AA.channel_masks(nx, ny, 1)
    full = (9, ny, nx)
    seen = {"aa": [], "ab": []}
    orig = (torch.empty, torch.empty_like, torch.zeros, torch.zeros_like)

    def spy(fn, bucket):
        def wrapped(*a, **kw):
            t = fn(*a, **kw)
            if hasattr(t, "shape"):
                bucket.append(tuple(t.shape))
            return t
        return wrapped

    for tag, kernel in (("aa", AA.aa_fused_push_pull_step), ("ab", AB.fused_push_pull_step)):
        torch.empty = spy(orig[0], seen[tag])
        torch.empty_like = spy(orig[1], seen[tag])
        torch.zeros = spy(orig[2], seen[tag])
        torch.zeros_like = spy(orig[3], seen[tag])
        try:
            f = rest(nx, ny)
            seen[tag].clear()
            if tag == "aa":
                kernel(f, 0.9, solid=solid)
            else:
                kernel(f, 0.9, solid=solid)
        finally:
            torch.empty, torch.empty_like, torch.zeros, torch.zeros_like = orig
    res["second_field_allocations"] = {k: int(full in v) for k, v in seen.items()}

    # ---- equivalence with the fused A-B reference ---------------------------
    eq = {}
    _, _, d = equivalence_series(8, 7, 500, tau=0.9)
    eq["periodic"] = {"steps": 500, "max_abs_diff": max(d), "final_abs_diff": d[-1]}
    solid_c = AA.channel_masks(4, 10, 1)
    _, _, d = equivalence_series(4, 10, 500, solid=solid_c, tau=0.9)
    eq["walled"] = {"steps": 500, "max_abs_diff": max(d), "final_abs_diff": d[-1]}
    wall = torch.zeros(2, 10, 4, dtype=torch.float64)
    wall[0, -1, :] = 0.02
    _, _, d = equivalence_series(4, 10, 500, solid=solid_c, wall_u=wall, tau=0.9)
    eq["moving_wall"] = {"steps": 500, "max_abs_diff": max(d), "final_abs_diff": d[-1]}
    force = torch.zeros(2, 10, 4, dtype=torch.float64)
    force[0] = G
    _, _, d = equivalence_series(4, 10, 500, solid=solid_c, force=force, tau=TAU, perturb=False)
    eq["guo_forced"] = {"steps": 500, "max_abs_diff": max(d), "final_abs_diff": d[-1]}
    res["equivalence"] = eq

    # ---- independent scatter oracle (numpy, from scratch) ------------------
    from test_aa_fused_independent_oracle import scatter_step, _numpy_rest  # noqa: E402

    steps = 40
    ny, nx = 8, 5
    solid_np = np.zeros((ny, nx), dtype=bool)
    solid_np[0, :] = solid_np[-1, :] = True
    f_np = _numpy_rest(ny, nx, perturb=True)
    f_aa = torch.tensor(f_np.copy())
    for _ in range(steps):
        f_np = scatter_step(f_np, 0.9, solid=solid_np)
        AA.aa_fused_push_pull_step(f_aa, 0.9, solid=torch.tensor(solid_np))
    fluid = ~solid_np
    res["scatter_oracle"] = {
        "steps": steps,
        "max_abs_diff_fluid_nodes": float(np.abs(f_np[:, fluid] - f_aa.numpy()[:, fluid]).max()),
    }

    # ---- physics: analytic benchmarks --------------------------------------
    res["gates"] = {
        "equivalence_abs": 1e-13,
        "scatter_oracle_abs": 1e-13,
        "mass_rel_drift": 1e-12,
        "precision_pct": 3.0,
    }
    path.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("poiseuille", "couette")}, indent=2))
    for r in res["poiseuille"]:
        print(f"n={r['n_fluid']:3d}  u_max err {100*r['u_max_rel_err']:.3f}%  "
              f"L2 {100*r['profile_rel_l2']:.3f}%  {1e3*r['sec_per_step']:.2f} ms/step")
    print(f"couette max rel err {res['couette']['max_rel_err']:.2e}")
    print(f"mass drift (500 steps) {res['mass']['max_rel_drift']:.2e}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())