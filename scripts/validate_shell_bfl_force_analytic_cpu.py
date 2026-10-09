#!/usr/bin/env python
"""CPU validation probe: octree **shell BFL force chain** on *analytic* flow fields.

Motivation
----------
The integrated framework's shell-BFL drag came out ~2.2x the single-card
converged value (integrated R6 d_max=2 -> Cd 2.77 vs single-card 1.24).  The
verified Taylor-Couette benchmark (benchmarks/verified/taylor_couette,
"disclosure 5") documents that the library's *linkwise moving-wall
momentum-exchange* primitive overestimates the wall torque by ~2.05-2.14x for
exactly the "collide-on-solid-cells + post-stream bounce-back" composition the
shell uses.  This probe isolates the shell BFL force chain from the coupled
dynamics by evaluating it on **frozen analytic fields** whose exact answer is
known, at two shell refinements (d_max = 1, 2).

Cases
-----
A. Uniform equilibrium stream  u = U x_hat, STATIONARY wall.
   Reference: the plain-bounce-back link sum  sum_links c_d (f_d + f_opp)
   is EXACTLY ZERO on any closed surface (each leaf: sum_d 2 w rho c_d = 0).
   So the *plain-BB* force is an analytic 0 for every d_max; the shell's
   Bouzidi value minus it isolates the interpolation correction, and the
   d_max = 1 -> 2 change tests refinement invariance of the chain.

B. Rigid-body rotation about the sphere axis, MOVING wall with the SAME
   u_w = Omega z_hat x (r - r_c)  (the wall rotates with the fluid).
   Reference: zero viscous torque (T_z = 0) and zero net force for ANY Omega
   and ANY resolution -- the rotating-wall / Taylor-Couette class transplanted
   to the shell.  An Omega sweep separates a linear (chain) from a cubic
   (compressibility) residual.

C. Convention spread on the SAME links/field: shell BFL  c_d (f_prev[f] + f_bc)
   vs the library ideal-halfway-reflection linkwise MEM
   f_r = f_o - 6 w rho (c . u_w).

Every number is a single force-chain evaluation on a field written down
analytically; nothing is tuned, no simulation coupling.

Usage
-----
    python scripts/validate_shell_bfl_force_analytic_cpu.py
        [--radius 12] [--dmax 1 2] [--u-in 0.06] [--omega 0.002 0.004]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.d3q27 import W as W27  # noqa: E402
from tensorlbm.d3q27 import equilibrium27  # noqa: E402
from tensorlbm.octree_boundary.bfl import (  # noqa: E402
    bfl_apply_gather,
    leaf_force_spatial_weights,
    leaf_force_weights,
)
from tensorlbm.octree_boundary.geometry import build_octree_shell  # noqa: E402
from tensorlbm.octree_boundary.stepping import build_ghost_plan  # noqa: E402


# --------------------------------------------------------------------------- #
# analytic fields
# --------------------------------------------------------------------------- #
def uniform_equilibrium(shape, u_in, device):
    """Exact LB equilibrium of a uniform stream u = (u_in, 0, 0)."""
    rho = torch.ones(shape, device=device)
    ux = torch.full(shape, u_in, device=device)
    z = torch.zeros(shape, device=device)
    return equilibrium27(rho, ux, z, z).to(torch.float64)


def rigid_rotation_equilibrium(shape, centre, omega, device):
    """Equilibrium of rigid-body rotation about z through ``centre``.

    ``u = Omega z_hat x (r - r_c)``: analytic incompressible field with zero
    strain rate -> analytic zero viscous torque for a co-rotating wall.
    """
    nz, ny, nx = shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float64),
        torch.arange(ny, device=device, dtype=torch.float64),
        torch.arange(nx, device=device, dtype=torch.float64),
        indexing="ij",
    )
    ux = -omega * (yy + 0.5 - centre[1])
    uy = omega * (xx + 0.5 - centre[0])
    uz = torch.zeros_like(ux)
    rho = torch.ones_like(ux)
    return equilibrium27(rho, ux, uy, uz).to(torch.float64)


def sample_host(field, host):
    return field[:, host[:, 0], host[:, 1], host[:, 2]].clone()


def wall_velocity_at_leaves(leaf_center, centre, omega):
    dx = leaf_center[:, 0].to(torch.float64) - float(centre[0])
    dy = leaf_center[:, 1].to(torch.float64) - float(centre[1])
    return -omega * dy, omega * dx, torch.zeros_like(dx)


# --------------------------------------------------------------------------- #
# force-chain evaluations
# --------------------------------------------------------------------------- #
def shell_bfl_force_torque(octree, f_prev, *, ghost_plan, ghost_vals, wall,
                           weights, origin):
    """One shell BFL pass; force + torque from the actual per-link impulses.

    ``weights`` MUST carry the per-leaf convective factor
    (``leaf_force_weights(octree, include_spatial=True)``): the shell mixes
    ``dx = 0.5`` (depth 1) and ``dx = 0.25`` (depth 2) leaves, so a leaf
    impulse must be mapped to a common lattice by ``(dx_leaf/dx_ref)^2 =
    2^-2(L-1)`` before summation.  ``F``/``T`` are therefore already in the
    common (level-1) lattice and are refinement-invariant; per level the
    pre-fix (time-only, per-leaf-lattice) sum is also reported for contrast.
    """
    link_sum = torch.zeros(3, dtype=torch.float64)
    torque_sum = torch.zeros(3, dtype=torch.float64)
    n_links = [0]
    by_level: dict[int, list] = {}

    def _accumulate(link):
        n_links[0] += int(link.shape[0])
        link_sum.add_(link.sum(dim=0))

    def sink(d, idx, link):
        _accumulate(link)
        r = octree.leaf_center[idx].to(torch.float64) - origin
        torque_sum.add_(torch.cross(r, link, dim=1).sum(dim=0))
        lv = octree.leaf_level[idx]
        for L in (1, 2):
            sel = lv == L
            if bool(sel.any()):
                e = by_level.setdefault(L, [torch.zeros(3, dtype=torch.float64), 0])
                e[0].add_(link[sel].sum(dim=0))
                e[1] += int(sel.sum())

    bfl_apply_gather(
        octree, f_prev.clone(), f_prev.clone(),
        ghost_plan=ghost_plan, ghost_vals=ghost_vals,
        wall_velocity=wall,
        wall_density=(torch.ones(octree.n_leaf, dtype=torch.float64)
                      if wall is not None else None),
        force_weights=weights, return_force=True, link_sink=sink,
    )
    levels = {}
    for L, (F, n) in sorted(by_level.items()):
        # F already carries the 2^-2(L-1) convective factor (weights include
        # it); recover the pre-fix (time-only, per-leaf-lattice) sum by
        # dividing it back out, purely for the bug-contrast report.
        F_raw = F * (2.0 ** (2.0 * (L - 1)))
        levels[L] = {
            "n_links": n,
            "F": F.tolist(),
            "F_raw_time_only": F_raw.tolist(),
        }
    # ``link_sum`` is already in the common (reference-level) lattice because
    # the weights fold the spatial factor; keep the field for report
    # backward-compat as an equality cross-check.
    f_rescaled = link_sum.clone()
    return link_sum, torque_sum, n_links[0], levels, f_rescaled


def plain_bb_force_torque(octree, f_prev, *, weights, origin):
    """Analytic reference: plain bounce-back link sum (f_d + f_opp) c_d.

    Exactly zero for a uniform equilibrium field at every d_max; the residual
    measures the link-set / weighting bookkeeping alone.
    """
    mask = octree.bfl_mask.clone()
    mask[0] = False
    opp = octree._opp
    c_vec = octree._c_vec.to(torch.float64)
    idx_d, idx_i = torch.nonzero(mask, as_tuple=False).T
    f_out = f_prev[idx_d, idx_i].to(torch.float64)
    f_in = f_prev[opp[idx_d], idx_i].to(torch.float64)
    link = (f_out + f_in).unsqueeze(1) * c_vec[idx_d] * weights[idx_i].unsqueeze(1)
    r = octree.leaf_center[idx_i].to(torch.float64) - origin
    return (link.sum(dim=0),
            torch.cross(r, link, dim=1).sum(dim=0),
            int(idx_d.numel()))


def ideal_reflection_force_torque(octree, f_prev, *, wall, weights, origin):
    """Library convention: ideal halfway-reflection linkwise MEM (force on wall)."""
    mask = octree.bfl_mask.clone()
    mask[0] = False
    c_vec = octree._c_vec.to(torch.float64)
    w = W27.to(torch.float64)
    d_c, i_c = torch.nonzero(mask, as_tuple=False).T
    if d_c.numel() == 0:
        z = torch.zeros(3, dtype=torch.float64)
        return z, z
    f_o = f_prev[d_c, i_c].to(torch.float64)
    if wall is None:
        uw = torch.zeros(d_c.shape[0], 3, dtype=torch.float64)
    else:
        uw = torch.stack([wall[0][i_c], wall[1][i_c], wall[2][i_c]], dim=1).to(torch.float64)
    f_r = f_o - 6.0 * w[d_c] * (c_vec[d_c] * uw).sum(dim=1)
    link = -(f_o + f_r).unsqueeze(1) * c_vec[d_c] * weights[i_c].unsqueeze(1)
    r = octree.leaf_center[i_c].to(torch.float64) - origin
    return -link.sum(dim=0), -torch.cross(r, link, dim=1).sum(dim=0)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def build_shell(radius, d_max, device, lattice="D3Q27"):
    l1 = int(round(radius * 2.0)) + 22
    shape = (l1, l1, l1 + 30)
    centre = ((shape[2] - 1) / 2.0, (shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0)
    octree = build_octree_shell(
        shape, centre, float(radius), bl_thickness_cells=max(2.0, radius / 2.0),
        d_max=d_max, transition=1, device=device, lattice=lattice,
    )
    return octree, shape, centre


def eval_case(octree, shape, centre_cells, weights, origin, f_field, wall):
    host = octree.leaf_host_cell
    f_prev = sample_host(f_field, host)
    gplan = build_ghost_plan(octree, shape)
    ghost_vals = torch.zeros(octree.Q, max(gplan.n_ghost, 1), dtype=torch.float64)
    if gplan.n_ghost:
        ghost_vals = f_field[:, gplan.z0.to(torch.long), gplan.y0.to(torch.long),
                             gplan.x0.to(torch.long)].clone()
    F, T, n_links, levels, F_rescaled = shell_bfl_force_torque(
        octree, f_prev, ghost_plan=gplan, ghost_vals=ghost_vals,
        wall=wall, weights=weights, origin=origin)
    Fp, Tp, _ = plain_bb_force_torque(octree, f_prev, weights=weights, origin=origin)
    Fi, Ti = ideal_reflection_force_torque(
        octree, f_prev, wall=wall, weights=weights, origin=origin)
    return dict(
        n_leaf=int(octree.n_leaf), n_links=n_links,
        F=F.tolist(), T=T.tolist(),
        F_plain_bb=Fp.tolist(), T_plain_bb=Tp.tolist(),
        F_ideal_reflection=Fi.tolist(), T_ideal_reflection=Ti.tolist(),
        F_bouzidi_correction=(F - Fp).tolist(),
        T_bouzidi_correction=(T - Tp).tolist(),
        per_level_force={str(k): v for k, v in levels.items()},
        F_level_rescaled=F_rescaled.tolist(),
        Fx_shell_over_ideal_reflection=(float(F[0]) / float(Fi[0])
                                        if abs(float(Fi[0])) > 1e-30 else float("nan")),
    )


def main():
    ap = argparse.ArgumentParser(description="shell BFL force chain on analytic fields")
    ap.add_argument("--radius", type=float, default=12.0)
    ap.add_argument("--dmax", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--u-in", type=float, default=0.06)
    ap.add_argument("--omega", type=float, nargs="+", default=[0.002, 0.004])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    radius, u_in = args.radius, args.u_in
    dyn = 0.5 * u_in ** 2 * math.pi * (2.0 * radius) ** 2

    print("=" * 84)
    print(f"shell BFL force chain on analytic fields   R={radius:g} U={u_in:g} "
          f"D3Q27 {device}")
    print("=" * 84)

    report = {"meta": vars(args), "dyn_ref": dyn, "cases": {}}
    for d_max in args.dmax:
        octree, shape, centre_cells = build_shell(radius, d_max, device)
        origin = torch.tensor(centre_cells, dtype=torch.float64)
        # FIXED weights: time weight x per-leaf convective (dx_leaf/dx_ref)^2
        # (reference level 1) so mixed-depth / cross-d_max impulses share one
        # lattice.  (Pre-fix this was the time-only weight and the frozen-field
        # force grew x3.71 from d_max=1 -> 2 instead of staying invariant.)
        weights = leaf_force_weights(octree, include_spatial=True)
        assert bool(
            torch.allclose(
                weights,
                leaf_force_weights(octree)
                * leaf_force_spatial_weights(octree),
            ),
        ), "combined weight must equal time x spatial"

        # ---- Case A: uniform stream, stationary wall -----------------------
        fA = uniform_equilibrium(shape, u_in, device)
        A = eval_case(octree, shape, centre_cells, weights, origin, fA, None)
        report["cases"][f"A_uniform_dmax{d_max}"] = A
        print(f"\n[A] uniform u={u_in:g}, stationary wall, d_max={d_max}  "
              f"n_leaf={A['n_leaf']} n_links={A['n_links']}")
        print(f"    shell BFL        Fx = {A['F'][0]:+.6e}   "
              f"(Cd_eq {A['F'][0] / dyn:.4f})")
        print(f"    plain-BB (ana 0) Fx = {A['F_plain_bb'][0]:+.6e}   "
              f"|Fy|={abs(A['F_plain_bb'][1]):.2e} |Fz|={abs(A['F_plain_bb'][2]):.2e}")
        print(f"    Bouzidi corr.    Fx = {A['F_bouzidi_correction'][0]:+.6e}")
        print(f"    ideal-reflect.   Fx = {A['F_ideal_reflection'][0]:+.6e}  "
              f"-> shell/ideal = {A['Fx_shell_over_ideal_reflection']:.4f}")
        for L, v in A["per_level_force"].items():
            print(f"    level {L}: n_links={v['n_links']:7d} "
                  f"Fx(corrected)={v['F'][0]:+.6e} "
                  f"per-link={v['F'][0] / max(v['n_links'], 1):+.4e}   "
                  f"Fx(time-only/raw)={v['F_raw_time_only'][0]:+.6e}")
        print(f"    fixed weigh. Fx = {A['F_level_rescaled'][0]:+.6e} "
              f"(weights fold dx^2 = 2^-2(L-1) per level into the aggregation)")

        # ---- Case B: rigid rotation, co-rotating wall (analytic T=0, F=0) --
        for om in args.omega:
            fB = rigid_rotation_equilibrium(shape, centre_cells, om, device)
            wall = wall_velocity_at_leaves(octree.leaf_center, centre_cells, om)
            B = eval_case(octree, shape, centre_cells, weights, origin, fB, wall)
            report["cases"][f"B_rot_om{om:g}_dmax{d_max}"] = B
            uw = om * radius
            print(f"\n[B] rigid rot Omega={om:g} (u_wall={uw:.4f}, Ma={uw / math.sqrt(1/3):.3f}),"
                  f" co-rotating wall, d_max={d_max}  n_links={B['n_links']}")
            print(f"    shell BFL  Fx={B['F'][0]:+.4e} Fy={B['F'][1]:+.4e} "
                  f"Tz={B['T'][2]:+.6e}   <-- ANALYTIC: F=0, Tz=0")
            print(f"    plain-BB   Fx={B['F_plain_bb'][0]:+.4e} Tz={B['T_plain_bb'][2]:+.6e}")
            print(f"    Bouzidi corr. Tz={B['T_bouzidi_correction'][2]:+.6e}")

    # ---- summaries --------------------------------------------------------
    print("\n" + "-" * 84)
    print("REFINEMENT INVARIANCE on the frozen field (per-root-step ledger value =")
    print("  the single weighted pass for a frozen field).  Fixed weights fold the")
    print("  per-level dx^2 = 2^-2(L-1); raw = pre-fix time-only weight:")
    prev = None
    prev_raw = None
    for d_max in args.dmax:
        A = report["cases"][f"A_uniform_dmax{d_max}"]
        nsub = 1 << d_max
        ledger = float(A["F"][0])            # single weighted pass == mem_force
        raw = float(A["per_level_force"].get("1", {}).get(
            "F_raw_time_only", [0.0] * 3)[0]) + float(
            A["per_level_force"].get("2", {}).get(
                "F_raw_time_only", [0.0] * 3)[0])
        print(f"  d_max={d_max}: n_substeps={nsub} n_links={A['n_links']:6d} ")
        print(f"           Fx(corrected)={ledger:.6e}   Fx(raw time-only)={raw:.6e}")
        if prev is not None:
            print(f"           ratio vs d_max={prev[0]}: "
                  f"corrected={ledger / prev[1]:.4f}  raw={raw / (prev_raw or 1.0):.4f}  "
                  f"(link-count ratio {A['n_links'] / prev[2]:.4f})")
        prev = (d_max, ledger, A["n_links"])
        prev_raw = raw
    first = report["cases"][f"A_uniform_dmax{args.dmax[0]}"]["F"][0]
    last = report["cases"][f"A_uniform_dmax{args.dmax[-1]}"]["F"][0]
    ratio = float(last) / float(first) if abs(float(first)) > 0 else float("nan")
    verdict = "PASS" if abs(ratio - 1.0) < 0.15 else "FAIL"
    print(f"  -> corrected d_max={args.dmax[-1]}/{args.dmax[0]} force ratio = "
          f"{ratio:.4f} (target ~1, pre-fix ~3.71): {verdict}")

    print("\nANALYTIC NULL TEST (case B, co-rotating wall): expect Tz == 0")
    for d_max in args.dmax:
        for om in args.omega:
            B = report["cases"][f"B_rot_om{om:g}_dmax{d_max}"]
            print(f"  d_max={d_max} Omega={om:g}: Tz={B['T'][2]:+.6e}  "
                  f"Tz_plain_bb={B['T_plain_bb'][2]:+.6e}")
    if len(args.omega) >= 2 and len(args.dmax) >= 1:
        d = args.dmax[0]
        t_lo = report["cases"][f"B_rot_om{args.omega[0]:g}_dmax{d}"]["T"][2]
        t_hi = report["cases"][f"B_rot_om{args.omega[1]:g}_dmax{d}"]["T"][2]
        if abs(t_lo) > 1e-30:
            print(f"  scaling d_max={d}: Tz(Omega={args.omega[1]:g}) / "
                  f"Tz(Omega={args.omega[0]:g}) = {t_hi / t_lo:.4f}  "
                  f"(Omega ratio {args.omega[1] / args.omega[0]:.4f})")
    print("-" * 84)

    out = Path("shell_bfl_force_analytic_report.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()