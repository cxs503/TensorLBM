#!/usr/bin/env python3
"""Octree shell validation against analytic solutions.

Three exact-reference cases that expose shell/interpolation/interface bugs
without needing a long drag convergence:

  1. Couette flow — linear profile ux = U*y/H.  For a linear field the
     trilinear ghost fill is exact, the BGK collision leaves a pure
     equilibrium state untouched and a uniform shear is an exact discrete
     BGK solution, so any drift is a genuine shell-machinery error
     (interpolation, interface mapping, streaming, restriction).
     Acceptance: max |ux drift| < 1e-3.

  2. Poiseuille flow — parabolic profile ux = 4*Uc*y*(H-y)/H^2.  The
     second-order field exposes ghost-fill interpolation error and
     non-equilibrium rescale bugs in fill_ghost / restriction.
     Acceptance: max |ux drift| < 1e-2.

  3. Resting sphere in quiescent fluid — the BFL momentum-exchange force
     must vanish.  A non-zero force means the shell/BFL geometry has a
     spurious asymmetry (pseudo-force).  Acceptance: |Fx| < 1e-6.

Cases 1-2 use the plane-degenerate shell (``build_plane_shell``) so the
analytic plane flow is not perturbed by a spherical wall; case 3 uses the
real sphere shell with ``bfl_apply_gather`` + ``ShellForceLedger``.

Non-equilibrium construction (``--neq-parent``, default on)
-----------------------------------------------------------
The parent and leaf fields carry the exact leading-order discrete-BGK
steady-state shear non-equilibrium ``g_i = -3*tau*w_i*rho*c_ix*c_iy*du/dy``
(gradient in the lattice's own units, leaf gradient ``dudy*2^-level``).
With ``tau_f = 0.5 + 2*(tau_c - 0.5)`` the coarse->leaf rescale factor is
exactly ``tau_f/(2*tau_c)`` — the factor ``fill_ghost`` applies — so this
neq exercises the ghost rescale path with a clean analytic signal.  A pure
equilibrium field (``--no-neq-parent``) exercises interpolation/collision/
interface exactness only.

Bug found and fixed (2026-08, P3 debug): ``fill_ghost`` injected the L1
*pre-collision* population as the leaf's incoming value, while
``stream_gather`` pulls *post-collision* states from real leaf neighbours.
For non-equilibrium fields the interface therefore over-injected the neq by
``tau_f/(tau_f - 1)`` (~3.5x at tau_f = 0.7), forming a spurious stress
layer at the AMR interface: driven-Couette ux error ~1.2e-3 (was
unobservable with pure-equilibrium test fields, where the collision is a
no-op).  ``fill_ghost`` now relaxes the rescaled neq with the leaf's tau
before injection; with the fix the Couette/Poiseuille drifts drop to
~1e-5 / ~8.5e-4.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/octree_analytic_check.py [--steps N]
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.octree_boundary.bfl import bfl_apply_gather, leaf_force_weights
from tensorlbm.octree_boundary.force import ShellForceLedger
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.stepping import (
    build_ghost_plan,
    build_plane_shell,
    step_octree_shell,
)
from tensorlbm.refinement import BoxRegion
from tensorlbm.solver3d import collide_bgk3d

Q = 19
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Analytic fields
# ---------------------------------------------------------------------------


def _shear_neq(
    dudy: torch.Tensor, tau: float, device,
) -> torch.Tensor:
    """Exact leading-order discrete-BGK steady-state neq for a shear profile.

    ``g_i = -3 * tau * w_i * rho * c_ix * c_iy * du/dy`` solves the discrete
    BGK fixed point ``f(x) = collide(stream(f))(x)`` for a linear velocity
    field to first order in the gradient.  Its coarse->fine rescale factor
    (tau_f/(2*tau_c), gradient halved at half spacing) is exactly what
    ``fill_ghost`` applies, so this neq exercises the non-equilibrium
    rescale path with a clean analytic signal.  Returns ``(Q, ...)`` with
    the same trailing shape as ``dudy`` and zero mass/momentum.
    """
    from tensorlbm.d3q19 import C, W

    c = C.to(device).to(DTYPE)
    w = W.to(device).to(DTYPE)
    cx = c[:, 0].view(Q, *((1,) * dudy.ndim))
    cy = c[:, 1].view(Q, *((1,) * dudy.ndim))
    wq = w.view(Q, *((1,) * dudy.ndim))
    return -3.0 * tau * wq * cx * cy * dudy.unsqueeze(0)


def _analytic_ux_leaf(octree, kind: str, U: float, H: float) -> torch.Tensor:
    """Analytic ux at each leaf centre (y = leaf_center[:, 1], world units)."""
    y = octree.leaf_center[:, 1].to(DTYPE)
    if kind == "couette":
        return U * y / H
    if kind == "poiseuille":
        return 4.0 * U * y * (H - y) / (H * H)
    raise ValueError(kind)


def _parent_field(
    shape: tuple[int, int, int], kind: str, U: float, H: float,
    tau: float, device, *, with_neq: bool = True,
) -> torch.Tensor:
    """(Q, nz, ny, nx) analytic L1 field at L1 cell centres.

    ``with_neq`` adds the discrete-BGK steady-state shear non-equilibrium so
    the ghost-fill rescale (``fill_ghost``) operates on a real neq signal.
    """
    nz, ny, nx = shape
    yy = torch.arange(ny, dtype=DTYPE, device=device).view(1, ny, 1) + 0.5
    if kind == "couette":
        ux = U * yy / H
        dudy = torch.full_like(ux, U / H)
    else:
        ux = 4.0 * U * yy * (H - yy) / (H * H)
        dudy = 4.0 * U * (H - 2.0 * yy) / (H * H)
    rho = torch.ones_like(ux)
    zero = torch.zeros_like(ux)
    feq = equilibrium3d(rho, ux, zero, zero)  # (Q, 1, ny, 1)
    if with_neq:
        feq = feq + _shear_neq(dudy, tau, device)
    return feq.expand(Q, nz, ny, nx).contiguous()


def _leaf_field(
    octree, kind: str, U: float, H: float, tau_leaf: float,
    device, *, with_neq: bool = True,
) -> torch.Tensor:
    """(Q, n_leaf) analytic leaf field at leaf centres.

    The leaf neq uses the gradient in the *leaf's own lattice units*:
    ``leaf_center`` is stored in world (L1) coordinates, so the lattice
    gradient is ``dudy_world * dx_leaf`` with ``dx_leaf = 2^-level``.  With
    ``tau_f = 0.5 + 2*(tau_c - 0.5)`` this makes the leaf neq exactly
    ``tau_f/(2*tau_c)`` times the L1 neq — the inverse of the ghost rescale
    in ``fill_ghost``, so boundary injection is consistent.
    """
    n = octree.n_leaf
    rho = torch.ones(1, 1, n, dtype=DTYPE, device=device)
    y = octree.leaf_center[:, 1].to(DTYPE).view(1, 1, n)
    if kind == "couette":
        ux = U * y / H
        dudy = torch.full_like(ux, U / H)
    else:
        ux = 4.0 * U * y * (H - y) / (H * H)
        dudy = 4.0 * U * (H - 2.0 * y) / (H * H)
    zero = torch.zeros(1, 1, n, dtype=DTYPE, device=device)
    feq = equilibrium3d(rho, ux, zero, zero).view(Q, n)  # (Q, 1, 1, n) -> (Q, n)
    if with_neq:
        dx_leaf = (2.0 ** (-octree.leaf_level.to(DTYPE))).view(1, 1, n)
        neq = _shear_neq(dudy * dx_leaf, tau_leaf, device)
        feq = feq + neq.view(Q, n)
    return feq


def _leaf_ux(f: torch.Tensor) -> torch.Tensor:
    """Per-leaf ux from (Q, n) populations."""
    _, ux, _, _ = macroscopic3d(f.view(Q, 1, 1, -1))
    return ux.view(-1)


def _bgk_advance(f: torch.Tensor, tau: float, level: int, substep: int):
    """Shell advance callback: BGK collide only (the stepper streams)."""
    del level, substep
    return collide_bgk3d(f.view(Q, 1, 1, -1), tau).view_as(f)


# ---------------------------------------------------------------------------
# Case runners
# ---------------------------------------------------------------------------


def run_plane_case(
    kind: str,
    steps: int,
    U: float,
    H: float,
    shape: tuple[int, int, int],
    box: BoxRegion,
    tau: float,
    device,
    *,
    with_neq: bool = True,
) -> dict:
    """Advance the plane shell against a fixed analytic L1 parent and measure
    how much the leaf ux profile drifts from the analytic reference."""
    from tensorlbm.static_block_amr import convective_refined_tau

    shell = build_plane_shell(shape, box, device=device)
    ghost_plan = build_ghost_plan(shell, shape)
    tau_leaf = convective_refined_tau(tau)          # depth-1 leaf tau (ratio 2)
    parent = _parent_field(shape, kind, U, H, tau, device, with_neq=with_neq)
    profile = _analytic_ux_leaf(shell, kind, U, H)
    shell.f_leaf = _leaf_field(
        shell, kind, U, H, tau_leaf, device, with_neq=with_neq,
    )

    max_drift = 0.0
    last_drift = 0.0
    for s in range(steps):
        # l1_old stays the analytic reference (never mutated); l1_f is a
        # fresh clone each step — the restriction/reflux write into it is
        # discarded so the parent field seen by the ghost fill is always
        # exactly the analytic profile.
        l1_f = parent.clone()
        step_octree_shell(
            shell, _bgk_advance, parent, l1_f,
            tau_coarse=tau, shell_level=1, ghost_plan=ghost_plan,
            reflux=False,
        )
        ux = _leaf_ux(shell.f_leaf)
        last_drift = float((ux - profile).abs().max().item())
        max_drift = max(max_drift, last_drift)
        if steps <= 20 or s in (0, steps // 2, steps - 1):
            print(f"      step {s + 1:4d}/{steps}: max |ux drift| = {last_drift:.3e}")
    return {
        "kind": kind,
        "steps": steps,
        "max_ux_drift": max_drift,
        "last_ux_drift": last_drift,
    }


def run_resting_sphere(
    steps: int,
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radius: float,
    tau: float,
    device,
    d_max: int = 2,
) -> dict:
    """Resting sphere in quiescent fluid: the BFL MEM force must be ~0.

    The shell is fully embedded (block 32^3, R=4, band 3+1 -> [8,24] margins).
    Leaves and L1 parent start at uniform equilibrium (u=0); with a stationary
    wall the reconstruction keeps the state at equilibrium, so a non-zero
    mean force is a pure geometry/mask asymmetry (pseudo-force).
    """
    shell = build_octree_shell(
        shape, center=center, radius=radius,
        bl_thickness_cells=3, d_max=d_max, device=device,
    )
    ghost_plan = build_ghost_plan(shell, shape)
    nz, ny, nx = shape
    rho = torch.ones(nz, ny, nx, dtype=DTYPE, device=device)
    zero = torch.zeros_like(rho)
    parent = equilibrium3d(rho, zero, zero, zero)
    shell.f_leaf = _leaf_field(shell, "couette", 0.0, 1.0, tau, device,
                               with_neq=False)
    weights = leaf_force_weights(shell)
    ledger = ShellForceLedger(shell)

    def bfl_callback(octree_, out, post, ghost_plan_, ghost_vals, *, substep):
        del substep
        return bfl_apply_gather(
            octree_, out, post,
            ghost_plan=ghost_plan_, ghost_vals=ghost_vals,
            force_weights=weights, return_force=True,
        )

    forces = []
    for s in range(steps):
        l1_f = parent.clone()
        step_octree_shell(
            shell, _bgk_advance, parent, l1_f,
            tau_coarse=tau, shell_level=1, ghost_plan=ghost_plan,
            reflux=False, bfl_fn=bfl_callback, force_ledger=ledger,
        )
        forces.append(ledger.mem_force.clone())
        ledger.reset()
    F = torch.stack(forces).mean(dim=0)
    return {
        "kind": "resting_sphere",
        "steps": steps,
        "force_x": float(F[0]),
        "force_y": float(F[1]),
        "force_z": float(F[2]),
        "max_abs_force": float(F.abs().max().item()),
        "n_leaf": int(shell.n_leaf),
        "n_bfl_links": int(shell.bfl_mask.sum().item()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="/tmp/octree_analytic.json")
    p.add_argument("--plane-shape", default="32,16,16", help="(nz,ny,nx)")
    p.add_argument("--sphere-shape", default="32,32,32")
    p.add_argument("--sphere-center", default="16,16,16")
    p.add_argument("--sphere-radius", type=float, default=4.0)
    p.add_argument(
        "--neq-parent", dest="neq_parent", action="store_true", default=True,
        help="carry the analytic shear non-equilibrium in parent/leaf fields "
             "(exercises the ghost rescale + collision path; default)",
    )
    p.add_argument(
        "--no-neq-parent", dest="neq_parent", action="store_false",
        help="pure-equilibrium fields only (interpolation/collision/interface "
             "exactness)",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    shape = tuple(int(v) for v in args.plane_shape.split(","))
    if len(shape) != 3:
        p.error("--plane-shape must be nz,ny,nx")
    H = float(shape[1])
    box = BoxRegion(
        x0=2, x1=shape[2] - 2,
        y0=2, y1=shape[1] - 2,
        z0=4, z1=shape[0] - 4,
    )
    tau = 0.6
    U = 0.05

    results: dict = {"neq_parent": args.neq_parent, "U": U, "H": H, "tau": tau}
    ok_all = True
    for kind, tol in (("couette", 1e-3), ("poiseuille", 1e-2)):
        print(f"=== {kind} (plane shell, H={H:.0f}, U={U}, "
              f"neq_parent={args.neq_parent}) ===")
        r = run_plane_case(
            kind, args.steps, U, H, shape, box, tau, device,
            with_neq=args.neq_parent,
        )
        ok = r["max_ux_drift"] < tol
        ok_all = ok_all and ok
        r["target"] = tol
        r["pass"] = ok
        results[kind] = r
        print(
            f"[{'PASS' if ok else 'FAIL'}] {kind:10s} max|ux drift| = "
            f"{r['max_ux_drift']:.3e}  (last step {r['last_ux_drift']:.3e}, "
            f"target < {tol:.0e})",
            flush=True,
        )

    print("=== resting_sphere (sphere shell, quiescent) ===")
    sshape = tuple(int(v) for v in args.sphere_shape.split(","))
    center = tuple(float(v) for v in args.sphere_center.split(","))
    r = run_resting_sphere(
        args.steps, sshape, center, args.sphere_radius, tau, device,
    )
    ok = abs(r["force_x"]) < 1e-6
    ok_all = ok_all and ok
    r["target"] = 1e-6
    r["pass"] = ok
    results["resting_sphere"] = r
    print(
        f"[{'PASS' if ok else 'FAIL'}] resting_sphere  F = "
        f"({r['force_x']:.3e}, {r['force_y']:.3e}, {r['force_z']:.3e})  "
        f"|F|max = {r['max_abs_force']:.3e}  (n_leaf={r['n_leaf']}, "
        f"bfl_links={r['n_bfl_links']}, target |Fx| < 1e-6)",
        flush=True,
    )

    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {args.output}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
