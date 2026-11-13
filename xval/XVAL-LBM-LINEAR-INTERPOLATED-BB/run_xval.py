#!/usr/bin/env python3
"""XVAL-LBM-LINEAR-INTERPOLATED-BB — wall-distance sweep of the 2-point linear
interpolated (Bouzidi-Firdaouss-Lallemand) bounce-back against the exact
plane-Poiseuille solution.

Physics / benchmark
-------------------
Force-driven fully developed plane Poiseuille flow in a channel that is periodic
in ``x`` and bounded by two *straight* solid rows in ``y``.  The no-slip walls
are placed at a controlled sub-grid distance ``delta`` from the first/last fluid
node, so the physical channel width is

    H(delta) = (N - 1) + 2 * delta                     (N = number of fluid rows)

and the exact solution of the Stokes/Poiseuille problem (White, *Viscous Fluid
Flow*, ch. 3; Bouzidi, Firdaouss & Lallemand, Phys. Fluids 13 (2001) 3452
§III.A) is

    u(y) = g / (2 nu) * (H^2/4 - (y - H/2)^2),   Q = g H^3 / (12 nu)

with ``g`` the body force per unit mass along ``x`` and ``nu`` the kinematic
viscosity.  Fluid node ``j = 1..N`` sits at ``y_j = delta + (j - 1)``.

Why this is the right test for *linear* interpolated bounce-back
----------------------------------------------------------------
The half-way bounce-back operator places the no-slip plane half a lattice unit
away from the last fluid node, so for a wall that is *not* half-way located its
effective wall position is displaced by ``1 - 2*delta`` cells (a purely
geometric, resolution-independent offset).  The 2-point linear interpolation
closure (Bouzidi et al. 2001; the ``BFLBounceBack`` / ``AlphaScheme`` family in
OSS LBM codes) reconstructs the reflected population from the true wall
distance, and for a *straight* wall reproduces the no-slip location at
``O(h^2)`` -- this is the claim the external anchors state and the one measured
here.  For curved/inclined (staircase) walls the same operator is only first
order and is *not* claimed to remove the effective offset; that scope boundary
is measured separately in ``docs/STAIRCASE_WALL_OFFSET_20261111.md`` (honest
FAIL of the inclined-wall variant) and is restated in the report.

Two arms, identical solid mask, identical forcing, identical resolution:

``halfway``  repository production operator ``bounce_back_cells`` (staircase
             face, i.e. wall at 0.5).
``interp``   ``tensorlbm.boundary.interpolated_bounce_back_2d`` (2-point
             Bouzidi closure, ``q`` from the analytic signed distance).

Observables (no free parameters, no tuning)
-------------------------------------------
``Q_num``    sum of ``u_x`` over the fluid nodes;
``Q_ref``    the *same discrete sum* of the exact solution sampled at the same
             node coordinates (so no quadrature bias enters the comparison);
``delta_eff`` effective wall offset obtained by inverting the *continuous* flux
             law ``H_eff = (12 nu Q / g)^(1/3)`` on the per-unit-width flux
             (``delta_eff = H_eff - H``); the O(h^2) quadrature bias of this
             inversion is reported in the report (< 1e-3 cells here);
``l2_rel``   ||u_num - u_ref||_2 / ||u_ref||_2 over the fluid nodes, the primary
             accuracy metric of the anchor (< 15 % acceptance);
``max_rel``  max_j |u_num - u_ref| / max_j u_ref.

Usage
-----
    PYTHONPATH=src python3 xval/XVAL-LBM-LINEAR-INTERPOLATED-BB/run_xval.py \
        --out artifacts            # full sweep, writes results.json + table.csv
    ... run_xval.py --quick --out artifacts_quick   # 2 configs, for tests
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.boundaries import bounce_back_cells  # noqa: E402
from tensorlbm.boundary.interpolated_bounce_back import (  # noqa: E402
    interpolated_bounce_back_2d,
)
from tensorlbm.d2q9 import C as C9  # noqa: E402
from tensorlbm.d2q9 import W as W9  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

ANCHOR = "XVAL-LBM-LINEAR-INTERPOLATED-BB"


def setup_channel(n_fluid: int, delta: float, nx: int = 2, n_wall: int = 3):
    """Periodic-in-x channel; walls at ``delta`` cells from the first/last fluid row.

    Node row ``j`` (0 .. N+1) sits at the physical height ``y_j = delta + (j - 1)``
    (``j = 1..N`` are fluid, ``j = 0`` and ``j = N+1`` are the two solid rows), so
    the physical walls ``y = 0`` and ``y = H`` are ``delta`` cells away from the
    outermost fluid nodes and the lattice spacing is uniform.
    """
    ny = n_fluid + 2 * n_wall
    height = (n_fluid - 1) + 2.0 * delta
    j = torch.arange(ny, dtype=torch.float64)
    # fluid row index = n_wall .. n_wall+n_fluid-1, at y_j = delta + (j - n_wall)
    y = delta + (j - float(n_wall))
    dist = torch.minimum(y, height - y)
    fluid1d = torch.zeros(ny, dtype=torch.bool)
    fluid1d[n_wall : n_wall + n_fluid] = True
    sdf1d = torch.where(fluid1d, dist, -dist.abs())
    if bool((sdf1d[fluid1d] <= 0).any()) or bool((sdf1d[~fluid1d] >= 0).any()):
        raise AssertionError("sdf sign convention violated")
    sdf = sdf1d.view(ny, 1).expand(ny, nx).contiguous()
    fluid = fluid1d.view(ny, 1).expand(ny, nx).contiguous()
    return sdf, fluid, ~fluid, y.numpy(), height


def _force_source(fx: torch.Tensor) -> torch.Tensor:
    """Momentum source ``s_i = 3 w_i c_ix F``: ``sum_i s_i c_i = F`` exactly."""
    out = torch.zeros((9,) + fx.shape, dtype=torch.float64)
    for i in range(9):
        out[i] = 3.0 * W9[i] * C9[i, 0] * fx
    return out


def _rest_row(nx: int, dtype=torch.float64) -> torch.Tensor:
    """Rest equilibrium populations at one row (``rho = 1``, ``u = 0``)."""
    w = W9.to(dtype)
    return w.view(9, 1).expand(9, nx).clone()


def stream_open_y(f: torch.Tensor) -> torch.Tensor:
    """Streaming with periodic ``x`` and *non-periodic* (open) ``y``.

    ``tensorlbm.solver.stream`` rolls every axis, so it is periodic in ``y`` too.
    With two solid blocks that wrap into each other, the populations the wall
    operators leave in the *outbound* slots of one wall reach the opposite wall
    and re-enter the fluid there, so the two arms would differ by a wrap artefact
    instead of by their boundary model.  The floor of the wrap (the two outermost
    rows, which are >=2 rows inside the solid blocks) is reset to the rest
    equilibrium so the wrap carries no signal; the fluid is untouched because
    each wall block is three rows thick.
    """
    out = stream(f)
    out[:, 0, :] = _rest_row(out.shape[-1], out.dtype)
    out[:, -1, :] = _rest_row(out.shape[-1], out.dtype)
    return out


def run_arm(
    n_fluid: int,
    delta: float,
    wall: str,
    *,
    tau: float = 0.8,
    u_max_target: float = 0.02,
    nx: int = 2,
    n_wall: int = 3,
    max_steps: int = 60000,
    check_every: int = 100,
    tol: float = 1e-9,
) -> dict:
    """Drive one (arm, resolution, wall distance) point to steady state."""
    nu = (tau - 0.5) / 3.0
    sdf, fluid, solid, y, height = setup_channel(n_fluid, delta, nx, n_wall)
    g = 8.0 * nu * u_max_target / height**2

    ny = n_fluid + 2 * n_wall
    f = equilibrium(
        torch.ones((ny, nx), dtype=torch.float64),
        torch.zeros((ny, nx), dtype=torch.float64),
        torch.zeros((ny, nx), dtype=torch.float64),
    )
    fx = torch.zeros((ny, nx), dtype=torch.float64)
    fx[fluid] = g
    src = _force_source(fx)

    def step(f: torch.Tensor) -> torch.Tensor:
        f_pre = collide_bgk(f, tau) + src
        f_st = stream_open_y(f_pre)
        if wall == "halfway":
            return bounce_back_cells(f_st, solid)
        if wall == "interp":
            return interpolated_bounce_back_2d(f_st, f_pre, sdf, solid)
        raise ValueError(f"unknown wall arm {wall!r}")

    ref = None
    steps = 0
    t0 = time.time()
    converged = False
    while steps < max_steps:
        for _ in range(check_every):
            f = step(f)
        steps += check_every
        _, ux, _ = macroscopic(f)
        prof = ux[fluid[:, 0], 0].clone()
        if ref is not None:
            d = float((prof - ref).abs().max().item())
            if d < tol * u_max_target:
                converged = True
                break
        ref = prof

    _, ux, uy = macroscopic(f)
    u = ux[fluid[:, 0], 0].numpy().astype(float)  # fluid rows, one x-column
    v = uy[fluid].abs().max().item()

    y_j = y[n_wall : n_wall + n_fluid]
    u_ref = g / (2.0 * nu) * (height**2 / 4.0 - (y_j - height / 2.0) ** 2)
    q_num = float(u.sum())
    q_ref = float(u_ref.sum())
    # continuous-flux inversion: per unit width, walls at y = 0 and y = H
    h_eff = (12.0 * nu * q_num / g) ** (1.0 / 3.0)
    l2 = float(torch.linalg.vector_norm(torch.tensor(u - u_ref)).item() / torch.linalg.vector_norm(torch.tensor(u_ref)).item())
    max_rel = float(abs(u - u_ref).max() / u_ref.max())

    return {
        "wall": wall,
        "n_fluid": n_fluid,
        "delta": delta,
        "H": height,
        "nu": nu,
        "tau": tau,
        "g": g,
        "u_max_target": u_max_target,
        "steps": steps,
        "converged": converged,
        "u": [float(x) for x in u],
        "u_ref": [float(x) for x in u_ref],
        "q_num": q_num,
        "q_ref": q_ref,
        "q_rel_err": (q_num - q_ref) / q_ref,
        "delta_eff": h_eff - height,
        "delta_eff_pred_halfway": 1.0 - 2.0 * delta,
        "l2_rel": l2,
        "max_rel": max_rel,
        "max_cross_flow": float(v),
        "wall_time_s": time.time() - t0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("artifacts")))
    ap.add_argument("--deltas", default="0.25,0.5,0.75,0.9")
    ap.add_argument("--ns", default="16,32")
    ap.add_argument("--arms", default="halfway,interp")
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--quick", action="store_true", help="2 configs only (test fixture)")
    args = ap.parse_args(argv)

    deltas = [float(x) for x in args.deltas.split(",")]
    ns = [int(x) for x in args.ns.split(",")]
    arms = args.arms.split(",")
    if args.quick:
        deltas, ns, arms = [0.5, 0.9], [8], ["halfway", "interp"]

    rows = []
    for n_fluid in ns:
        for delta in deltas:
            for wall in arms:
                r = run_arm(n_fluid, delta, wall, tau=args.tau)
                rows.append(r)
                print(
                    f"N={n_fluid:3d} delta={delta:.2f} {wall:8s} "
                    f"H={r['H']:6.2f} delta_eff={r['delta_eff']:+.4f} "
                    f"l2_rel={100 * r['l2_rel']:6.2f}% max_rel={100 * r['max_rel']:6.2f}% "
                    f"Q_err={100 * r['q_rel_err']:+6.2f}% steps={r['steps']} "
                    f"conv={r['converged']} t={r['wall_time_s']:.1f}s",
                    flush=True,
                )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "anchor": ANCHOR,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "setup": {
            "tau": args.tau,
            "nu": (args.tau - 0.5) / 3.0,
            "deltas": deltas,
            "n_fluid": ns,
            "arms": arms,
            "nx": 2,
            "u_max_target": 0.02,
        },
        "cases": rows,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=1))

    hdr = "n_fluid,delta,wall,H,delta_eff,delta_eff_pred_halfway,l2_rel,max_rel,q_rel_err,steps,converged"
    lines = [hdr]
    for r in rows:
        lines.append(
            f"{r['n_fluid']},{r['delta']:.3f},{r['wall']},{r['H']:.4f},"
            f"{r['delta_eff']:.6f},{r['delta_eff_pred_halfway']:+.6f},"
            f"{r['l2_rel']:.6f},{r['max_rel']:.6f},{r['q_rel_err']:.6f},"
            f"{r['steps']},{int(r['converged'])}"
        )
    (out / "table.csv").write_text("\n".join(lines) + "\n")
    print(f"wrote {out/'results.json'} and {out/'table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())