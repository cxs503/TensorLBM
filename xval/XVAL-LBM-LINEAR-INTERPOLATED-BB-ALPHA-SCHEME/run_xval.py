#!/usr/bin/env python3
"""XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME.

Cross-validation of TensorLBM's interpolated bounce-back against the OSS
`alpha`-scheme family, on the exact benchmark the OSS test suite uses:
force-driven, fully developed plane Poiseuille flow between two straight walls
placed a sub-grid distance ``delta`` from the first/last fluid node.

OSS anchors (source read, no code copied)
-----------------------------------------
* ``~/oss-cache/pylbm/pylbm/boundary.py:920-993`` ``LinearBounceBack``:
  ``alpha_hat = -1 + alpha (1/2 + delta)``, ``beta = 1 - alpha delta``,
  ``beta_hat = 1 - alpha/2``, ``K_hat = 1 - alpha_hat`` (``Par*`` subclasses);
  ``BFLBounceBack`` alpha = 2 (delta<=1/2) / 1/delta, ``CLIBounceBack``
  alpha = 4/(1+2 delta) (so ``alpha_hat = 1``), ``YLIBounceBack`` 2/(1+delta).
* ``~/oss-cache/pylbm/tests/test_linear_bounce_back.py``: the channel is
  periodic in x with N fluid nodes, walls ``delta`` outside the first/last one
  (``H = N - 1 + 2 delta``); the scheme is exact when the TRT product is
  ``Lambda+ Lambda- = delta^2 / 2`` with the force added after the collision
  with isotropic weights, and the reported velocity is the stored momentum
  minus ``F/2``.  That exactness statement is the reference of this XVAL.
* ``site-packages/lbmpy/boundaries/boundaryconditions.py:149-230``
  ``NoSlipLinearBouzidi``: same three-branch closure on ``q`` (``q>=1/2``:
  ``(f_q + (2q-1) f_qbar)/(2q)``; ``q<1/2``: ``2q f_q + (1-2q) f*``), i.e. the
  same algebra as BFL, an independent second OSS implementation.

Arms
----
carrier: ``bgk`` (tau = 0.8), ``trt_d2`` (pylbm exactness product
``Lambda+ Lambda- = delta^2/2``), ``trt_c`` (classical ``3/16``), ``trt_off``
(``Lambda+ Lambda- = 0.1``, control that the magic product is what matters).
closure: ``halfway`` (production ``bounce_back_cells``), ``bfl``,
``bfl_par`` (+ post-collision correction), ``cli`` (``alpha_hat = 1``).

Usage
-----
    PYTHONPATH=src python3 xval/XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME/run_xval.py
    ... --quick          # 2 configs, for the CI fixture
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.boundaries import bounce_back_cells  # noqa: E402
from tensorlbm.boundary import interpolated_bounce_back as ibb  # noqa: E402
from tensorlbm.boundary.interpolated_bounce_back import (  # noqa: E402
    interpolated_bounce_back_2d,
)
from tensorlbm.d2q9 import C as C9  # noqa: E402
from tensorlbm.d2q9 import W as W9  # noqa: E402
from tensorlbm.solver import (  # noqa: E402
    collide_bgk,
    collide_trt,
    equilibrium,
    macroscopic,
    stream,
)

ANCHOR = "XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME"
CLOSURES = ("halfway", "bfl", "bfl_par", "cli")
CARRIERS = ("bgk", "trt_d2", "trt_c", "trt_off")


def setup_channel(n_fluid: int, delta: float, nx: int = 2, n_wall: int = 3):
    """Channel periodic in x; walls ``delta`` cells outside the first/last fluid row."""
    ny = n_fluid + 2 * n_wall
    height = (n_fluid - 1) + 2.0 * delta
    j = torch.arange(ny, dtype=torch.float64)
    y = delta + (j - float(n_wall))           # physical height of every row
    fluid1d = torch.zeros(ny, dtype=torch.bool)
    fluid1d[n_wall : n_wall + n_fluid] = True
    dist = torch.minimum(y, height - y)        # distance to the nearest wall
    sdf1d = torch.where(fluid1d, dist, -dist.abs())
    assert bool((sdf1d[fluid1d] > 0).all()) and bool((sdf1d[~fluid1d] < 0).all())
    sdf = sdf1d.view(ny, 1).expand(ny, nx).contiguous()
    fluid = fluid1d.view(ny, 1).expand(ny, nx).contiguous()
    return sdf, fluid, ~fluid, y.numpy(), height, n_wall


def source_linear(fx: torch.Tensor) -> torch.Tensor:
    """``s_i = 3 w_i c_ix F`` (isotropic weights), as in the OSS test scheme."""
    return (3.0 * W9.to(torch.float64)).view(9, 1, 1) * C9[:, 0].to(torch.float64).view(
        9, 1, 1
    ) * fx.unsqueeze(0)


def stream_open_y(f: torch.Tensor) -> torch.Tensor:
    """Streaming periodic in x and open in y (outermost solid rows reset to rest).

    ``solver.stream`` rolls both axes; with two solid blocks the walls would wrap
    into each other and the populations left in one wall's outbound slots would
    re-enter the fluid at the other wall.  Resetting the two outermost rows (>=2
    rows inside each 3-row block) kills that artefact without touching fluid.
    """
    out = stream(f)
    rest = W9.to(f.dtype).view(9, 1).expand(9, out.shape[-1])
    out[:, 0, :] = rest
    out[:, -1, :] = rest
    return out


def trt_params(carrier: str, nu: float, delta: float) -> tuple[float, float] | None:
    """(tau_plus, Lambda) of the TRT carrier; ``None`` for BGK."""
    if carrier == "bgk":
        return None
    lam_p = 3.0 * nu
    if carrier == "trt_d2":
        lam_prod = 0.5 * delta**2            # OSS exactness product
    elif carrier == "trt_c":
        lam_prod = 3.0 / 16.0                # classical half-way magic number
    elif carrier == "trt_off":
        lam_prod = 0.1
    elif carrier.startswith("trt:"):
        lam_prod = float(carrier.split(":", 1)[1])   # explicit magic product (scan)
    else:
        raise ValueError(carrier)
    return 0.5 + lam_p, lam_prod


def run_arm(
    n_fluid: int,
    delta: float,
    carrier: str,
    closure: str,
    *,
    nu: float = 0.1,
    u_max_target: float = 0.02,
    nx: int = 2,
    n_wall: int = 3,
    max_steps: int = 40000,
    check_every: int = 200,
    tol: float = 1e-12,
) -> dict:
    sdf, fluid, solid, y, height, nw = setup_channel(n_fluid, delta, nx, n_wall)
    g = 8.0 * nu * u_max_target / height**2       # body force per unit mass
    tp = trt_params(carrier, nu, delta)
    tau = 0.5 + 3.0 * nu                          # BGK relaxation time

    ny = n_fluid + 2 * n_wall
    f = equilibrium(
        torch.ones((ny, nx), dtype=torch.float64),
        torch.zeros((ny, nx), dtype=torch.float64),
        torch.zeros((ny, nx), dtype=torch.float64),
    )
    fx = torch.zeros((ny, nx), dtype=torch.float64)
    fx[fluid] = g
    src = source_linear(fx)

    def collide(f: torch.Tensor) -> torch.Tensor:
        if tp is None:
            return collide_bgk(f, tau)
        return collide_trt(f, tp[0], tp[1])

    def step(f: torch.Tensor) -> torch.Tensor:
        f_old = f
        f_pre = collide(f) + src                  # post-collision, pre-streaming
        f_st = stream_open_y(f_pre)
        if closure == "halfway":
            return bounce_back_cells(f_st, solid)
        kw = {}
        if closure == "cli":
            kw["alpha"] = ibb.alpha_cli
        elif closure == "bfl_par":
            kw["post_collision_correction"] = True
            kw["f_old"] = f_old
        return interpolated_bounce_back_2d(f_st, f_pre, sdf, solid, **kw)

    ref = None
    steps = 0
    converged = False
    t0 = time.time()
    while steps < max_steps:
        for _ in range(check_every):
            f = step(f)
        steps += check_every
        if steps < 4000:                          # let the transient die first
            continue
        _, ux, _ = macroscopic(f)
        prof = ux[fluid[:, 0], 0].clone()
        if ref is not None:
            if float((prof - ref).abs().max()) < tol * u_max_target:
                converged = True
                break
        ref = prof

    rho, ux, uy = macroscopic(f)
    mom = (rho * ux)[fluid[:, 0], 0].numpy().astype(float)
    u = mom - 0.5 * g                             # OSS convention: momentum - F/2
    v = float(uy[fluid].abs().max())

    yj = y[nw : nw + n_fluid]
    u_ref = g / (2.0 * nu) * yj * (height - yj)
    flow = u_ref > 1e-6 * u_ref.max()
    l2 = float(np.linalg.norm(u - u_ref) / np.linalg.norm(u_ref))
    max_rel = float(np.abs(u - u_ref).max() / np.abs(u_ref).max())
    l2_raw = float(np.linalg.norm(mom - u_ref) / np.linalg.norm(u_ref))
    q_err = float((u.sum() - u_ref.sum()) / u_ref.sum())

    # effective wall position: roots of the fitted parabola (w = 0 <=> wall at delta)
    coef = np.polyfit(yj, u, 2)
    roots = np.sort(np.roots(coef))
    w_bot = float(yj[0] - roots[0])
    w_top = float(roots[1] - yj[-1])
    fit_res = float(np.linalg.norm(np.polyval(coef, yj) - u) / max(np.linalg.norm(u), 1e-300))

    return {
        "n_fluid": n_fluid,
        "delta": delta,
        "carrier": carrier,
        "closure": closure,
        "H": height,
        "nu": nu,
        "tau_plus": None if tp is None else tp[0],
        "lambda_product": None if tp is None else tp[1],
        "g": g,
        "u_max_target": u_max_target,
        "Ma": float(np.sqrt(u_ref.max() ** 2 / (1.0 / 3.0))),
        "Re_H": float(u_ref.max() * height / nu),
        "steps": steps,
        "converged": converged,
        "u": [float(x) for x in u],
        "u_ref": [float(x) for x in u_ref],
        "y": [float(x) for x in yj],
        "l2_rel": l2,
        "max_rel": max_rel,
        "l2_rel_no_half_force_shift": l2_raw,
        "q_err": q_err,
        "w_bot": w_bot,
        "w_top": w_top,
        "fit_res": fit_res,
        "max_cross_flow": v,
        "wall_time_s": time.time() - t0,
        "flow_nodes": int(flow.sum()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("artifacts")))
    ap.add_argument("--deltas", default="0.125,0.25,0.5,0.75,0.9")
    ap.add_argument("--ns", default="8,16")
    ap.add_argument("--carriers", default=",".join(CARRIERS))
    ap.add_argument("--closures", default=",".join(CLOSURES))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    deltas = [float(x) for x in args.deltas.split(",")]
    ns = [int(x) for x in args.ns.split(",")]
    carriers = args.carriers.split(",")
    closures = args.closures.split(",")
    if args.quick:
        deltas, ns, carriers, closures = [0.25, 0.5], [8], ["bgk", "trt_d2"], ["halfway", "bfl_par"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in ns:
        for delta in deltas:
            for carrier in carriers:
                for closure in closures:
                    r = run_arm(n, delta, carrier, closure)
                    rows.append(r)
                    (Path(args.out) / "partial.json").write_text(json.dumps(rows, indent=1))
                    print(
                        f"N={n:3d} d={delta:5.3f} {carrier:8s} {closure:8s} "
                        f"lam={r['lambda_product'] if r['lambda_product'] is not None else float('nan'):.4f} "
                        f"l2={100*r['l2_rel']:9.3e}% max={100*r['max_rel']:9.3e}% "
                        f"w=({r['w_bot']:+.3f},{r['w_top']:+.3f}) fit={r['fit_res']:.1e} "
                        f"conv={int(r['converged'])} t={r['wall_time_s']:.1f}s",
                        flush=True,
                    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "anchor": ANCHOR,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch": torch.__version__,
        "setup": {
            "nu": 0.1,
            "u_max_target": 0.02,
            "nx": nx if (nx := 2) else 2,
            "deltas": deltas,
            "n_fluid": ns,
            "carriers": carriers,
            "closures": closures,
            "history": "kernel-equivalence self-check: bfl == pre-mission operator "
            "(max 5.6e-17), trt via solver.collide_trt",
        },
        "cases": rows,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=1))
    hdr = ("n_fluid,delta,carrier,closure,lambda_product,l2_rel,max_rel,q_err,"
           "w_bot,w_top,fit_res,steps,converged")
    lines = [hdr]
    for r in rows:
        lam = "" if r["lambda_product"] is None else f"{r['lambda_product']:.6f}"
        lines.append(
            f"{r['n_fluid']},{r['delta']:.4f},{r['carrier']},{r['closure']},{lam},"
            f"{r['l2_rel']:.6e},{r['max_rel']:.6e},{r['q_err']:.6e},"
            f"{r['w_bot']:.6f},{r['w_top']:.6f},{r['fit_res']:.3e},{r['steps']},{int(r['converged'])}"
        )
    (out / "table.csv").write_text("\n".join(lines) + "\n")
    print(f"wrote {out/'results.json'} and {out/'table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
