"""Solver-in-the-loop closure calibration: data -> closure -> solver.

Roadmap step B3 on top of the differentiable bounded path
(``tensorlbm.autograd_calib``): a Reynolds-dependent Smagorinsky closure
``C_s(Re) = c0 (Re/re_ref)^b`` is identified from drag observations at a
few Re by backpropagating through bounded rollouts, then evaluated at a
held-out Re and contrasted with a constant-``C_s`` fit on the same data.

Verification mode: the "observations" are produced by the same solver with
a known truth closure, so recovery is exact-by-construction; the example
demonstrates the machinery end to end (real campaign data enters the same
way as ``DragTarget`` rows).

Regime note (identifiability): the bounded domain is kept at
``tau <= 0.58`` — there the windowed drag responds 13-18% over a 12x
``C_s`` range, so the closure is identifiable; at ``tau >= 0.65`` the
response collapses to 2-7% and calibration from drag is ill-posed.

Usage
-----
    python examples/closure_calibration.py                    # power closure, CPU
    python examples/closure_calibration.py --kind scalar      # constant baseline
    python examples/closure_calibration.py --device cuda --grid gpu   # (10,14,30) Re 40-110
    python examples/closure_calibration.py --iters 40         # faster smoke run
"""

from __future__ import annotations

import argparse

import torch

from tensorlbm.autograd_calib import (
    BoxCase,
    bounded_drag,
    calibrate,
    cs_power,
    evaluate,
    synthetic_targets,
)

GRIDS = {
    # name: (box kwargs, train Re, held-out Re, truth closure)
    "small": (
        dict(nz=6, ny=8, nx=18, radius=2, u_in=0.20, steps=80, window_start=60),
        (30.0, 48.0, 70.0),
        58.0,
        cs_power(0.08, -1.2, re_ref=40.0),
    ),
    "gpu": (
        dict(nz=10, ny=14, nx=30, radius=3, u_in=0.20, steps=300, window_start=200),
        (40.0, 70.0, 110.0),
        90.0,
        cs_power(0.09, -1.0, re_ref=60.0),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kind", choices=("power", "scalar"), default="power")
    parser.add_argument("--grid", choices=tuple(GRIDS), default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iters", type=int, default=110)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--cs0", type=float, default=0.05, help="initial constant guess")
    parser.add_argument(
        "--baseline", action="store_true", help="also fit the other closure kind and contrast"
    )
    parser.add_argument("--seed", type=int, default=0, help="unused: rollouts are deterministic")
    args = parser.parse_args()
    if args.seed:
        torch.manual_seed(args.seed)

    box_kwargs, train_res, held, truth = GRIDS[args.grid]
    box = BoxCase(device=args.device, **box_kwargs)
    print(
        f"box {box.nz}x{box.ny}x{box.nx} r={box.radius} u_in={box.u_in} "
        f"steps={box.steps} window>= {box.window_start}  device={box.device}"
    )
    for re in train_res:
        tau = float(box.tau_of_re(re))
        print(f"  Re {re:5.1f}: tau {tau:.3f}  truth C_s {truth(re):.4f}")

    targets = synthetic_targets(box, train_res, truth)
    print("targets:", ", ".join(f"Re {t.re:g} -> C_D {t.cd:.4f}" for t in targets))

    result = calibrate(
        targets,
        box,
        kind=args.kind,
        cs0=args.cs0,
        iters=args.iters,
        lr=args.lr,
        log_every=max(args.iters // 5, 1),
    )
    loss0, loss1 = result.loss_history[0], result.loss_history[-1]
    if result.kind == "power":
        print(
            f"fit: c0={result.params['c0']:.5f} (at re_ref={result.re_ref:.2f}) "
            f"b={result.params['b']:+.4f}"
        )
    else:
        print(f"fit: C_s={result.params['cs']:.5f}")
    print(f"loss {loss0:.6e} -> {loss1:.6e} (x{loss0 / max(loss1, 1e-300):.1f})")
    for re, row in evaluate(result, targets, box).items():
        print(
            f"  Re {re:>5}: target {row['target']:.4f}  pred {row['pred']:.4f}  "
            f"err {row['rel_err_pct']:.2f}%"
        )

    def rel_err(closure, re: float) -> float:
        cd_true = bounded_drag(box, re=re, cs=truth(re)).item()
        cd_hat = bounded_drag(box, re=re, cs=closure(re)).item()
        return abs(cd_hat - cd_true) / cd_true

    print(f"held-out Re {held:g}: rel err {rel_err(result.closure, held) * 100:.2f}%")

    if args.baseline:
        other = "scalar" if args.kind == "power" else "power"
        base = calibrate(targets, box, kind=other, cs0=args.cs0, iters=args.iters, lr=args.lr)
        ends = (train_res[0], train_res[-1])
        for name, res in ((args.kind, result), (other, base)):
            worst = max(rel_err(res.closure, re) for re in ends)
            print(
                f"  {other if name == args.kind else args.kind} worst endpoint err: {worst * 100:.2f}%"
            )


if __name__ == "__main__":
    main()
