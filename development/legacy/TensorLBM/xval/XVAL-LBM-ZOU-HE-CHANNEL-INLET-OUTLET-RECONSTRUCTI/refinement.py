#!/usr/bin/env python
"""Grid-refinement (h-) study of the Zou-He channel closure.

The campaign in ``case.py`` reports one number per (arm, H).  A single number at
a single resolution cannot tell a converged discretisation error from an
unlucky transient, and all of those runs ended on ``max_steps`` with the block
drift above tolerance (``converged = False``).  This driver answers the two
questions that were left open:

1. *Is the steady state reached?*  Every level is marched until the drift of the
   running time average between consecutive blocks falls below ``--tol``, or
   ``--steps`` is exhausted; the drift history is recorded so the residual can be
   inspected instead of assumed.
2. *Does the discretisation error converge, and at what order?*  With a fixed
   aspect ratio ``L/H = 8`` the exact plane-Poiseuille profile is the same
   function of ``yt/H`` at every level, so the observed order of accuracy
   ``p = log(e_coarse/e_fine) / log(h_coarse/h_fine)`` is meaningful, and a
   Richardson extrapolation gives the h -> 0 limit of the centre-line velocity.

Usage
-----
    PYTHONPATH=src python .../refinement.py --height 32 --length 256 \
        --steps 40000 --tol 1e-5 --burn-in 12000 --avg-window 8000 \
        --out artifacts/xval_zouhe/refine_H32.json

All metrics in ``REFINEMENT.md`` are read back from these records.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch  # noqa: E402

from tensorlbm.verification.zou_he_channel import (  # noqa: E402
    ChannelCase,
    analytic_profile,
    run_channel,
)
from tensorlbm.verification.zou_he_channel.solver import profile_metrics  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--tau", type=float, default=0.65)
    p.add_argument("--drho", type=float, default=4.0e-3)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--burn-in", type=int, required=True)
    p.add_argument("--avg-window", type=int, required=True)
    p.add_argument("--tol", type=float, default=1.0e-5)
    p.add_argument("--arm", default="pressure-pressure")
    p.add_argument("--init", default="analytic", choices=("analytic", "uniform"))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    case = ChannelCase(
        height=args.height,
        length=args.length,
        tau=args.tau,
        drho=args.drho,
        arm=args.arm,
        init=args.init,
        tol=args.tol,
        max_steps=args.steps,
        check_every=max(500, args.avg_window // 2),
        burn_in=args.burn_in,
        avg_window=args.avg_window,
        ramp_steps=0,
    )
    t0 = time.time()
    res = run_channel(case, verbose=False)
    wall = time.time() - t0

    ux = res["ux"]
    rho = res["rho"]
    interior = slice(1, case.ny - 1)
    g_meas = res["g_measured"]

    m_meas = profile_metrics(case, res["u_profile"], g_meas)
    m_nom = profile_metrics(case, res["u_profile"], case.g_nominal)

    # semi-local (mid-channel) profile, normalised by H, for a cross-level overlay
    yt = (np.arange(1, case.ny - 1) - 0.5) / float(case.height)
    u_prof = res["u_profile"].numpy()
    u_ref = analytic_profile(case, g_meas).numpy()

    # how much of the total error is a *bulk* amplitude offset (the g-fit channel)
    # vs a genuine shape defect: the shape-only metric is the conservative one.
    q = res["q_col"].numpy()
    record = {
        "arm": args.arm,
        "init": case.init,
        "height": case.height,
        "length": case.nx,
        "ny": case.ny,
        "nx": case.nx,
        "aspect": case.nx / float(case.height),
        "h": 1.0 / case.height,
        "tau": case.tau,
        "nu": case.nu,
        "drho": case.drho,
        "tol": case.tol,
        "max_steps": case.max_steps,
        "steps": res["steps"],
        "burn_in": case.burn_in,
        "avg_window": case.avg_window,
        "navg": res["navg"],
        "converged": bool(res["converged"]),
        "drift_hist": res["drift_hist"],
        "drift_steps": res["drift_steps"],
        "drift_last": res["drift_last"],
        "drift_profile_hist": res["drift_profile_hist"],
        "drift_profile_last": res["drift_profile_last"],
        "wall_seconds": wall,
        "g_measured": g_meas,
        "g_nominal": case.g_nominal,
        "g_rel_error": abs(g_meas - case.g_nominal) / case.g_nominal,
        "u_max_sim": m_meas["u_max_sim"],
        "u_max_ref": m_meas["u_max_ref"],
        "metrics_vs_measured_gradient": m_meas,
        "metrics_vs_nominal_gradient": m_nom,
        "mass_defect": float(res["mass_defect"]),
        "q_in": float(q[0]),
        "q_out": float(q[-1]),
        "q_interior_mean": float(q[1:-1].mean()),
        "q_interior_spread": float((q[1:-1].max() - q[1:-1].min()) / abs(q[1:-1].mean())),
        "yt_over_H": yt.tolist(),
        "u_profile_sim": u_prof.tolist(),
        "u_profile_exact": u_ref.tolist(),
        "rho_mean": float(rho[interior, :].mean()),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(record, fh, indent=1)

    print(
        f"[H={case.height} nx={case.nx}] steps={res['steps']} "
        f"converged={res['converged']} drift={res['drift_last']:.2e} "
        f"drift_prof={res['drift_profile_last']:.2e} "
        f"rel_linf={m_meas['rel_linf']:.3e} shape={m_meas['shape_rel_l2']:.3e} "
        f"mass_defect={res['mass_defect']:.2e} wall={wall:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())