#!/usr/bin/env python
"""Run one arm of XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI.

Usage
-----
    PYTHONPATH=src python xval/XVAL-LBM-ZOU-HE-.../case.py --arm pressure-pressure \
        --height 32 --length 160 --steps 120000 --out artifacts/case_H32_pp.json

Writes a JSON record (metrics + diagnostics + profiles) and, when ``--npz`` is
given, the time-averaged velocity/density fields for figure generation.  All
numbers in the platform report are read back from these JSON records, never
transcribed by hand.
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

from tensorlbm.verification.zou_he_channel import ChannelCase, run_channel  # noqa: E402
from tensorlbm.verification.zou_he_channel.solver import (  # noqa: E402
    analytic_profile,
    profile_metrics,
)

ARMS = ("pressure-pressure", "velocity-pressure", "equilibrium-pressure")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--tau", type=float, default=0.65)
    p.add_argument("--drho", type=float, default=4.0e-3)
    p.add_argument("--steps", type=int, default=120000)
    p.add_argument("--burn-in", type=int, default=20000)
    p.add_argument("--avg-window", type=int, default=2000)
    p.add_argument("--ramp", type=int, default=5000)
    p.add_argument("--tol", type=float, default=1.0e-6)
    p.add_argument("--init", default="analytic", choices=("analytic", "uniform"))
    p.add_argument("--out", required=True)
    p.add_argument("--npz", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    case = ChannelCase(
        height=args.height,
        length=args.length,
        tau=args.tau,
        drho=args.drho,
        arm=args.arm,
        init=args.init,
        tol=args.tol,
        max_steps=args.steps,
        check_every=max(500, args.avg_window),
        burn_in=args.burn_in,
        avg_window=args.avg_window,
        ramp_steps=args.ramp,
    )
    t0 = time.time()
    res = run_channel(case, verbose=True)
    wall = time.time() - t0

    ux = res["ux"].numpy()
    rho = res["rho"].numpy()
    ny, nx = ux.shape
    fluid = slice(1, ny - 1)

    # profiles: mid-channel (developed), two stations for a flatness check
    stations = {"x0.25": nx // 4, "x0.5": nx // 2, "x0.75": 3 * nx // 4}
    prof = {name: ux[fluid, x].tolist() for name, x in stations.items()}

    g = res["g_measured"]
    m_meas = profile_metrics(case, res["u_profile"], g)
    m_nom = profile_metrics(case, res["u_profile"], case.g_nominal)
    u_ref_meas = analytic_profile(case, g).numpy()

    record = {
        "arm": args.arm,
        "height": case.height,
        "length": case.nx,
        "ny": ny,
        "nx": nx,
        "tau": case.tau,
        "nu": case.nu,
        "drho": case.drho,
        "init": case.init,
        "ramp_steps": case.ramp_steps,
        "burn_in": case.burn_in,
        "avg_window": case.avg_window,
        "steps": res["steps"],
        "navg": res["navg"],
        "converged": bool(res["converged"]),
        "wall_seconds": wall,
        "g_measured": g,
        "g_nominal": case.g_nominal,
        "g_rel_error": abs(g - case.g_nominal) / case.g_nominal,
        "rho_in_mean": float(rho[fluid, 0].mean()),
        "rho_out_mean": float(rho[fluid, -1].mean()),
        "rho_mean": float(rho[fluid, :].mean()),
        "mass_defect": float(res["mass_defect"]),
        "q_in": float(res["q_col"][0]),
        "q_out": float(res["q_col"][-1]),
        "q_interior_mean": float(res["q_col"][1:-1].mean()),
        "q_interior_spread": float(
            (res["q_col"][1:-1].max() - res["q_col"][1:-1].min())
            / max(abs(res["q_col"][1:-1].mean()), 1e-30)
        ),
        "u_max_nominal": case.u_max_nominal,
        "metrics_vs_measured_gradient": m_meas,
        "metrics_vs_nominal_gradient": m_nom,
        "profile_mid": res["u_profile"].tolist(),
        "profile_stations": prof,
        "profile_analytic_measured_g": u_ref_meas.tolist(),
        "y_fluid": list(range(1, ny - 1)),
        "ripple": {
            "ux_field_max": float(np.abs(ux[fluid, :]).max()),
            "u_at_station_std_over_x": float(
                np.std(ux[fluid, nx // 4: 3 * nx // 4], axis=1).mean()
            ),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(record, fh, indent=1)
    if args.npz:
        np.savez_compressed(
            args.npz,
            ux=ux,
            uy=res["uy"].numpy(),
            rho=rho,
            q_col=res["q_col"].numpy(),
            y=np.arange(ny),
            x=np.arange(nx),
            arm=args.arm,
            height=case.height,
        )
    print(
        f"[{args.arm} H={case.height} nx={case.nx}] steps={res['steps']} "
        f"rel_linf={m_meas['rel_linf']:.3e} rel_umax={m_meas['rel_umax']:.3e} "
        f"shape={m_meas['shape_rel_l2']:.3e} mass_defect={res['mass_defect']:.2e} "
        f"wall={wall:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())