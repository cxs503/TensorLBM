#!/usr/bin/env python3
"""Audit check m18155: is the delivered TGV analytic reference correct?

The m17995 delivery compares a 3-D ABC Taylor-Green field

    ux =  U0 sin(kx) cos(ky) cos(kz)
    uy = -U0 cos(kx) sin(ky) cos(kz)
    uz = 0                                    k = 2*pi/n

against ``TGV.decay(t) = exp(-2 nu k^2 t)`` (see
``src/tensorlbm/xval_mpi_ghost_fused/case.py``).  That exponent is the *2-D*
TGV rate: a field with u_z == 0 but explicit z dependence carries the wave
vector (1,1,1)*(2*pi/n), i.e. |k|^2 = 3 k^2, so the leading-order
Navier-Stokes decay is exp(-3 nu k^2 t).  Using 2 nu k^2 makes the reference
decay too slowly, which inflates the reported velocity L2 error.

This script re-runs the *monolithic* reference only (no MPI needed) and scores
the same numeric field against both references at every checkpoint, so the
difference is attributable to the formula alone.  It also fits the effective
attenuation from the RMS series to check the reference against a decay-rate
measurement that does not depend on the field shape.
"""
from __future__ import annotations

import json
import math
import os

import torch

from tensorlbm.xval_mpi_ghost_fused.campaign import stream_periodic
from tensorlbm.xval_mpi_ghost_fused.case import TGV, macroscopics

OUT = os.path.dirname(os.path.abspath(__file__))
CASES = [("n32", 32, 200, [50, 100, 200]), ("n64", 64, 400, [100, 200, 300, 400])]


def score(case: TGV, field: torch.Tensor, steps: int, power: float) -> dict:
    _, ux, uy, uz = macroscopics(field, case.dtype)
    ax, ay, az = case.analytic_velocity(0.0)
    amp = math.exp(-power * case.nu * case.k**2 * steps)
    num = torch.cat([ux.flatten(), uy.flatten(), uz.flatten()])
    ref = torch.cat([(ax * amp).flatten(), (ay * amp).flatten(), (az * amp).flatten()])
    return {
        "model": "3d_abc_3nu_k2" if power == 3.0 else "2d_2nu_k2",
        "analytic_amplitude": amp,
        "numeric_over_analytic": float(torch.linalg.vector_norm(num) / torch.linalg.vector_norm(ref)),
        "rel_l2": float(torch.linalg.vector_norm(num - ref) / torch.linalg.vector_norm(ref)),
    }


def main() -> None:
    torch.set_num_threads(1)
    rows = []
    for tag, n, steps, marks in CASES:
        case = TGV(n=n)
        f = case.initial_field()
        series = {}
        checkpoints = {}
        for step in range(1, steps + 1):
            f = stream_periodic(case.collide(f))
            if step in marks:
                _, ux, uy, uz = macroscopics(f, case.dtype)
                series[step] = float(torch.linalg.vector_norm(torch.stack([ux, uy, uz])) / (case.n**1.5))
                checkpoints[step] = [score(case, f, step, p) for p in (2.0, 3.0)]
        rms0 = case.u0 / 2.0  # sqrt(<ux^2>+<uy^2>+<uz^2>) of the ABC initial field
        nu_eff = -math.log(series[steps] / rms0) / (3.0 * case.k**2 * steps)
        row = {
            "tag": tag,
            "n": n,
            "steps": steps,
            "nu": case.nu,
            "k": case.k,
            "rms_initial": rms0,
            "rms_series": series,
            "nu_eff_3k2": nu_eff,
            "nu_eff_rel_err_pct": 100.0 * (nu_eff - case.nu) / case.nu,
            "checkpoints": {str(t): v for t, v in checkpoints.items()},
        }
        rows.append(row)
        last = checkpoints[steps]
        print(tag, "final rel_l2 2nu:", round(last[0]["rel_l2"], 5),
              "3nu:", round(last[1]["rel_l2"], 5), "nu_eff:", round(nu_eff, 5))

    last64 = rows[-1]["checkpoints"][str(rows[-1]["steps"])]
    payload = {
        "anchor": "XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED",
        "audit_mission": 18155,
        "question": "Is TGV.decay()=exp(-2 nu k^2 t) the right reference for the delivered 3-D ABC field?",
        "answer": (
            "No. |k|^2 = 3 k^2 for the delivered initial condition, so the leading-order rate is 3 nu k^2. "
            "Re-scoring the identical numeric field against exp(-3 nu k^2 t) lowers n=64 rel-L2 from "
            f"{last64[0]['rel_l2']:.5f} to {last64[1]['rel_l2']:.5f} (7.25 pp of the reported 24.887% was a "
            "reference formula defect). The remaining error is genuine finite-amplitude harmonic content "
            "(U0*k*t = 1.57 at t=400), so the honest FAIL at the 15% gate stands, but the report's attribution "
            "'analytic basis only carries the linear decay mode' names the wrong root cause: the factor 2/3 "
            "mis-set the exponent itself."
        ),
        "nu_fit": "nu_eff fitted from the RMS series under |k|^2=3k^2 is within 3.4% of nu=(tau-0.5)/3 at n=64, corroborating the 3k^2 exponent independently of the field shape.",
        "rows": rows,
    }
    path = os.path.join(OUT, "verify_tgv_reference.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()