#!/usr/bin/env python
"""One-point contact-angle probe (used by the shell sweep driver)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "sc_surface_tension"))

import torch  # noqa: E402

from tensorlbm.sc_surface_tension import contact_angle_from_mask, psi_exp  # noqa: E402
from run import _jsonable, run_contact_angle  # noqa: E402
from contact_sweep import correct, synthetic_cap_bias  # noqa: E402


def psi_of_rho(rho_w: float) -> float:
    return float(psi_exp(torch.tensor(float(rho_w))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho-w", type=float, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ny", type=int, default=72)
    ap.add_argument("--nx", type=int, default=112)
    ap.add_argument("--R0", type=float, default=18.0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    psi_w = psi_of_rho(args.rho_w)
    pl, pg = psi_of_rho(1.957), psi_of_rho(0.1596)
    theta_young = math.degrees(math.acos(max(-1.0, min(1.0, (psi_w - pg) / (pl - pg)))))
    res = run_contact_angle(theta_young, R0=args.R0, ny=args.ny, nx=args.nx,
                            steps=args.steps, sample=args.steps, verbose=not args.quiet,
                            wall_override=psi_w)
    table = synthetic_cap_bias(ny=args.ny, nx=args.nx)
    # mask-based (linear-fit) estimator -- the primary instrument
    rho = torch.tensor(res["rho_field"])
    solid = torch.zeros((args.ny, args.nx), dtype=torch.bool)
    solid[:2, :] = True
    solid[args.ny - 2:, :] = True
    qm = contact_angle_from_mask(rho, solid)
    row = {
        "rho_w": args.rho_w,
        "psi_w": psi_w,
        "theta_young_deg": theta_young,
        "theta_measured_deg": res["theta_measured"],
        "theta_mask_deg": qm["theta_deg"],
        "theta_mask_corrected_deg": correct(qm["theta_deg"], table),
        "R_mask": qm["R_fit"],
        "z_mask": qm["z_c"],
        "n_rows_mask": qm["n_rows"],
        "status_mask": qm["status"],
        "theta_corrected_deg": correct(res["theta_measured"], table),
        "R_fit": res["R_fit"],
        "z_c": res["z_c"],
        "n_contour": res["n_contour"],
        "mass_drift": res["mass_drift"],
        "steps": res["steps"],
        "status": res.get("status", "ok"),
    }
    print("RESULT " + json.dumps(_jsonable(row)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())