#!/usr/bin/env python
"""Run one flapping-foil LBM case and dump JSON + CSV (node P162W1-N07).

Example::

    python run_case.py --label k0.5_heave --chord 32 --k 0.5 --h0c 0.02 \
        --a0 0 --phi 0 --outdir artifacts_p162n07

The JSON contains the measured force/power coefficients, the Garrick closed
form at the same parameters, the cycle-resolved time series summary, the
energy budget and (for locked cases) an independent control-volume momentum
balance.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensorlbm import flapping_lbm_2d as M  # noqa: E402
from tensorlbm.flapping_propulsion import (  # noqa: E402
    garrick_coefficients,
    kinematics_diagnostics,
)


def momentum_balance(f: np.ndarray, U: float) -> dict:
    """Control-volume force from the flux deficit between inlet and outlet."""
    rho, ux, uy = M.momenta(f)
    # 2D, per unit span; pressure p = rho/3
    p = rho / 3.0
    i_in, i_out = 2, f.shape[2] - 3
    mom_in = float(np.sum(rho[:, i_in] * ux[:, i_in] ** 2 + p[:, i_in]))
    mom_out = float(np.sum(rho[:, i_out] * ux[:, i_out] ** 2 + p[:, i_out]))
    return {
        "momentum_in": mom_in,
        "momentum_out": mom_out,
        "force_on_fluid_x": mom_out - mom_in,
        "drag_on_body_x": mom_in - mom_out,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--chord", type=int, default=32)
    ap.add_argument("--re", type=float, default=500.0)
    ap.add_argument("--U", type=float, default=0.15)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--h0c", type=float, default=0.02)
    ap.add_argument("--a0", type=float, default=2.0)
    ap.add_argument("--phi", type=float, default=90.0)
    ap.add_argument("--axis", type=float, default=-0.5)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--locked", action="store_true")
    ap.add_argument("--no-ibm", action="store_true")
    ap.add_argument("--phase-offset", type=float, default=0.0)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--periodic-x", action="store_true")
    ap.add_argument("--snapshots", type=int, default=0)
    ap.add_argument("--outdir", default="artifacts_p162n07")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    cfg = M.LBMConfig(
        chord=args.chord,
        re=args.re,
        U=args.U,
        h0_over_c=args.h0c,
        alpha0_deg=args.a0,
        k=args.k,
        phi_deg=args.phi,
        pitch_axis_a=args.axis,
        n_cycles=args.cycles,
        warmup_cycles=args.warmup,
        locked=args.locked,
        ibm=not args.no_ibm,
        phase_offset_deg=args.phase_offset,
        n_forcing_passes=args.passes,
        periodic_x=args.periodic_x,
    )
    csv_path = os.path.join(args.outdir, f"{args.label}.csv")
    solver = M.FlappingFoil2D(cfg)
    res = solver.run(
        csv_path=csv_path,
        collect_fields_every=(cfg.steps_per_cycle() // 4 if args.snapshots else 0),
    )
    avg = M.cycle_average(res, cfg.U, cfg.chord)

    # theory at the same parameters (Garrick 1936 / Theodorsen 1935)
    kin = cfg.kinematic
    theory = {}
    if (not args.locked) and (cfg.h0_over_c > 0.0 or cfg.alpha0_deg > 0.0):
        g = garrick_coefficients(kin, cfg.k)
        theory = {
            "C_T": g.C_T,
            "C_S": g.C_S,
            "C_T_tilt": g.C_T_tilt,
            "C_P": g.C_P,
            "C_P_loss": g.C_P_loss,
            "eta": g.eta,
            "F": g.C.real,
            "G": g.C.imag,
            "absC": abs(g.C),
        }

    # energy budget over the averaging window
    rows = res["rows"]
    W = float(np.sum([r["W"] for r in rows]))
    dKE = float(rows[-1]["ke"] - rows[0]["ke"])
    phi = float(np.sum([r["phi"] for r in rows]))
    phi_neq = float(np.sum([r.get("phi_neq", 0.0) for r in rows]))
    flux = float(np.sum([r["flux"] for r in rows]))
    denom = abs(W) if abs(W) > 0 else 1.0
    budget = {
        "W_body": W,
        "dKE": dKE,
        "phi_visc": phi,
        "phi_neq": phi_neq,
        "flux_out": flux,
        "residual": W - dKE - phi - flux,
        "relative_residual": abs(W - dKE - phi - flux) / denom,
        "n_steps": int(rows[-1]["step"] - rows[0]["step"]),
    }
    energy_per_step = {
        "W_body": W / budget["n_steps"],
        "dKE": dKE / budget["n_steps"],
        "phi_visc": phi / budget["n_steps"],
        "flux_out": flux / budget["n_steps"],
        "relative_residual": abs(
            W / budget["n_steps"] - dKE / budget["n_steps"] - phi / budget["n_steps"]
            - flux / budget["n_steps"]
        ) / max(abs(W / budget["n_steps"]), 1e-30),
    }

    out = {
        "label": args.label,
        "config": res["config"],
        "dims": res["dims"],
        "tau": res["tau"],
        "nu": res["nu"],
        "period": res["period"],
        "kinematics": kinematics_diagnostics(kin, cfg.k),
        "measured": avg,
        "theory": theory,
        "energy_budget": budget,
        "energy_budget_per_step": energy_per_step,
        "csv": os.path.abspath(csv_path),
        "csv_sha256": M.file_sha256(csv_path),
    }

    # independent control-volume check (only meaningful without motion)
    if args.locked or args.no_ibm:
        rho, ux, uy = solver.macros()
        out["momentum_balance"] = momentum_balance(solver.f, cfg.U)

    # snapshots for the report figures
    if args.snapshots and res["snapshots"]:
        snaps = res["snapshots"]
        npz = os.path.join(args.outdir, f"{args.label}_fields.npz")
        np.savez_compressed(
            npz,
            t_over_T=np.array([s["t_over_T"] for s in snaps]),
            ux=np.array([s["ux"] for s in snaps]),
            uy=np.array([s["uy"] for s in snaps]),
            ny=solver.ny,
            nx=solver.nx,
            U=cfg.U,
        )
        out["fields_npz"] = os.path.abspath(npz)

    path = os.path.join(args.outdir, f"{args.label}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("label", "measured", "theory") if k in out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())