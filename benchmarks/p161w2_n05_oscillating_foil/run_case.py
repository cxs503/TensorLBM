#!/usr/bin/env python
"""Run one oscillating-foil LBM case and dump a self-describing JSON payload.

Node ``p161w2-arb-n05``.  Every number that later enters the acceptance report
is produced here (nothing is typed by hand into the report).

Example::

    python run_case.py --label heave_h025_k1 --mode heave --k 1.0 --h0c 0.25

The payload contains

* the case configuration and grid,
* cycle-averaged coefficients ``C_T, C_L, C_P, eta`` (LBM),
* the same coefficients from the Garrick closed form (analytic anchor),
* the energy ledger (wall work / dKE / dissipation, two estimators / boundary
  flux),
* an independent control-volume drag estimate for locked runs,
* a SHA-256 of the payload (bit-level reproducibility, criterion C8).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "src")
sys.path.insert(0, SRC)

from tensorlbm.flapping_propulsion import FlappingKinematics, garrick_coefficients  # noqa: E402
from tensorlbm.oscillating_foil_lbm import (  # noqa: E402
    OscillatingFoil2D,
    OscillatingFoilConfig,
    cycle_statistics,
    dissipation_rate,
)


def independent_drag(solver: OscillatingFoil2D) -> dict:
    """Control-volume streamwise force on a *locked* body (steady state).

    Steady momentum balance over the box delimited by the inlet plane ``x1`` and
    a downstream plane ``x2`` (free-slip lateral walls contribute nothing):

        F_on_body = sum_y [rho u^2 + p]_{x1} - sum_y [rho u^2 + p]_{x2}

    This is independent of the immersed-boundary force integration, so it
    catches a broken IBM force accounting.
    """
    rho, ux, uy = __import__("tensorlbm.flapping_lbm_2d", fromlist=["momenta"]).momenta(
        solver.f
    )
    p = rho / 3.0
    x1 = 2
    x2 = solver.nx - 3
    mom1 = float(np.sum(rho[:, x1] * ux[:, x1] ** 2 + p[:, x1]))
    mom2 = float(np.sum(rho[:, x2] * ux[:, x2] ** 2 + p[:, x2]))
    q = 0.5 * solver.cfg.U**2 * solver.cfg.chord
    return {"drag_force": mom1 - mom2, "C_D_cv": (mom1 - mom2) / q}


def payload_hash(payload: dict) -> str:
    body = {k: v for k, v in payload.items() if k != "sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def run(args) -> dict:
    cfg = OscillatingFoilConfig(
        chord=args.chord,
        re=args.re,
        U=args.U,
        h0_over_c=args.h0c,
        alpha0_deg=args.a0,
        k=args.k,
        phi_deg=args.phi,
        mode=args.mode,
        n_cycles=args.cycles,
        warmup_cycles=args.warmup,
        periodic_x=args.periodic,
        periodic_y=args.periodic,
        solid_interior=not args.no_solid,
        kernel=args.kernel,
        n_forcing_passes=args.passes,
        diag_stride=args.diag_stride,
        marker_spacing=args.marker_spacing,
    )
    if args.noibm:
        # negative control: no immersed body at all
        cfg.mode = "locked"
    solver = OscillatingFoil2D(cfg)
    if args.noibm:
        solver.step = _step_without_ibm(solver)  # type: ignore[assignment]

    per = cfg.steps_per_cycle()
    rows: list[dict] = []
    t0 = time.time()
    for _ in range(cfg.total_steps()):
        rows.append(solver.step(diagnostics=True))
    dt = time.time() - t0

    stats = cycle_statistics(rows, cfg)
    kin = FlappingKinematics(
        U=cfg.U,
        c=float(cfg.chord),
        h0=cfg.motion.h0,
        alpha0=cfg.motion.alpha0,
        phi=cfg.motion.phi,
        a=cfg.motion.a,
    )
    theory = {}
    if cfg.mode != "locked":
        g = garrick_coefficients(kin, cfg.k)
        theory = {
            "C_T": g.C_T,
            "C_S": g.C_S,
            "C_T_tilt": g.C_T_tilt,
            "C_P": g.C_P,
            "eta": g.eta,
            "F": g.C.real,
            "G": g.C.imag,
            "absC": abs(g.C),
            "strouhal": kin.strouhal(cfg.k),
        }
    else:
        theory = {"C_T": 0.0, "C_P": 0.0, "eta": float("nan")}

    # energy ledger over the averaging window (integer number of cycles)
    use = [r for r in rows if r["step"] > 0]
    n_use = (len(use) // per) * per
    win = use[-n_use:] if n_use >= per else use
    work = float(np.sum([r["body_work"] for r in win]))
    phi_cont = float(np.sum([r["phi"] for r in win]))
    flux = float(np.sum([r["flux"] for r in win]))
    dke = float(win[-1]["ke"] - win[0]["ke"])
    rho, ux, uy = __import__("tensorlbm.flapping_lbm_2d", fromlist=["momenta"]).momenta(
        solver.f
    )
    from tensorlbm.flapping_lbm_2d import dissipation_neq  # noqa: PLC0415

    phi_neq = float(dissipation_neq(solver.f, cfg.tau))

    extra = {}
    if cfg.mode == "locked":
        extra["independent"] = independent_drag(solver)

    stride = max(1, per // args.samples_per_cycle)
    series = [
        {
            "step": r["step"],
            "Fx_fluid": r["Fx_fluid"],
            "Fy_fluid": r["Fy_fluid"],
            "body_work": r["body_work"],
        }
        for r in rows[::stride]
    ]

    payload = {
        "label": args.label,
        "node": "p161w2-arb-n05",
        "config": {
            "chord": cfg.chord,
            "re": cfg.re,
            "U": cfg.U,
            "k": cfg.k,
            "h0_over_c": cfg.h0_over_c,
            "alpha0_deg": cfg.alpha0_deg,
            "phi_deg": cfg.phi_deg,
            "mode": cfg.mode,
            "periodic": bool(cfg.periodic_x and cfg.periodic_y),
            "kernel": cfg.kernel,
            "n_forcing_passes": cfg.n_forcing_passes,
            "solid_interior": cfg.solid_interior,
            "marker_spacing": cfg.marker_spacing,
            "n_cycles": cfg.n_cycles,
            "warmup_cycles": cfg.warmup_cycles,
            "noibm_control": bool(args.noibm),
        },
        "grid": [solver.nx, solver.ny],
        "tau": cfg.tau,
        "nu": cfg.nu,
        "steps": cfg.total_steps(),
        "steps_per_cycle": per,
        "ms_per_step": 1000.0 * dt / max(1, cfg.total_steps()),
        "coeffs": stats,
        "theory": theory,
        "energy": {
            "work": work,
            "dke": dke,
            "phi_cont": phi_cont,
            "phi_neq_total": phi_neq,
            "flux": flux,
            "cycles": n_use / per,
            "rel_residual_cont": abs(work - (dke + phi_cont + flux))
            / max(abs(work), abs(phi_cont), 1e-30),
        },
        "extra": extra,
        "series": series,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    payload["sha256"] = payload_hash(payload)
    return payload


def _step_without_ibm(solver: OscillatingFoil2D):
    """Return a ``step`` bound method identical to the solver's minus the IBM."""

    def step(diagnostics: bool = True):
        orig = solver.cfg.solid_interior
        solver.cfg.solid_interior = False
        try:
            # temporarily replace the marker set with an empty one
            bx, by = solver._bx, solver._by
            solver._bx = np.zeros(0)
            solver._by = np.zeros(0)
            out = OscillatingFoil2D.step(solver, diagnostics=diagnostics)
        finally:
            solver._bx, solver._by = bx, by
            solver.cfg.solid_interior = orig
        out["Fx_fluid"] = 0.0
        out["Fy_fluid"] = 0.0
        out["body_work"] = 0.0
        return out

    return step


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--mode", default="coupled", choices=["coupled", "pitch", "heave", "locked"])
    ap.add_argument("--chord", type=int, default=40)
    ap.add_argument("--re", type=float, default=500.0)
    ap.add_argument("--U", type=float, default=0.1)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--h0c", type=float, default=0.25)
    ap.add_argument("--a0", type=float, default=10.0)
    ap.add_argument("--phi", type=float, default=90.0)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--kernel", default="4pt", choices=["4pt", "hat"])
    ap.add_argument("--marker-spacing", type=float, default=0.5)
    ap.add_argument("--diag-stride", type=int, default=1)
    ap.add_argument("--samples-per-cycle", type=int, default=64)
    ap.add_argument("--periodic", action="store_true")
    ap.add_argument("--no-solid", action="store_true")
    ap.add_argument("--noibm", action="store_true")
    ap.add_argument("--outdir", default="artifacts")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    payload = run(args)
    out = os.path.join(args.outdir, f"{args.label}.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)
    c = payload["coeffs"]
    print(
        f"{args.label:24s} grid={payload['grid']} C_T={c['C_T']:+.4f} "
        f"C_P={c['C_P']:+.4f} eta={c['eta']:+.3f} "
        f"th_C_T={payload['theory']['C_T']:+.4f} ms/step={payload['ms_per_step']:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())