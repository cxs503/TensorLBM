#!/usr/bin/env python
"""Capture vorticity/velocity snapshots of one oscillating-foil case.

Writes ``artifacts/fields_<label>.npz`` with four phase-locked snapshots
(``t/T = 0, 0.25, 0.5, 0.75`` of the last cycle) plus the marker positions, and
``artifacts/fields_<label>.json`` with the case parameters.  Used by
``make_figures.py`` to draw the field contour panels of the report.

Usage::

    python make_fields.py --label fields_heave_k2 --k 2.0 --h0c 0.15 --chord 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "src")
sys.path.insert(0, SRC)

from tensorlbm.flapping_lbm_2d import momenta  # noqa: E402
from tensorlbm.oscillating_foil_lbm import OscillatingFoil2D, OscillatingFoilConfig  # noqa: E402


def vorticity(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    w = np.zeros_like(ux)
    w[1:-1, 1:-1] = 0.5 * (uy[1:-1, 2:] - uy[1:-1, :-2]) - 0.5 * (
        ux[2:, 1:-1] - ux[:-2, 1:-1]
    )
    return w


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="fields_heave_k2")
    ap.add_argument("--chord", type=int, default=32)
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--h0c", type=float, default=0.15)
    ap.add_argument("--a0", type=float, default=0.0)
    ap.add_argument("--mode", default="heave")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--U", type=float, default=0.15)
    ap.add_argument("--outdir", default=os.path.join(HERE, "artifacts"))
    args = ap.parse_args(argv)

    cfg = OscillatingFoilConfig(
        chord=args.chord,
        U=args.U,
        re=500.0,
        k=args.k,
        h0_over_c=args.h0c,
        alpha0_deg=args.a0,
        mode=args.mode,
        n_cycles=args.cycles,
        warmup_cycles=args.warmup,
    )
    solver = OscillatingFoil2D(cfg)
    per = cfg.steps_per_cycle()
    snapshots = []
    phases = {0.0, 0.25, 0.5, 0.75}
    for i in range(cfg.total_steps()):
        solver.step(diagnostics=False)
        frac = ((solver.step_count % per) / per)
        if solver.step_count > cfg.warmup_cycles * per:
            for ph in sorted(phases):
                if abs(frac - ph) < 1.0 / per:
                    rho, ux, uy = momenta(solver.f)
                    px, py, _, _, _ = solver.body_points(solver.time)
                    snapshots.append(
                        {
                            "phase": ph,
                            "ux": ux.astype(np.float32),
                            "uy": uy.astype(np.float32),
                            "vort": vorticity(ux, uy).astype(np.float32),
                            "mx": px.astype(np.float32),
                            "my": py.astype(np.float32),
                        }
                    )
                    phases.discard(ph)
            if not phases:
                break

    os.makedirs(args.outdir, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.outdir, f"{args.label}.npz"),
        **{f"{k}_{i}": s[k] for i, s in enumerate(snapshots) for k in ("ux", "uy", "vort", "mx", "my")},
        phases=np.array([s["phase"] for s in snapshots]),
    )
    with open(os.path.join(args.outdir, f"{args.label}.json"), "w") as fh:
        json.dump(
            {
                "label": args.label,
                "cfg": {
                    "chord": cfg.chord,
                    "k": cfg.k,
                    "h0_over_c": cfg.h0_over_c,
                    "alpha0_deg": cfg.alpha0_deg,
                    "mode": cfg.mode,
                    "U": cfg.U,
                    "Re": cfg.re,
                },
                "grid": [solver.nx, solver.ny],
                "phases": [s["phase"] for s in snapshots],
            },
            fh,
            indent=1,
        )
    print(f"wrote {len(snapshots)} snapshots to {args.outdir}/{args.label}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())