#!/usr/bin/env python
"""Development probe: does the LBM side reproduce the analytic thrust?

Runs a small matrix of short cases and prints the measured force coefficients
next to the Garrick closed form plus the wall-clock cost per step.  Used to tune
the immersed-boundary set-up (resolution, kernel, forcing passes).

Usage::

    python tools/probe_force.py --chords 48,64 --cycles 4 --workers 6
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "..", "src")
sys.path.insert(0, SRC)

from tensorlbm.flapping_propulsion import FlappingKinematics, garrick_coefficients  # noqa: E402
from tensorlbm.oscillating_foil_lbm import (  # noqa: E402
    OscillatingFoil2D,
    OscillatingFoilConfig,
    cycle_statistics,
)


def theory(cfg: OscillatingFoilConfig) -> dict:
    kin = FlappingKinematics(
        U=cfg.U,
        c=float(cfg.chord),
        h0=cfg.motion.h0,
        alpha0=cfg.motion.alpha0,
        phi=cfg.motion.phi,
        a=cfg.motion.a,
    )
    g = garrick_coefficients(kin, cfg.k)
    return {"C_T": g.C_T, "C_P": g.C_P, "eta": g.eta}


def run_case(spec):
    cfg = OscillatingFoilConfig(
        chord=spec["chord"],
        re=spec.get("re", 500.0),
        U=spec.get("U", 0.15),
        h0_over_c=spec.get("h0c", 0.25),
        alpha0_deg=spec.get("a0", 0.0),
        k=spec["k"],
        phi_deg=spec.get("phi", 90.0),
        mode=spec["mode"],
        n_cycles=spec.get("cycles", 4),
        warmup_cycles=spec.get("warmup", 2),
        domain_x_over_c=spec.get("dx", 9.0),
        domain_y_over_c=spec.get("dy", 4.0),
        solid_interior=spec.get("solid", True),
        kernel=spec.get("kernel", "4pt"),
        n_forcing_passes=spec.get("passes", 1),
        marker_spacing=spec.get("spacing", 0.5),
    )
    solver = OscillatingFoil2D(cfg)
    rows = []
    t0 = time.time()
    for _ in range(cfg.total_steps()):
        rows.append(solver.step(diagnostics=False))
    dt = time.time() - t0
    st = cycle_statistics(rows, cfg)
    q = 0.5 * cfg.U**2 * cfg.chord
    tail = rows[-cfg.steps_per_cycle() :]
    st["label"] = spec["label"]
    st["grid"] = [solver.nx, solver.ny]
    st["ms_per_step"] = 1000.0 * dt / cfg.total_steps()
    st["C_interior"] = sum(r["Fx_interior"] for r in tail) / len(tail) / q
    st["theory"] = theory(cfg) if cfg.mode != "locked" else {"C_T": 0.0, "C_P": 0.0, "eta": 0.0}
    st["cd0"] = -st["C_T"] if cfg.mode == "locked" else None
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chords", default="48,64")
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--h0c", type=float, default=0.25)
    ap.add_argument("--a0", type=float, default=10.0)
    ap.add_argument("--U", type=float, default=0.15)
    ap.add_argument("--dx", type=float, default=9.0)
    ap.add_argument("--dy", type=float, default=4.0)
    ap.add_argument("--passes", type=int, default=1)
    args = ap.parse_args()

    specs = []
    for c in [int(x) for x in args.chords.split(",")]:
        for label, mode, extra in (
            ("locked", "locked", {}),
            ("heave", "heave", {"h0c": args.h0c}),
            ("pitch", "pitch", {"a0": args.a0}),
        ):
            specs.append(
                dict(
                    label=f"{label}_c{c}",
                    mode=mode,
                    chord=c,
                    k=args.k,
                    U=args.U,
                    dx=args.dx,
                    dy=args.dy,
                    cycles=args.cycles,
                    warmup=args.warmup,
                    passes=args.passes,
                    **extra,
                )
            )

    hdr = (
        f"{'case':12s} {'grid':>9s} {'C_T':>8s} {'C_T_th':>8s} {'C_P':>8s} "
        f"{'eta':>7s} {'eta_th':>7s} {'3sig':>8s} {'Cint':>8s} {'ms/st':>7s}"
    )
    print(hdr, flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(run_case, specs):
            print(
                f"{r['label']:12s} {r['grid'][0]:4d}x{r['grid'][1]:<4d} "
                f"{r['C_T']:8.4f} {r['theory']['C_T']:8.4f} {r['C_P']:8.4f} "
                f"{r['eta']:7.3f} {r['theory']['eta']:7.3f} "
                f"{3 * r['C_T_cycle_std']:8.1e} {r['C_interior']:8.4f} {r['ms_per_step']:7.1f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())