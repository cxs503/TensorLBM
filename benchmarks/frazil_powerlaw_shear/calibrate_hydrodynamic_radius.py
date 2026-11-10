#!/usr/bin/env python3
"""Stokes-drag gate: the hydrodynamic radius of the *discretised* crystal.

Why: ``phi`` measured from the lattice mask of a staircase sphere is not the
volume fraction the fluid responds to.  Half-way bounce-back resolves a body
whose drag is that of a slightly smaller smooth sphere, so the mask fraction
overestimates the hydrodynamic fraction at coarse resolution.  That distinction
is not cosmetic here: the Krieger–Dougherty comparison at ``c = 0.20`` is a
statement about the hydrodynamic fraction, so the conversion has to be measured
in the same cell, at the same resolution, and reported.

The measurement is an independent one: a *fixed* sphere in a uniform stream
(both walls moving together, so no shear), whose momentum-exchange force is the
drag.  Inverting Stokes' law with the Haberman–Sayre confinement factor gives
``a_h``, and ``phi_h / phi_mask = (a_h / a_eff)^3`` with
``a_eff = (3 V_mask / 4 pi)^(1/3)``.

Writes ``artifacts/frazil_powerlaw_shear/drag_calibration.json``::

    python3 benchmarks/frazil_powerlaw_shear/calibrate_hydrodynamic_radius.py --radius 2.5 --box 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_OUT = os.path.join(ROOT, "artifacts", "frazil_powerlaw_shear",
                           "drag_calibration.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=2.5)
    ap.add_argument("--box", type=int, default=32)
    ap.add_argument("--nz", type=int, default=None)
    ap.add_argument("--nu", type=float, default=0.2)
    ap.add_argument("--re-p", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    from tensorlbm.ice_suspension_rheology.calibrate import run_drag_case

    rec = run_drag_case(radius=args.radius, box=args.box, nz=args.nz, re_p=args.re_p,
                        nu=args.nu, n_steps=args.steps, threads=args.threads)
    a_h = rec["drag"] / (6.0 * 3.141592653589793 * args.nu * rec["u_stream"]
                         * rec["haberman_sayre_K"])
    rec["a_hydrodynamic"] = float(a_h)
    rec["a_h_over_a_eff"] = float(a_h / rec["a_eff"])
    rec["phi_ratio"] = float(rec["a_h_over_a_eff"] ** 3)
    rec["note"] = ("a_h from F_drag / (6 pi mu U K_confinement); phi_h = phi_mask * "
                   "phi_ratio is the volume-fraction convention used in the KD comparison")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    print(json.dumps({k: rec[k] for k in ("radius", "a_eff", "a_hydrodynamic",
                                          "a_h_over_a_eff", "phi_ratio", "re_p",
                                          "err_pct", "err_pct_corrected", "out"
                                          if False else "drag")}, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())