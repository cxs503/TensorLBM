"""Round-4 probe: static-control parasitic current at the *coexistence* state.

Round 3 measured criterion ③ (interface spurious current) from the nominal
initialisation ``rho_heavy=0.7 / rho_light=0.3``.  The coexistence calibration
(``coexistence.py``) shows the model's own coexisting densities at
``G_12 = -2.5`` are ``rho_A ~ 0.874`` / ``rho_B ~ 0.137`` with a 10-90
interface thickness of ~4.6 cells: the nominal state was far off coexistence
and the interface kept thinning, which is why R3 saw the residual saturated at
``O(1e-2)``.

This probe runs the same static control (``G_x = 0``) from the calibrated
coexistence initialisation and records ``S_ifc(t)`` for both pair-force
discretisations, so the R4 method decision can be made on measured data before
the acceptance matrix is launched.

Usage::

    PYTHONPATH=src python cases/mcmp_stratified_poiseuille/r4_probe_static.py \
        --ny 49 --steps 30000 --sc-form edm \
        --out artifacts/mcmp_stratified_poiseuille/r4/probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from cases.mcmp_stratified_poiseuille.spurious_diagnostic import run  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ny", type=int, default=49)
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--g12", type=float, default=-2.5)
    ap.add_argument("--rho-heavy", type=float, default=0.87373)
    ap.add_argument("--rho-light", type=float, default=0.13706)
    ap.add_argument("--width", type=float, default=2.114)
    ap.add_argument("--sc-form", default="edm", choices=["standard", "edm"])
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    rec = run(ny=args.ny, nx=args.nx, tau_lower=1.0, tau_upper=0.75, gx=0.0,
              g12=args.g12, steps=args.steps, u_eq="viscous",
              force_scheme="guo", wall_mode="mirror", sc_form=args.sc_form,
              rho_heavy=args.rho_heavy, rho_light=args.rho_light,
              width=args.width, sample_every=args.sample_every)
    s = rec["series"]
    tail = s[-min(6, len(s)):]
    # steady = last 20 % of the run: S_ifc drift < 2 %
    win = [h for h in s if h["step"] >= 0.8 * s[-1]["step"]]
    smin, smax = min(h["S_ifc"] for h in win), max(h["S_ifc"] for h in win)
    summary = dict(
        ny=args.ny, g12=args.g12, sc_form=args.sc_form, steps=s[-1]["step"],
        rho_heavy=args.rho_heavy, rho_light=args.rho_light, width=args.width,
        S_ifc=tail[-1]["S_ifc"], S_all=tail[-1]["S_all"], S_bulk=tail[-1]["S_bulk"],
        S_ifc_rel_drift=(smax - smin) / max(smin, 1e-30),
        y_int=tail[-1]["y_int"], phi_contrast=tail[-1]["phi_contrast"],
        u_max_mix=tail[-1]["u_max_mix"],
        wall_time_s=rec["wall_time_s"],
    )
    print("PROBE " + json.dumps(summary))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(dict(summary=summary, series=s, config=rec["config"]), fh, indent=1)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())