"""Smoke check of the library entry point tensorlbm.cases.two_layer_poiseuille.

The analytic two-layer reference is imported from the benchmark case directory,
so the library module itself stays free of case-specific numbers.

Usage: PYTHONPATH=src:. python scripts/run_two_layer_poiseuille_check.py [--steps N]
"""
from __future__ import annotations

import argparse
import json
import sys

from tensorlbm.cases.two_layer_poiseuille import run_two_layer_poiseuille
from cases.mcmp_stratified_poiseuille.analytic import layer_viscosities, layered_poiseuille


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ny", type=int, default=25)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--sc-form", default="edm", choices=["standard", "edm"])
    args = ap.parse_args(argv)

    rec = run_two_layer_poiseuille(ny=args.ny, steps=args.steps, sc_form=args.sc_form)
    nu_lo, nu_up = layer_viscosities(1.0, 0.75, 0.7, 0.3, "mixture")
    ana = layered_poiseuille(args.ny, (args.ny - 2) / 2.0, 5e-5, nu_lo, nu_up, H=args.ny - 2)
    y_target = 0.5 + (args.ny - 2) / 2.0
    out = dict(
        ny=args.ny, steps=args.steps, sc_form=args.sc_form,
        Q_sim=rec["Q_mix"], Q_analytic=ana["Q_total"],
        err_Q=(rec["Q_mix"] - ana["Q_total"]) / ana["Q_total"],
        y_sim=rec["interface_crossing"], y_analytic=y_target,
        err_interface=(rec["interface_crossing"] - y_target) / y_target,
        spurious_mix=rec["spurious_mix"], u_max_mix=rec["u_max_mix"],
        phi_contrast=rec["phi_contrast"], steady=rec["steady"], finite=rec["finite"],
    )
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
