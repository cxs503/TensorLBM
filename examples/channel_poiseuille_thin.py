#!/usr/bin/env python3
"""Thin example main -- the *after* state of the example-per-main anti-pattern.

Everything that used to be copied into every example (lattice build, BGK +
Guo step, bounce-back, steady-state loop, post-processing) now lives in
:mod:`tensorlbm.example_harness`.  This file only selects a case, runs it and
writes the machine-refereed record, so it cannot drift from the library.

    PYTHONPATH=src python examples/channel_poiseuille_thin.py --ny 32

Reference: u_max = g H^2 / (8 nu) with H = ny - 2 (half-way bounce-back walls
at y = 1/2 and y = ny - 3/2).
"""

from __future__ import annotations

import argparse
import json
import os

from tensorlbm.example_harness import get_case, record_digest, run_case


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ny", type=int, default=32)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--out", default=None, help="write the run record JSON here")
    ns = ap.parse_args(argv)

    record = run_case(get_case("channel_poiseuille", ny=ns.ny, steps=ns.steps))
    record["record_digest"] = record_digest(record)
    kpis = record["kpis"]
    print(
        json.dumps(
            {
                "case": record["case"],
                "ny": ns.ny,
                "u_max": kpis["u_max"],
                "u_max_analytic": record["analytic"]["u_max"],
                "u_max_rel_err": kpis["relative_error_vs_analytic"],
                "profile_rel_l2": kpis["profile_rel_l2"],
                "mass_drift": kpis["mass_drift"],
                "steps_run": kpis["steps_run"],
                "record_digest": record["record_digest"],
            },
            indent=2,
        )
    )
    if ns.out:
        os.makedirs(os.path.dirname(os.path.abspath(ns.out)), exist_ok=True)
        with open(ns.out, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())