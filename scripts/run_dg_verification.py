#!/usr/bin/env python3
"""Runner for the IN70-N01 DG convection-diffusion verification ladder.

Examples
--------
    PYTHONPATH=src python scripts/run_dg_verification.py --bench all
    PYTHONPATH=src python scripts/run_dg_verification.py --bench cone --cone-n 16 32 64
    PYTHONPATH=src python scripts/run_dg_verification.py --bench figure-data   # fields only

Every benchmark writes ``artifacts/IN70_N01/<name>.json``; the stashed numpy
fields of the last run are written to ``artifacts/IN70_N01/fields.npz`` for the
figure script.  Nothing is copied by hand: the report JSON is produced from the
in-memory results of this run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from tensorlbm.dg_verification import benchmarks as B

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "artifacts", "IN70_N01")

BENCHES = {
    "cone": B.bench_rotating_cone,
    "mms": B.bench_mms,
    "sod": B.bench_sod,
    "iso": B.bench_sod_isothermal,
    "square": B.bench_square_wave,
    "mass": B.bench_mass_ledger,
}


def _write(name: str, payload: dict) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print(f"  -> {path}", flush=True)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="all", choices=["all", *BENCHES, "figure-data"])
    ap.add_argument("--cone-n", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--cone-degrees", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--mms-n", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--mms-degrees", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--sod-n", type=int, nargs="+", default=[100, 200, 400])
    ap.add_argument("--sod-degrees", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--iso-n", type=int, default=2000)
    ap.add_argument("--iso-degrees", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--square-n", type=int, nargs="+", default=[100, 200, 400])
    ap.add_argument("--no-fields", action="store_true")
    args = ap.parse_args(argv)

    t_start = time.time()
    written = []

    if args.bench in ("all", "cone"):
        written.append(_write("rotating_cone", B.bench_rotating_cone(
            n_list=tuple(args.cone_n), degrees=tuple(args.cone_degrees))))
    if args.bench in ("all", "mms"):
        written.append(_write("mms", B.bench_mms(
            n_list=tuple(args.mms_n), degrees=tuple(args.mms_degrees))))
    if args.bench in ("all", "sod"):
        written.append(_write("sod", B.bench_sod(
            n_list=tuple(args.sod_n), degrees=tuple(args.sod_degrees))))
    if args.bench in ("all", "iso"):
        written.append(_write("sod_isothermal", B.bench_sod_isothermal(
            n_cells=args.iso_n, degrees=tuple(args.iso_degrees))))
    if args.bench in ("all", "square"):
        written.append(_write("square_wave", B.bench_square_wave(
            n_list=tuple(args.square_n))))
    if args.bench in ("all", "mass"):
        written.append(_write("mass_ledger", B.bench_mass_ledger()))

    if args.bench == "figure-data":
        # cheap field snapshots only (cone on a mid grid, both degrees)
        B.bench_rotating_cone(n_list=(16, 64), degrees=(1, 2), dt_refine_check=False)
        written.append(_write("rotating_cone_figure", {"rows": []}))

    if B.FIELDS and not args.no_fields:
        os.makedirs(OUT_DIR, exist_ok=True)
        tag = args.bench if args.bench != "all" else "all"
        npz = os.path.join(OUT_DIR, f"fields_{tag}.npz")
        np.savez_compressed(npz, **B.FIELDS)
        print(f"  -> {npz} ({len(B.FIELDS)} arrays)", flush=True)

    print(f"done in {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())