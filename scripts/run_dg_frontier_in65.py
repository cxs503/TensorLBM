#!/usr/bin/env python
"""IN65-N04 -- driver for the DG convection/diffusion verification frontier.

Each ``--case`` writes ``artifacts/IN65_N04/<case>.json`` and prints one
``@@JSON <case> <payload>`` line, so a cluster log can be parsed back into the
exact same artefacts when the node filesystem is not available.

Usage
-----
    PYTHONPATH=src python scripts/run_dg_frontier_in65.py --case all
    PYTHONPATH=src python scripts/run_dg_frontier_in65.py --case cone --out-dir artifacts/IN65_N04

Cases: ``cone`` ``hump`` ``gaussian`` ``kink`` ``mms`` ``sod`` ``sod_lbm``
``square`` ``ledger``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorlbm.dg_verification import frontier as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _dump(case: str, payload: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{case}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    line = json.dumps(payload, default=float, separators=(",", ":"))
    print(f"@@JSON {case} {line}", flush=True)
    print(f"[{case}] written {path}", flush=True)


CASES = ("cone", "hump", "gaussian", "smooth", "kink", "mms", "sod", "sod_lbm",
         "square", "ledger")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "artifacts", "IN65_N04"))
    ap.add_argument("--fast", action="store_true", help="reduced sweep for smoke tests")
    ap.add_argument("--medium", action="store_true",
                    help="workstation-sized sweep: one refinement less than the full one")
    args = ap.parse_args()

    cases = CASES if args.case == "all" else (args.case,)
    for case in cases:
        if case in ("cone", "hump", "gaussian", "smooth"):
            n_list = (16, 32) if args.fast else (16, 32, 64, 128)
            degs = (1, 2) if args.fast else (1, 2, 3)
            if case == "smooth" and not args.fast:
                n_list = (16, 32, 64, 96)  # sigma=0.05 companion: 96 is already < 1e-5
            if args.medium and case != "smooth":
                n_list, degs = (16, 32, 64), (1, 2)
            payload = F.bench_rotating(n_list=n_list, degrees=degs, initial=case)
        elif case == "kink":
            n_list = (32, 64) if args.fast else (32, 64, 128, 256, 512)
            degs = (1, 2) if args.fast else (1, 2, 3)
            if args.medium:
                n_list, degs = (32, 64, 128, 256), (1, 2, 3)
            payload = {
                "kink": F.bench_kink_order(n_list=n_list, degrees=degs, initial="tent"),
                "smooth": F.bench_kink_order(n_list=n_list, degrees=degs, initial="gaussian"),
            }
        elif case == "mms":
            n_list = (8, 16) if args.fast else (8, 16, 32, 64)
            degs = (1, 2) if args.fast else (1, 2, 3)
            if args.medium:
                n_list, degs = (8, 16, 32, 64), (1, 2)
            payload = F.bench_mms_varcoeff(n_list=n_list, degrees=degs)
        elif case == "sod":
            n_list = (100, 200) if args.fast else (100, 200, 400, 800)
            degs = (1,) if args.fast else (1, 2)
            if args.medium:
                n_list, degs = (100, 200, 400), (1, 2)
            payload = F.bench_sod_hllc(
                n_list=n_list, degrees=degs, fluxes=("hllc", "rusanov"),
                tvb_M=0.0, t_end=0.2, cfl=0.3,
            )
            payload["sod_highM"] = F.bench_sod_hllc(
                n_list=n_list, degrees=degs, fluxes=("hllc", "rusanov"),
                tvb_M=50.0, t_end=0.2, cfl=0.3,
            )
        elif case == "sod_lbm":
            times = (100, 200) if args.fast else (100, 200, 300, 400)
            degs = (1,) if args.fast else (1, 2)
            if args.medium:
                times, degs = (100, 200, 400), (1, 2)
            payload = F.bench_sod_lbm_crosscheck(times=times, degrees=degs)
        elif case == "square":
            n_list = (64,) if args.fast else (64, 128, 256, 512)
            degs = (1, 2) if args.fast else (1, 2, 3)
            if args.medium:
                n_list, degs = (64, 128, 256), (1, 2)
            payload = F.bench_square_wave_metrics(n_list=n_list, degrees=degs)
        elif case == "ledger":
            payload = F.bench_euler_mass_ledger(
                n_cells=200 if not args.fast else 60, degrees=(1, 2), fluxes=("hllc", "rusanov")
            )
        else:  # pragma: no cover
            raise SystemExit(f"unknown case {case!r}")
        _dump(case, payload, args.out_dir)


if __name__ == "__main__":
    main()