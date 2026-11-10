#!/usr/bin/env python3
"""Run one LBM-LES plane-channel verification case (campaign p84w3-n05).

Geometry (campaign specification): ``Lx x H x Lz = 2 pi h x 2h x pi h`` with
``h`` the half-height in lattice units, periodic in ``x``/``z``, no-slip
(Bouzidi interpolated bounce-back) walls in ``y``, driven by the constant body
force ``a = 2 u_tau^2 / H``.

Everything written to the artifact is produced by the run itself: wall-unit
profiles, Reynolds stresses, spectra with their a-priori-band fits, the wall
momentum-exchange flux, and the four campaign verdicts computed by
:func:`tensorlbm.turbulence.les_channel_verification.evaluate_criteria`.
No number is typed in by hand.

Example
-------
    PYTHONPATH=src python benchmarks/p84w3_n05_les_channel/run_case.py \
        --h 16 --steps 36000 --warmup 12000 --out artifacts/p84w3_n05_les_channel/case_re180_h16.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from tensorlbm.turbulence.les_channel_verification import (  # noqa: E402
    run_case,
    write_artifact,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--h", type=int, default=16, help="half-height in lattice units")
    ap.add_argument("--re-tau", type=float, default=180.0)
    ap.add_argument("--u-tau", type=float, default=0.01)
    ap.add_argument("--cs", type=float, default=0.12, help="Smagorinsky constant")
    ap.add_argument("--van-driest-a-plus", type=float, default=26.0)
    ap.add_argument("--strain-source", default="velocity_gradient",
                    choices=("velocity_gradient", "fneq"))
    ap.add_argument("--wall-model", default="none", choices=("none", "spalding"))
    ap.add_argument("--no-van-driest", action="store_true")
    ap.add_argument("--tau-minus-max", type=float, default=None)
    ap.add_argument("--steps", type=int, default=36000)
    ap.add_argument("--warmup", type=int, default=12000)
    ap.add_argument("--sample-interval", type=int, default=10)
    ap.add_argument("--report-interval", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    label = args.label or f"re{int(args.re_tau)}_h{args.h}_{args.strain_source}"
    out = args.out or (ROOT / "artifacts" / "p84w3_n05_les_channel" / f"case_{label}.json")

    t0 = time.time()
    artifact = run_case(
        half_height=args.h,
        steps=args.steps,
        warmup_steps=args.warmup,
        u_tau=args.u_tau,
        re_tau=args.re_tau,
        cs=args.cs,
        strain_source=args.strain_source,
        wall_model=args.wall_model,
        van_driest=not args.no_van_driest,
        tau_minus_max=args.tau_minus_max,
        sample_interval=args.sample_interval,
        report_interval=args.report_interval,
        van_driest_a_plus=args.van_driest_a_plus,
        label=label,
        verbose=True,
    )
    artifact["command"] = {
        "argv": sys.argv[1:],
        "label": label,
        "wall_clock_seconds": time.time() - t0,
    }
    path = write_artifact(artifact, out)
    summary = artifact["summary"]
    print("\n== summary ==")
    print(json.dumps(summary, indent=1, default=str))
    print("\n== criteria ==")
    for name, check in artifact["criteria"]["checks"].items():
        print(f"{check['status']:9s} {name}: metric={check.get('metric')}")
    print(f"\nartifact: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
