"""Run one P399W2-N04 validation case (single job, JSON out).

Used to spread the case matrix over several processes / cluster slots::

    PYTHONPATH=src python scripts/p399w2_n04_curved_force/run_case.py \
        --label re100_quad_M64 --re 100 --diameter 16 --box 15 10 \
        --markers 64 --scheme quadrature --normal-passes 1 \
        --steps 9000 --transient 4000 --u-target 0.06 --save-fields

The per-case JSON lands in ``artifacts/p399w2_n04/case_<label>.json`` and is
merged by ``merge_cases.py`` into the stage files consumed by the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_validation import ROOT, run_cylinder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--re", type=float, required=True)
    ap.add_argument("--diameter", type=float, default=20.0)
    ap.add_argument("--box", type=float, nargs=2, default=(24.0, 16.0))
    ap.add_argument("--markers", type=int, default=64)
    ap.add_argument("--scheme", default="quadrature", choices=["quadrature", "raw"])
    ap.add_argument("--correctors", type=int, default=2)
    ap.add_argument("--normal-passes", type=int, default=1)
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--transient", type=int, default=8000)
    ap.add_argument("--u-target", type=float, default=0.05)
    ap.add_argument("--sample-every", type=int, default=40)
    ap.add_argument("--out", default="artifacts/p399w2_n04")
    ap.add_argument("--save-fields", action="store_true")
    ap.add_argument("--stokes-check", action="store_true")
    ap.add_argument("--perturbation", type=float, default=1e-3)
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_cylinder(
        label=args.label,
        reynolds=args.re,
        diameter=args.diameter,
        box=(args.box[0], args.box[1]),
        n_markers=args.markers,
        scheme=args.scheme,
        n_correctors=args.correctors,
        n_normal_passes=args.normal_passes,
        n_steps=args.steps,
        n_transient=args.transient,
        u_target=args.u_target,
        sample_every=args.sample_every,
        save_fields=args.save_fields,
        perturbation=args.perturbation,
        stokes_check=args.stokes_check,
    )
    path = out_dir / f"case_{args.label}.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"[case] {args.label} -> {path}")
    print(json.dumps(res["metrics"], indent=1))


if __name__ == "__main__":
    main()