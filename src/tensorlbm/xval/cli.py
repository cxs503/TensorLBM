"""Command-line entry point for the XVAL-LBM-TAYLORGREEN cross-validation.

Examples
--------
Full campaign (all three tools; unavailable ones are recorded as frictions)::

    python -m tensorlbm.xval --out runs/xval_taylor_green

Stationary + Galilean-boosted case in one call, writing a machine-readable
summary to stdout::

    python -m tensorlbm.xval --out runs/xval_taylor_green --boost

Reproduce one tool only (used by the CI smoke test)::

    python -m tensorlbm.xval --out /tmp/xval --tools tensorlbm --n 32 --steps 600
"""

from __future__ import annotations

import argparse
import json
import sys

from .campaign import run_and_report
from .case_spec import DEFAULT_SPEC, TG2DSpec, boosted_spec


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output directory for artefacts")
    p.add_argument("--n", type=int, default=DEFAULT_SPEC.n, help="grid size (default 128)")
    p.add_argument("--re", type=float, default=DEFAULT_SPEC.re, help="Reynolds number (default 100)")
    p.add_argument("--u0", type=float, default=DEFAULT_SPEC.u0, help="peak initial velocity")
    p.add_argument("--steps", type=int, default=DEFAULT_SPEC.steps)
    p.add_argument("--sample-every", type=int, default=DEFAULT_SPEC.sample_every)
    p.add_argument("--field-steps", default=",".join(str(s) for s in DEFAULT_SPEC.field_steps))
    p.add_argument("--boost", action="store_true", help="run the Galilean-boosted companion case")
    p.add_argument(
        "--tools",
        default="tensorlbm,palabos,openfoam",
        help="comma separated tool list (tensorlbm,palabos,openfoam)",
    )
    p.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    p.add_argument("--threads", type=int, default=8)
    return p


def spec_from_args(args: argparse.Namespace) -> TG2DSpec:
    spec = TG2DSpec(
        n=args.n,
        re=args.re,
        u0=args.u0,
        steps=args.steps,
        sample_every=args.sample_every,
        field_steps=tuple(int(s) for s in str(args.field_steps).split(",") if s),
    )
    return boosted_spec(spec) if args.boost else spec


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = spec_from_args(args)
    tools = tuple(t.strip() for t in args.tools.split(",") if t.strip())
    payload = run_and_report(spec, args.out, tools=tools, dtype=args.dtype, threads=args.threads)
    summary = {
        "out": args.out,
        "case": payload["case"],
        "overall": payload["overall"],
        "criteria": payload["criteria"],
        "bug_suspects": payload["bug_suspects"],
        "frictions": payload["frictions"],
        "tools_available": [k for k, v in payload["tools"].items() if v["available"]],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())