"""Unified CLI runner for TensorLBM examples.

Replaces the "example-per-main" antipattern with a single entry point::

    python -m tensorlbm.examples.runner lid_driven_cavity --nx 64 --re 400
    python -m tensorlbm.examples.runner --list
    python -m tensorlbm.examples.runner --run-all --n-steps 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .base import ExampleResult
from . import registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensorlbm.examples",
        description="Run TensorLBM examples via a unified interface.",
    )
    parser.add_argument(
        "example",
        nargs="?",
        help="Example name to run (omit with --list or --run-all)",
    )
    parser.add_argument("--list", action="store_true", help="List all registered examples")
    parser.add_argument("--run-all", action="store_true", help="Run every registered example")
    parser.add_argument(
        "--n-steps", type=int, default=None, help="Override step count for all examples"
    )
    parser.add_argument("--device", choices=["cpu", "sdaa", "cuda"], default=None)
    parser.add_argument("--output-root", default=None, help="Override output directory")
    parser.add_argument(
        "--json", action="store_true", help="Print results as JSON (machine-readable)"
    )
    return parser


def _apply_global_overrides(args) -> dict:
    overrides = {}
    if args.n_steps is not None:
        overrides["n_steps"] = args.n_steps
    if args.device is not None:
        overrides["device"] = args.device
    if args.output_root is not None:
        overrides["output_root"] = args.output_root
    return overrides


def main(argv=None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)

    if args.list:
        names = registry.list_examples()
        if args.json:
            print(json.dumps(names))
        else:
            print("Registered examples:")
            for name in names:
                ex = registry.get(name)
                desc = ex.description.split("\n")[0] if ex else ""
                print(f"  {name:30s} {desc}")
        return 0

    if args.run_all:
        overrides = _apply_global_overrides(args)
        results = registry.run_all(**overrides)
        _print_results(results, args.json)
        failed = [n for n, r in results.items() if not r.success]
        return 1 if failed else 0

    if args.example is None:
        parser.print_help()
        return 1

    example = registry.get(args.example)
    if example is None:
        print(f"Error: unknown example '{args.example}'", file=sys.stderr)
        print(f"Available: {registry.list_examples()}", file=sys.stderr)
        return 1

    # Build per-example parser for extra args
    ex_parser = argparse.ArgumentParser(add_help=False)
    example.add_cli_args(ex_parser)
    ex_args, _ = ex_parser.parse_known_args(extra)

    result = example.run_from_cli(ex_args)
    _print_results({args.example: result}, args.json)
    return 0 if result.success else 1


def _print_results(results: dict, as_json: bool) -> None:
    if as_json:
        out = {}
        for name, r in results.items():
            out[name] = {
                "success": r.success,
                "elapsed_s": r.elapsed_s,
                "metrics": r.metrics,
                "artifacts": r.artifacts,
                "error": r.error,
            }
        print(json.dumps(out, indent=2))
    else:
        for name, r in results.items():
            status = "PASS" if r.success else "FAIL"
            print(f"[{status}] {name} ({r.elapsed_s:.1f}s)")
            if r.metrics:
                for k, v in r.metrics.items():
                    print(f"       {k}: {v}")
            if r.error:
                print(f"       error: {r.error}")


if __name__ == "__main__":
    sys.exit(main())
