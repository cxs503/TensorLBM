"""Entry point: ``python -m tensorlbm.model_matrix <subcommand>``.

Subcommands
-----------
``list``    print the registry (model keys, lattice, implementation, cases)
``guard``   construct every registry entry and run one step (compile guard,
            the analogue of the upstream GPU-pipeline guard)
``run``     execute the model x case matrix, write coverage/benchmark
            artifacts and exit non-zero when the gate fails

Examples
--------
::

    python -m tensorlbm.model_matrix list
    python -m tensorlbm.model_matrix guard --out artifacts/model_matrix_ci
    python -m tensorlbm.model_matrix run --out artifacts/model_matrix_ci/run
    python -m tensorlbm.model_matrix run --tau 0.6 --res 32 --models d3q19/bgk,d3q27/kbc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import coverage as cov
from .cases import Grid, case_names
from .registry import get_model, model_names, nu_from_tau


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _split(arg: str | None, default: list[str]) -> list[str]:
    if not arg:
        return default
    return [s.strip() for s in arg.split(",") if s.strip()]


def _dtype(name: str):
    import torch  # noqa: PLC0415

    return {"float32": torch.float32, "float64": torch.float64}[name]


def cmd_list(args: argparse.Namespace) -> int:
    rows = []
    for name in model_names():
        spec = get_model(name)
        rows.append(
            {
                "model": name,
                "lattice": spec.lattice,
                "implementation": spec.source,
                "tau_arg": spec.tau_arg,
                "params": spec.params,
                "family": spec.family,
                "enforcement": spec.enforcement,
                "cases": list(spec.cases),
            }
        )
    print(json.dumps(rows, indent=2))
    return 0


def cmd_guard(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = [cov.guard_one(n, dtype=_dtype(args.dtype)) for n in _split(args.models, model_names())]
    payload = {
        "meta": cov.environment_meta(_repo_root()),
        "guard": results,
        "verdict": "PASS" if all(r["status"] == "GUARD_PASS" for r in results) else "FAIL",
    }
    (out / "guard.json").write_text(json.dumps(payload, indent=2) + "\n")
    for r in results:
        print(f"[guard] {r['model']:22s} {r['status']} {r['reason']}")
    print(f"[guard] verdict={payload['verdict']} -> {out / 'guard.json'}")
    return 0 if payload["verdict"] == "PASS" else 1


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    taus = [float(t) for t in args.tau.split(",")]
    models = _split(args.models, model_names())
    cases = _split(args.cases, case_names())
    grid = Grid(nx=args.res, ny=args.res, nz=args.nz)

    run = cov.run_matrix(
        models,
        cases,
        grid=grid,
        taus=taus,
        dtype=_dtype(args.dtype),
        progress=not args.quiet,
    )
    summary = cov.summarize(run, min_coverage=args.min_coverage)
    meta = cov.environment_meta(_repo_root())
    meta["repro_cmd"] = args.repro_cmd or (
        f"PYTHONPATH=src python -m tensorlbm.model_matrix run --out {out} "
        f"--tau {args.tau} --res {args.res} --dtype {args.dtype} "
        f"--min-coverage {args.min_coverage}"
    )

    out.mkdir(parents=True, exist_ok=True)
    cov.write_json(run, summary, out / "model_matrix.json", **meta)
    cov.write_csv(run, out / "matrix.csv")
    cov.write_coverage_md(run, summary, out / "COVERAGE.md")
    cov.write_benchmark_md(run, summary, out / "BENCHMARK_REF_VS_GOT.md")
    figs = cov.make_figures(run, summary, out / "figures") if not args.no_figures else []

    print(
        f"[matrix] coverage={summary['coverage_fraction'] * 100:.1f}% "
        f"({summary['cells_covered']}/{summary['cells_declared']}) "
        f"gate_failures={len(summary['gate_failures'])} verdict={summary['verdict']}"
    )
    for f in [out / "model_matrix.json", out / "COVERAGE.md", out / "BENCHMARK_REF_VS_GOT.md", *figs]:
        print(f"[matrix] wrote {f}")
    if not args.non_gating and summary["verdict"] != "PASS":
        print("[matrix] GATE FAILED", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tensorlbm.model_matrix", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    lp = sub.add_parser("list", help="print the model registry")
    lp.set_defaults(func=cmd_list)

    gp = sub.add_parser("guard", help="construct + one-step guard for every model")
    gp.add_argument("--out", default="artifacts/model_matrix_ci")
    gp.add_argument("--models", default=None)
    gp.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    gp.set_defaults(func=cmd_guard)

    rp = sub.add_parser("run", help="run the model x case matrix and write evidence")
    rp.add_argument("--out", default="artifacts/model_matrix_ci/run")
    rp.add_argument("--models", default=None, help="comma-separated registry keys")
    rp.add_argument("--cases", default=None, help="comma-separated case names")
    rp.add_argument("--tau", default="0.6,0.8")
    rp.add_argument("--res", type=int, default=32, help="nx = ny")
    rp.add_argument("--nz", type=int, default=1)
    rp.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    rp.add_argument("--min-coverage", type=float, default=0.85)
    rp.add_argument("--no-figures", action="store_true")
    rp.add_argument("--non-gating", action="store_true", help="always exit 0 (report only)")
    rp.add_argument("--repro-cmd", default=None)
    rp.add_argument("--quiet", action="store_true")
    rp.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())