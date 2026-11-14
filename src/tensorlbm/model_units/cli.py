"""CLI for model units: ``list`` / ``build`` / ``test`` / ``ci-matrix`` / ``doctor``.

Mirrors the TCLB workflow where the Makefile generates a per-model target:
``make model-<name>`` -> :func:`build_and_test`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .build import ARTIFACT_ROOT, build, doctor, test
from .coverage import (
    DEFAULT_MODELS_ROOT,
    coverage_report,
    discover_directories,
    matrix_cells,
)
from .registry import DEFAULT_ROOT, ci_matrix, discover, makefile_targets


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _select(units, names: list[str] | None):
    if not names:
        return units
    wanted = set(names)
    chosen = [u for u in units if u.name in wanted or u.unit_id in wanted]
    missing = wanted - {u.name for u in chosen} - {u.unit_id for u in chosen}
    if missing:
        raise SystemExit(f"unknown model unit(s): {sorted(missing)}")
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tensorlbm.model_units")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="model unit root directory")
    parser.add_argument(
        "--models-root",
        default=str(DEFAULT_MODELS_ROOT),
        help="root holding every model directory, whatever its manifest dialect",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list discovered model units")

    b = sub.add_parser("build", help="build units (manifest validation + isolated import)")
    b.add_argument("--unit", action="append")
    b.add_argument("--out", default=str(ARTIFACT_ROOT))

    t = sub.add_parser("test", help="run each unit's case against its declared reference")
    t.add_argument("--unit", action="append")

    bt = sub.add_parser("build-and-test", help="the CI matrix cell command")
    bt.add_argument("--unit", required=True)
    bt.add_argument("--out", default=str(ARTIFACT_ROOT))

    m = sub.add_parser("ci-matrix", help="emit the generated CI matrix")
    m.add_argument("--out", default=None)
    m.add_argument(
        "--github-output", action="store_true", help="write matrix=... to $GITHUB_OUTPUT"
    )

    mk = sub.add_parser("makefile", help="emit the generated Makefile fragment")
    mk.add_argument("--out", default=str(ARTIFACT_ROOT / "Makefile.models"))

    sub.add_parser("doctor", help="registry consistency gates (CI plan job)")

    v = sub.add_parser("verify", help="doctor + generated-matrix coverage + artifact drift")
    v.add_argument("--matrix-out", default=str(ARTIFACT_ROOT / "ci_matrix.json"))

    sub.add_parser("manifest-list", help="print model.toml paths (make dependency list)")

    args = parser.parse_args(argv)
    units = discover(args.root)

    if args.command == "list":
        _print(
            {
                "root": str(Path(args.root)),
                "count": len(units),
                "units": [u.to_dict() for u in units],
            }
        )
        return 0

    if args.command == "manifest-list":
        for unit in discover(args.root):
            print(unit.path / "model.toml")
        return 0

    if args.command == "ci-matrix":
        cov = coverage_report(models_root=args.models_root)
        matrix = {"include": cov["cells_detail"]}
        text = json.dumps(matrix, sort_keys=True)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
        if args.github_output:
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
                fh.write(f"matrix={text}\n")
        print(text)
        return 0 if cov["ok"] else 1

    if args.command == "makefile":
        directories = discover_directories(args.models_root)
        extra = [
            cell
            for cell in matrix_cells(directories, models_root=args.models_root)
            if cell["kind"] != "unit_toml"
        ]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            makefile_targets(units, extra_cells=extra) + "\n", encoding="utf-8"
        )
        print(args.out)
        return 0

    if args.command == "verify":
        report = doctor(args.root, models_root=args.models_root)
        cov = coverage_report(models_root=args.models_root)
        matrix = {"include": cov["cells_detail"]}
        ci_names = sorted(u.name for u in units if u.ci)
        on_matrix = sorted(cell["unit"] for cell in matrix["include"])
        missing_ci = sorted(set(ci_names) - set(on_matrix))
        out = Path(args.matrix_out)
        drift: list[str] = []
        if out.is_file():
            committed = json.loads(out.read_text())
            key = lambda c: (c.get("kind", "unit_toml"), c["unit"])  # noqa: E731
            old = {key(c): c.get("cmd") for c in committed.get("include", [])}
            new = {key(c): c.get("cmd") for c in matrix["include"]}
            if old != new:
                drift = sorted(map(str, set(old.items()) ^ set(new.items()))) or [
                    "cmd changed"
                ]
        coverage_ok = bool(cov["ok"] and not missing_ci)
        report["matrix"] = {
            "ci_units": ci_names,
            "on_matrix": on_matrix,
            "missing_ci": missing_ci,
            "coverage_ok": coverage_ok,
            "drift": drift,
        }
        report["ok"] = bool(report["ok"] and coverage_ok and not drift)
        _print(report)
        return 0 if report["ok"] else 1

    if args.command == "doctor":
        report = doctor(args.root)
        _print(report)
        return 0 if report["ok"] else 1

    if args.command == "build":
        records = [build(u, args.out) for u in _select(units, args.unit)]
        _print(
            {
                "built": [
                    {
                        "unit": r["unit"]["unit"],
                        "artifact": r["artifact"],
                        "kernel_modules": r["kernel_import"]["n_modules"],
                    }
                    for r in records
                ]
            }
        )
        return 0

    if args.command == "test":
        results = [test(u) for u in _select(units, args.unit)]
        _print(
            {
                r["unit"]["unit"]: {
                    "kpi": r["kpi"],
                    "value": r["value"],
                    "threshold": r["threshold"],
                    "status": r["status"],
                }
                for r in results
            }
        )
        return 0 if all(r["status"] == "passed" for r in results) else 1

    if args.command == "build-and-test":
        chosen = _select(units, [args.unit])[0]
        built = build(chosen, args.out)
        result = test(chosen)
        _print(
            {
                "unit": chosen.unit_id,
                "build": {
                    "artifact": built["artifact"],
                    "kernel_modules": built["kernel_import"]["n_modules"],
                },
                "test": {
                    "kpi": result["kpi"],
                    "value": result["value"],
                    "threshold": result["threshold"],
                    "status": result["status"],
                },
            }
        )
        return 0 if result["status"] == "passed" else 1

    return 2  # pragma: no cover - argparse rejects unknown commands


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
