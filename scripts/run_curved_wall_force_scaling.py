#!/usr/bin/env python3
"""Driver for the curved-wall force-error scaling campaign (IN19-T1).

Sub-commands
------------
``case``      run one cylinder case and write ``<out>/<label>.json``
``couette``   run the Couette wall-format validation set
``domain``    run the domain-height sensitivity study at a fixed diameter
``assemble``  build the factorial table, criteria and the final report JSON
``all-local`` small smoke ladder used by the pytest gate

All heavy runs are intended to be launched in the background (one process per
case) because the shared host delivers only ~2e6 cell-updates/s per process.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.curved_wall_force_scaling import (  # noqa: E402
    CD_REFERENCE_RE100,
    CylinderCaseConfig,
    evaluate_criteria,
    run_couette,
    run_cylinder_case,
    summarise_runs,
)

ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "curved_wall_force_scaling"


def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
    tmp.replace(path)


def case_label(diameter: int, fmt: str) -> str:
    return f"N{diameter:04d}_{fmt}"


def cmd_case(args: argparse.Namespace) -> int:
    cfg = CylinderCaseConfig(
        diameter=args.diameter,
        format=args.format,
        steps_per_diameter=args.steps_per_diameter,
        lattice_speed=args.lattice_speed,
        upstream_diameters=args.upstream,
        downstream_diameters=args.downstream,
        half_height_diameters=args.half_height,
        threads=args.threads,
        rest_repair=args.rest_repair,
        cv_margin_cells=args.cv_margin,
        cv_sample_every=args.cv_every,
        snapshot_path=args.snapshot,
    )
    label = case_label(args.diameter, args.format)
    out = Path(args.out)
    started = time.time()
    result = run_cylinder_case(cfg, max_steps=args.max_steps, label=label)
    result["config"] = {
        "diameter": cfg.diameter,
        "format": cfg.format,
        "steps_per_diameter": cfg.steps_per_diameter,
        "lattice_speed": cfg.lattice_speed,
        "upstream_diameters": cfg.upstream_diameters,
        "downstream_diameters": cfg.downstream_diameters,
        "half_height_diameters": cfg.half_height_diameters,
        "rest_repair": cfg.rest_repair,
        "reynolds": cfg.reynolds,
    }
    result["wall_clock_seconds"] = time.time() - started
    _dump(out / f"{label}.json", result)
    print(
        f"{label}: CD_me={result['cd_me_steady']:.4f} CD_cv={result['cd_cv_steady']:.4f} "
        f"rel_err={abs(result['cd_me_steady'] - CD_REFERENCE_RE100) / CD_REFERENCE_RE100:.4f} "
        f"({result['wall_seconds']:.0f}s)"
    )
    return 0


def cmd_couette(args: argparse.Namespace) -> int:
    cases = [
        ("halfway", 0.5),
        ("bouzidi_linear", 0.5),
        ("bouzidi_linear", 0.3),
        ("bouzidi_linear", 0.7),
        ("bouzidi_quadratic", 0.5),
        ("bouzidi_quadratic", 0.3),
        ("bouzidi_quadratic", 0.7),
    ]
    payload = []
    for fmt, q in cases:
        res = run_couette(
            nx=4, ny=args.ny, q_bottom=q, u_wall=0.05, tau=1.0,
            steps=args.steps, format=fmt, threads=args.threads,
        )
        payload.append(res)
        print(
            f"couette {fmt:20s} q={q:.2f} shear_err_bottom={100 * res['bottom_relative_error']:+.3f}% "
            f"profile_err={100 * res['profile_relative_error']:+.3f}%"
        )
    _dump(Path(args.out) / "couette_validation.json", {"cases": payload})
    return 0


def cmd_domain(args: argparse.Namespace) -> int:
    payload = []
    for half in [float(h) for h in args.heights.split(",")]:
        cfg = CylinderCaseConfig(
            diameter=args.diameter,
            format=args.format,
            half_height_diameters=half,
            steps_per_diameter=args.steps_per_diameter,
            lattice_speed=args.lattice_speed,
            threads=args.threads,
        )
        res = run_cylinder_case(cfg, max_steps=args.max_steps, label=f"domain_h{half:g}")
        payload.append(
            {
                "half_height_diameters": half,
                "diameter": args.diameter,
                "format": args.format,
                "cd_me_steady": res["cd_me_steady"],
                "cd_cv_steady": res["cd_cv_steady"],
                "grid_cells": res["grid"]["cells"],
                "steps": res["steps"],
                "wall_seconds": res["wall_seconds"],
            }
        )
        print(
            f"domain half={half:g}: CD_me={res['cd_me_steady']:.4f} CD_cv={res['cd_cv_steady']:.4f}"
        )
    _dump(Path(args.out) / "domain_sensitivity.json", {"cases": payload})
    return 0


def _load_runs(run_dir: Path) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for path in sorted(run_dir.glob("N*_*.json")):
        data = json.loads(path.read_text())
        runs[data["label"]] = data
    return runs


def cmd_assemble(args: argparse.Namespace) -> int:
    root = Path(args.out)
    runs = _load_runs(root / "runs")
    if not runs:
        raise SystemExit("no per-case runs found")
    formats = ("bouzidi_linear", "bouzidi_quadratic")
    diameters = sorted({r["diameter"] for r in runs.values() if r["format"] in formats})
    expected = len(diameters) * len(formats) * 2 * 2
    table = summarise_runs(runs, formats=formats, expected_rows=expected)
    table["grid_ladder_executed"] = diameters
    table["grid_ladder_requested"] = [64, 128, 256, 512]
    table["runs"] = {
        label: {
            "diameter": r["diameter"],
            "format": r["format"],
            "grid": r["grid"],
            "steps": r["steps"],
            "cd_me_steady": r["cd_me_steady"],
            "cd_cv_steady": r["cd_cv_steady"],
            "cd_me_transient": r["cd_me_transient"],
            "cd_cv_transient": r["cd_cv_transient"],
            "cd_me_std_last_window": r["cd_me_std_last_window"],
            "cd_me_steady_density_norm": r.get("cd_me_steady_density_norm"),
            "cd_cv_steady_density_norm": r.get("cd_cv_steady_density_norm"),
            "mean_density_last": r.get("mean_density_last"),
            "cumulative_mass_leak": r["cumulative_mass_leak"],
            "boundary": r["boundary"],
            "control_volume_box": r["control_volume_box"],
            "wall_seconds": r["wall_seconds"],
            "cell_steps_per_second": r["cell_steps_per_second"],
            "tau": r["tau"],
            "nu": r["nu"],
        }
        for label, r in runs.items()
    }
    criteria = evaluate_criteria(
        table, finest=diameters[-1], coarsest=diameters[0], cd_reference=CD_REFERENCE_RE100
    )
    couette_path = root / "couette_validation.json"
    domain_path = root / "domain_sensitivity.json"
    payload = {
        "campaign": "IN19-T1",
        "module": "tensorlbm.curved_wall_force_scaling",
        "cd_reference": CD_REFERENCE_RE100,
        "reference_note": "pinned experimental/high-resolution CD of the unbounded Re=100 cylinder",
        "grid_ladder_requested": [64, 128, 256, 512],
        "grid_ladder_executed": diameters,
        "factorial": {"diameters": diameters, "formats": list(formats),
                      "force_methods": ["momentum_exchange", "control_volume"],
                      "windows": ["steady", "transient"]},
        "table": table,
        "criteria": criteria,
        "couette_validation": json.loads(couette_path.read_text()) if couette_path.exists() else None,
        "domain_sensitivity": json.loads(domain_path.read_text()) if domain_path.exists() else None,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _dump(root / "sweep_results.json", payload)
    print(json.dumps(criteria, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("case")
    p.add_argument("--diameter", type=int, required=True)
    p.add_argument("--format", required=True)
    p.add_argument("--steps-per-diameter", type=float, default=100.0)
    p.add_argument("--lattice-speed", type=float, default=0.1)
    p.add_argument("--upstream", type=float, default=3.0)
    p.add_argument("--downstream", type=float, default=6.0)
    p.add_argument("--half-height", type=float, default=8.0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--cv-margin", type=int, default=6)
    p.add_argument("--cv-every", type=int, default=20)
    p.add_argument("--rest-repair", action="store_true")
    p.add_argument("--snapshot", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out", default=str(ARTIFACT_ROOT / "runs"))
    p.set_defaults(func=cmd_case)

    p = sub.add_parser("couette")
    p.add_argument("--ny", type=int, default=32)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", default=str(ARTIFACT_ROOT))
    p.set_defaults(func=cmd_couette)

    p = sub.add_parser("domain")
    p.add_argument("--diameter", type=int, default=32)
    p.add_argument("--format", default="bouzidi_quadratic")
    p.add_argument("--heights", default="6,8,12")
    p.add_argument("--steps-per-diameter", type=float, default=100.0)
    p.add_argument("--lattice-speed", type=float, default=0.1)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out", default=str(ARTIFACT_ROOT))
    p.set_defaults(func=cmd_domain)

    p = sub.add_parser("assemble")
    p.add_argument("--out", default=str(ARTIFACT_ROOT))
    p.set_defaults(func=cmd_assemble)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())