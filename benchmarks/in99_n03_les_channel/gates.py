#!/usr/bin/env python3
"""IN99-N03 gate evaluator: LES channel statistics vs MKM-1999 DNS.

Consumes only run artefacts produced by ``run_case.py`` (native kernel JSON),
recomputes every number here and writes ``gates.json`` fail-closed: a gate with
missing inputs is ``withheld``, never ``pass``.

    python3 benchmarks/in99_n03_les_channel/gates.py \
        --run-dir runs/in99_n03 --out runs/in99_n03/gates.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from tensorlbm.turbulence.les_channel import (  # noqa: E402
    compare_to_mkm,
    evaluate_gates,
    wall_units,
)
from tensorlbm.validation import mkm1999  # noqa: E402

#: case name -> nominal MKM Reynolds-number anchor
CASE_ANCHOR = {
    "R180_h24": 180,
    "R180_h32": 180,
    "R180_h40": 180,
    "R180_h32_nosgs": 180,
    "R395_h40": 395,
    "R590_h48": 590,
}


def load_case(run_dir: Path, name: str) -> dict | None:
    path = run_dir / f"{name}.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    probe = result.get("profiles", {}).get("mean_ux", [])
    if not probe or any(v != v for v in probe[:4]):  # NaN guard: a diverged run
        return None
    profile = wall_units(result)
    anchor = CASE_ANCHOR.get(name)
    mkm = mkm1999.load_profile(anchor) if anchor else None
    comparison = compare_to_mkm(profile, mkm)
    return {
        "result": result,
        "profile": profile,
        "comparison": comparison,
        "anchor": anchor,
        "path": str(path.relative_to(_REPO)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/in99_n03")
    ap.add_argument("--out", default=None)
    ap.add_argument("--production", default="R180_h40")
    ap.add_argument("--grid-levels", nargs="*", default=["R180_h24", "R180_h32", "R180_h40"])
    ap.add_argument("--negative-control", default="R180_h32_nosgs")
    ap.add_argument("--control-reference", default="R180_h32")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = _REPO / run_dir
    names = sorted(set(CASE_ANCHOR) | {args.production, args.negative_control,
                                       args.control_reference, *args.grid_levels})
    cases = {name: load_case(run_dir, name) for name in names}
    cases = {k: v for k, v in cases.items() if v is not None}

    report = evaluate_gates(
        cases,
        production=args.production,
        grid_levels=tuple(args.grid_levels),
        negative_control=args.negative_control,
        control_reference=args.control_reference,
    )
    # Reynolds-number family: informative comparison at all three MKM anchors.
    family = {}
    for name, case in cases.items():
        comp = case["comparison"]
        if not comp.get("available"):
            continue
        family[name] = {
            "re_tau": case["result"]["configuration"]["re_tau"],
            "grid": {k: case["result"]["configuration"][k] for k in ("nx", "ny", "nz")},
            "eddy_turnovers": case["result"]["derived"].get("statistics_eddy_turnovers"),
            "buffer_rms_pct": comp.get("buffer", {}).get("rms_relative_pct"),
            "log_rms_pct": comp.get("log", {}).get("rms_relative_pct"),
            "u_rms_peak": comp.get("u_rms_peak"),
            "minus_uv_peak": comp.get("minus_uv_peak"),
            "cf_relative_pct": comp.get("cf", {}).get("relative_pct"),
            "u_bulk_plus_relative_pct": comp.get("u_bulk_plus", {}).get("relative_pct"),
            "wall_law": case["profile"]["wall_law"],
            "u_tau": case["profile"]["u_tau"],
            "max_pairwise_u_tau_deviation_pct": case["profile"][
                "max_pairwise_u_tau_deviation_pct"
            ],
            "anisotropy": case["profile"]["anisotropy"],
            "sgs_cs": case["result"]["configuration"]["cs"],
            "provenance": {
                "elapsed_seconds": case["result"]["provenance"]["elapsed_seconds"],
                "mega_cell_steps_per_second": case["result"]["provenance"][
                    "mega_cell_steps_per_second"
                ],
                "kernel_source_sha256": case["result"]["provenance"]["kernel_source_sha256"],
                "binary_sha256": case["result"]["provenance"]["binary_sha256"],
            },
        }
    report["reynolds_family"] = family
    report["cases_present"] = sorted(cases)
    report["run_dir"] = str(run_dir)
    out = Path(args.out) if args.out else run_dir / "gates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("verdict", "summary")}, indent=1))
    for gate in report["gates"]:
        print(f"  {gate['status']:9s} {gate['gate']}: {gate['problems'] or 'ok'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())