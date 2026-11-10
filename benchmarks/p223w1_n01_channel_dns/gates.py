#!/usr/bin/env python3
"""P223W1-N01 gate evaluator: DNS-arm channel statistics vs MKM-1999 DNS.

Consumes only the artefacts produced by ``run_case.py`` (native kernel JSON plus
the optional ``_spec.bin/_spec.json`` plane dumps), recomputes every number here
and writes ``gates.json`` fail-closed: a gate with missing inputs is ``withheld``,
never ``pass``.

    python3 benchmarks/p223w1_n01_channel_dns/gates.py \
        --run-dir runs/p223w1_n01_channel_dns \
        --out runs/p223w1_n01_channel_dns/gates.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "benchmarks" / "p223w1_n01_channel_dns"))

from run_case import CASES, REQUIRED  # noqa: E402
from tensorlbm.turbulence.channel_dns import (  # noqa: E402
    dns_cells_required,
    evaluate_dns_gates,
    load_dns_case,
)


def _discover(run_dir: Path) -> dict[str, int]:
    """Every case artefact in ``run_dir``, with its MKM anchor.

    Ladder legs come from ``run_case.CASES``; any extra artefact (a stability
    probe, an SGS-sensitivity rerun) is picked up from its ``R180``/``R395``/
    ``R590`` name prefix so ad-hoc runs are graded too instead of being ignored.
    """
    found = {name: spec.mkm_anchor for name, spec in CASES.items()
             if (run_dir / f"{name}.json").exists()}
    nominal = (180, 395, 590)
    for path in sorted(run_dir.glob("*.json")):
        name = path.stem
        if name.endswith(("_meta", "_spec")) or name == "gates" or name in found:
            continue
        import re

        match = re.search(r'"re_tau":\s*([0-9.]+)', path.read_text(encoding="utf-8"))
        if match is None:
            continue
        re_tau = float(match.group(1))
        found[name] = min(nominal, key=lambda a: abs(a - re_tau))
    return found


def _fmtnum(value) -> str:
    return "  n/a" if value is None else f"{float(value):6.2f}"


def build(run_dir: Path) -> dict:
    anchors = _discover(run_dir)
    cases = {
        name: load_dns_case(run_dir, name, re_tau_anchor=anchor)
        for name, anchor in anchors.items()
    }
    gates = evaluate_dns_gates(cases, require=tuple(n for n in REQUIRED if n in anchors))
    metrics = {}
    for name, case in cases.items():
        spec = CASES.get(name)
        row = {
            "re_tau": (spec.re_tau if spec else (case or {}).get("anchor")),
            "grid": [spec.nx, spec.ny, spec.nz] if spec else None,
            "half_height": spec.half_height if spec else None,
            "u_tau": spec.u_tau if spec else None,
            "nu": spec.nu if spec else None,
            "dy_plus": spec.spacing_plus if spec else None,
            "lx_over_h": spec.nx / spec.half_height if spec else None,
            "lz_over_h": spec.nz / spec.half_height if spec else None,
            "steps": spec.steps if spec else None,
            "warmup": spec.warmup if spec else None,
            "planned_eddy_turnovers": spec.eddy_turnovers if spec else None,
            "cs": spec.cs if spec else None,
            "artefact": str((run_dir / f"{name}.json").resolve().relative_to(_REPO))
            if (run_dir / f"{name}.json").exists()
            and str((run_dir / f"{name}.json").resolve()).startswith(str(_REPO))
            else (str(run_dir / f"{name}.json") if (run_dir / f"{name}.json").exists() else None),
        }
        if case is not None and spec is None and not case.get("diverged"):
            res = case["resolution"]
            row.update(
                {
                    "re_tau": float(case["result"]["configuration"]["re_tau"]),
                    "grid": [res["nx"], res["ny"], res["nz"]],
                    "half_height": res["half_height"],
                    "u_tau": res["u_tau"],
                    "nu": res["nu"],
                    "dy_plus": res["dy_plus"],
                    "lx_over_h": res["lx_over_h"],
                    "lz_over_h": res["lz_over_h"],
                    "steps": int(case["result"]["configuration"]["steps"]),
                    "warmup": int(case["result"]["configuration"]["warmup"]),
                    "eddy_turnovers": case["eddy_turnovers"],
                    "cs": float(case["result"]["configuration"].get("cs", float("nan"))),
                }
            )
        if case is None:
            row["status"] = "WITHHELD"
        elif case.get("diverged"):
            row["status"] = "DIVERGED"
            res = case.get("resolution") or {}
            row.update(
                {
                    "re_tau": res.get("re_tau", case.get("anchor")),
                    "grid": [res["nx"], res["ny"], res["nz"]] if res else None,
                    "dy_plus": res.get("dy_plus"),
                    "lx_over_h": res.get("lx_over_h"),
                    "lz_over_h": res.get("lz_over_h"),
                    "nu": res.get("nu"),
                    "cs": case.get("cs"),
                    "regularized": case.get("regularized"),
                    "u_tau": case.get("u_tau"),
                    "steps": case.get("steps"),
                    "max_speed": case.get("max_speed"),
                    "grade": (res or {}).get("grade"),
                }
            )
        else:
            prof = case["profile"]
            row.update(
                {
                    "grade": case["resolution"]["grade"],
                    "dns_resolved": case["resolution"]["dns_resolved"],
                    "eddy_turnovers": case["eddy_turnovers"],
                    "samples": case["samples"],
                    "u_bulk_plus": prof["u_bulk_plus"],
                    "cf": prof["cf"],
                    "cf_mkm": case["cf"].get("cf_mkm"),
                    "cf_mkm_relative_pct": case["cf"].get("mkm_relative_pct"),
                    "cf_dean": case["cf"]["cf_dean"],
                    "cf_dean_relative_pct": case["cf"]["dean_relative_pct"],
                    "re_bulk": case["cf"]["re_bulk"],
                    "u_plus_buffer_rms_pct": case["comparison"]
                    .get("buffer", {})
                    .get("rms_relative_pct"),
                    "u_plus_log_rms_pct": case["comparison"]
                    .get("log", {})
                    .get("rms_relative_pct"),
                    "u_rms_peak": prof["peaks"]["u_rms"],
                    "u_rms_peak_mkm_pct": case["stress"]
                    .get("u_rms", {})
                    .get("value_relative_pct"),
                    "minus_uv_peak": prof["peaks"]["minus_uv"],
                    "minus_uv_peak_mkm_pct": case["stress"]
                    .get("minus_uv", {})
                    .get("value_relative_pct"),
                    "kappa": prof.get("wall_law", {}).get("kappa"),
                    "B": prof.get("wall_law", {}).get("B"),
                    "wall_law_rms_residual": prof.get("wall_law", {}).get(
                        "rms_residual"
                    ),
                    "u_tau_route_spread_pct": prof.get(
                        "max_pairwise_u_tau_deviation_pct"
                    ),
                    "batch_bulk_spread_pct": case["convergence"].get(
                        "bulk_velocity_spread_pct"
                    ),
                    "momentum_balance_rms_pct": (case.get("momentum_balance") or {}).get(
                        "relative_rms_pct"
                    ),
                    "turbulence": case.get("turbulence"),
                    "cs_runtime": case.get("cs"),
                    "regularized_runtime": case.get("regularized"),
                    "spectrum": None
                    if not case.get("spectra")
                    else {
                        "legs": [
                            {
                                "y_plus": leg["y_plus"],
                                "slope": leg["fit"].get("slope"),
                                "r2": leg["fit"].get("r2"),
                                "valid": leg["fit"].get("valid"),
                            }
                            for leg in case["spectra"]["streamwise"]
                        ]
                    },
                    "status": gates["cases"][name]["status"],
                    "failed_gates": gates["cases"][name]["failed_gates"],
                    "withheld_gates": gates["cases"][name]["withheld_gates"],
                }
            )
        metrics[name] = row

    metrics = dict(sorted(metrics.items(), key=lambda kv: (kv[1].get("re_tau") or 0,
                                                           kv[1].get("dy_plus") or 0)))
    # resolution trend: how the DNS-comparison error moves with the wall-normal
    # resolution at a fixed Reynolds number.  Two grids cannot yield a formal
    # observed order of accuracy, so the ratio is reported as a trend only.
    trend = {}
    for re_key in ("180", "395", "590"):
        rows = [
            m
            for m in metrics.values()
            if m.get("re_tau") is not None
            and str(int(round(m["re_tau"]))) == re_key
            and m.get("u_plus_log_rms_pct") is not None
        ]
        rows.sort(key=lambda m: m["dy_plus"])
        trend[f"re_tau_{re_key}"] = {
            "points": [
                {
                    "dy_plus": r["dy_plus"],
                    "u_plus_log_rms_pct": r["u_plus_log_rms_pct"],
                    "u_plus_buffer_rms_pct": r.get("u_plus_buffer_rms_pct"),
                    "cf_mkm_relative_pct": r.get("cf_mkm_relative_pct"),
                    "u_rms_peak_mkm_pct": r.get("u_rms_peak_mkm_pct"),
                }
                for r in rows
            ],
            "log_band_error_ratio_coarse_over_fine": (
                rows[-1]["u_plus_log_rms_pct"] / rows[0]["u_plus_log_rms_pct"]
                if len(rows) >= 2 and rows[0]["u_plus_log_rms_pct"]
                else None
            ),
        }

    projection = {
        f"re_tau_{int(round(spec.re_tau))}": dns_cells_required(spec.re_tau)
        for spec in CASES.values()
    }
    return {
        "node": "p223w1_n01_lbm_turbulent_channel",
        "solver": "tensorlbm.turbulence.les_chan (D3Q19 TRT, cs=0 DNS arm)",
        "reference": "MKM-1999 DNS (SHA-256 verified) + Dean (1978) C_f",
        "study_status": gates["study_status"],
        "cases": gates["cases"],
        "metrics": metrics,
        "dns_cell_projection": projection,
        "resolution_trend": trend,
        "gates": gates["gates"],
        "criteria": gates["criteria"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir", default=str(_REPO / "runs" / "p223w1_n01_channel_dns")
    )
    ap.add_argument(
        "--out",
        default=str(_REPO / "runs" / "p223w1_n01_channel_dns" / "gates.json"),
    )
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    payload = build(run_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    print(f"study_status={payload['study_status']}")
    for name, row in payload["metrics"].items():
        print(
            f"{name:>22s} Re={_fmtnum(row.get('re_tau'))} dy+={_fmtnum(row.get('dy_plus'))} "
            f"grade={row.get('grade', '-'):>14s} status={row['status']:>8s} "
            f"failed={row.get('failed_gates', [])}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())