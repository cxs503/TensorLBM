"""End-to-end IN101-N01 driver: campaign -> criteria -> figures -> artifact.

Usage (from a repo checkout with ``PYTHONPATH=src``)::

    python -m tensorlbm.ice_keel_drag.report --run-dir artifacts/IN101_N01/run \
        --out artifacts/IN101_N01

The artifact ``criteria.json`` is the machine-readable verdict set that the
mission report quotes; every number in it is computed from the stored records.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from . import campaign, cases, criteria, figures


def build_artifact(run_dir: str | Path, out_dir: str | Path, *,
                   make_figures: bool = True) -> dict[str, Any]:
    run = Path(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = campaign.load_records(run)
    verdicts = criteria.evaluate(records)
    artifact = {
        "mission": "IN101-N01",
        "title": "Ice-underside keel morphology: skin/form drag decomposition",
        "n_cases": len(records),
        "cases": {cid: {
            "family": r["family"], "tags": r["tags"], "grid": r["grid"],
            "h_eff": r["h_eff"], "nu": r["nu"], "re_tau": r["re_tau"],
            "re_h": r["re_h"], "cf": r["cf"], "cd_total": r["cd_total"],
            "cd_skin": r["cd_skin"], "cd_form": r["cd_form"],
            "tau_total": r["tau_total"], "tau_form": r["tau_form"],
            "tau_skin": r.get("tau_skin"), "tau_me": r.get("tau_me"),
            "tau_flux": r.get("tau_flux"),
            "S_fr_over_plan": r["S_fr_over_plan"],
            "analytic_S_fr_over_plan": r["analytic_S_fr_over_plan"],
            "closure_residual": r.get("closure_residual"),
            "momentum_balance_residual": r.get("momentum_balance_residual"),
            "stationarity_drift": r.get("stationarity_drift"),
            "u_plus_bulk": r.get("u_plus_bulk"),
            "log_fit": r.get("log_fit"),
            "wall_seconds": r.get("wall_seconds"),
        } for cid, r in sorted(records.items())},
        "criteria": verdicts["criteria"],
        "verdicts": verdicts["verdicts"],
        "n_passed": verdicts["n_passed"],
        "n_criteria": verdicts["n_criteria"],
        "all_passed": verdicts["all_passed"],
        "figures": [],
    }
    if make_figures:
        try:
            made = figures.make_all(records, verdicts, out / "figures")
            artifact["figures"] = [{"name": n, "path": str(p)} for n, p in made]
        except Exception as exc:  # pragma: no cover - figures are not the verdict
            artifact["figures_error"] = f"{type(exc).__name__}: {exc}"
    path = out / "criteria.json"
    path.write_text(json.dumps(artifact, indent=1, sort_keys=True, allow_nan=True))
    return artifact


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(description="IN101-N01 report/artifact builder")
    ap.add_argument("--run-dir", default="artifacts/IN101_N01/run")
    ap.add_argument("--out", default="artifacts/IN101_N01")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    art = build_artifact(args.run_dir, args.out, make_figures=not args.no_figures)
    print(json.dumps({k: art[k] for k in ("n_cases", "verdicts", "n_passed",
                                         "n_criteria", "all_passed")}, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())