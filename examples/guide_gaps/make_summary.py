#!/usr/bin/env python3
"""Aggregate the three guide-gap artifacts into ``artifacts/guide_gaps/summary.json``.

Usage::

    python3 examples/guide_gaps/make_summary.py            # aggregate (builds missing)
    python3 examples/guide_gaps/make_summary.py --no-run   # aggregate only

The aggregation is deliberately mechanical: it reads the per-case artifacts,
reports the primary judged quantity, the ``< 15 %`` / ``< 3 %`` verdicts and the
provenance string of that quantity's reference.  Nothing is retyped by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _common as C  # noqa: E402

#: checks registered at the tight (3 %) physical-accuracy tolerance are the
#: *judged quantities*; every other check is a numerical guard (conservation,
#: symmetry, truncation floor, convergence order, ...).
JUDGED_TOL = C.TOL_TIGHT


def _build_missing(quick: bool = True) -> None:
    import dam_break_sc
    import forchheimer
    import kelvin_helmholtz

    builders = {
        "dam_break_sc": dam_break_sc.run,
        "forchheimer": forchheimer.run,
        "kelvin_helmholtz": kelvin_helmholtz.run,
    }
    for case, fn in builders.items():
        if not C.artifact_path(case).exists():
            print(f"[summary] building missing artifact {case} ...")
            C.write_artifact(case, fn(quick=quick))


def build_summary(quick: bool = True, run_missing: bool = True, write: bool = True) -> dict:
    if run_missing:
        _build_missing(quick=quick)
    cases: dict[str, dict] = {}
    for case in C.CASES:
        art = C.load_artifact(case)
        checks = art.get("checks", [])
        judged = [c for c in checks if abs(float(c.get("tol", -1.0)) - JUDGED_TOL) < 1e-15]
        failed = [c["name"] for c in checks if not c.get("pass", False)]
        cases[case] = {
            "case": case,
            "quantity": art.get("quantity"),
            "computed": art.get("computed"),
            "reference": art.get("reference"),
            "rel_err": art.get("rel_err"),
            "evidence_grade": art.get("evidence_grade"),
            "reference_source": art.get("reference_source"),
            "lt_15pct": bool(art.get("rel_err") is not None and art["rel_err"] < C.TOL_ACCEPT),
            "lt_3pct": bool(art.get("rel_err") is not None and art["rel_err"] < C.TOL_TIGHT),
            "judged_checks": [c["name"] for c in judged],
            "worst_judged_rel_err": max((c["rel_err"] for c in judged), default=None),
            "worst_judged_lt_15pct": bool(
                judged and max(c["rel_err"] for c in judged) < C.TOL_ACCEPT
            ),
            "worst_judged_lt_3pct": bool(
                judged and max(c["rel_err"] for c in judged) < C.TOL_TIGHT
            ),
            "n_checks": len(checks),
            "n_checks_failed": len(failed),
            "failed_checks": failed,
            "generated_utc": art.get("generated_utc"),
            "scope": art.get("scope", ""),
        }
    summary = {
        "mission": "m21096",
        "pack": "guide_gaps",
        "generated_utc": C.now_utc(),
        "n_cases": len(cases),
        "tolerances": {"accept": C.TOL_ACCEPT, "tight": C.TOL_TIGHT},
        "judged_tolerance_rule": "checks registered with tol == %.2f are the judged "
        "physical quantities; all other checks are numerical guards" % JUDGED_TOL,
        "cases": cases,
        "all_lt_15pct": all(v["lt_15pct"] for v in cases.values()),
        "all_lt_3pct": all(v["lt_3pct"] for v in cases.values()),
        "all_judged_lt_15pct": all(v["worst_judged_lt_15pct"] for v in cases.values()),
        "all_judged_lt_3pct": all(v["worst_judged_lt_3pct"] for v in cases.values()),
        "n_checks_total": sum(v["n_checks"] for v in cases.values()),
        "n_checks_failed": sum(v["n_checks_failed"] for v in cases.values()),
        "grades": {k: v["evidence_grade"] for k, v in cases.items()},
        "grades_acceptable": all(
            v["evidence_grade"] in C.ACCEPTED_GRADES for v in cases.values()
        ),
        "withheld_or_not_supported": {
            k: v for k, v in cases.items() if v["evidence_grade"] not in ("REAL",)
        },
        "commands": {
            "pytest": "python3 -m pytest tests/test_guide_gaps.py -q",
            "cases": ["python3 examples/guide_gaps/%s.py --quick" % c for c in C.CASES],
            "summary": "python3 examples/guide_gaps/make_summary.py",
            "report": "python3 examples/guide_gaps/make_report.py",
        },
    }
    if write:
        C.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = C.ARTIFACT_DIR / "summary.json"
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-run", action="store_true", help="do not build missing artifacts")
    args = ap.parse_args(argv)
    s = build_summary(run_missing=not args.no_run)
    print(f"[summary] -> {C.ARTIFACT_DIR / 'summary.json'}")
    for case, row in s["cases"].items():
        print(
            "  %-18s grade=%-8s rel_err=%.3e  <15%%=%s  <3%%=%s  checks=%d (failed %d)"
            % (
                case,
                row["evidence_grade"],
                row["rel_err"],
                row["lt_15pct"],
                row["lt_3pct"],
                row["n_checks"],
                row["n_checks_failed"],
            )
        )
        print(f"      ref: {row['reference_source'][:110]}")
    print(
        "  ALL  <15%%=%s  <3%%=%s | judged-only <15%%=%s  <3%%=%s | failed checks=%d"
        % (
            s["all_lt_15pct"],
            s["all_lt_3pct"],
            s["all_judged_lt_15pct"],
            s["all_judged_lt_3pct"],
            s["n_checks_failed"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())