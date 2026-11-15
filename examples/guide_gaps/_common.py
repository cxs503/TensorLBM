"""Shared helpers for the ``examples/guide_gaps`` evidence pack.

The three guide gaps (``dam_break_sc``, ``forchheimer``, ``kelvin_helmholtz``)
are closed by machine-checkable artifact files, one per case, plus an
aggregated ``summary.json``.  Every artifact carries the *same* minimal schema
so the acceptance gate can be evaluated mechanically:

    {case, quantity, computed, reference, rel_err, evidence_grade, ...}

``evidence_grade`` follows the ecosystem convention ``REAL / MOCK / STUB /
WITHHELD / NOT_SUPPORTED``.  ``WITHHELD`` is only allowed together with an
explicit ``withheld_reason`` string: inventing a reference value and calling it
experimental is exactly the failure mode this pack exists to prevent.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "guide_gaps"
DOC_DIR = REPO_ROOT / "docs" / "guide_gaps"
FIG_DIR = ARTIFACT_DIR / "figures"

CASES: tuple[str, ...] = ("dam_break_sc", "forchheimer", "kelvin_helmholtz")

GRADES = ("REAL", "MOCK", "STUB", "WITHHELD", "NOT_SUPPORTED")
#: grades that are acceptable for a *closed* guide gap
ACCEPTED_GRADES = ("REAL", "WITHHELD")

#: platform accuracy standard: >15 % is not acceptable
TOL_ACCEPT = 0.15
#: "publication" grade target
TOL_TIGHT = 0.03


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel_err(computed: float, reference: float) -> float:
    """Relative error, robust for a zero reference (falls back to absolute).

    A pure relative error is undefined when ``reference == 0``; in that case the
    absolute difference is reported, which is honest and still machine-checkable.
    """
    computed = float(computed)
    reference = float(reference)
    if reference == 0.0:
        return abs(computed - reference)
    return abs(computed - reference) / abs(reference)


def check(
    name: str,
    quantity: str,
    computed: float,
    reference: float,
    tol: float,
    reference_source: str,
    note: str = "",
) -> dict[str, Any]:
    """One machine-checkable comparison (numeric + provenance string)."""
    err = rel_err(computed, reference)
    row = {
        "name": name,
        "quantity": quantity,
        "computed": float(computed),
        "reference": float(reference),
        "rel_err": err,
        "tol": float(tol),
        "pass": bool(err <= tol),
        "reference_source": reference_source,
    }
    if note:
        row["note"] = note
    return row


def artifact_path(case: str) -> Path:
    return ARTIFACT_DIR / f"{case}.json"


def write_artifact(case: str, payload: dict[str, Any]) -> Path:
    """Write ``artifacts/guide_gaps/<case>.json`` with the mandatory schema."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    grade = payload.get("evidence_grade")
    if grade not in GRADES:
        raise ValueError(f"{case}: evidence_grade {grade!r} not in {GRADES}")
    if grade == "WITHHELD" and not payload.get("withheld_reason"):
        raise ValueError(f"{case}: WITHHELD requires an explicit withheld_reason")
    if grade == "REAL":
        for key in ("computed", "reference"):
            if payload.get(key) is None:
                raise ValueError(f"{case}: REAL evidence needs a numeric {key}")
    payload = dict(payload)
    payload.setdefault("case", case)
    payload["generated_utc"] = now_utc()
    payload.setdefault("primary_tol", TOL_TIGHT)
    payload["meta"] = {
        "schema": "guide-gap-artifact/v1",
        "tols": {"accept": TOL_ACCEPT, "tight": TOL_TIGHT},
        **payload.get("meta", {}),
    }
    path = artifact_path(case)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def load_artifact(case: str) -> dict[str, Any]:
    path = artifact_path(case)
    if not path.exists():
        raise FileNotFoundError(f"missing artifact {path}; run the example first")
    return json.loads(path.read_text(encoding="utf-8"))


def all_pass(checks: Iterable[dict[str, Any]], tol: float | None = None) -> bool:
    rows = list(checks)
    if tol is not None:
        return all(r.get("rel_err", math.inf) <= tol for r in rows)
    return all(bool(r.get("pass")) for r in rows)


def grade_of(checks: Iterable[dict[str, Any]]) -> str:
    """REAL when every recorded comparison has a numeric analytic reference."""
    rows = list(checks)
    ok = bool(rows) and all(r.get("reference") is not None for r in rows)
    return "REAL" if ok else "WITHHELD"


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.3f}%"