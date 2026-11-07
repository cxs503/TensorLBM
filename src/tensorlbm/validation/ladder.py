"""Assembly of the high-Reynolds wall-treatment validation ladder.

A *ladder* is a machine-checkable verdict document: every case contributes
named criteria, each criterion carries its measured number, the gate it is
compared against, the comparison direction and the artifact that produced it.
The overall verdict is fail-closed: a case passes only when every criterion
passes, and an honest FAIL is preserved verbatim in the payload.

The intent is auditability — a reviewer can recompute any number from the
referenced artifact without re-running a simulation, and no criterion can be
"satisfied" by prose.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "Criterion",
    "assemble_ladder",
    "criterion",
    "relative_error_pct",
    "verdict",
    "write_payload",
]

_COMPARISONS = {"<=", ">=", "in_band", "abs<=", "abs>=", ">", "<"}


@dataclass(frozen=True)
class Criterion:
    """One machine-checkable acceptance item."""

    name: str
    description: str
    value: float | None
    threshold: float | tuple[float, float] | None
    comparison: str
    passed: bool
    artifact: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def criterion(
    name: str,
    description: str,
    value: float | None,
    threshold: float | tuple[float, float] | None,
    comparison: str,
    *,
    artifact: str | None = None,
) -> Criterion:
    """Build a criterion and evaluate it (fail-closed on missing values)."""
    if comparison not in _COMPARISONS:
        raise ValueError(f"unknown comparison {comparison!r}; expected {sorted(_COMPARISONS)}")
    if value is None or threshold is None:
        return Criterion(
            name=name,
            description=description,
            value=value,
            threshold=threshold,
            comparison=comparison,
            passed=False,
            artifact=artifact,
        )
    if not math.isfinite(float(value)):
        return Criterion(
            name=name,
            description=description,
            value=float(value),
            threshold=threshold,
            comparison=comparison,
            passed=False,
            artifact=artifact,
        )
    numeric = float(value)
    passed: bool
    if comparison == "<=":
        passed = numeric <= float(threshold)  # type: ignore[arg-type]
    elif comparison == "<":
        passed = numeric < float(threshold)  # type: ignore[arg-type]
    elif comparison == ">=":
        passed = numeric >= float(threshold)  # type: ignore[arg-type]
    elif comparison == ">":
        passed = numeric > float(threshold)  # type: ignore[arg-type]
    elif comparison == "abs<=":
        passed = abs(numeric) <= float(threshold)  # type: ignore[arg-type]
    elif comparison == "abs>=":
        passed = abs(numeric) >= float(threshold)  # type: ignore[arg-type]
    else:  # in_band
        low, high = threshold  # type: ignore[misc]
        passed = low <= numeric <= high
    return Criterion(
        name=name,
        description=description,
        value=numeric,
        threshold=threshold,
        comparison=comparison,
        passed=passed,
        artifact=artifact,
    )


def relative_error_pct(value: float, reference: float) -> float:
    """Signed relative error in percent (solver convention: sim vs reference)."""
    if reference == 0.0:
        raise ValueError("reference value must be non-zero")
    return (value - reference) / reference * 100.0


def verdict(items: Iterable[Criterion]) -> dict[str, Any]:
    """Fail-closed verdict for a list of criteria."""
    collected = list(items)
    failed = [item.name for item in collected if not item.passed]
    return {
        "verdict": "PASS" if collected and not failed else "FAIL",
        "criteria_count": len(collected),
        "failed_criteria": failed,
        "criteria": [item.to_dict() for item in collected],
    }


def assemble_ladder(
    cases: Mapping[str, Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine per-case payloads (each with a ``criteria`` list) into one ladder."""
    payload_cases: dict[str, Any] = {}
    failed_cases: list[str] = []
    for name, case in cases.items():
        criteria = [item for item in case.get("criteria", []) if isinstance(item, Criterion)]
        case_verdict = verdict(criteria)
        entry = dict(case)
        entry.update(
            {
                "verdict": case_verdict["verdict"],
                "criteria": case_verdict["criteria"],
                "failed_criteria": case_verdict["failed_criteria"],
            }
        )
        payload_cases[name] = entry
        if case_verdict["verdict"] != "PASS":
            failed_cases.append(name)
    return {
        "schema": "tensorlbm.wall-ladder/v1",
        "metadata": dict(metadata or {}),
        "cases": payload_cases,
        "overall_verdict": "PASS" if not failed_cases else "FAIL",
        "failed_cases": failed_cases,
        "note": (
            "Fail-closed ladder: a case passes only when every criterion passes. "
            "Accuracy gates that are not met are reported as FAIL and are never "
            "downgraded by convergence evidence."
        ),
    }


def _json_default(obj: Any) -> Any:
    """Serialise dataclass payloads (``Criterion``) transparently."""
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serialisable")


def write_payload(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write a ladder payload as sorted, indented JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    return target


def criteria_from_dicts(items: Sequence[Mapping[str, Any]]) -> list[Criterion]:
    """Rehydrate criteria serialised by :meth:`Criterion.to_dict`."""
    restored = []
    for item in items:
        restored.append(
            Criterion(
                name=str(item["name"]),
                description=str(item["description"]),
                value=item["value"],
                threshold=item["threshold"],
                comparison=str(item["comparison"]),
                passed=bool(item["passed"]),
                artifact=item.get("artifact"),
            )
        )
    return restored