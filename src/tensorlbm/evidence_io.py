"""JSON evidence output helpers shared by the AMR sphere validation runners.

Handles the boilerplate that every validation script repeats at the end of a
run: recursive non-finite-float sanitisation (``ForceStationarityReport`` may
hold ``math.inf`` autocorrelation fields and tuples; ``json.dumps`` would
emit non-standard ``Infinity`` tokens), dataclass-to-dict conversion for the
stationarity report, and the mkdir + write step.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    """Recursively map non-finite floats to None for strict JSON validity."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def stationarity_dict(report: Any) -> Any:
    """Convert a dataclass stationarity report to a plain dict."""
    return asdict(report) if hasattr(report, "__dataclass_fields__") else report


def write_evidence(result: dict[str, Any], path: str | Path) -> None:
    """mkdir the parent directory and write the result as indented JSON.

    Non-finite floats are mapped to ``null`` via :func:`json_safe`, so the
    output is always strict JSON.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")


def common_schema_fields(prefix: str) -> dict[str, Any]:
    """Standard schema/status/physical_validation header for a runner's result."""
    return {
        "schema": f"tensorlbm-{prefix}",
        "status": "measured_candidate",
        "physical_validation": False,
    }


__all__ = [
    "common_schema_fields",
    "json_safe",
    "stationarity_dict",
    "write_evidence",
]
