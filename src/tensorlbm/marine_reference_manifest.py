"""Explicit, hashable marine resistance reference records."""
from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite
from typing import Any


def build_marine_reference_manifest(*, case: str, coefficient: float, source: str) -> dict[str, object]:
    if not isinstance(case, str) or not case:
        raise ValueError("case must be a non-empty string")
    if not isinstance(coefficient, (int, float)) or isinstance(coefficient, bool) or not isfinite(float(coefficient)) or coefficient <= 0:
        raise ValueError("reference coefficient must be finite and positive")
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a non-empty string")
    unsigned: dict[str, Any] = {"schema": "marine-reference-manifest-v1", "case": case,
                                "coefficient": float(coefficient), "source": source}
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {**unsigned, "sha256": sha256(encoded).hexdigest()}
