"""Canonical fail-closed artifact joining observation, provenance and reference."""
from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite
from typing import Any, Mapping


def _digest(value: Mapping[str, Any]) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def build_marine_resistance_artifact(
    observation: Mapping[str, Any], provenance: Mapping[str, Any], reference: Mapping[str, Any],
) -> dict[str, object]:
    """Build a canonical artifact; withheld evidence is never converted to PASS."""
    if observation.get("schema") != "suboff-resistance-runtime-observation-v1":
        raise ValueError("unsupported runtime observation schema")
    if provenance.get("schema") != "marine-run-provenance-v1" or provenance.get("observation_sha256") != _digest(observation):
        raise ValueError("provenance does not bind runtime observation")
    unsigned_reference = {key: value for key, value in reference.items() if key != "sha256"}
    if reference.get("schema") != "marine-reference-manifest-v1" or reference.get("sha256") != _digest(unsigned_reference):
        raise ValueError("invalid reference manifest hash")
    if observation.get("case") != reference.get("case"):
        raise ValueError("observation and reference case differ")
    measured = observation.get("resistance")
    coefficient = measured.get("coefficient") if isinstance(measured, Mapping) else None
    if not isinstance(coefficient, (int, float)) or isinstance(coefficient, bool) or not isfinite(float(coefficient)):
        raise ValueError("observation has no finite measured coefficient")
    ref = float(reference["coefficient"])
    error = abs(float(coefficient) - ref) / ref * 100.0
    completion = observation.get("completion")
    return {
        # Version 2 is the runtime contract. Version 1 remains reserved for
        # pre-binding canonical gate artifacts.
        "kind": "marine_resistance_kpi", "schema_version": 2, "case": observation["case"],
        # Keep the complete evidence in the signed-by-hash artifact.  A gate must
        # never accept detached hashes whose source records cannot be checked.
        "binding": {"observation_sha256": _digest(observation), "provenance_sha256": _digest(provenance),
                    "reference_sha256": reference["sha256"]},
        "evidence": {"observation": dict(observation), "provenance": dict(provenance),
                     "reference": dict(reference)},
        "completion": dict(completion) if isinstance(completion, Mapping) else {"state": "WITHHELD"},
        # Validation facts are copied only from the bound runtime observation.
        # Physics is deliberately never promoted by artifact construction.
        "preflight": dict(observation.get("preflight", {})),
        "numerics": dict(observation.get("numerics", {})), 
        "conservation": dict(observation.get("conservation", {})),
        "resistance": {"status": "measured_not_validated", "pass": False, "coefficient": float(coefficient),
                       "reference_coefficient": ref, "relative_error_pct": error},
        "physics": {"status": "withheld", "pass": False},
    }
