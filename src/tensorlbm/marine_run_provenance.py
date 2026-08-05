"""Hash-bound provenance for an already executed marine runner observation."""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def build_marine_run_provenance(observation: Mapping[str, Any], *, runner: str) -> dict[str, object]:
    """Bind an observation to the runner identity without inventing validation."""
    if not isinstance(observation, Mapping) or observation.get("schema") != "suboff-resistance-runtime-observation-v1":
        raise ValueError("unsupported runtime observation schema")
    if not isinstance(runner, str) or not runner:
        raise ValueError("runner must be a non-empty string")
    return {
        "schema": "marine-run-provenance-v1",
        "runner": runner,
        "observation_sha256": sha256(_canonical(observation)).hexdigest(),
    }
