"""Evaluate existing benchmark artifacts without launching a solver."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def _load_gate_module():
    """Load the standalone gate without importing the optional AI package tree."""
    module_path = Path(__file__).parents[1] / "src" / "tensorlbm" / "regression_gate.py"
    spec = importlib.util.spec_from_file_location("tensorlbm_regression_gate", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load regression gate from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GATE = _load_gate_module()
evaluate_acoustic_campaign_gate = _GATE.evaluate_acoustic_campaign_gate
evaluate_regression_gate = _GATE.evaluate_regression_gate
write_regression_manifest = _GATE.write_regression_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True, help="root containing completed run directories")
    parser.add_argument("--manifest", type=Path, required=True, help="gate specification JSON")
    parser.add_argument("--report", type=Path, required=True, help="output regression manifest JSON")
    args = parser.parse_args()

    payload: Any = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        parser.error("manifest must be a JSON object")
    if "status_file" in payload:
        report = evaluate_acoustic_campaign_gate(args.artifacts, payload)
    elif isinstance(payload.get("cases"), dict):
        report = evaluate_regression_gate(args.artifacts, payload["cases"])
    else:
        parser.error("manifest must define generic 'cases' or acoustic 'status_file' + 'cases'")
    write_regression_manifest(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
