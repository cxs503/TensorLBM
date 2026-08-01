from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

EXAMPLES = Path(__file__).parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
MODULE_PATH = EXAMPLES / "suboff_nested_static_amr_smoke.py"
SPEC = importlib.util.spec_from_file_location("suboff_nested_static_amr_smoke", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _args(tmp_path: Path, *, preflight: bool = False):
    values = [
        "--device", "cpu",
        "--nx", "80", "--ny", "40", "--nz", "40",
        "--hull-length", "24", "--center-x-fraction", "0.35",
        "--outer-wall-margin", "4", "--outer-wake-cells", "8",
        "--inner-wall-margin", "2", "--inner-wake-cells", "0",
        "--cv-margin", "2", "--steps", "2", "--ramp-steps", "0",
        "--resolved-reynolds", "2000", "--sponge-width", "3",
        "--memory-bytes-per-cell", "742",
        "--output", str(tmp_path / "nested-smoke.json"),
    ]
    if preflight:
        values.append("--preflight-only")
    return MODULE.parser().parse_args(values)


def test_nested_suboff_smoke_closes_force_and_both_interfaces(tmp_path: Path) -> None:
    result = MODULE.run(_args(tmp_path))

    assert result["status"] == "integration_smoke_pass"
    assert result["physical_validation"] is False
    assert result["geometry"]["geometry_owner_level"] == 2
    assert result["geometry"]["force_owner_level"] == 2
    assert result["result"]["maximum_source_corrected_observer_difference_pct"] < 0.1
    assert max(result["result"]["maximum_reflux_residual_by_interface"]) < 1.0e-6
    assert result["acceptance"]["resistance_accuracy_assessed"] is False


def test_nested_suboff_preflight_does_not_claim_physics(tmp_path: Path) -> None:
    result = MODULE.run(_args(tmp_path, preflight=True))

    assert result["status"] == "preflight_only"
    assert result["physical_validation"] is False
    assert result["planning"]["total_allocated_cells"] > 0
    assert result["planning"]["memory_estimate_bytes_per_cell"] == 742.0
