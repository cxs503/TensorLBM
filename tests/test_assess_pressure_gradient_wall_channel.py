from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "assess_pressure_gradient_wall_channel.py"
)
SPEC = importlib.util.spec_from_file_location("assess_pressure_gradient_wall_channel", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_channel_assessor_reads_symmetric_profile_and_preserves_sources(
    tmp_path: Path,
) -> None:
    run = tmp_path / "channel"
    run.mkdir()
    metadata = {
        "config": {"u_tau": 0.005, "re_tau": 100.0},
        "derived": {"nu": 0.00025, "body_force": 2.5e-6, "H": 20},
        "averaging_samples": 100,
        "log_law_rms_error": 0.2,
        "diagnostics": [
            {"max_speed": 0.1},
            {"max_speed": 0.1001},
            {"max_speed": 0.1},
        ],
    }
    (run / "run_metadata.json").write_text(json.dumps(metadata))
    with (run / "velocity_profile.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("y", "y_plus", "u_plus", "u_plus_loglaw"))
        for y in range(1, 21):
            writer.writerow((y, y, 12.0, 12.0))
    result = MODULE.assess(
        run,
        minimum_y_plus=10.0,
        maximum_y_plus=200.0,
    )
    assert result["configuration"]["expected_u_tau"] == 0.005
    assert result["models"]["duprat"]["sample_count"] == 20
    assert result["models"]["duprat"]["attached_samples"] == 20
    assert result["wall_resolved_reference"]["stationarity_target_met"] is True
    assert result["production_force_changed"] is False
    assert len(result["source"]["metadata_sha256"]) == 64
