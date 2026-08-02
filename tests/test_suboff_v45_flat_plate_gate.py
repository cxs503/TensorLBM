from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_v45_queue_and_parameterised_suboff_runner_are_valid_shell() -> None:
    for name in (
        "run_suboff_mixed_precision_long.sh",
        "run_suboff_v45_after_physical_re_flat_plate.sh",
    ):
        subprocess.run(
            ["bash", "-n", str(ROOT / "scripts" / name)],
            check=True,
        )


def test_v45_is_gated_by_canonical_flat_plate_and_viscosity() -> None:
    queue = (
        ROOT / "scripts" / "run_suboff_v45_after_physical_re_flat_plate.sh"
    ).read_text(encoding="utf-8")
    assert 'assessment.get("admitted") is not True' in queue
    assert "all_levels_recover_configured_viscosity" in queue
    assert "configured_reynolds_sequence_admitted" in queue
    assert "rational_binary64_cast_to_runtime_dtype_v1" in queue
    assert "TENSORLBM_STRESS_EXCHANGE_DISTANCE=8.4375" in queue
    assert "TENSORLBM_WALL_EXCHANGE_RATIO_TARGET=0.01171875" in queue
    assert "TENSORLBM_WALL_Y_PLUS_UPPER_BOUND=10000" in queue
    assert "87.4" not in queue
