from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_physical_re_flat_plate_scripts_are_valid_shell() -> None:
    for name in (
        "run_flat_plate_physical_re_level.sh",
        "run_flat_plate_physical_re_chain.sh",
        "assess_flat_plate_physical_re_when_ready.sh",
    ):
        subprocess.run(
            ["bash", "-n", str(ROOT / "scripts" / name)],
            check=True,
        )

    chain = (
        ROOT / "scripts" / "run_flat_plate_physical_re_chain.sh"
    ).read_text(encoding="utf-8")
    assert "flat_plate_convergence_assess.py" in chain
    assert "flat-plate-v6-physical-re13p213m-convergence.json" in chain


def test_physical_re_level_preflight_uses_project_source(tmp_path: Path) -> None:
    environment = os.environ | {
        "TENSORLBM_PREFLIGHT_ONLY": "1",
        "TENSORLBM_PYTHON": os.sys.executable,
    }
    completed = subprocess.run(
        [
            str(ROOT / "scripts" / "run_flat_plate_physical_re_level.sh"),
            "L256",
            "GPU-test-uuid",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert str(ROOT / "src" / "tensorlbm") in completed.stdout
