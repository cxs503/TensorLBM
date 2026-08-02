from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_sphere_projected_pressure_runner_is_valid_and_diagnostic() -> None:
    path = ROOT / "scripts" / "run_sphere_v10_projected_pressure_r9.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    source = path.read_text(encoding="utf-8")
    assert "--projected-pressure-interval 30" in source
    assert "--projected-pressure-reconstruction linear" in source
    assert "--radius 9" in source
