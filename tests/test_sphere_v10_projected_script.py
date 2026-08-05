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


def test_projected_pressure_assessment_watcher_is_valid_shell() -> None:
    path = ROOT / "scripts" / "run_sphere_projected_assess_when_ready.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    source = path.read_text(encoding="utf-8")
    assert "assess_sphere_projected_pressure.py" in source
    assert "sphere-v10-projected-linear-r9-assessment.json" in source


def test_projected_pressure_convergence_queue_is_fail_closed() -> None:
    for name in (
        "run_sphere_v10_projected_pressure_level.sh",
        "run_sphere_projected_convergence_after_r9.sh",
    ):
        subprocess.run(
            ["bash", "-n", str(ROOT / "scripts" / name)],
            check=True,
        )
    queue = (
        ROOT / "scripts" / "run_sphere_projected_convergence_after_r9.sh"
    ).read_text(encoding="utf-8")
    assert 'get("single_grid_candidate") is not True' in queue
    level = (
        ROOT / "scripts" / "run_sphere_v10_projected_pressure_level.sh"
    ).read_text(encoding="utf-8")
    assert "projected_interval=30" in level
    assert "projected_interval=40" in level
    assert "projected_interval=50" in level
