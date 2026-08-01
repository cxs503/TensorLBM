from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("script", "level"),
    (
        ("run_suboff_v8_equivalent_level.sh", "L90"),
        ("run_suboff_nested_v3_equivalent_level.sh", "L90"),
        ("run_sphere_v3_equivalent_level.sh", "R9"),
        ("run_cylinder_v4_equivalent_level.sh", "R9"),
        ("run_flat_plate_v4_equivalent_level.sh", "L256"),
    ),
)
def test_campaign_launcher_preflight_imports_current_checkout(
    tmp_path: Path,
    script: str,
    level: str,
) -> None:
    env = os.environ.copy()
    env.update({
        "TENSORLBM_PYTHON": sys.executable,
        "TENSORLBM_PREFLIGHT_ONLY": "1",
    })
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / script), level, "0", str(tmp_path)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    imported = Path(completed.stdout.strip()).resolve()
    assert imported == (ROOT / "src" / "tensorlbm" / "__init__.py").resolve()
