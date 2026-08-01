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
        ("run_suboff_v9_equivalent_level.sh", "L90"),
        ("run_suboff_nested_v3_equivalent_level.sh", "L90"),
        ("run_suboff_nested_v4_continuation_level.sh", "L90"),
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


def test_nested_v4_launcher_expands_audited_l150_continuation(
    tmp_path: Path,
) -> None:
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["TENSORLBM_PYTHON"] = str(fake_python)
    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_suboff_nested_v4_continuation_level.sh"),
            "L150",
            "0",
            str(tmp_path),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    arguments = completed.stdout.splitlines()
    assert "--regularize-restriction" in arguments
    assert "--enforce-transfer-positivity" in arguments
    assert arguments[arguments.index("--ghost-interpolation") + 1] == "trilinear"
    assert arguments[arguments.index("--inner-wall-margin") + 1] == "8"
    assert arguments[arguments.index("--resolved-reynolds-start") + 1] == "2000"
    assert arguments[arguments.index("--viscosity-ramp-start-step") + 1] == "500"
    assert arguments[arguments.index("--viscosity-ramp-end-step") + 1] == "1000"
    assert arguments[arguments.index("--health-interval") + 1] == "100"
    assert arguments[arguments.index("--interface-filter-width") + 1] == "0"
    assert arguments[arguments.index("--interface-filter-strength") + 1] == "0"
    assert arguments[
        arguments.index("--minimum-target-reynolds-convective-times") + 1
    ] == "7.5"
    output = arguments[arguments.index("--output") + 1]
    assert "suboff-nested-v4-equivalent-l150-20k.json" in output


def test_suboff_v9_launcher_uses_quadratic_inlet_pressure_observer(
    tmp_path: Path,
) -> None:
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["TENSORLBM_PYTHON"] = str(fake_python)
    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_suboff_v9_equivalent_level.sh"),
            "L120",
            "0",
            str(tmp_path),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    arguments = completed.stdout.splitlines()
    assert arguments[arguments.index("--pressure-reference") + 1] == "inlet"
    assert arguments[
        arguments.index("--surface-pressure-extrapolation") + 1
    ] == "quadratic"
    output = arguments[arguments.index("--output") + 1]
    assert "suboff-v9-equivalent-l120-16k.json" in output
