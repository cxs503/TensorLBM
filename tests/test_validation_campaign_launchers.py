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
        ("run_suboff_nested_v10_scaled_level.sh", "L90"),
        ("run_sphere_v3_equivalent_level.sh", "R9"),
        ("run_cylinder_v4_equivalent_level.sh", "R9"),
        ("run_cylinder_v4_domain_width.sh", "W30"),
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
    assert arguments[arguments.index("--reflux-correction-stencil") + 1] == (
        "exterior_cells"
    )
    assert arguments[arguments.index("--inner-wall-margin") + 1] == "8"
    assert arguments[arguments.index("--resolved-reynolds-start") + 1] == "2000"
    assert arguments[arguments.index("--viscosity-ramp-start-step") + 1] == "500"
    assert arguments[arguments.index("--viscosity-ramp-end-step") + 1] == "1000"
    assert arguments[arguments.index("--health-interval") + 1] == "100"
    assert arguments[arguments.index("--interface-filter-width") + 1] == "0"
    assert arguments[arguments.index("--interface-filter-strength") + 1] == "0"
    assert arguments[
        arguments.index("--maximum-reflux-applied-correction-fraction") + 1
    ] == "0.001"
    assert arguments[arguments.index("--cs-smag") + 1] == "0.05"
    assert arguments[arguments.index("--collision-model") + 1] == (
        "cumulant_smagorinsky"
    )
    assert arguments[arguments.index("--wale-cw") + 1] == "0.5"
    assert arguments[arguments.index("--vreman-cv") + 1] == "0.025"
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


def test_cylinder_domain_launcher_changes_only_lateral_clearance(
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
            str(ROOT / "scripts" / "run_cylinder_v4_domain_width.sh"),
            "W30",
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
    assert arguments[arguments.index("--nx") + 1] == "360"
    assert arguments[arguments.index("--ny") + 1] == "540"
    assert arguments[arguments.index("--radius") + 1] == "9"
    assert arguments[arguments.index("--steps") + 1] == "54000"
    output = arguments[arguments.index("--output") + 1]
    assert "cylinder-v4-domain-w30d-r9-54k.json" in output


def test_sphere_launcher_names_natural_kbc_variant_separately(
    tmp_path: Path,
) -> None:
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update({
        "TENSORLBM_PYTHON": str(fake_python),
        "TENSORLBM_COLLISION_MODEL": "natural_kbc_d3q19",
    })
    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_sphere_v3_equivalent_level.sh"),
            "R9",
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
    assert arguments[arguments.index("--collision-model") + 1] == (
        "natural_kbc_d3q19"
    )
    output = arguments[arguments.index("--output") + 1]
    assert "sphere-v3-natural-kbc-equivalent-r9-7200.json" in output


def test_nested_v10_launcher_scales_all_inner_physical_locations(
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
    env["TENSORLBM_WALL_NORMAL_RAMP_STEPS"] = "0"
    env["TENSORLBM_WALL_SHEAR_RAMP_STEPS"] = "5000"
    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_suboff_nested_v10_scaled_level.sh"),
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
    assert arguments[arguments.index("--inner-wall-margin") + 1] == "15"
    assert arguments[arguments.index("--inner-wake-cells") + 1] == "20"
    assert arguments[arguments.index("--cv-margin") + 1] == "10"
    assert arguments[arguments.index("--aux-cv-margins") + 1] == "5,15"
    assert arguments[arguments.index("--stress-exchange-distance") + 1] == (
        "3.515625"
    )
    assert "--regularize-prolongation" in arguments
    assert arguments[arguments.index("--interface-filter-width") + 1] == "2"
    assert arguments[arguments.index("--interface-filter-strength") + 1] == "1.0"
    assert arguments[arguments.index("--wall-normal-ramp-steps") + 1] == "0"
    assert arguments[arguments.index("--wall-shear-ramp-steps") + 1] == "5000"
    output = arguments[arguments.index("--output") + 1]
    assert "suboff-nested-v10-equivalent-l150-20k.json" in output
