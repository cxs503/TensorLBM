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
        ("run_suboff_nested_v11_scaled_wall_level.sh", "L90"),
        ("run_suboff_nested_v12_four_level_l90.sh", "L90"),
        ("run_suboff_nested_v13_mass_conservative_l90.sh", "L90"),
        ("run_suboff_nested_v16_aff8_allocation_probe.sh", "L90"),
        ("run_sphere_v3_equivalent_level.sh", "R9"),
        ("run_cylinder_v4_equivalent_level.sh", "R9"),
        ("run_cylinder_v4_domain_width.sh", "W30"),
        ("run_flat_plate_v4_equivalent_level.sh", "L256"),
        ("run_flat_plate_v5_mass_conservative_level.sh", "L256"),
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
    assert arguments[arguments.index("--stress-exchange-distance") + 1] == (
        "7.03125"
    )
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


def test_four_level_l90_launcher_preserves_physical_cv_and_fails_closed_memory(
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
        "TENSORLBM_RUN_PREFLIGHT_ONLY": "1",
        "TENSORLBM_COMPILE_NATURAL_KBC": "1",
    })
    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_suboff_nested_v12_four_level_l90.sh"),
            "L90",
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
    assert arguments[arguments.index("--deep-wall-margin") + 1] == "7"
    assert arguments[arguments.index("--deep-wake-cells") + 1] == "14"
    assert arguments[arguments.index("--cv-margin") + 1] == "8"
    assert arguments[arguments.index("--aux-cv-margins") + 1] == "4,12"
    assert arguments[arguments.index("--memory-bytes-per-cell") + 1] == "1100"
    assert arguments[arguments.index("--stress-exchange-distance") + 1] == (
        "1.0"
    )
    assert arguments[arguments.index("--collision-model") + 1] == "natural_kbc"
    assert "--compile-natural-kbc" in arguments
    assert "--preflight-only" in arguments


def test_mass_conservative_l90_launcher_can_form_clean_inlet_sponge_pair(
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
        "TENSORLBM_CAMPAIGN_GENERATION": "v14",
        "TENSORLBM_SPONGE_INLET": "1",
    })
    completed = subprocess.run(
        [
            "bash",
            str(
                ROOT
                / "scripts"
                / "run_suboff_nested_v13_mass_conservative_l90.sh"
            ),
            "L90",
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
    assert arguments[arguments.index("--collision-model") + 1] == "natural_kbc"
    assert arguments[arguments.index("--stress-exchange-distance") + 1] == (
        "4.21875"
    )
    assert arguments[arguments.index("--deep-wall-margin") + 1] == "0"
    assert "--sponge-inlet" in arguments
    output = arguments[arguments.index("--output") + 1]
    assert "suboff-nested-v14-equivalent-l90-12k.json" in output


def test_l120_multigpu_probe_preserves_geometric_similarity_and_memory_gates(
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
            str(
                ROOT
                / "scripts"
                / "run_suboff_nested_v20_l120_multigpu_allocation_probe.sh"
            ),
            "GPU-a,GPU-b,GPU-c",
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
    assert arguments[arguments.index("--level-devices") + 1] == (
        "cuda:0,cuda:0,cuda:1,cuda:2"
    )
    assert arguments[arguments.index("--hull-length") + 1] == "120"
    assert arguments[arguments.index("--inner-wall-margin") + 1] == "11"
    assert arguments[arguments.index("--inner-wake-cells") + 1] == "16"
    assert arguments[arguments.index("--deep-wall-margin") + 1] == "9"
    assert arguments[arguments.index("--deep-wake-cells") + 1] == "19"
    assert arguments[arguments.index("--cv-margin") + 1] == "11"
    assert arguments[arguments.index("--aux-cv-margins") + 1] == "5,16"
    assert arguments[arguments.index("--memory-bytes-per-cell") + 1] == "900"
    assert arguments[arguments.index("--collision-chunk-cells") + 1] == (
        "262144"
    )
    assert "--low-memory-wall-macroscopic" in arguments


def test_aff8_bounded_probe_keeps_exact_geometry_and_fail_closed_memory(
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
            str(
                ROOT
                / "scripts"
                / "run_suboff_nested_v21_aff8_bounded_allocation_probe.sh"
            ),
            "GPU-a",
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
    assert arguments[arguments.index("--hull-type") + 1] == "full"
    assert arguments[arguments.index("--deep-wall-margin") + 1] == "7"
    assert arguments[arguments.index("--deep-wake-cells") + 1] == "14"
    assert arguments[arguments.index("--memory-bytes-per-cell") + 1] == "900"
    assert arguments[arguments.index("--collision-chunk-cells") + 1] == (
        "262144"
    )
    assert "--low-memory-wall-macroscopic" in arguments


def test_flat_plate_v5_uses_a_distinct_checkpoint_generation(
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
            str(
                ROOT / "scripts" / "run_flat_plate_v5_mass_conservative_level.sh"
            ),
            "L256",
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
    checkpoint = arguments[arguments.index("--checkpoint") + 1]
    output = arguments[arguments.index("--output") + 1]
    assert "flat-plate-v5-equivalent-l256-32000.ckpt" in checkpoint
    assert "flat-plate-v5-equivalent-l256-32000.json" in output


def test_nested_launcher_routes_optional_inlet_sponge(tmp_path: Path) -> None:
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update({
        "TENSORLBM_PYTHON": str(fake_python),
        "TENSORLBM_SPONGE_INLET": "1",
    })
    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_suboff_nested_v3_equivalent_level.sh"),
            "L90",
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
    assert "--sponge-inlet" in completed.stdout.splitlines()


def test_nested_launcher_can_extend_an_identical_checkpoint(
    tmp_path: Path,
) -> None:
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    seed = tmp_path / "v12-l90-3k.ckpt"
    seed.write_bytes(b"nested continuation provenance fixture")
    result_dir = tmp_path / "continued"
    env = os.environ.copy()
    env.update({
        "TENSORLBM_PYTHON": str(fake_python),
        "TENSORLBM_CAMPAIGN_GENERATION": "v12",
        "TENSORLBM_STEPS": "12000",
        "TENSORLBM_CONTINUE_FROM_CHECKPOINT": str(seed),
    })

    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_suboff_nested_v3_equivalent_level.sh"),
            "L90",
            "0",
            str(result_dir),
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
    assert arguments[arguments.index("--steps") + 1] == "12000"
    assert "--resume" in arguments
    checkpoint = arguments[arguments.index("--checkpoint") + 1]
    output = arguments[arguments.index("--output") + 1]
    assert checkpoint.endswith("suboff-nested-v12-equivalent-l90-12k.ckpt")
    assert output.endswith("suboff-nested-v12-equivalent-l90-12k.json")
    assert Path(checkpoint).read_bytes() == seed.read_bytes()


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


def test_suboff_v9_launcher_can_extend_only_time_and_statistics(
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
        "TENSORLBM_SUBOFF_STEPS": "32000",
        "TENSORLBM_SUBOFF_STATISTICS_WINDOW_STEPS": "26000",
        "TENSORLBM_SUBOFF_RUN_LABEL": "32k-continuation",
    })
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
    assert arguments[arguments.index("--steps") + 1] == "32000"
    assert arguments[arguments.index("--warmup-steps") + 1] == "6000"
    assert arguments[arguments.index("--statistics-window-steps") + 1] == (
        "26000"
    )
    output = arguments[arguments.index("--output") + 1]
    checkpoint = arguments[arguments.index("--checkpoint") + 1]
    assert output.endswith("suboff-v9-equivalent-l120-32k-continuation.json")
    assert checkpoint.endswith("suboff-v9-equivalent-l120-32k-continuation.ckpt")


@pytest.mark.parametrize(
    ("name", "value", "message"),
    (
        ("TENSORLBM_SUBOFF_STEPS", "not-an-integer", "positive integer"),
        (
            "TENSORLBM_SUBOFF_STATISTICS_WINDOW_STEPS",
            "999999",
            "must fit after warmup",
        ),
        ("TENSORLBM_SUBOFF_RUN_LABEL", "bad/path", "unsupported characters"),
    ),
)
def test_suboff_launcher_rejects_invalid_continuation_overrides(
    tmp_path: Path,
    name: str,
    value: str,
    message: str,
) -> None:
    env = os.environ.copy()
    env.update({
        "TENSORLBM_PYTHON": sys.executable,
        name: value,
    })
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

    assert completed.returncode == 2
    assert message in completed.stderr


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


def test_cylinder_domain_launcher_can_extend_an_identical_checkpoint(
    tmp_path: Path,
) -> None:
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    seed = tmp_path / "w30-54k.ckpt"
    seed.write_bytes(b"checkpoint provenance fixture")
    result_dir = tmp_path / "continued"
    env = os.environ.copy()
    env.update({
        "TENSORLBM_PYTHON": str(fake_python),
        "TENSORLBM_STEPS": "63000",
        "TENSORLBM_CONTINUE_FROM_CHECKPOINT": str(seed),
    })

    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_cylinder_v4_domain_width.sh"),
            "W30",
            "0",
            str(result_dir),
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
    assert arguments[arguments.index("--steps") + 1] == "63000"
    assert "--resume" in arguments
    checkpoint = arguments[arguments.index("--checkpoint") + 1]
    output = arguments[arguments.index("--output") + 1]
    assert checkpoint.endswith("cylinder-v4-domain-w30d-r9-63k.ckpt")
    assert output.endswith("cylinder-v4-domain-w30d-r9-63k.json")
    assert Path(checkpoint).read_bytes() == seed.read_bytes()


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


def test_gradient_sgs_pilot_launcher_records_vreman_coefficient(
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
            str(ROOT / "scripts" / "run_suboff_nested_l90_gradient_sgs_pilot.sh"),
            "vreman",
            "0.1",
            "v31",
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
        "cumulant_vreman"
    )
    assert arguments[arguments.index("--vreman-cv") + 1] == "0.1"
    assert arguments[arguments.index("--wale-cw") + 1] == "0.5"
    output = arguments[arguments.index("--output") + 1]
    assert "suboff-nested-v31-l90-vreman-masked-2400.json" in output


def test_cylinder_launcher_names_natural_kbc_variant_separately(
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
            str(ROOT / "scripts" / "run_cylinder_v4_equivalent_level.sh"),
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
    assert "cylinder-v4-natural-kbc-equivalent-r9-54000.json" in output


def test_cylinder_r18_extends_the_same_scaled_grid_family(
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
            str(ROOT / "scripts" / "run_cylinder_v4_equivalent_level.sh"),
            "R18",
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
    assert arguments[arguments.index("--nx") + 1] == "720"
    assert arguments[arguments.index("--ny") + 1] == "720"
    assert arguments[arguments.index("--radius") + 1] == "18"
    assert arguments[arguments.index("--steps") + 1] == "108000"
    assert arguments[arguments.index("--warmup-steps") + 1] == "63000"
    assert arguments[arguments.index("--statistics-window-steps") + 1] == (
        "45000"
    )
    assert arguments[arguments.index("--sponge-width") + 1] == "36"
    assert arguments[arguments.index("--cv-margin") + 1] == "12"


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


@pytest.mark.parametrize(
    ("level", "exchange"),
    (("L90", "4.21875"), ("L120", "5.625"), ("L150", "7.03125")),
)
def test_nested_v11_launcher_scales_flat_plate_wall_exchange(
    tmp_path: Path,
    level: str,
    exchange: str,
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
            str(ROOT / "scripts" / "run_suboff_nested_v11_scaled_wall_level.sh"),
            level,
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
    assert arguments[arguments.index("--stress-exchange-distance") + 1] == exchange
    output = arguments[arguments.index("--output") + 1]
    assert f"suboff-nested-v11-equivalent-l{level[1:]}" in output


def test_v22_compiled_allocation_probe_uses_tensor_tau_executor(
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
            str(
                ROOT
                / "scripts"
                / "run_suboff_nested_v22_compiled_allocation_probe.sh"
            ),
            "L90",
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
    assert "--compile-natural-kbc" in arguments
    output = arguments[arguments.index("--output") + 1]
    assert "suboff-nested-v22-aff1-four-level-l90" in output


def test_v28_allocation_probe_uses_scaled_wall_exchange(
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
            str(
                ROOT
                / "scripts"
                / "run_suboff_nested_v28_scaled_wall_allocation_probe.sh"
            ),
            "L90",
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
    assert "--compile-natural-kbc" in arguments
    assert arguments[arguments.index("--stress-exchange-distance") + 1] == (
        "8.4375"
    )
    output = arguments[arguments.index("--output") + 1]
    assert "suboff-nested-v28-aff1-four-level-l90" in output


@pytest.mark.parametrize(
    ("launcher", "generation", "steps", "resolved_reynolds"),
    (
        ("run_suboff_nested_v23_re200k_compiled_l90.sh", "v23", "3000", "200000"),
        ("run_suboff_nested_v24_compiled_long_l90.sh", "v24", "12000", "100000"),
    ),
)
def test_compiled_production_launchers_lock_corrected_boundary_and_memory_path(
    tmp_path: Path,
    launcher: str,
    generation: str,
    steps: str,
    resolved_reynolds: str,
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
            str(ROOT / "scripts" / launcher),
            "L90",
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
    assert "--compile-natural-kbc" in arguments
    assert "--sponge-inlet" in arguments
    assert "--low-memory-wall-macroscopic" in arguments
    assert arguments[arguments.index("--collision-chunk-cells") + 1] == "262144"
    assert arguments[arguments.index("--resolved-reynolds") + 1] == resolved_reynolds
    expected_exchange = "8.4375" if generation == "v24" else "1.0"
    assert arguments[arguments.index("--stress-exchange-distance") + 1] == (
        expected_exchange
    )
    output = arguments[arguments.index("--output") + 1]
    assert f"suboff-nested-{generation}-equivalent-l90-{int(steps) // 1000}k" in output


def test_generic_compiled_reynolds_pilot_records_generation_and_target(
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
            str(
                ROOT
                / "scripts"
                / "run_suboff_nested_compiled_reynolds_pilot.sh"
            ),
            "v25",
            "500000",
            "L90",
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
    assert arguments[arguments.index("--resolved-reynolds") + 1] == "500000"
    assert "--compile-natural-kbc" in arguments
    assert "--sponge-inlet" in arguments
    output = arguments[arguments.index("--output") + 1]
    assert "suboff-nested-v25-equivalent-l90-3k" in output


def test_v29_re200k_pilot_changes_only_scaled_wall_location(
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
            str(
                ROOT
                / "scripts"
                / "run_suboff_nested_v29_re200k_scaled_wall_l90.sh"
            ),
            "L90",
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
    assert arguments[arguments.index("--resolved-reynolds") + 1] == "200000"
    assert arguments[arguments.index("--stress-exchange-distance") + 1] == (
        "8.4375"
    )
    assert "--compile-natural-kbc" in arguments
    assert "--sponge-inlet" in arguments
    output = arguments[arguments.index("--output") + 1]
    assert "suboff-nested-v29-equivalent-l90-3k" in output


def test_sphere_v4_launcher_locks_bounded_compiled_family(
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
            str(
                ROOT
                / "scripts"
                / "run_sphere_v4_bounded_natural_kbc_level.sh"
            ),
            "R15",
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
    assert arguments[arguments.index("--collision-chunk-cells") + 1] == "262144"
    assert "--compile-natural-kbc" in arguments
    assert "--sponge-inlet" in arguments
    output = arguments[arguments.index("--output") + 1]
    assert "sphere-v4-natural-kbc-equivalent-r15-12000" in output


def test_sphere_v4_convergence_assessor_uses_only_matching_family(
    tmp_path: Path,
) -> None:
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    for radius, steps in ((9, 7200), (12, 9600), (15, 12000)):
        (tmp_path / (
            f"sphere-v4-natural-kbc-equivalent-r{radius}-{steps}.json"
        )).write_text("{}\n", encoding="utf-8")
    env = os.environ.copy()
    env["TENSORLBM_PYTHON"] = str(fake_python)
    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_sphere_v4_convergence_assess.sh"),
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
    inputs = arguments[1:4]
    assert all("sphere-v4-natural-kbc-equivalent" in value for value in inputs)
    assert arguments[arguments.index("--output") + 1].endswith(
        "sphere-v4-natural-kbc-r9-r12-r15-convergence.json"
    )


def test_cylinder_r18_assessor_extends_matching_grid_family(
    tmp_path: Path,
) -> None:
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    for radius, steps in ((9, 54000), (12, 72000), (15, 90000), (18, 108000)):
        (tmp_path / (
            f"cylinder-v4-equivalent-r{radius}-{steps}.json"
        )).write_text("{}\n", encoding="utf-8")
    env = os.environ.copy()
    env["TENSORLBM_PYTHON"] = str(fake_python)
    completed = subprocess.run(
        [
            "bash",
            str(
                ROOT
                / "scripts"
                / "run_cylinder_v4_r18_convergence_assess.sh"
            ),
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
    inputs = arguments[1:5]
    assert len(inputs) == 4
    assert all("cylinder-v4-equivalent-r" in value for value in inputs)
    assert arguments[arguments.index("--output") + 1].endswith(
        "cylinder-v4-r9-r12-r15-r18-convergence.json"
    )
