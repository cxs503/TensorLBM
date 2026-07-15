"""Physical-to-lattice contract for comparable propeller refinement."""
from __future__ import annotations

from pathlib import Path

import pytest

from tensorlbm.propeller_benchmark import (
    PhysicalPropellerRefinementSpec,
    map_physical_propeller_refinement,
)


def test_fixed_physical_rotation_mapping_derives_all_lattice_units(tmp_path: Path) -> None:
    spec = PhysicalPropellerRefinementSpec(
        diameter_m=0.25,
        advance_speed_ms=2.5,
        rotation_rps=10.0,
        nu_m2s=1.0e-6,
        diameter_lu_levels=(32, 48, 64),
        steps_per_revolution=100_000,
        output_root=tmp_path,
    )

    evidence = map_physical_propeller_refinement(spec)

    levels = evidence["levels"]
    assert [level["dx_m"] for level in levels] == pytest.approx([0.25 / 32, 0.25 / 48, 0.25 / 64])
    assert [level["dt_s"] for level in levels] == pytest.approx([1.0e-6] * 3)
    assert [level["rpm_lu"] for level in levels] == pytest.approx([1.0e-5] * 3)
    assert [level["u_lu"] for level in levels] == pytest.approx([0.00032, 0.00048, 0.00064])
    assert [level["nu_lu"] for level in levels] == pytest.approx([1.6384e-8, 3.6864e-8, 6.5536e-8])
    assert all(level["re_d_preserved"] for level in levels)
    assert all(level["j_preserved"] for level in levels)
    assert evidence["status"] == "fail_closed"
    assert evidence["campaign"]["status"] == "not_run"
    assert evidence["metric_convergence"]["status"] == "withheld"
    assert any(v["constraint"] == "tip_mach_preservation" for v in evidence["violations"])
    assert (tmp_path / "propeller_owt" / "physical_lattice_refinement.json").is_file()


def test_physical_mapping_reports_tau_and_low_mach_violations_per_level(tmp_path: Path) -> None:
    evidence = map_physical_propeller_refinement(PhysicalPropellerRefinementSpec(
        diameter_m=1.0,
        advance_speed_ms=1.0,
        rotation_rps=1.0,
        nu_m2s=1.0e-2,
        diameter_lu_levels=(10, 20, 30),
        steps_per_revolution=100,
        tau_max=0.51,
        output_root=tmp_path,
    ))

    violations = evidence["violations"]
    assert any(v["constraint"] == "tau_range" and v["level"] == "level_2" for v in violations)
    assert any(v["constraint"] == "low_mach" and v["level"] == "level_2" for v in violations)
    assert all({"constraint", "level", "actual", "required", "operator"} <= set(v) for v in violations)
