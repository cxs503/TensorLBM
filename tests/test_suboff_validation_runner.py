"""TDD tests for SUBOFF bare-hull D3Q19+MRT admission→run→force/Ct chain.

These tests verify the full chain end-to-end on a small grid:
  1. wall_function_admission gate works in real config
  2. real run produces force/Ct time series
  3. measured_candidate evidence artifact is produced
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from tensorlbm.suboff_validation_runner import (
    SuboffValidationConfig,
    SuboffValidationEvidence,
    run_suboff_d3q19_mrt_validation,
)
from tensorlbm.wall_function_contract import (
    ValidationLevel,
    WallFunctionCapability,
    WallFunctionCompatibilityError,
)


# ---------------------------------------------------------------------------
# Small-grid config factory
# ---------------------------------------------------------------------------

def _small_config(**overrides: Any) -> SuboffValidationConfig:
    defaults: dict[str, Any] = dict(
        nx=48,
        ny=24,
        nz=24,
        n_steps=20,
        warmup=5,
        u_in=0.06,
        re=200.0,
        hull_length=24.0,
        device="cpu",
        use_wall_function=False,
    )
    defaults.update(overrides)
    return SuboffValidationConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. Admission gate in real config
# ---------------------------------------------------------------------------

class TestAdmissionGateInRealConfig:
    """Verify the wall_function_admission gate behaves correctly in a real run."""

    def test_gate_skipped_when_wall_function_disabled(self) -> None:
        """use_wall_function=False must skip the gate entirely."""
        config = _small_config(use_wall_function=False)
        evidence = run_suboff_d3q19_mrt_validation(config)
        assert evidence.admission["status"] == "skipped"
        assert "use_wall_function=False" in evidence.admission["reason"]

    def test_gate_admits_d3q19_mrt_smagorinsky_at_implementation_only(self) -> None:
        """use_wall_function=True + D3Q19/MRT_SMAGORINSKY must be admitted."""
        config = _small_config(use_wall_function=True)
        evidence = run_suboff_d3q19_mrt_validation(config)
        assert evidence.admission["status"] == "admitted"
        assert evidence.admission["validation"] == ValidationLevel.IMPLEMENTATION_ONLY.name
        assert evidence.admission["capability"] == WallFunctionCapability.LOG_LAW_BODY_FORCE.value
        assert evidence.admission["lattice"] == "D3Q19"
        assert evidence.admission["collision"] == "MRT_SMAGORINSKY"

    def test_gate_withholds_d3q27_lattice(self) -> None:
        """A D3Q27 lattice must be withheld by the admission gate."""
        config = _small_config(use_wall_function=True, lattice="D3Q27")
        with pytest.raises(WallFunctionCompatibilityError, match="WITHHELD_UNVERIFIED_COMBINATION"):
            run_suboff_d3q19_mrt_validation(config)

    def test_gate_withholds_free_surface(self) -> None:
        """Free-surface physics must be withheld by the admission gate."""
        config = _small_config(use_wall_function=True, free_surface=True)
        with pytest.raises(WallFunctionCompatibilityError, match="WITHHELD_UNVERIFIED_COMBINATION"):
            run_suboff_d3q19_mrt_validation(config)


# ---------------------------------------------------------------------------
# 2. Real run produces force/Ct time series
# ---------------------------------------------------------------------------

class TestForceCtTimeSeries:
    """Verify the real solver run produces force and Ct time series."""

    def test_run_completes_all_steps(self) -> None:
        config = _small_config()
        evidence = run_suboff_d3q19_mrt_validation(config)
        assert evidence.runtime["completed_steps"] == config.n_steps
        assert evidence.runtime["requested_steps"] == config.n_steps

    def test_force_time_series_is_non_empty_and_finite(self) -> None:
        config = _small_config()
        evidence = run_suboff_d3q19_mrt_validation(config)
        series = evidence.force_time_series
        assert len(series) == config.n_steps
        for sample in series:
            assert isinstance(sample["step"], int)
            assert isinstance(sample["fx"], float)
            assert isinstance(sample["fy"], float)
            assert isinstance(sample["fz"], float)
            assert all(isinstance(sample[k], float) and abs(sample[k]) < 1e6
                       for k in ("fx", "fy", "fz"))

    def test_ct_time_series_is_non_empty_and_finite(self) -> None:
        config = _small_config()
        evidence = run_suboff_d3q19_mrt_validation(config)
        series = evidence.ct_time_series
        assert len(series) == config.n_steps
        for sample in series:
            assert isinstance(sample["step"], int)
            assert isinstance(sample["ct"], float)
            assert isinstance(sample["ct_fric"], float)
            assert isinstance(sample["ct_pres"], float)

    def test_force_series_step_indices_are_sequential(self) -> None:
        config = _small_config()
        evidence = run_suboff_d3q19_mrt_validation(config)
        steps = [s["step"] for s in evidence.force_time_series]
        assert steps == list(range(1, config.n_steps + 1))

    def test_all_populations_finite_at_every_step(self) -> None:
        config = _small_config()
        evidence = run_suboff_d3q19_mrt_validation(config)
        assert evidence.runtime["all_populations_finite"] is True
        assert evidence.runtime["finite_population_checks"] == config.n_steps

    def test_density_range_is_physical(self) -> None:
        config = _small_config()
        evidence = run_suboff_d3q19_mrt_validation(config)
        assert evidence.runtime["density_min"] > 0.0
        assert evidence.runtime["density_min"] <= evidence.runtime["density_max"]


# ---------------------------------------------------------------------------
# 3. measured_candidate evidence artifact
# ---------------------------------------------------------------------------

class TestMeasuredCandidateEvidence:
    """Verify the evidence artifact has the correct measured_candidate status."""

    def test_evidence_status_is_measured_candidate(self) -> None:
        config = _small_config()
        evidence = run_suboff_d3q19_mrt_validation(config)
        assert evidence.status == "measured_candidate"

    def test_physical_validation_is_false(self) -> None:
        config = _small_config()
        evidence = run_suboff_d3q19_mrt_validation(config)
        assert evidence.physical_validation is False

    def test_steady_state_is_diagnostic_withheld(self) -> None:
        config = _small_config()
        evidence = run_suboff_d3q19_mrt_validation(config)
        assert evidence.steady_state == "diagnostic_withheld"

    def test_evidence_records_solver_configuration(self) -> None:
        config = _small_config()
        evidence = run_suboff_d3q19_mrt_validation(config)
        assert evidence.config["lattice"] == "D3Q19"
        assert evidence.config["collision"] == "MRT"
        assert evidence.config["boundary"] == "bounce_back"
        assert evidence.config["wall"] == "static"
        assert evidence.config["hull_type"] == "bare_hull"
        assert evidence.config["nx"] == config.nx
        assert evidence.config["ny"] == config.ny
        assert evidence.config["nz"] == config.nz

    def test_evidence_records_wetted_area_and_dynamic_pressure(self) -> None:
        config = _small_config()
        evidence = run_suboff_d3q19_mrt_validation(config)
        assert evidence.wetted_area > 0.0
        assert evidence.dynamic_pressure > 0.0

    def test_evidence_is_json_serializable(self) -> None:
        config = _small_config()
        evidence = run_suboff_d3q19_mrt_validation(config)
        artifact = evidence.to_artifact()
        # Must round-trip through JSON without losing the measured_candidate status
        serialized = json.dumps(artifact, sort_keys=True)
        deserialized = json.loads(serialized)
        assert deserialized["status"] == "measured_candidate"
        assert deserialized["physical_validation"] is False
        assert deserialized["steady_state"] == "diagnostic_withheld"
        assert len(deserialized["force_time_series"]) == config.n_steps
        assert len(deserialized["ct_time_series"]) == config.n_steps

    def test_evidence_artifact_written_to_file(self, tmp_path: Path) -> None:
        config = _small_config()
        evidence = run_suboff_d3q19_mrt_validation(config)
        path = tmp_path / "suboff_d3q19_mrt_evidence.json"
        evidence.write_artifact(path)
        assert path.exists()
        artifact = json.loads(path.read_text())
        assert artifact["status"] == "measured_candidate"
        assert artifact["physical_validation"] is False

    def test_wall_function_enabled_also_produces_measured_candidate(self) -> None:
        """The wall-function-admitted path must also produce measured_candidate."""
        config = _small_config(use_wall_function=True)
        evidence = run_suboff_d3q19_mrt_validation(config)
        assert evidence.status == "measured_candidate"
        assert evidence.physical_validation is False
        assert evidence.admission["status"] == "admitted"
