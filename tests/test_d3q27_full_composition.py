"""TDD contract tests for D3Q27 MRT **full** composition evidence (R1).

Whereas ``test_d3q27_composition_evidence.py`` establishes *component-level*
evidence for ``collide_mrt27`` in isolation, this module establishes
**composition** evidence: bounce-back + streaming + macroscopic + force
working together as an integrated pipeline.

The probe still marks physical validation as WITHHELD — it proves executable
consistency of the composition, not physical accuracy.

RED phase: the probe module does not exist yet, so these tests fail on import.
GREEN phase: after creating ``tensorlbm.d3q27_full_composition_test``, all
tests pass and the machine-readable artifact is produced.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tensorlbm.d3q27 import (
    C as C27,
    OPPOSITE as OPPOSITE27,
    W as W27,
    collide_mrt27,
    equilibrium27,
    macroscopic27,
    stream27,
)
from tensorlbm.boundaries_d3q27 import bounce_back_cells_27
from tensorlbm.obstacles import compute_obstacle_forces_27
from tensorlbm.d3q27_full_composition_test import (
    D3Q27_FULL_COMPOSITION_VERSION,
    run_d3q27_full_composition_probe,
)


# ---------------------------------------------------------------------------
# Probe existence and version
# ---------------------------------------------------------------------------

def test_probe_version_is_declared() -> None:
    assert D3Q27_FULL_COMPOSITION_VERSION == "d3q27-full-composition-evidence-r1"


# ---------------------------------------------------------------------------
# Probe execution and machine-readable artifact
# ---------------------------------------------------------------------------

def test_probe_returns_machine_readable_artifact() -> None:
    artifact = run_d3q27_full_composition_probe()

    assert isinstance(artifact, dict)
    assert artifact["artifact_id"] == "d3q27-full-composition-evidence-r1"
    assert artifact["version"] == D3Q27_FULL_COMPOSITION_VERSION
    assert artifact["lattice"] == "D3Q27"
    assert artifact["collision"] == "MRT"
    assert artifact["entrypoint"] == "tensorlbm.d3q27.collide_mrt27"


def test_probe_config_is_reproducible() -> None:
    artifact = run_d3q27_full_composition_probe()
    cfg = artifact["probe_config"]

    assert cfg["dtype"] == "float32"
    assert cfg["device"] == "cpu"
    assert cfg["seed"] == 31415
    assert cfg["tau"] == 0.8


# ---------------------------------------------------------------------------
# 1. Bounce-back wall composition checks
# ---------------------------------------------------------------------------

def test_check_bounce_back_involution_passes() -> None:
    artifact = run_d3q27_full_composition_probe()
    check = artifact["checks"]["bounce_back_involution"]
    assert check["status"] == "PASS"
    assert check["max_abs_delta"] <= check["tolerance"]


def test_check_bounce_back_mass_conservation_passes() -> None:
    artifact = run_d3q27_full_composition_probe()
    check = artifact["checks"]["bounce_back_mass_conservation"]
    assert check["status"] == "PASS"
    assert check["max_abs_delta"] <= check["tolerance"]


def test_check_bounce_back_momentum_reflection_passes() -> None:
    artifact = run_d3q27_full_composition_probe()
    check = artifact["checks"]["bounce_back_momentum_reflection"]
    assert check["status"] == "PASS"
    assert check["max_abs_delta"] <= check["tolerance"]


# ---------------------------------------------------------------------------
# 2. Equilibrium + collision + streaming one-step composition checks
# ---------------------------------------------------------------------------

def test_check_full_step_shape_passes() -> None:
    artifact = run_d3q27_full_composition_probe()
    check = artifact["checks"]["full_step_shape"]
    assert check["status"] == "PASS"


def test_check_full_step_mass_periodic_passes() -> None:
    artifact = run_d3q27_full_composition_probe()
    check = artifact["checks"]["full_step_mass_periodic"]
    assert check["status"] == "PASS"
    assert check["max_abs_delta"] <= check["tolerance"]


def test_check_full_step_equilibrium_fixed_point_passes() -> None:
    artifact = run_d3q27_full_composition_probe()
    check = artifact["checks"]["full_step_equilibrium_fixed_point"]
    assert check["status"] == "PASS"
    assert check["max_abs_delta"] <= check["tolerance"]


def test_check_full_step_finite_passes() -> None:
    artifact = run_d3q27_full_composition_probe()
    check = artifact["checks"]["full_step_finite"]
    assert check["status"] == "PASS"
    assert check["all_finite"] is True


# ---------------------------------------------------------------------------
# 3. Macroscopic recovery composition checks
# ---------------------------------------------------------------------------

def test_check_macroscopic_roundtrip_passes() -> None:
    artifact = run_d3q27_full_composition_probe()
    check = artifact["checks"]["macroscopic_roundtrip"]
    assert check["status"] == "PASS"
    assert check["max_abs_delta"] <= check["tolerance"]


def test_check_macroscopic_finite_after_step_passes() -> None:
    artifact = run_d3q27_full_composition_probe()
    check = artifact["checks"]["macroscopic_finite_after_step"]
    assert check["status"] == "PASS"
    assert check["all_finite"] is True


def test_check_macroscopic_mass_after_step_passes() -> None:
    artifact = run_d3q27_full_composition_probe()
    check = artifact["checks"]["macroscopic_mass_after_step"]
    assert check["status"] == "PASS"
    assert check["max_abs_delta"] <= check["tolerance"]


# ---------------------------------------------------------------------------
# 4. Wall-link force extraction composition checks
# ---------------------------------------------------------------------------

def test_check_force_empty_zero_passes() -> None:
    artifact = run_d3q27_full_composition_probe()
    check = artifact["checks"]["force_empty_zero"]
    assert check["status"] == "PASS"
    assert check["max_abs_delta"] <= check["tolerance"]


def test_check_force_finite_passes() -> None:
    artifact = run_d3q27_full_composition_probe()
    check = artifact["checks"]["force_finite"]
    assert check["status"] == "PASS"
    assert check["all_finite"] is True


def test_check_force_momentum_balance_passes() -> None:
    artifact = run_d3q27_full_composition_probe()
    check = artifact["checks"]["force_momentum_balance"]
    assert check["status"] == "PASS"
    assert check["max_abs_delta"] <= check["tolerance"]


def test_check_force_drag_sign_passes() -> None:
    artifact = run_d3q27_full_composition_probe()
    check = artifact["checks"]["force_drag_sign"]
    assert check["status"] == "PASS"
    assert check["drag_positive"] is True


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_check_determinism_passes() -> None:
    artifact = run_d3q27_full_composition_probe()
    check = artifact["checks"]["determinism"]
    assert check["status"] == "PASS"
    assert check["bitwise_identical"] is True


# ---------------------------------------------------------------------------
# All checks pass → composition_contract tier
# ---------------------------------------------------------------------------

def test_all_checks_pass_yields_composition_contract_tier() -> None:
    artifact = run_d3q27_full_composition_probe()

    all_pass = all(c["status"] == "PASS" for c in artifact["checks"].values())
    assert all_pass
    assert artifact["composition_evidence_tier"] == "composition_contract"


# ---------------------------------------------------------------------------
# WITHHELD physical validation
# ---------------------------------------------------------------------------

def test_withheld_physical_validation_is_documented() -> None:
    artifact = run_d3q27_full_composition_probe()
    withheld = artifact["withheld_physical_validation"]

    expected_aspects = {
        "wall_treatment",
        "geometry",
        "boundary",
        "streaming_collision_coupling",
        "force_observation",
        "physical_accuracy",
    }
    assert set(withheld.keys()) == expected_aspects
    for aspect, entry in withheld.items():
        assert entry["status"] == "WITHHELD"
        assert isinstance(entry["reason"], str) and entry["reason"].strip()


def test_capability_matrix_cross_reference_is_honest() -> None:
    artifact = run_d3q27_full_composition_probe()
    xref = artifact["capability_matrix_cross_reference"]

    assert xref["general_capability_matrix"] == "WITHHELD"
    assert xref["evidence_tier"] == "no_composition_evidence"
    assert xref["reason_code"] == "WITHHELD_D3Q27_COMPOSITION"
    assert xref["component_evidence_tier"] == "component_contract"
    assert xref["composition_evidence_tier"] == "composition_contract"


# ---------------------------------------------------------------------------
# Artifact integrity
# ---------------------------------------------------------------------------

def test_artifact_has_sha256_self_hash() -> None:
    artifact = run_d3q27_full_composition_probe()
    assert "artifact_sha256" in artifact
    assert len(artifact["artifact_sha256"]) == 64


def test_artifact_is_json_serializable() -> None:
    artifact = run_d3q27_full_composition_probe()
    payload = json.dumps(artifact, sort_keys=True, ensure_ascii=True)
    assert isinstance(payload, str)
    restored = json.loads(payload)
    assert restored["artifact_id"] == artifact["artifact_id"]


# ---------------------------------------------------------------------------
# Probe is deterministic across repeated calls
# ---------------------------------------------------------------------------

def test_probe_is_deterministic_across_calls() -> None:
    a1 = run_d3q27_full_composition_probe()
    a2 = run_d3q27_full_composition_probe()
    for key in a1["checks"]:
        assert a1["checks"][key] == a2["checks"][key], f"check {key} differs across calls"


# ---------------------------------------------------------------------------
# Documentation artifact exists
# ---------------------------------------------------------------------------

def test_documentation_artifact_is_present() -> None:
    document = Path(__file__).parents[1] / "docs" / "d3q27_full_composition_evidence_r1.md"
    assert document.is_file()
