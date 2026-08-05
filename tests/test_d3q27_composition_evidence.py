"""TDD contract tests for D3Q27 MRT component-level composition evidence.

This probe establishes that D3Q27 MRT has **component-level** executable
consistency evidence (equilibrium fixed point, mass/momentum invariants,
finite output, bitwise determinism, source hash binding), while explicitly
documenting that complete wall/geometry/boundary/output composition remains
WITHHELD by the general capability matrix.

RED phase: the probe module does not exist yet, so these tests fail on import.
GREEN phase: after creating ``tensorlbm.d3q27_composition_evidence``, all
tests pass and the machine-readable artifact is produced.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tensorlbm.d3q27 import collide_mrt27, equilibrium27, macroscopic27
from tensorlbm.d3q27_composition_evidence import (
    D3Q27_COMPOSITION_EVIDENCE_VERSION,
    EXPECTED_SOURCE_SHA256,
    run_d3q27_mrt_composition_probe,
)


# ---------------------------------------------------------------------------
# Probe existence and version
# ---------------------------------------------------------------------------

def test_probe_version_is_declared() -> None:
    assert D3Q27_COMPOSITION_EVIDENCE_VERSION == "d3q27-composition-evidence-r1"


def test_expected_source_hash_matches_existing_audit() -> None:
    """The source hash must match the one already locked by the consistency audit."""
    assert EXPECTED_SOURCE_SHA256 == "4b1b55bf7b2aae49857f22d261e75666765764f5eeeb37050f105a17bafc10b5"


# ---------------------------------------------------------------------------
# Probe execution and machine-readable artifact
# ---------------------------------------------------------------------------

def test_probe_returns_machine_readable_artifact() -> None:
    artifact = run_d3q27_mrt_composition_probe()

    assert isinstance(artifact, dict)
    assert artifact["artifact_id"] == "d3q27-mrt-composition-evidence-r1"
    assert artifact["version"] == D3Q27_COMPOSITION_EVIDENCE_VERSION
    assert artifact["lattice"] == "D3Q27"
    assert artifact["collision"] == "MRT"
    assert artifact["entrypoint"] == "tensorlbm.d3q27.collide_mrt27"


def test_probe_config_is_reproducible() -> None:
    artifact = run_d3q27_mrt_composition_probe()
    cfg = artifact["probe_config"]

    assert cfg["shape"] == [2, 3, 4]
    assert cfg["dtype"] == "float32"
    assert cfg["device"] == "cpu"
    assert cfg["seed"] == 2718
    assert cfg["tau"] == 0.8


# ---------------------------------------------------------------------------
# Five component-level checks (all must PASS)
# ---------------------------------------------------------------------------

def test_check_equilibrium_fixed_point_passes() -> None:
    artifact = run_d3q27_mrt_composition_probe()
    check = artifact["checks"]["equilibrium_fixed_point"]

    assert check["status"] == "PASS"
    assert check["max_abs_delta"] <= check["tolerance"]
    assert check["tolerance"] == 1e-6


def test_check_mass_invariant_passes() -> None:
    artifact = run_d3q27_mrt_composition_probe()
    check = artifact["checks"]["mass_invariant"]

    assert check["status"] == "PASS"
    assert check["max_abs_delta"] <= check["tolerance"]
    assert check["tolerance"] == 1e-6


def test_check_momentum_invariant_passes() -> None:
    artifact = run_d3q27_mrt_composition_probe()
    check = artifact["checks"]["momentum_invariant"]

    assert check["status"] == "PASS"
    assert check["max_abs_delta"] <= check["tolerance"]
    assert check["tolerance"] == 1e-6


def test_check_finite_output_passes() -> None:
    artifact = run_d3q27_mrt_composition_probe()
    check = artifact["checks"]["finite_output"]

    assert check["status"] == "PASS"
    assert check["all_finite"] is True


def test_check_determinism_passes() -> None:
    artifact = run_d3q27_mrt_composition_probe()
    check = artifact["checks"]["determinism"]

    assert check["status"] == "PASS"
    assert check["bitwise_identical"] is True


def test_check_source_hash_passes() -> None:
    artifact = run_d3q27_mrt_composition_probe()
    check = artifact["checks"]["source_hash"]

    assert check["status"] == "PASS"
    assert check["sha256"] == EXPECTED_SOURCE_SHA256


# ---------------------------------------------------------------------------
# All checks pass → component_contract tier
# ---------------------------------------------------------------------------

def test_all_checks_pass_yields_component_contract_tier() -> None:
    artifact = run_d3q27_mrt_composition_probe()

    all_pass = all(c["status"] == "PASS" for c in artifact["checks"].values())
    assert all_pass
    assert artifact["component_evidence_tier"] == "component_contract"


# ---------------------------------------------------------------------------
# WITHHELD complete composition aspects
# ---------------------------------------------------------------------------

def test_withheld_composition_aspects_are_documented() -> None:
    artifact = run_d3q27_mrt_composition_probe()
    withheld = artifact["withheld_composition"]

    expected_aspects = {
        "wall_treatment",
        "geometry",
        "boundary",
        "output",
        "streaming_collision_coupling",
        "force_observation",
    }
    assert set(withheld.keys()) == expected_aspects
    for aspect, entry in withheld.items():
        assert entry["status"] == "WITHHELD"
        assert isinstance(entry["reason"], str) and entry["reason"].strip()


def test_capability_matrix_cross_reference_is_honest() -> None:
    artifact = run_d3q27_mrt_composition_probe()
    xref = artifact["capability_matrix_cross_reference"]

    assert xref["general_capability_matrix"] == "WITHHELD"
    assert xref["evidence_tier"] == "no_composition_evidence"
    assert xref["reason_code"] == "WITHHELD_D3Q27_COMPOSITION"
    assert xref["advanced_collision_contract"] == "AVAILABLE"


# ---------------------------------------------------------------------------
# Artifact integrity
# ---------------------------------------------------------------------------

def test_artifact_has_sha256_self_hash() -> None:
    artifact = run_d3q27_mrt_composition_probe()
    assert "artifact_sha256" in artifact
    assert len(artifact["artifact_sha256"]) == 64


def test_artifact_is_json_serializable() -> None:
    artifact = run_d3q27_mrt_composition_probe()
    payload = json.dumps(artifact, sort_keys=True, ensure_ascii=True)
    assert isinstance(payload, str)
    # Round-trip
    restored = json.loads(payload)
    assert restored["artifact_id"] == artifact["artifact_id"]


# ---------------------------------------------------------------------------
# Probe is deterministic across repeated calls
# ---------------------------------------------------------------------------

def test_probe_is_deterministic_across_calls() -> None:
    a1 = run_d3q27_mrt_composition_probe()
    a2 = run_d3q27_mrt_composition_probe()
    # All measured numeric values must be identical
    for key in a1["checks"]:
        assert a1["checks"][key] == a2["checks"][key], f"check {key} differs across calls"


# ---------------------------------------------------------------------------
# Documentation artifact exists
# ---------------------------------------------------------------------------

def test_documentation_artifact_is_present() -> None:
    document = Path(__file__).parents[1] / "docs" / "d3q27_composition_evidence_r1.md"
    assert document.is_file()
