"""TDD contract for the reproducible advanced-collision consistency evidence."""

from __future__ import annotations

import json

from tensorlbm.collision_matrix_cross_validation import (
    EVIDENCE_SCHEMA_VERSION,
    run_collision_matrix_cross_validation,
    write_collision_matrix_evidence,
)


def test_runner_records_only_available_mrt_and_explicitly_skips_withheld_families() -> None:
    evidence = run_collision_matrix_cross_validation()

    assert evidence.schema_version == EVIDENCE_SCHEMA_VERSION
    assert [(item.lattice, item.family) for item in evidence.combinations] == [
        ("D3Q19", "MRT"),
        ("D3Q19", "CM"),
        ("D3Q19", "KBC"),
        ("D3Q19", "CUMULANT"),
        ("D3Q27", "MRT"),
        ("D3Q27", "CM"),
        ("D3Q27", "KBC"),
        ("D3Q27", "CUMULANT"),
    ]
    available = [item for item in evidence.combinations if item.status == "PASS"]
    assert [(item.lattice, item.family) for item in available] == [
        ("D3Q19", "MRT"),
        ("D3Q19", "CUMULANT"),
        ("D3Q27", "MRT"),
        ("D3Q27", "CUMULANT"),
    ]
    for item in available:
        assert [probe.name for probe in item.probes] == [
            "equilibrium_fixed_point",
            "mass_momentum_collision_invariants",
            "finite_non_equilibrium",
        ]
        assert all(probe.status == "PASS" for probe in item.probes)
        assert all(probe.max_abs_error is not None for probe in item.probes)
        assert item.source_provenance
        assert all(len(source.sha256) == 64 for source in item.source_provenance)

    withheld = [item for item in evidence.combinations if item.status == "SKIPPED_WITHHELD"]
    assert len(withheld) == 4
    assert all(item.probes == () and item.withheld_reason.startswith("WITHHELD_") for item in withheld)


def test_runner_is_reproducible_and_writer_emits_machine_readable_hashable_json(tmp_path) -> None:
    first = run_collision_matrix_cross_validation()
    second = run_collision_matrix_cross_validation()
    assert first == second

    output = tmp_path / "collision-matrix-cross-validation-r1.json"
    written = write_collision_matrix_evidence(output)
    raw = output.read_bytes()
    payload = json.loads(raw)
    assert written == output
    assert payload["schema_version"] == EVIDENCE_SCHEMA_VERSION
    assert len(payload["canonical_payload_sha256"]) == 64
    assert payload["combinations"] == json.loads(json.dumps(first.to_dict()))["combinations"]
