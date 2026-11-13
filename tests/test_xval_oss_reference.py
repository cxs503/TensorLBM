"""Tests for the negative-no-oss-reference XVAL discipline (TensorLBM).

Covers ``src/tensorlbm/xval/oss_reference.py`` (probe / coverage / gate /
negative-knowledge payloads) plus one small live parity check of the lbmpy OSS
adapter against the in-house D2Q9 solver.

Run: ``python -m pytest tests/test_xval_oss_reference.py -q``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.xval import oss_reference as OR  # noqa: E402


# --------------------------------------------------------------- probe
def test_probe_record_is_fingerprinted_and_structured():
    probe = OR.probe_oss_references(generated_at="2026-11-13T00:00:00+00:00")
    assert probe["schema"] == OR.PROBE_SCHEMA
    assert probe["generated_at"] == "2026-11-13T00:00:00+00:00"
    assert len(probe["record_sha256"]) == 64
    for rec in probe["references"]:
        for key in ("oss_id", "kind", "available", "version", "path", "sha256", "error"):
            assert key in rec
        assert isinstance(rec["available"], bool)
        if rec["available"]:
            assert rec["error"] is None
        else:
            assert rec["error"]
    assert probe["available_ids"] == [r["oss_id"] for r in probe["references"] if r["available"]]


def test_probe_is_deterministic_for_the_same_environment():
    a = OR.probe_oss_references(generated_at="fixed")
    b = OR.probe_oss_references(generated_at="fixed")
    assert a["record_sha256"] == b["record_sha256"]
    assert a["available_ids"] == b["available_ids"]


def test_probe_reports_absent_module_instead_of_raising():
    probe = OR.probe_oss_references(
        [{"oss_id": "phantom", "kind": "python_module", "target": "definitely_not_a_module_m17996"}],
        generated_at="fixed",
    )
    rec = probe["references"][0]
    assert rec["available"] is False
    assert "not importable" in rec["error"]
    assert probe["available_ids"] == []


def test_probe_reports_absent_binary_instead_of_raising():
    probe = OR.probe_oss_references(
        [{"oss_id": "ghostbin", "kind": "binary", "path_names": ("definitely-not-a-binary-m17996",)}],
        generated_at="fixed",
    )
    rec = probe["references"][0]
    assert rec["available"] is False
    assert "not found" in rec["error"]


# --------------------------------------------------------------- coverage
def test_coverage_marks_available_and_missing_capabilities():
    probe = OR.probe_oss_references(
        [{"oss_id": "jsonpkg", "kind": "python_module", "target": "json"}], generated_at="fixed"
    )
    reqs = [
        {"capability": "core_thing", "evidence": {"jsonpkg": ("json.decoder",)}, "note": "n"},
        {"capability": "missing_thing", "evidence": {"jsonpkg": ("json.definitely_not_here",)}, "note": "n"},
        {"capability": "no_reference_at_all", "evidence": {}, "note": "n"},
    ]
    cov = OR.audit_capability_coverage(probe, reqs)
    by_cap = {row["capability"]: row for row in cov["rows"]}
    assert by_cap["core_thing"]["status"] == "OSS_REFERENCE_AVAILABLE"
    assert by_cap["core_thing"]["supporting_references"] == ["jsonpkg"]
    assert by_cap["missing_thing"]["status"] == "NO_OSS_REFERENCE"
    assert by_cap["no_reference_at_all"]["status"] == "NO_OSS_REFERENCE"
    assert cov["capabilities_with_oss_reference"] == 1
    assert cov["capabilities_without_oss_reference"] == 2


def test_coverage_records_look_alikes_that_were_rejected():
    probe = OR.probe_oss_references(
        [{"oss_id": "jsonpkg", "kind": "python_module", "target": "json"}], generated_at="fixed"
    )
    reqs = [
        {
            "capability": "looked_like_but_is_not",
            "evidence": {},
            "reject_evidence": {"jsonpkg": ("json.decoder",)},
            "note": "rejected because it is a parser, not the capability",
        }
    ]
    cov = OR.audit_capability_coverage(probe, reqs)
    row = cov["rows"][0]
    assert row["status"] == "NO_OSS_REFERENCE"
    assert row["look_alikes_rejected"][0]["oss_id"] == "jsonpkg"


def test_real_registry_has_both_states_and_a_declared_no_reference_family():
    """H2 falsifiability: the real registry must be able to produce NO_OSS_REFERENCE."""
    probe = OR.probe_oss_references()
    cov = OR.audit_capability_coverage(probe)
    statuses = {row["status"] for row in cov["rows"]}
    assert statuses <= {"OSS_REFERENCE_AVAILABLE", "NO_OSS_REFERENCE"}
    by_cap = {row["capability"]: row for row in cov["rows"]}
    # families with no evidence declared at all can never be covered
    assert by_cap["immersed_boundary_ibm"]["status"] == "NO_OSS_REFERENCE"
    assert by_cap["free_surface_volume_tracking"]["status"] == "NO_OSS_REFERENCE"


# --------------------------------------------------------------- gate
def _probe_with(available_id: str) -> dict:
    return OR.probe_oss_references(
        [{"oss_id": available_id, "kind": "python_module", "target": "json"}], generated_at="fixed"
    )


def test_gate_pass_when_reference_resolves_and_errors_are_small():
    gate = OR.XvalReferenceGate(_probe_with("oss_a"))
    verdict = gate.decide(
        OR.Claim("c1", "core", "oss", "oss_a", {"field_l2_rel_pct": 0.02, "gamma_rel_pct": 1e-3})
    )
    assert verdict["status"] == "PASS"
    assert verdict["negative_knowledge_required"] is False
    assert verdict["reference"]["resolved"] is True


def test_gate_fail_when_reference_resolves_but_error_exceeds_band():
    gate = OR.XvalReferenceGate(_probe_with("oss_a"), tolerance_pct=15.0)
    verdict = gate.decide(OR.Claim("c2", "core", "oss", "oss_a", {"a": 3.0, "b": 21.0}))
    assert verdict["status"] == "FAIL"
    assert verdict["culprits"] == ["b"]


def test_gate_withholds_unresolvable_oss_reference_and_wants_negative_knowledge():
    gate = OR.XvalReferenceGate(_probe_with("oss_a"))
    verdict = gate.decide(OR.Claim("c3", "free_surface", "oss", "oss_not_installed", {"a": 0.1}))
    assert verdict["status"] == "WITHHELD"
    assert verdict["negative_knowledge_required"] is True
    assert "not available in probe" in verdict["reference"]["detail"]


def test_gate_withholds_undeclared_analytic_reference():
    gate = OR.XvalReferenceGate(_probe_with("oss_a"))
    verdict = gate.decide(OR.Claim("c4", "core", "analytic", "remembered_table_1974", {"a": 0.1}))
    assert verdict["status"] == "WITHHELD"
    assert verdict["negative_knowledge_required"] is True


def test_gate_accepts_declared_analytic_reference():
    gate = OR.XvalReferenceGate(_probe_with("oss_a"))
    verdict = gate.decide(OR.Claim("c5", "core", "analytic", "taylor_green_2d_exact", {"a": 0.05}))
    assert verdict["status"] == "PASS"
    assert "formula" in verdict["reference"]


def test_gate_refuses_claims_without_reference_or_kpi():
    gate = OR.XvalReferenceGate(_probe_with("oss_a"))
    with pytest.raises(OR.UnreferencedClaimError):
        gate.decide(OR.Claim("c6", "core", "oss", "", {"a": 0.1}))
    with pytest.raises(OR.UnreferencedClaimError):
        gate.decide(OR.Claim("c7", "core", "oss", "oss_a", {}))


def test_there_is_no_code_path_from_unresolved_reference_to_pass():
    """The whole point of the practice: unresolved reference can never PASS."""
    gate = OR.XvalReferenceGate(_probe_with("oss_a"), tolerance_pct=1e9)
    verdict = gate.decide(OR.Claim("c8", "core", "oss", "absent", {"a": 1e-12}))
    assert verdict["status"] == "WITHHELD"


# --------------------------------------------------------------- payloads
def test_negative_knowledge_payloads_match_register_schema():
    probe = OR.probe_oss_references(
        [{"oss_id": "jsonpkg", "kind": "python_module", "target": "json"}], generated_at="fixed"
    )
    reqs = [{"capability": "no_ref_cap", "evidence": {}, "note": "n"}]
    cov = OR.audit_capability_coverage(probe, reqs)
    payloads = OR.negative_knowledge_payloads(cov, mission_id=17996, source_artifacts=["artifacts/x/probe.json"])
    assert len(payloads) == 1
    p = payloads[0]
    register_fields = {
        "title", "hypothesis", "failure_kind", "mission_id", "falsification_evidence", "failure_reason",
        "applicable_conditions", "keywords", "domain_tags", "differentiation_required", "source_artifacts", "confidence",
        "registered_by",
    }
    body = {k: v for k, v in p.items() if not k.startswith("_")}
    assert set(body) <= register_fields
    assert body["failure_kind"] == OR.FAILURE_KIND
    assert body["failure_kind"] in {"hypothesis_falsified", "method_failed", "blocked", "infeasible", "regression"}
    assert body["domain_tags"] == ["TensorLBM"]
    assert "no_ref_cap" in body["keywords"]
    evidence = json.loads(body["falsification_evidence"])
    assert evidence["probe_sha256"] == probe["record_sha256"]
    assert evidence["mission_id"] == 17996


def test_no_payload_for_capabilities_that_have_a_reference():
    probe = OR.probe_oss_references(
        [{"oss_id": "jsonpkg", "kind": "python_module", "target": "json"}], generated_at="fixed"
    )
    reqs = [{"capability": "covered", "evidence": {"jsonpkg": ("json.decoder",)}, "note": "n"}]
    cov = OR.audit_capability_coverage(probe, reqs)
    assert OR.negative_knowledge_payloads(cov) == []


def test_write_audit_bundle_writes_hashed_files(tmp_path):
    probe = OR.probe_oss_references(
        [{"oss_id": "jsonpkg", "kind": "python_module", "target": "json"}], generated_at="fixed"
    )
    cov = OR.audit_capability_coverage(probe, [{"capability": "c", "evidence": {}, "note": "n"}])
    payloads = OR.negative_knowledge_payloads(cov, mission_id=1)
    manifest = OR.write_audit_bundle(tmp_path, probe=probe, coverage=cov, payloads=payloads, mission_id=1)
    for name, digest in manifest["files"].items():
        assert (tmp_path / name).is_file()
        assert len(digest) == 64
    assert manifest["probe_sha256"] == probe["record_sha256"]


# --------------------------------------------------------------- live parity
def _lbmpy_importable() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("lbmpy") is not None
    except Exception:
        return False


@pytest.mark.skipif(not _lbmpy_importable(), reason="lbmpy not installed")
def test_inhouse_and_lbmpy_agree_on_a_small_taylor_green_case(tmp_path):
    """Same TG2DSpec through both arms: energy, orientation and field must agree."""
    import numpy as np

    from tensorlbm.xval.case_spec import TG2DSpec
    from tensorlbm.xval.runners import load_field, run_lbmpy, run_tensorlbm

    spec = TG2DSpec(n=16, re=8.0, u0=0.05, boost=0.0, steps=20, sample_every=10, field_steps=(0, 20))
    inhouse = run_tensorlbm(spec, tmp_path / "tensorlbm", threads=1)
    oss = run_lbmpy(spec, tmp_path / "lbmpy")

    k_i = np.asarray(inhouse["history"]["kinetic_energy"], dtype=float)
    k_o = np.asarray(oss["history"]["kinetic_energy"], dtype=float)
    assert k_i.shape == k_o.shape
    assert abs(k_i[-1] - k_o[-1]) / k_o[-1] < 1e-2

    ux_i0, uy_i0 = load_field(tmp_path / "tensorlbm" / "field_tensorlbm_0.csv")
    ux_o0, uy_o0 = load_field(tmp_path / "lbmpy" / "field_lbmpy_0.csv")
    denom = float(np.sqrt((ux_o0**2 + uy_o0**2).mean()))
    direct = float(np.sqrt(((ux_i0 - ux_o0) ** 2 + (uy_i0 - uy_o0) ** 2).mean())) / denom
    transposed = float(np.sqrt(((ux_i0 - ux_o0.T) ** 2 + (uy_i0 - uy_o0.T) ** 2).mean())) / denom
    assert direct < 1e-8, f"orientation/IC mismatch: direct={direct}, transposed={transposed}"

    ux_i, uy_i = load_field(tmp_path / "tensorlbm" / "field_tensorlbm_20.csv")
    ux_o, uy_o = load_field(tmp_path / "lbmpy" / "field_lbmpy_20.csv")
    field_rel = float(np.sqrt(((ux_i - ux_o) ** 2 + (uy_i - uy_o) ** 2).mean())) / float(
        np.sqrt((ux_o**2 + uy_o**2).mean())
    )
    assert field_rel < 0.05, f"field mismatch {field_rel:.3e}"