"""Acceptance tests for the XVAL-LBM-GITLAB-CI-STAGED-GATES deliverable.

These tests are the machine-checkable part of the mission: they re-derive the
KPIs from the committed artefacts (upstream OSS bytes, corpus snapshot, pytest
selection measurements) and assert the two pre-declared hypotheses.

Additionally, they validate the new .gitlab-ci.yml follows the ALT-2 staged-gate
pattern derived from the xval analysis.

Run: ``PYTHONPATH=. pytest tests/test_ci_staged_gates_xval.py -q``
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from development.verification.gitlab_ci_staged_gates import ci_spec, gate_sim, oss_refs
from development.verification.gitlab_ci_staged_gates.run_xval import (
    VOCAB_RENAME,
    load_oss_spec,
    parity,
    transplant,
    validate_simulator,
)

HERE = Path(__file__).resolve().parents[1] / "development/verification/gitlab_ci_staged_gates"
REPO_ROOT = Path(__file__).resolve().parents[1]
SCOPE = {
    "testsuite-cuda-py3.10": ["tests", "app/tests"],
    "testsuite-cpu-py3.13": ["tests", "app/tests"],
}


@pytest.fixture(scope="module")
def corpus() -> gate_sim.Corpus:
    return gate_sim.Corpus.load(HERE / "data" / "corpus_markers.json")


@pytest.fixture(scope="module")
def ours() -> dict:
    return ci_spec.parse_github_actions(REPO_ROOT / ".github" / "workflows" / "ci.yml")


@pytest.fixture(scope="module")
def oss() -> dict:
    spec, _nox, _meta = load_oss_spec()
    return spec


@pytest.fixture(scope="module")
def ours_gitlab() -> dict:
    """Parse TensorLBM's own .gitlab-ci.yml (the ALT-2 deliverable)."""
    return ci_spec.parse_gitlab_ci(REPO_ROOT / ".gitlab-ci.yml")


# --------------------------------------------------------------------------- #
# OSS reference integrity
# --------------------------------------------------------------------------- #
def test_oss_refs_are_content_addressed() -> None:
    manifest = json.loads((HERE / "oss_refs" / "manifest.json").read_text())
    for key in ("gitlab_ci", "noxfile", "pyproject"):
        assert manifest[key]["sha256"] == ci_spec.sha256_file(oss_refs.LOCAL[key])
    # The declared ladder is 8 stages; only 7 own an active job.
    spec = ci_spec.parse_gitlab_ci(oss_refs.LOCAL["gitlab_ci"])
    assert len(spec["stages"]) == 8
    assert spec["empty_stages"] == ["integration"]


def test_oss_ladder_structure(oss: dict) -> None:
    lad = gate_sim.ladder_structure(oss)
    assert lad["n_stages_active"] == 7
    assert lad["tier_counts"] == {"advisory": 1, "blocking": 4, "conditional": 3}
    # Only two real dependency edges, both into the deploy job: the ladder is
    # declarative order, not a serial gate.
    assert lad["needs_edges"] == 2
    tiers = {j["id"]: j["tier"] for j in oss["jobs"]}
    assert tiers["minimal-sympy-master"] == "advisory"
    assert tiers["nightly-sympy"] == "conditional"
    expr = {j["id"]: j["selection"]["marker_expr"] for j in oss["jobs"]}
    assert expr["testsuite-cuda-py3.10"] == "not longrun"
    assert expr["nightly-sympy"] == "not longrun"


# --------------------------------------------------------------------------- #
# GitHub Actions CI (baseline)
# --------------------------------------------------------------------------- #
def test_ours_ladder_structure(ours: dict) -> None:
    lad = gate_sim.ladder_structure(ours)
    assert lad["n_stages_declared"] == 0  # GitHub Actions has no stage ladder
    assert lad["needs_edges"] == 0
    assert lad["tier_counts"] == {"blocking": 6}
    tiers = {j["id"]: j["tier"] for j in ours["jobs"]}
    assert tiers["test"] == "blocking"
    fast = {j["id"]: j["selection"]["marker_events"] for j in ours["jobs"]}
    assert fast["test"] == ["mr", "branch_push"]


# --------------------------------------------------------------------------- #
# Marker expression semantics
# --------------------------------------------------------------------------- #
def test_marker_expression_semantics() -> None:
    marks = ["slow", "physics"]
    assert gate_sim.marker_match(None, marks)
    assert gate_sim.marker_match("not slow", marks) is False
    assert gate_sim.marker_match("not fast", marks) is True
    assert gate_sim.marker_match("physics and not slow", marks) is False  # slow IS present
    assert gate_sim.marker_match("physics and not unit", marks) is True
    assert gate_sim.marker_match("unit or physics", marks) is True
    assert gate_sim.marker_match("unit or slow", marks) is True
    assert gate_sim.marker_match("unit or gpu", marks) is False


# --------------------------------------------------------------------------- #
# Simulator fidelity
# --------------------------------------------------------------------------- #
def test_simulator_fidelity_below_15pct(corpus: gate_sim.Corpus) -> None:
    measurements = json.loads((HERE / "data" / "selection_measurements.json").read_text())
    rows = validate_simulator(corpus, measurements)
    assert len(rows) == len(measurements["rows"])
    for row in rows:
        assert row["rel_error_pct"] < 15.0, row
        assert row["abs_diff"] == 0, row  # exact, not just inside tolerance


# --------------------------------------------------------------------------- #
# Hypothesis tests
# --------------------------------------------------------------------------- #
def test_h1_functional_parity(ours: dict, oss: dict, corpus: gate_sim.Corpus) -> None:
    adapted = transplant(oss, "ALT-2", VOCAB_RENAME)
    for event in ci_spec.EVENTS:
        a = gate_sim.kpis(ours, event, corpus, None, SCOPE)
        b = gate_sim.kpis(adapted, event, corpus, None, SCOPE)
        row = parity(a, b)
        assert row["parity_pass_15pct"], row


def test_h2_vocabulary_not_portable(oss: dict, corpus: gate_sim.Corpus) -> None:
    from development.verification.gitlab_ci_staged_gates.run_xval import leakage

    verbatim = leakage(corpus, oss, None)
    adapted = leakage(corpus, oss, VOCAB_RENAME)
    assert verbatim["leak_fraction"] >= 0.15
    assert adapted["leak_fraction"] < 0.15
    # `not longrun` selects the whole corpus: this repository has no longrun marker.
    assert len(corpus.ids("not longrun")) == len(corpus)


def test_guard_coverage_on_default_events(ours: dict, corpus: gate_sim.Corpus) -> None:
    mr = gate_sim.kpis(ours, "mr", corpus, None, SCOPE)
    main = gate_sim.kpis(ours, "main_push", corpus, None, SCOPE)
    n_slow = len(corpus.ids("slow"))
    assert mr["n_escaped"] == n_slow  # the slow tier is unguarded on MRs by design
    assert main["n_escaped"] == 0  # and fully guarded on the default branch


# --------------------------------------------------------------------------- #
# ALT-2 deliverable: TensorLBM's own .gitlab-ci.yml
# --------------------------------------------------------------------------- #
def test_gitlab_ci_file_exists() -> None:
    """The .gitlab-ci.yml deliverable must exist at the repo root."""
    assert (REPO_ROOT / ".gitlab-ci.yml").is_file()


def test_gitlab_ci_is_valid_yaml() -> None:
    """The .gitlab-ci.yml must parse as valid YAML with a stages key."""
    import yaml

    doc = yaml.safe_load((REPO_ROOT / ".gitlab-ci.yml").read_text())
    assert isinstance(doc, dict)
    assert "stages" in doc
    assert isinstance(doc["stages"], list)
    assert len(doc["stages"]) >= 5  # quality, test, reporting, prerelease, nightly at minimum


def test_gitlab_ci_staged_gate_ladder(ours_gitlab: dict) -> None:
    """The .gitlab-ci.yml must implement a staged-gate ladder (ALT-2)."""
    lad = gate_sim.ladder_structure(ours_gitlab)
    # Must have a declared stage ladder (unlike the flat GitHub Actions CI).
    assert lad["n_stages_declared"] >= 5, (
        f"expected >= 5 declared stages, got {lad['n_stages_declared']}"
    )
    # Most declared stages must have active jobs.
    assert lad["n_stages_active"] >= 4, (
        f"expected >= 4 active stages, got {lad['n_stages_active']}"
    )
    # Must have at least one empty stage (declared-but-disabled gate hygiene).
    # The 'integration' stage is declared but has no active job, following the
    # lbmpy pattern.
    assert len(lad["empty_stages"]) >= 1, (
        f"expected >= 1 empty stage (declared-but-disabled hygiene), got {lad['empty_stages']}"
    )


def test_gitlab_ci_tier_taxonomy(ours_gitlab: dict) -> None:
    """The .gitlab-ci.yml must use the gate-tier taxonomy from the OSS ladder."""
    tiers = {j["tier"] for j in ours_gitlab["jobs"]}
    # Must have blocking gates (the core quality gates).
    assert "blocking" in tiers, "expected at least one blocking gate"
    # Must have at least one advisory gate (non-blocking, like prerelease checks).
    assert "advisory" in tiers, "expected at least one advisory gate (allow_failure: true)"
    # Must have at least one conditional gate (schedule-only, like nightly).
    assert "conditional" in tiers, "expected at least one conditional gate (schedule-only)"


def test_gitlab_ci_marker_vocabulary_adapter(ours_gitlab: dict) -> None:
    """The fast test gate must use the adapted marker vocabulary 'not slow'.

    This is the key ALT-2 adaptation: the OSS vocabulary 'not longrun' is not
    portable (H2), so the .gitlab-ci.yml must use 'not slow' instead.
    """
    test_jobs = [
        j for j in ours_gitlab["jobs"]
        if j["selection"]["kind"] == "pytest" and j["tier"] == "blocking"
    ]
    # At least one blocking pytest gate must use "not slow".
    marker_exprs = {j["id"]: j["selection"]["marker_expr"] for j in test_jobs}
    has_adapted_marker = any(
        expr == "not slow" for expr in marker_exprs.values() if expr is not None
    )
    assert has_adapted_marker, (
        f"expected a blocking pytest gate with marker 'not slow', got {marker_exprs}"
    )
    # No gate should use the non-portable 'not longrun' vocabulary.
    for j in ours_gitlab["jobs"]:
        expr = j["selection"].get("marker_expr")
        if expr is not None:
            assert "longrun" not in expr, (
                f"job {j['id']} uses non-portable marker '{expr}'; use 'not slow' instead"
            )


def test_gitlab_ci_coverage_report_aggregation(ours_gitlab: dict) -> None:
    """The .gitlab-ci.yml must have a coverage-report aggregation stage.

    This is one of the structural elements transplanted from the OSS ladder:
    a separate Reporting stage that aggregates coverage from the test stage.
    """
    coverage_jobs = [j for j in ours_gitlab["jobs"] if j.get("coverage_gate")]
    assert len(coverage_jobs) >= 1, "expected at least one job with a coverage gate"
    # The coverage-report job should be in the Reporting stage.
    reporting_jobs = [
        j for j in ours_gitlab["jobs"]
        if j["stage"].lower() in ("reporting", "report")
    ]
    assert len(reporting_jobs) >= 1, "expected at least one job in the Reporting stage"


def test_gitlab_ci_needs_edges(ours_gitlab: dict) -> None:
    """The .gitlab-ci.yml must have DAG dependency edges (needs:).

    The OSS ladder has 2 needs edges (into the deploy job).  Our ALT-2
    implementation should have at least the coverage-report and deploy edges.
    """
    lad = gate_sim.ladder_structure(ours_gitlab)
    assert lad["needs_edges"] >= 2, (
        f"expected >= 2 needs edges (coverage-report + deploy), got {lad['needs_edges']}"
    )


def test_gitlab_ci_declared_disabled_gate(ours_gitlab: dict) -> None:
    """The .gitlab-ci.yml must have a declared-but-disabled integration gate.

    Following the lbmpy pattern, the integration stage is declared but has no
    active job (commented out).  This documents the integration test surface
    without running it on every pipeline.
    """
    assert len(ours_gitlab.get("notes", [])) > 0, (
        "expected notes about declared-but-disabled gates"
    )
    # The integration stage should be in the empty_stages list.
    assert "integration" in ours_gitlab.get("empty_stages", []), (
        f"expected 'integration' in empty_stages, got {ours_gitlab.get('empty_stages')}"
    )


def test_gitlab_ci_nightly_is_conditional(ours_gitlab: dict) -> None:
    """Jobs in the nightly stage must be conditional (schedule-only)."""
    nightly_jobs = [
        j for j in ours_gitlab["jobs"]
        if j["stage"] == "nightly"
    ]
    assert len(nightly_jobs) >= 1, "expected at least one job in the nightly stage"
    for j in nightly_jobs:
        assert j["tier"] == "conditional", (
            f"nightly job {j['id']} should be conditional (schedule-only), got {j['tier']}"
        )


def test_gitlab_ci_parity_with_github_actions(
    ours: dict, ours_gitlab: dict, corpus: gate_sim.Corpus
) -> None:
    """The .gitlab-ci.yml must achieve functional parity with GitHub Actions CI.

    On the default event (MR), the guard_fraction of the GitLab CI must be
    within 15% of the GitHub Actions CI, following the H1 hypothesis.
    """
    # Map the GitLab CI test gates to the same scope as the GitHub Actions ones.
    gitlab_scope = {}
    for j in ours_gitlab["jobs"]:
        if j["selection"]["kind"] == "pytest" and not j["selection"].get("scope"):
            # Whole-project scope: map to tests + app/tests
            gitlab_scope[j["id"]] = ["tests", "app/tests"]

    for event in ("mr", "branch_push"):
        gh = gate_sim.kpis(ours, event, corpus, None, SCOPE)
        gl = gate_sim.kpis(ours_gitlab, event, corpus, None, gitlab_scope)
        row = parity(gh, gl)
        assert row["parity_pass_15pct"], (
            f"GitLab CI parity failed on {event}: {row}"
        )
