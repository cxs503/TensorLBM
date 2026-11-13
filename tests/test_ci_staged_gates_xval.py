"""Acceptance tests for the XVAL-LBM-GITLAB-CI-STAGED-GATES deliverable.

These tests are the machine-checkable part of the mission: they re-derive the
KPIs from the committed artefacts (upstream OSS bytes, corpus snapshot, pytest
selection measurements) and assert the two pre-declared hypotheses.

Run: ``PYTHONPATH=src pytest tests/test_ci_staged_gates_xval.py -q``
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xval.gitlab_ci_staged_gates import ci_spec, gate_sim, oss_refs
from xval.gitlab_ci_staged_gates.run_xval import (
    VOCAB_RENAME,
    load_oss_spec,
    parity,
    transplant,
    validate_simulator,
)

HERE = Path(__file__).resolve().parents[1] / "xval" / "gitlab_ci_staged_gates"
SCOPE = {
    "testsuite-cuda-py3.10": ["tests", "app/tests"],
    "testsuite-cpu-py3.13": ["tests", "app/tests"],
}


@pytest.fixture(scope="module")
def corpus() -> gate_sim.Corpus:
    return gate_sim.Corpus.load(HERE / "data" / "corpus_markers.json")


@pytest.fixture(scope="module")
def ours() -> dict:
    return ci_spec.parse_github_actions(HERE.parents[1] / ".github" / "workflows" / "ci.yml")


@pytest.fixture(scope="module")
def oss() -> dict:
    spec, _nox, _meta = load_oss_spec()
    return spec


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


def test_ours_ladder_structure(ours: dict) -> None:
    lad = gate_sim.ladder_structure(ours)
    assert lad["n_stages_declared"] == 0  # GitHub Actions has no stage ladder
    assert lad["needs_edges"] == 0
    assert lad["tier_counts"] == {"blocking": 6}
    tiers = {j["id"]: j["tier"] for j in ours["jobs"]}
    assert tiers["test"] == "blocking"
    fast = {j["id"]: j["selection"]["marker_events"] for j in ours["jobs"]}
    assert fast["test"] == ["mr", "branch_push"]


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


def test_simulator_fidelity_below_15pct(corpus: gate_sim.Corpus) -> None:
    measurements = json.loads((HERE / "data" / "selection_measurements.json").read_text())
    rows = validate_simulator(corpus, measurements)
    assert len(rows) == len(measurements["rows"])
    for row in rows:
        assert row["rel_error_pct"] < 15.0, row
        assert row["abs_diff"] == 0, row  # exact, not just inside tolerance


def test_h1_functional_parity(ours: dict, oss: dict, corpus: gate_sim.Corpus) -> None:
    adapted = transplant(oss, "ALT-2", VOCAB_RENAME)
    for event in ci_spec.EVENTS:
        a = gate_sim.kpis(ours, event, corpus, None, SCOPE)
        b = gate_sim.kpis(adapted, event, corpus, None, SCOPE)
        row = parity(a, b)
        assert row["parity_pass_15pct"], row


def test_h2_vocabulary_not_portable(oss: dict, corpus: gate_sim.Corpus) -> None:
    from xval.gitlab_ci_staged_gates.run_xval import leakage

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
