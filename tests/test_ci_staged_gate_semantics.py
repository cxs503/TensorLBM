"""Executable acceptance tests for the staged-gate ladder (XVAL-LBM-GITLAB-CI-STAGED-GATES).

The pre-existing ``tests/test_ci_staged_gates_xval.py`` census-checks the
*declared* structure.  These tests exercise the *semantics*: they falsify the
four claims in the ``.gitlab-ci.yml`` header by fault injection, validate the
artefact contracts, and assert the rung-1 ratchet is green -- i.e. that the
ladder is actually reachable instead of being decorative.

Run: ``PYTHONPATH=. pytest tests/test_ci_staged_gate_semantics.py -q``
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from xval.gitlab_ci_staged_gates.stage_semantics import (
    ALLOWED_FAILURE,
    PRUNED_BARRIER,
    SUCCEEDED,
    Pipeline,
    claim_checks,
    summarize,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_PATH = REPO_ROOT / ".gitlab-ci.yml"


@pytest.fixture(scope="module")
def pipe() -> Pipeline:
    return Pipeline.load(CI_PATH)


@pytest.fixture(scope="module")
def receipt(pipe: Pipeline) -> dict:
    return summarize(pipe)


# --------------------------------------------------------------------------- #
# The real pipeline: no structural debt, all four claims verified
# --------------------------------------------------------------------------- #
def test_pipeline_parses_with_stage_ladder(pipe: Pipeline) -> None:
    assert len(pipe.stages) >= 5
    assert len(pipe.jobs) >= 10
    # Every job must land on a declared stage and every needs target must exist.
    assert [f for f in pipe.structural_findings() if f["severity"] == "error"] == []


def test_artifact_contracts_close(receipt: dict) -> None:
    """Reports must sit inside artifacts:paths; coverage must have a producer."""
    assert [f for f in receipt["findings"] if f["severity"] == "error"] == []


def test_all_ladder_claims_verified(receipt: dict) -> None:
    claims = receipt["claims"]
    assert claims["n_checks"] >= 40, "fault injection must cover every gate and event"
    assert claims["n_failed"] == 0, claims["failed"]


def test_no_decorative_blocking_gate(receipt: dict) -> None:
    """Every blocking gate outside the final stage must prune something, somewhere."""
    last_stage = receipt["stages"][-1]
    gate_names = set(receipt["gates"])
    power = {name: 0 for name in gate_names}
    for payload in receipt["events"].values():
        for name, value in payload["gate_power"].items():
            power[name] = max(power[name], value)
    jobs = Pipeline.load(CI_PATH).jobs
    useless = [
        name
        for name in gate_names
        if power[name] == 0 and jobs[name].stage != last_stage and name != "nightly-full"
    ]
    assert useless == [], f"blocking gates that can never stop anything: {useless}"


def test_event_scoping(pipe: Pipeline) -> None:
    """nightly is schedule-only; deploy is default-branch-only."""
    assert pipe.jobs["nightly-full"].runs_on("schedule")
    assert not any(pipe.jobs["nightly-full"].runs_on(e) for e in ("mr", "branch_push", "main_push"))
    assert pipe.jobs["pages"].runs_on("main_push")
    assert not pipe.jobs["pages"].runs_on("mr")
    assert not pipe.jobs["pages"].runs_on("branch_push")


def test_quality_rung_is_reachable(receipt: dict) -> None:
    """The first rung must be green, else no pipeline ever reaches stage 2.

    ``lint`` and ``format-check`` therefore run the ratchet (no new debt)
    instead of a plain ruff gate that is red by construction.
    """
    for gate in ("lint", "format-check"):
        status = subprocess.run(
            [sys.executable, "tools/ci_ratchet.py", "ruff" if gate == "lint" else "format"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert status.returncode == 0, f"{gate} ratchet is red:\n{status.stdout}{status.stderr}"


def test_baseline_file_is_committed_and_matches() -> None:
    baseline = json.loads((REPO_ROOT / "tools" / "ci_ratchet_baseline.json").read_text())
    assert baseline["ruff"]["total"] > 0
    assert baseline["format"]["total"] > 0
    assert baseline["ruff_version"].startswith("ruff")


# --------------------------------------------------------------------------- #
# Simulator self-checks on synthetic ladders (the claims must be falsifiable)
# --------------------------------------------------------------------------- #
SYNTH = """
stages: [quality, test, deploy]

blocking-gate:
  stage: quality
  needs: []
  script: [echo gate]

advisory-gate:
  stage: quality
  needs: []
  allow_failure: true
  script: [echo advisory]

barrier-job:
  stage: test
  needs: []
  script: [echo barrier]

dag-job:
  stage: test
  needs: [blocking-gate]
  script: [echo dag]

deploy-job:
  stage: deploy
  needs: [dag-job]
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
    - when: never
  script: [echo deploy]
"""


def test_blocking_failure_prunes_later_stages() -> None:
    pipe = Pipeline.from_text(SYNTH)
    state = pipe.simulate("mr", failing={"blocking-gate"})
    assert state["blocking-gate"] == "failed"
    assert state["barrier-job"] == PRUNED_BARRIER  # C2: needs: [] does not dodge the barrier
    assert state["dag-job"] == PRUNED_BARRIER


def test_advisory_failure_does_not_prune() -> None:
    pipe = Pipeline.from_text(SYNTH)
    state = pipe.simulate("mr", failing={"advisory-gate"})
    assert state["advisory-gate"] == ALLOWED_FAILURE
    assert state["barrier-job"] == SUCCEEDED
    assert state["dag-job"] == SUCCEEDED


def test_needs_edge_prunes_when_dependency_fails() -> None:
    pipe = Pipeline.from_text(SYNTH)
    state = pipe.simulate("mr", failing={"blocking-gate"})
    # deploy-job is skipped on mr anyway; main_push proves the DAG edge.
    state_main = pipe.simulate("main_push", failing={"blocking-gate"})
    assert state_main["deploy-job"] != SUCCEEDED
    assert state["deploy-job"] != SUCCEEDED


def test_broken_artifact_report_is_detected() -> None:
    broken = """
stages: [test]
junit-job:
  stage: test
  needs: []
  script: [pytest]
  artifacts:
    paths: [artifacts/ci/pytest.log]
    reports:
      junit: artifacts/ci/junit.xml
"""
    findings = Pipeline.from_text(broken).artifact_findings()
    assert any(f["severity"] == "error" and "never be uploaded" in f["message"] for f in findings)


def test_coverage_without_producer_is_detected() -> None:
    broken = """
stages: [test, reporting]
t:
  stage: test
  needs: []
  script: [pytest -q]
c:
  stage: reporting
  needs: [t]
  script: [coverage combine .coverage.*]
"""
    findings = Pipeline.from_text(broken).artifact_findings()
    assert any("coverage combine" in f["message"] for f in findings)


def test_needs_into_later_stage_is_rejected() -> None:
    broken = """
stages: [a, b]
early:
  stage: a
  needs: [late]
  script: [echo]
late:
  stage: b
  needs: []
  script: [echo]
"""
    findings = Pipeline.from_text(broken).structural_findings()
    assert any("later stage" in f["message"] for f in findings)


def test_unverifiable_condition_is_withheld_not_guessed() -> None:
    odd = """
stages: [test]
weird:
  stage: test
  needs: []
  script: [echo]
  rules:
    - if: $CI_PIPELINE_SOURCE =~ /push/
"""
    pipe = Pipeline.from_text(odd)
    assert pipe.unverifiable, "an undecidable rule must be reported, not silently resolved"


def test_claims_are_falsifiable_by_mutation() -> None:
    """Config-level claims must go red when the declaration stops backing them.

    C1/C2 (fail-fast, barrier-survives-``needs: []``) hold *by construction* in
    any conforming GitLab pipeline and therefore cannot be broken by a YAML
    mutation -- what is falsifiable is whether the configuration actually
    leverages them.  The two config-level claims are C4b (default-branch-only
    deploy) and the gate-power property (no gate may be decorative).
    """
    # Keep the default-branch *declaration* but make the condition always true:
    # the claim must then go red instead of quietly disappearing.
    unconditional = SYNTH.replace(
        "  rules:\n    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH\n    - when: never\n",
        "  rules:\n    - if: $CI_DEFAULT_BRANCH\n",
    )
    mutated = Pipeline.from_text(unconditional)
    assert mutated.jobs["deploy-job"].runs_on("mr"), "mutation should expose deploy on MRs"
    assert any(not c["ok"] for c in claim_checks(mutated)), "C4b must go red"
    assert all(c["ok"] for c in claim_checks(Pipeline.from_text(SYNTH)))

    # Decorative-gate mutation: a blocking gate in the final stage gates nothing.
    moved = SYNTH.replace("blocking-gate:\n  stage: quality", "blocking-gate:\n  stage: deploy")
    assert decorative_gates(Pipeline.from_text(moved), last_stage="deploy") != []


def pipe_decorative_gates(pipe: Pipeline) -> list[str]:
    return decorative_gates(pipe, last_stage=pipe.stages[-1])


def decorative_gates(pipe: Pipeline, last_stage: str) -> list[str]:
    """Blocking gates (outside the final stage) that can never prune anything."""
    power = {name: 0 for name, job in pipe.jobs.items() if job.is_gate}
    for event in ("mr", "branch_push", "main_push", "schedule"):
        for name, value in pipe.gate_power(event).items():
            power[name] = max(power.get(name, 0), value)
    return [
        name
        for name, value in power.items()
        if value == 0 and pipe.jobs[name].stage != last_stage and name != "nightly-full"
    ]
