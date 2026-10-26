"""Exercise CI scope selection and diagnostic pipelines without running solvers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text())


@pytest.mark.parametrize(
    "event,ref,fast",
    [
        ("pull_request", "refs/pull/1/merge", True),
        ("push", "refs/heads/feature", True),
        ("push", "refs/heads/main", False),
        ("schedule", "refs/heads/main", False),
        ("workflow_dispatch", "refs/heads/main", False),
        ("workflow_dispatch", "refs/heads/feature", False),
    ],
)
def test_ci_scope_selection(event, ref, fast, tmp_path):
    step = next(s for s in _workflow()["jobs"]["test"]["steps"] if s.get("id") == "scope")
    script = (
        step["run"].replace("${{ github.event_name }}", event).replace("${{ github.ref }}", ref)
    )
    output, summary = tmp_path / "output", tmp_path / "summary"
    env = dict(os.environ, GITHUB_OUTPUT=str(output), GITHUB_STEP_SUMMARY=str(summary))
    result = subprocess.run(["bash", "-e", "-o", "pipefail", "-c", script], env=env)
    assert result.returncode == 0
    expected = 'args=-m "not slow" --timeout=600' if fast else "args=--timeout=1800"
    assert output.read_text().strip() == expected
    assert ("scope=fast" if fast else "scope=full") in summary.read_text()


@pytest.mark.parametrize("job", ["test", "platform-test"])
def test_ci_pytest_pipeline_preserves_failure_and_log(job, tmp_path):
    workflow = _workflow()
    assert workflow["defaults"]["run"]["shell"] == "bash"
    steps = workflow["jobs"][job]["steps"]
    step = next(s for s in steps if "--junitxml=" in s.get("run", ""))
    assert not step.get("continue-on-error", False)
    assert "-o timeout_func_only=false" in step["run"]
    script = step["run"].replace("${{ steps.scope.outputs.args }}", "--timeout=600")
    (tmp_path / "artifacts" / "ci").mkdir(parents=True)
    # Exercise the actual workflow pipeline with a known failing pytest stand-in.
    script = 'pytest() { echo "diagnostic before failure"; return 7; };\n' + script
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 7
    assert "diagnostic before failure" in (tmp_path / "artifacts/ci/pytest.log").read_text()
    upload = next(s for s in steps if s.get("with", {}).get("path") == "artifacts/ci/")
    assert upload["if"] == "always()"
    inventory = next(s for s in steps if s.get("name") == "Record test environment")
    assert inventory["if"] == "always()"
    assert "pip freeze --all" in inventory["run"]
    assert workflow["jobs"][job]["timeout-minutes"] > 0
