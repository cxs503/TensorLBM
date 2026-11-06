"""Exercise CI scope selection and diagnostic pipelines without running solvers."""

from __future__ import annotations

import os
import re
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


# --- concurrency policy ------------------------------------------------------
#
# GitHub keeps at most one *running* and one *pending* run per concurrency
# group, and a newly queued run cancels the previously *pending* one even when
# ``cancel-in-progress`` is false (only a running run needs the flag).  The
# group expression is therefore load-bearing policy, so the tests below execute
# it rather than matching it textually: the ref-wide main group that shipped
# until 2026-11-06 collapsed every pair of main commits into one group, so each
# merge cancelled the previous pending run and main showed red with zero steps
# executed.  The tests pin the corrected behaviour and fail loudly if the group
# expression is dropped back to a ref-wide form.

_EXPR_TOKEN = re.compile(
    r"\s*(?:(?P<literal>'[^']*')|(?P<operator>==|!=|&&|\|\||\(|\)|,)|(?P<name>[A-Za-z_][\w.]*))"
)
MAIN_REF = "refs/heads/main"
SHA_A = "a" * 40
SHA_B = "b" * 40


def _tokenize(expression):
    tokens, pos = [], 0
    while pos < len(expression):
        match = _EXPR_TOKEN.match(expression, pos)
        if match is None:
            if not expression[pos:].strip():
                break
            raise AssertionError(f"unsupported CI expression syntax: {expression[pos:]!r}")
        pos = match.end()
        tokens.append(match.group(match.lastgroup))
    return tokens


def _truthy(value):
    return not (value is False or value is None or value == "")


class _ExprParser:
    """Evaluate the subset of the GitHub expression language used by ci.yml."""

    def __init__(self, expression, context):
        self.tokens = _tokenize(expression)
        self.pos = 0
        self.context = context

    def parse(self):
        value = self._or()
        assert self.pos == len(self.tokens), f"trailing tokens in {self.tokens[self.pos :]}"
        return value

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _or(self):
        value = self._and()
        while self._peek() == "||":
            self.pos += 1
            other = self._and()
            value = value if _truthy(value) else other
        return value

    def _and(self):
        value = self._comparison()
        while self._peek() == "&&":
            self.pos += 1
            other = self._comparison()
            value = other if _truthy(value) else value
        return value

    def _comparison(self):
        value = self._primary()
        while self._peek() in ("==", "!="):
            operator = self.tokens[self.pos]
            self.pos += 1
            other = self._primary()
            value = (value == other) if operator == "==" else (value != other)
        return value

    def _primary(self):
        token = self._peek()
        assert token is not None, "unexpected end of expression"
        self.pos += 1
        if token.startswith("'"):
            return token[1:-1]
        if token == "(":
            value = self._or()
            assert self._peek() == ")", "unbalanced parenthesis"
            self.pos += 1
            return value
        if token == "format":
            assert self._peek() == "(", "format requires an argument list"
            self.pos += 1
            template = self._primary()
            self.pos += 1  # comma
            argument = self._or()
            assert self._peek() == ")", "unbalanced format() call"
            self.pos += 1
            return template.replace("{0}", str(argument))
        assert token in self.context, f"expression reads undefined context value {token!r}"
        return self.context[token]


def _evaluate(expression, event, ref, sha):
    return _ExprParser(
        expression,
        {
            "github.event_name": event,
            "github.ref": ref,
            "github.sha": sha,
            "github.workflow": "CI",
        },
    ).parse()


def _stringify(value):
    if value is False:
        return "false"
    if value is None:
        return ""
    return str(value)


def _group(template, event="push", ref=MAIN_REF, sha=SHA_A):
    """Render a ``concurrency.group`` template the way GitHub interpolates it."""
    rendered, pos = [], 0
    while True:
        start = template.find("${{", pos)
        if start < 0:
            rendered.append(template[pos:])
            return "".join(rendered)
        end = template.find("}}", start)
        assert end > start, f"unterminated expression in {template!r}"
        rendered.append(template[pos:start])
        rendered.append(_stringify(_evaluate(template[start + 3 : end], event, ref, sha)))
        pos = end + 2


def test_main_commits_get_distinct_concurrency_groups():
    template = _workflow()["concurrency"]["group"]
    first = _group(template, sha=SHA_A)
    second = _group(template, sha=SHA_B)
    assert first == f"ci-CI-{MAIN_REF}-{SHA_A}"
    assert first != second, (
        "two main commits must not share one group (pending run would be cancelled)"
    )


def test_ref_wide_main_group_is_the_regression_this_guards():
    # Negative control: the pre-2026-11-06 expression did collapse main commits,
    # so the assertion above is not vacuous.
    legacy = "ci-${{ github.workflow }}-${{ github.ref }}"
    assert _group(legacy, sha=SHA_A) == _group(legacy, sha=SHA_B)
    assert _group(_workflow()["concurrency"]["group"], sha=SHA_A) != _group(legacy, sha=SHA_A)


def test_branch_groups_stay_ref_scoped_so_new_pushes_supersede():
    template = _workflow()["concurrency"]["group"]
    branch = "refs/heads/agent/m4689"
    first = _group(template, ref=branch, sha=SHA_A)
    second = _group(template, ref=branch, sha=SHA_B)
    assert first == second == f"ci-CI-{branch}"


def test_pull_request_runs_share_no_group_with_main_pushes():
    template = _workflow()["concurrency"]["group"]
    pr = _group(template, event="pull_request", ref="refs/pull/1/merge", sha=SHA_A)
    main = _group(template, event="push", ref=MAIN_REF, sha=SHA_A)
    assert pr != main, "a merge push must not cancel the pull_request verdict for the same commit"


def _unwrap(template):
    """Return the payload of a value that is a single ``${{ ... }}`` block."""
    text = template.strip()
    assert text.startswith("${{") and text.endswith("}}"), template
    return text[3:-2]


@pytest.mark.parametrize("ref", [MAIN_REF, "refs/heads/agent/m4689", "refs/pull/7/merge"])
def test_cancel_in_progress_never_kills_main(ref):
    expression = _unwrap(_workflow()["concurrency"]["cancel-in-progress"])
    cancelled = _evaluate(expression, "push", ref, SHA_A)
    assert cancelled is (ref != MAIN_REF)
