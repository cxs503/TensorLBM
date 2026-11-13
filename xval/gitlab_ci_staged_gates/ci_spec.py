"""Normalize a CI definition into a gate-policy spec (GitLab CI and GitHub Actions).

Why this module exists
----------------------
The XVAL question is *policy-level*: what does a CI definition actually gate?
To answer it for two different CI systems in one coordinate system, both
definitions are reduced to the same record set:

    {"name", "system", "stage_semantics", "stages", "jobs": [...], "notes": [...]}

with one record per job::

    {"id", "stage", "tier", "needs", "events", "selection", "allows_failure",
     "artifacts", "coverage_gate", "timeout_min", "parallelism", "evidence"}

Nothing here is transcribed by hand: every record carries the source evidence
that produced it (``evidence``), so a reviewer can diff the spec against the raw
file, and every source file is identified by sha256.

Tier taxonomy (derived from the OSS ladder under study):

* ``blocking``    -- runs on every event, failure fails the pipeline
* ``advisory``    -- runs but failure cannot fail the pipeline (``allow_failure``)
* ``conditional`` -- only runs on a restricted event subset (e.g. schedules)
* ``manual``      -- needs an explicit trigger

Event vocabulary: ``mr`` (pull/merge request), ``branch_push`` (push to a
non-default branch), ``main_push`` (push to the default branch), ``schedule``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

EVENTS = ("mr", "branch_push", "main_push", "schedule")

_LINT_TOOLS = ("ruff ", "flake8", "mypy", "black ", "isort", "pylint")
_BUILD_TOOLS = ("python -m build", "python -m pip", "pip install", "sphinx-build", "pip ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


_MODULE_INVOCATIONS = {"pytest", "pip", "build", "coverage", "nox", "compileall", "venv"}
_MARKER_RE = re.compile(r"-m\s+(?:\"([^\"]+)\"|'([^']+)'|([A-Za-z0-9_.-]+))")


def marker_expr(commands: list[str]) -> str | None:
    """Return the pytest ``-m`` expression, ignoring ``python -m <module>`` forms."""
    for line in commands:
        if "pytest" not in line and "args=" not in line and "--timeout" not in line:
            continue
        if "pip install" in line and "pytest" not in line:
            continue
        m = _MARKER_RE.search(line)
        if m:
            tok = next(g for g in m.groups() if g)
            if tok in _MODULE_INVOCATIONS:
                continue
            return tok
    return None


def path_scope(commands: list[str]) -> list[str]:
    """Path arguments of the test/lint invocation (empty = whole project)."""
    scope: list[str] = []
    for line in commands:
        if "pytest" in line:
            scope.extend(re.findall(r"(?:^|\s)((?:app/)?tests(?:/[\w./-]+)?)(?=\s|$)", line))
        else:
            scope.extend(re.findall(r"python3?\s+((?:app/)?tests/[\w./-]+\.py)", line))
    return sorted({t.rstrip("/") for t in scope})


def classify_selection(commands: list[str]) -> str:
    """Classify what a job actually inspects: test corpus / lint / build / targeted."""
    text = "\n".join(c for c in commands if c)
    if "nox -s" in text:
        return "nox"  # resolved to its session body by oss_refs.resolve_noxfile()
    if "pytest" in text:
        if (
            re.search(r"pytest[^\n]*?\s(tests/[\w/]+\.py|app/tests/[\w/]+\.py)", text)
            and "-m " not in text
        ):
            return "targeted"
        return "pytest"
    if re.search(r"python3?\s+(tests|app/tests)/[\w/]+\.py", text):
        return "targeted"
    if any(t in text for t in _LINT_TOOLS):
        return "lint"
    if any(t in text for t in _BUILD_TOOLS):
        return "build"
    return "other"


def _clause_events(clause: str) -> list[str]:
    c = clause.replace(" ", "")
    found: list[str] = []
    if "CI_PIPELINE_SOURCE" in c:
        if '"schedule"' in c:
            found.append("schedule")
        if "merge_request_event" in c:
            found.append("mr")
        if '"push"' in c or "web" in c:
            found.extend(["branch_push", "main_push"])
    if "CI_COMMIT_BRANCH" in c:
        found.extend(["branch_push", "main_push"])
    if "CI_PROJECT_PATH" in c:
        found.extend(EVENTS)
    return sorted(set(found), key=EVENTS.index)


def _events_from_condition(cond: str) -> list[str]:
    """Map a GitLab ``rules: if`` condition onto the event vocabulary.

    Clauses are combined the way GitLab evaluates them: a conjunction of
    equalities intersects, an inequality (``!=``) subtracts.
    """
    events = list(EVENTS)
    for clause in cond.split("&&"):
        evs = _clause_events(clause)
        if not evs:
            continue
        if "!=" in clause:
            events = [e for e in events if e not in evs]
        else:
            events = [e for e in events if e in evs]
    return events


def _tier(events: list[str], allows_failure: bool, when: str) -> str:
    if allows_failure:
        return "advisory"
    if when == "manual":
        return "manual"
    if len(events) < len(EVENTS):
        return "conditional"
    return "blocking"


# --------------------------------------------------------------------------- #
# GitLab CI
# --------------------------------------------------------------------------- #
_GITLAB_RESERVED = {
    "stages",
    "variables",
    "workflow",
    "include",
    "default",
    "image",
    "before_script",
    "after_script",
    "cache",
}


def parse_gitlab_ci(path: str | Path) -> dict[str, Any]:
    """Parse a GitLab CI file into the normalized gate-policy spec.

    ``extends`` inheritance is resolved (templates are the ``.name`` keys), the
    ``needs`` distinction between "key omitted" (stage barrier) and ``needs: []``
    (DAG edge removed) is preserved, and stages without an active job are
    recorded because they are the difference between the *declared* and the
    *active* ladder.
    """
    import yaml

    path = Path(path)
    text = path.read_text()
    doc = yaml.safe_load(text) or {}

    stages = list(doc.get("stages") or [])
    templates = {k: v for k, v in doc.items() if isinstance(k, str) and k.startswith(".")}
    jobs: list[dict[str, Any]] = []
    notes: list[str] = []

    for job_id, body in doc.items():
        if job_id in _GITLAB_RESERVED or not isinstance(body, dict) or str(job_id).startswith("."):
            continue
        body = _resolve_extends(body, templates)
        stage = body.get("stage", "test")
        script = body.get("script") or []
        if isinstance(script, str):
            script = [script]
        needs = body.get("needs")
        needs_declared = needs is not None
        events = _gitlab_events(body)
        allows_failure = bool(body.get("allow_failure", False))
        when = str(body.get("when", "on_success"))
        artifacts = body.get("artifacts") if isinstance(body.get("artifacts"), dict) else {}
        jobs.append(
            {
                "id": str(job_id),
                "stage": stage,
                "tier": _tier(events, allows_failure, when),
                "needs": list(needs or []),
                "needs_declared": needs_declared,
                "events": events,
                "selection": {
                    "kind": classify_selection(script),
                    "marker_expr": marker_expr(script),
                    "nox_session": _nox_session(script),
                    "scope": path_scope(script),
                    "commands": script,
                },
                "allows_failure": allows_failure,
                "artifacts": sorted(artifacts.get("paths") or []),
                "artifact_reports": sorted(artifacts.get("reports") or {}),
                "coverage_gate": "coverage" in body,
                "timeout_min": _duration_minutes(body.get("timeout")),
                "parallelism": _par_cores(script),
                "evidence": {
                    "stage": f"stage: {stage}",
                    "needs": "needs: []"
                    if needs == []
                    else (
                        "needs: omitted (stage barrier)" if not needs_declared else "needs: [...]"
                    ),
                    "allow_failure": f"allow_failure: {allows_failure}",
                    "trigger": _gitlab_trigger_evidence(body),
                },
            }
        )

    # Commented-out job blocks: column-0 comments of the form "# job-name:".
    for line in text.splitlines():
        if re.match(r"^# [A-Za-z_][\w-]*:\s*$", line):
            notes.append(f"declared-but-disabled job: {line.lstrip('# ').rstrip(':')}")

    active_stages = {j["stage"] for j in jobs}
    empty_stages = [s for s in stages if s not in active_stages]
    for s in empty_stages:
        notes.append(f"stage with no active job: {s}")

    return {
        "name": "lbmpy.gitlab-ci",
        "system": "gitlab-ci",
        "source": {"path": str(path), "sha256": sha256_file(path)},
        "stage_semantics": "barrier-with-needs-dag",
        "stages": stages,
        "empty_stages": empty_stages,
        "jobs": jobs,
        "notes": notes,
    }


def _resolve_extends(body: dict[str, Any], templates: dict[str, Any]) -> dict[str, Any]:
    """Merge ``extends:`` templates into a job body (one level of nesting)."""
    merged: dict[str, Any] = {}
    ext = body.get("extends")
    parents = [ext] if isinstance(ext, str) else list(ext or [])
    for p in parents:
        tpl = templates.get(p) or {}
        parent = _resolve_extends(tpl, templates) if tpl.get("extends") else tpl
        merged.update(parent)
    for k, v in body.items():
        if k == "extends":
            continue
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            d = dict(merged[k])
            d.update(v)
            merged[k] = d
        else:
            merged[k] = v
    return merged


def _nox_session(script: list[str]) -> str | None:
    for line in script:
        m = re.search(r"nox\s+-s\s+\"?([^\s\"(]+)", line)
        if m:
            return m.group(1)
    return None


def _duration_minutes(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    m = re.match(r"^(\d+)\s*(min|h|hour)", value.strip())
    if not m:
        return None
    n = float(m.group(1))
    return n * 60 if m.group(2).startswith("h") else n


def _par_cores(script: list[str]) -> dict[str, Any]:
    """Detect runner parallelism declarations in a job script."""
    for line in script:
        m = re.search(r"pytest[^\n]*?\s-n\s+([A-Za-z0-9_$@{}]+)", line)
        if m:
            return {"kind": "xdist", "workers": m.group(1)}
    return {"kind": "serial", "workers": 1}


def _gitlab_events(body: dict[str, Any]) -> list[str]:
    """Static event activation for a GitLab job (declared rules only)."""
    rules = body.get("rules")
    if rules:
        events: list[str] = []
        for rule in rules:
            cond = str(rule.get("if", ""))
            events.extend(_events_from_condition(cond) if cond else EVENTS)
        return sorted(set(events), key=EVENTS.index)
    excepted = body.get("except")
    if isinstance(excepted, dict) and (excepted.get("variables") or []):
        # e.g. `except: variables: [$ENABLE_NIGHTLY_BUILDS]`: the job is skipped
        # only when that variable is set, i.e. on nightly-style runs.
        return [e for e in EVENTS if e != "schedule"]
    return list(EVENTS)


def _gitlab_trigger_evidence(body: dict[str, Any]) -> str:
    if body.get("rules"):
        return "rules: " + "; ".join(str(r.get("if", "<always>")) for r in body["rules"])
    if body.get("except"):
        return "except: " + json.dumps(body["except"])[:80]
    return "default (all pipeline sources)"


# --------------------------------------------------------------------------- #
# GitHub Actions
# --------------------------------------------------------------------------- #
def parse_github_actions(path: str | Path) -> dict[str, Any]:
    """Parse a GitHub Actions workflow into the same normalized spec.

    GitHub Actions has no stage ladder: jobs start as soon as their ``needs``
    are satisfied, so a *declared* step order is not a gate.  Tiering has to be
    recovered from ``continue-on-error`` and from the event conditions a scope
    step applies to the test selection; the extraction is a documented rule
    table so the mapping stays auditable.
    """
    import yaml

    path = Path(path)
    doc = yaml.safe_load(path.read_text()) or {}
    triggers = doc["on"] if "on" in doc else doc.get(True)
    if isinstance(triggers, dict):
        workflow_events = sorted(triggers)
    elif isinstance(triggers, list):
        workflow_events = sorted(triggers)
    elif triggers is None:
        workflow_events = []
    else:
        workflow_events = [str(triggers)]

    jobs: list[dict[str, Any]] = []
    for job_id, body in (doc.get("jobs") or {}).items():
        steps = body.get("steps") or []
        script = [s.get("run", "") for s in steps if isinstance(s, dict) and s.get("run")]
        sel = _gh_selection(script)
        needs = body.get("needs") or []
        if isinstance(needs, str):
            needs = [needs]
        allows_failure = bool(body.get("continue-on-error", False))
        events = list(EVENTS)
        jobs.append(
            {
                "id": str(job_id),
                "stage": "<flat>" if not needs else "after:" + ",".join(needs),
                "tier": "advisory" if allows_failure else "blocking",
                "needs": list(needs),
                "needs_declared": bool(needs),
                "events": events,
                "selection": {
                    "kind": sel["kind"],
                    "marker_expr": sel["marker_expr"],
                    "marker_events": sel["marker_events"],
                    "scope": path_scope(script),
                    "nox_session": None,
                    "commands": [s.get("name", "") or "run:" for s in steps if isinstance(s, dict)],
                },
                "allows_failure": allows_failure,
                "artifacts": sorted(set(re.findall(r"path:\s*(\S+)", "\n".join(script)))),
                "artifact_reports": ["junit"] if "junitxml" in "\n".join(script) else [],
                "coverage_gate": any("--cov" in ln for ln in script),
                "timeout_min": body.get("timeout-minutes"),
                "parallelism": _par_cores(script),
                "evidence": {
                    "stage": "github-actions flat job (no stage ladder)",
                    "needs": "needs: " + (str(needs) if needs else "omitted (starts immediately)"),
                    "allow_failure": f"continue-on-error: {allows_failure}",
                    "trigger": sel["evidence"],
                },
            }
        )

    notes = []
    if not any(j["tier"] == "advisory" for j in jobs):
        notes.append("no advisory (non-blocking) gate: every gate failure fails the workflow")
    if not any(j["needs_declared"] for j in jobs):
        notes.append("no needs edges: all jobs start at once, so declared step order is not a gate")

    return {
        "name": "tensorlbm.github-actions-ci",
        "system": "github-actions",
        "source": {"path": str(path), "sha256": sha256_file(path)},
        "stage_semantics": "flat-dag",
        "stages": [],
        "empty_stages": [],
        "jobs": jobs,
        "notes": notes,
        "workflow_events": workflow_events,
    }


_GH_SCOPE_RULES = (
    # (substring of the shell scope step, events that get the filtered selection)
    ('github.event_name }}" = "pull_request', ("mr",)),
    ('github.ref }}" != "refs/heads/main', ("branch_push",)),
)


def _gh_selection(script: list[str]) -> dict[str, Any]:
    """Recover (marker expression, fast events) from the workflow's scope step."""
    joined = "\n".join(script)
    fast: set[str] = set()
    labels = []
    for needle, evs in _GH_SCOPE_RULES:
        if needle in joined:
            fast |= set(evs)
            labels.append(needle)
    marker = marker_expr(script)
    kind = classify_selection(script)
    return {
        "kind": kind,
        "marker_expr": marker,
        # The `-m` filter is applied only on the fast branch of the scope step;
        # the fall-through branch (default branch / schedule) runs the corpus.
        "marker_events": sorted(fast, key=EVENTS.index) if marker else [],
        "evidence": "; ".join(f"scope step matches {lbl!r}" for lbl in labels)
        or "no scope step found (no deselection)",
    }


def dump(spec: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(spec, indent=2, sort_keys=False) + "\n")
