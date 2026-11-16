"""Executable semantics of the GitLab staged-gate ladder (XVAL-LBM-GITLAB-CI-STAGED-GATES).

Motivation
----------
``.gitlab-ci.yml`` documents four ladder claims in its header:

    C1  "Each stage is a quality gate: a failure in a blocking gate stops the
         pipeline from progressing to later stages."
    C2  "``needs: []`` means the job starts immediately (no DAG dependency),
         but the stage barrier still applies."
    C3  "Advisory gates (``allow_failure: true``) run but cannot stop the
         pipeline."
    C4  "Conditional gates (rules-restricted) only run on specific events."

Until now those claims were prose: the existing XVAL harness censuses the
declared structure (``gate_sim.ladder_structure``) and measures marker
coverage, but *nothing* executed the barrier/fail-fast/advisory/event logic.
A declared-but-never-exercised gate ladder is exactly the failure mode the
anchor is about, so this module re-derives the semantics independently of
``ci_spec`` (a second, raw-YAML reading — two independent parsers disagreeing
is itself a finding) and makes C1-C4 falsifiable by fault injection.

Model (GitLab subset actually used by this repository)
-----------------------------------------------------
* A pipeline is created for one *event*: ``mr``, ``branch_push``,
  ``main_push`` or ``schedule``.  Event activation comes from ``rules:``
  (first matching rule wins; ``when: never`` deactivates the job).
* Jobs run stage by stage in the order of ``stages:``; a blocking failure in
  stage *i* prunes every job in stages *> i* (C1).  A failed
  ``allow_failure: true`` job does not prune (C3).
* ``needs:`` creates a DAG edge; a job whose needs are pruned is itself
  pruned.  ``needs: []`` removes DAG edges but **not** the stage barrier (C2).
* ``needs:`` pointing at a job that does not exist on this event makes the
  pipeline un-creatable (GitLab errors) unless ``optional: true`` -- reported
  as a structural finding rather than silently modelled as "fine".

Conditions that the evaluator cannot decide are never guessed: they are
returned as ``unverifiable`` and surfaced in the receipt (honest WITHHELD).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

EVENTS = ("mr", "branch_push", "main_push", "schedule")

#: Synthetic pipeline contexts.  Only variables that the repository's rules
#: actually reference need to be faithful.
CONTEXTS: dict[str, dict[str, str]] = {
    "mr": {
        "CI_PIPELINE_SOURCE": "merge_request_event",
        "CI_COMMIT_BRANCH": "feature/xval",
        "CI_COMMIT_REF_NAME": "feature/xval",
        "CI_DEFAULT_BRANCH": "main",
        "CI_MERGE_REQUEST_ID": "1",
        "CI_MERGE_REQUEST_TARGET_BRANCH_NAME": "main",
    },
    "branch_push": {
        "CI_PIPELINE_SOURCE": "push",
        "CI_COMMIT_BRANCH": "feature/xval",
        "CI_COMMIT_REF_NAME": "feature/xval",
        "CI_DEFAULT_BRANCH": "main",
    },
    "main_push": {
        "CI_PIPELINE_SOURCE": "push",
        "CI_COMMIT_BRANCH": "main",
        "CI_COMMIT_REF_NAME": "main",
        "CI_DEFAULT_BRANCH": "main",
    },
    "schedule": {
        "CI_PIPELINE_SOURCE": "schedule",
        "CI_COMMIT_BRANCH": "main",
        "CI_COMMIT_REF_NAME": "main",
        "CI_DEFAULT_BRANCH": "main",
    },
}

# Job states produced by the simulator.
SUCCEEDED = "succeeded"
FAILED = "failed"  # blocking failure -> prunes later stages
ALLOWED_FAILURE = "allowed_failure"  # failed but allow_failure: true
PRUNED_BARRIER = "pruned_barrier"  # earlier stage blocked the ladder (C1/C2)
PRUNED_NEEDS = "pruned_needs"  # a needs: dependency was pruned/failed
SKIPPED_EVENT = "skipped_event"  # rules: when: never / no matching rule
PRUNED_MANUAL = "manual"  # when: manual and not triggered

_RAN = (SUCCEEDED, ALLOWED_FAILURE)
_DID_NOT_RUN = (PRUNED_BARRIER, PRUNED_NEEDS, SKIPPED_EVENT, PRUNED_MANUAL)


class UnverifiableCondition(RuntimeError):
    """A ``rules: if`` expression outside the deterministic subset."""


# --------------------------------------------------------------------------- #
# Condition evaluation
# --------------------------------------------------------------------------- #
_ATOM = re.compile(r"^\$(\w+)\s*(?:(==|!=)\s*(.+))?$")


def _atom(expr: str, ctx: dict[str, str]) -> bool:
    expr = expr.strip()
    if not expr:
        return False
    m = _ATOM.match(expr)
    if not m:
        raise UnverifiableCondition(expr)
    var, op, rhs = m.group(1), m.group(2), m.group(3)
    if var not in ctx:
        raise UnverifiableCondition(expr)
    lhs_val = ctx[var]
    if op is None:
        return bool(lhs_val)
    rhs = rhs.strip()
    if rhs.startswith('"') and rhs.endswith('"') or rhs.startswith("'") and rhs.endswith("'"):
        ref = rhs[1:-1]
    elif rhs.startswith("$"):
        ref_var = rhs[1:]
        if ref_var not in ctx:
            raise UnverifiableCondition(expr)
        ref = ctx[ref_var]
    else:
        raise UnverifiableCondition(expr)
    return (lhs_val == ref) if op == "==" else (lhs_val != ref)


def evaluate(expr: str, ctx: dict[str, str]) -> bool:
    """Evaluate a GitLab ``rules: if`` expression over a pipeline context.

    Supports the ``$VAR <op> value`` atom subset combined with ``&&`` / ``||``
    and redundant parentheses.  Anything else raises ``UnverifiableCondition``.
    """
    text = expr.strip()
    while text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    if "||" in text:
        parts = [p for p in text.split("||")]
        return any(evaluate(p, ctx) for p in parts)
    if "&&" in text:
        return all(evaluate(p, ctx) for p in text.split("&&"))
    return _atom(text, ctx)


# --------------------------------------------------------------------------- #
# Pipeline model
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    name: str
    stage: str
    needs: tuple[str, ...] | None  # None = key omitted (stage barrier)
    allow_failure: bool
    script: str
    artifacts_paths: tuple[str, ...] = ()
    artifact_reports: tuple[str, ...] = ()
    rules: tuple[dict[str, Any], ...] = ()
    job_when: str = "on_success"

    # ---------------------------------------------------------------- events
    def runs_on(self, event: str) -> bool:
        """Resolve ``rules:`` for ``event`` (first match wins)."""
        ctx = CONTEXTS[event]
        if not self.rules:
            return self.job_when != "never"
        for rule in self.rules:
            cond = rule.get("if")
            matched = True if cond is None else evaluate(str(cond), ctx)
            if matched:
                return str(rule.get("when", "on_success")) != "never"
        return False  # no rule matched -> job absent from the pipeline

    def unverifiable_rules(self) -> list[str]:
        out = []
        for rule in self.rules:
            cond = rule.get("if")
            if cond is None:
                continue
            if not all(_decidable(str(cond), ctx) for ctx in CONTEXTS.values()):
                out.append(str(cond))
        return out

    # --------------------------------------------------------------- gating
    @property
    def is_gate(self) -> bool:
        """A blocking gate: its failure can stop the ladder."""
        return not self.allow_failure and self.job_when not in ("manual", "never")


def _decidable(cond: str, ctx: dict[str, str]) -> bool:
    try:
        evaluate(cond, ctx)
    except UnverifiableCondition:
        return False
    return True


def tier_of(job: Job) -> str:
    """Gate strength of a job, independent of its event scope.

    ``manual`` (needs an explicit trigger) > ``advisory`` (``allow_failure``,
    cannot stop the pipeline) > ``blocking``.  Event scope is reported
    separately as ``events`` so "blocking on three events" stays visible.
    """
    if job.job_when == "manual":
        return "manual"
    if job.allow_failure:
        return "advisory"
    return "blocking"


@dataclass
class Pipeline:
    stages: list[str]
    jobs: dict[str, Job]
    source: str = "<memory>"
    #: sha256 of the exact bytes the model was built from.  Without it a
    #: committed receipt cannot be told apart from a stale one.
    source_sha256: str = ""
    unverifiable: dict[str, list[str]] = field(default_factory=dict)

    # ------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str | Path, root: str | Path | None = None) -> "Pipeline":
        """Load a CI definition; ``root`` stores the source as a repo-relative path.

        A committed receipt must be reproducible on any checkout, so the source
        is recorded relative to ``root`` whenever the file lives under it.
        """
        p = Path(path).resolve()
        data = p.read_bytes()
        source = str(p)
        if root is not None:
            try:
                source = p.relative_to(Path(root).resolve()).as_posix()
            except ValueError:
                source = p.as_posix()
        return cls.from_text(data.decode("utf-8"), source=source)

    @classmethod
    def from_text(cls, text: str, source: str = "<memory>") -> "Pipeline":
        doc = yaml.safe_load(text) or {}
        stages = [str(s) for s in (doc.get("stages") or ["test"])]
        templates = {k: v for k, v in doc.items() if str(k).startswith(".")}
        jobs: dict[str, Job] = {}
        unverifiable: dict[str, list[str]] = {}
        for name, body in doc.items():
            if str(name).startswith(".") or not isinstance(body, dict):
                continue
            if "script" not in body and "trigger" not in body:
                continue
            resolved = _resolve_extends(body, templates)
            stage = str(resolved.get("stage", "test"))
            needs_raw = resolved.get("needs")
            needs = None if needs_raw is None else tuple(str(n) for n in needs_raw)
            artifacts = (
                resolved.get("artifacts") if isinstance(resolved.get("artifacts"), dict) else {}
            )
            reports = artifacts.get("reports") if isinstance(artifacts.get("reports"), dict) else {}
            script = resolved.get("script") or []
            if isinstance(script, str):
                script = [script]
            job = Job(
                name=str(name),
                stage=stage,
                needs=needs,
                allow_failure=bool(resolved.get("allow_failure", False)),
                script="\n".join(str(s) for s in script),
                artifacts_paths=tuple(str(p) for p in (artifacts.get("paths") or [])),
                artifact_reports=_report_paths(reports),
                rules=tuple(resolved.get("rules") or ())
                if isinstance(resolved.get("rules"), (list, tuple))
                else (),
                job_when=str(resolved.get("when", "on_success")),
            )
            jobs[job.name] = job
            bad = job.unverifiable_rules()
            if bad:
                unverifiable[job.name] = bad
        return cls(
            stages=stages,
            jobs=jobs,
            source=source,
            source_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            unverifiable=unverifiable,
        )

    # ------------------------------------------------------------ structure
    def structural_findings(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        idx = {s: i for i, s in enumerate(self.stages)}
        for job in self.jobs.values():
            if job.stage not in idx:
                out.append(_f("error", job.name, f"stage '{job.stage}' not declared in stages:"))
            for dep in job.needs or ():
                if dep not in self.jobs:
                    out.append(_f("error", job.name, f"needs unknown job '{dep}'"))
                    continue
                if job.stage in idx and self.jobs[dep].stage in idx:
                    if idx[self.jobs[dep].stage] > idx[job.stage]:
                        out.append(
                            _f(
                                "error",
                                job.name,
                                f"needs '{dep}' lives in later stage "
                                f"'{self.jobs[dep].stage}' (GitLab rejects this)",
                            )
                        )
            if job.allow_failure and job.stage == self.stages[-1] if self.stages else False:
                out.append(
                    _f("info", job.name, "advisory gate in the last stage cannot gate anything")
                )
        out.extend(self._cycle_findings())
        return out

    def _cycle_findings(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        colour: dict[str, int] = {}

        def visit(name: str, stack: list[str]) -> None:
            colour[name] = 1
            for dep in self.jobs[name].needs or ():
                if dep not in self.jobs:
                    continue
                if colour.get(dep, 0) == 1:
                    out.append(_f("error", name, f"needs cycle: {' -> '.join(stack + [dep])}"))
                elif colour.get(dep, 0) == 0:
                    visit(dep, stack + [dep])
            colour[name] = 2

        for name in self.jobs:
            if colour.get(name, 0) == 0:
                visit(name, [name])
        return out

    def artifact_findings(self) -> list[dict[str, str]]:
        """Producer/consumer closure for reports and coverage aggregation."""
        out: list[dict[str, str]] = []
        for job in self.jobs.values():
            for rep in job.artifact_reports:
                kind, _, path = rep.partition(":")
                if not path:
                    continue
                if not _covered(path, job.artifacts_paths):
                    out.append(
                        _f(
                            "error",
                            job.name,
                            f"{kind} report '{path}' is not inside artifacts:paths "
                            f"{list(job.artifacts_paths)} -- the report would never be uploaded",
                        )
                    )
            if "coverage combine" in job.script:
                producers = [
                    self.jobs[d].name
                    for d in (job.needs or ())
                    if "--cov" in self.jobs[d].script
                    and any(".coverage" in p for p in self.jobs[d].artifacts_paths)
                ]
                if not producers:
                    out.append(
                        _f(
                            "error",
                            job.name,
                            "runs 'coverage combine' but no needs: producer uploads "
                            ".coverage.* -- the job cannot see any data file",
                        )
                    )
        return out

    # ------------------------------------------------------- event + exec
    def active(self, event: str) -> dict[str, Job]:
        return {n: j for n, j in self.jobs.items() if j.runs_on(event)}

    def event_findings(self, event: str) -> list[dict[str, str]]:
        """Config that makes the pipeline un-creatable on ``event``."""
        out: list[dict[str, str]] = []
        active = self.active(event)
        for job in active.values():
            for dep in job.needs or ():
                if dep not in self.jobs:
                    continue  # already an error in structural_findings
                if dep not in active:
                    out.append(
                        _f(
                            "error",
                            job.name,
                            f"needs '{dep}' but that job is absent on event '{event}' "
                            f"(needs it:optional: true or align the rules)",
                        )
                    )
        return out

    def simulate(
        self,
        event: str,
        failing: Iterable[str] = (),
        approved_manual: bool = True,
    ) -> dict[str, str]:
        """Run the ladder for ``event`` with fault injection on ``failing``."""
        failing = set(failing)
        active = self.active(event)
        state: dict[str, str] = {n: SKIPPED_EVENT for n in self.jobs}
        blocked_from: int | None = None

        for si, stage in enumerate(self.stages):
            stopped = blocked_from is not None and si > blocked_from
            for job in active.values():
                if job.stage != stage:
                    continue
                if stopped:
                    state[job.name] = PRUNED_BARRIER
                    continue
                if job.job_when == "manual" and not approved_manual:
                    state[job.name] = PRUNED_MANUAL
                    continue
                if job.needs is not None:
                    bad = [d for d in job.needs if state.get(d) not in _RAN]
                    if bad:
                        state[job.name] = PRUNED_NEEDS
                        continue
                if job.name in failing:
                    state[job.name] = FAILED if not job.allow_failure else ALLOWED_FAILURE
                    if not job.allow_failure and blocked_from is None:
                        blocked_from = si
                else:
                    state[job.name] = SUCCEEDED
        return state

    # ------------------------------------------------------------- metrics
    def gate_power(self, event: str) -> dict[str, int]:
        """Downstream jobs pruned by each blocking gate failing on its own."""
        baseline = self.simulate(event)
        ran_baseline = {n for n, s in baseline.items() if s in _RAN}
        out: dict[str, int] = {}
        active = self.active(event)
        for name, job in active.items():
            if not job.is_gate:
                continue
            state = self.simulate(event, failing={name})
            pruned = sum(
                1
                for n, s in state.items()
                if s in _DID_NOT_RUN
                and (baseline.get(n) in _RAN or n in ran_baseline)
                and n != name
            )
            out[name] = pruned
        return out

    # ------------------------------------------------------------- inventory
    def ladder(self) -> list[dict[str, Any]]:
        """Per-stage inventory: which jobs, at which strength, on which events.

        This is the single machine-readable source of truth for the ladder
        tables in the receipt and in ``docs/ci/gitlab_ci_staged_gates.md``;
        the documents are generated from it, so prose cannot drift from the
        pipeline.
        """
        out: list[dict[str, Any]] = []
        for stage in self.stages:
            jobs = sorted(n for n, j in self.jobs.items() if j.stage == stage)
            out.append(
                {
                    "stage": stage,
                    "jobs": [
                        {
                            "name": name,
                            "tier": tier_of(self.jobs[name]),
                            "events": [e for e in EVENTS if self.jobs[name].runs_on(e)],
                            "needs": list(self.jobs[name].needs)
                            if self.jobs[name].needs is not None
                            else None,
                        }
                        for name in jobs
                    ],
                }
            )
        return out

    def receipt(self) -> dict[str, Any]:
        """Machine-readable verdict bundle for all four ladder claims."""
        events: dict[str, Any] = {}
        for event in EVENTS:
            state = self.simulate(event)
            events[event] = {
                "n_active": sum(1 for s in state.values() if s in _RAN),
                "n_skipped": sum(1 for s in state.values() if s == SKIPPED_EVENT),
                "ran": sorted(n for n, s in state.items() if s in _RAN),
                "gate_power": self.gate_power(event),
                "event_findings": self.event_findings(event),
            }
        findings = self.structural_findings() + self.artifact_findings()
        for event in EVENTS:
            findings.extend(events[event]["event_findings"])
        return {
            "source": self.source,
            "source_sha256": self.source_sha256,
            "ladder": self.ladder(),
            "stages": self.stages,
            "n_jobs": len(self.jobs),
            "gates": sorted(n for n, j in self.jobs.items() if j.is_gate),
            "advisory": sorted(n for n, j in self.jobs.items() if j.allow_failure),
            "immediate_jobs": sorted(n for n, j in self.jobs.items() if j.needs == ()),
            "unverifiable_rules": self.unverifiable,
            "findings": findings,
            "n_error": sum(1 for f in findings if f["severity"] == "error"),
            "events": events,
        }


def _report_paths(reports: dict[str, Any]) -> tuple[str, ...]:
    """Flatten ``artifacts:reports:`` into ``kind:path`` pairs.

    GitLab nests coverage reports one level deeper than JUnit, so both shapes
    (``junit: path`` and ``coverage_report: {path: ...}``) must be handled.
    """
    out: list[str] = []
    for kind, value in reports.items():
        if isinstance(value, str):
            out.append(f"{kind}:{value}")
        elif isinstance(value, dict):
            for sub in ("path", "paths"):
                item = value.get(sub)
                if isinstance(item, str):
                    out.append(f"{kind}:{item}")
                elif isinstance(item, (list, tuple)):
                    out.extend(f"{kind}:{i}" for i in item)
    return tuple(out)


def _f(severity: str, job: str, message: str) -> dict[str, str]:
    return {"severity": severity, "job": job, "message": message}


def _covered(path: str, declared: Iterable[str]) -> bool:
    for d in declared:
        d = d.rstrip("/")
        if path == d or path.startswith(d + "/"):
            return True
    return False


def _resolve_extends(body: dict[str, Any], templates: dict[str, Any]) -> dict[str, Any]:
    """Flatten ``extends:`` chains (list or single key, left-to-right merge)."""
    merged: dict[str, Any] = {}
    parent = body.get("extends")
    parents = [parent] if isinstance(parent, str) else list(parent or ())
    for pname in parents:
        tpl = templates.get(pname)
        if isinstance(tpl, dict):
            merged.update(_resolve_extends(tpl, templates))
    merged.update({k: v for k, v in body.items() if k != "extends"})
    return merged


# --------------------------------------------------------------------------- #
# Claim checks (C1-C4) -- the falsifiable part
# --------------------------------------------------------------------------- #
def claim_checks(pipe: Pipeline) -> list[dict[str, Any]]:
    """Falsify C1-C4 on the real pipeline via targeted fault injection."""
    checks: list[dict[str, Any]] = []
    events = list(EVENTS)
    stages = pipe.stages

    # C1: a blocking gate failing must prune >= 1 later-stage job, and must
    # prune *everything* downstream of its stage.
    for event in events:
        state0 = pipe.simulate(event)
        active = pipe.active(event)
        for name, job in active.items():
            if not job.is_gate:
                continue
            si = stages.index(job.stage) if job.stage in stages else -1
            later = [
                n
                for n, j in active.items()
                if j.stage in stages and si >= 0 and stages.index(j.stage) > si
            ]
            state = pipe.simulate(event, failing={name})
            pruned_later = [n for n in later if state[n] in _DID_NOT_RUN and state0[n] in _RAN]
            checks.append(
                {
                    "claim": "C1-fail-fast",
                    "event": event,
                    "gate": name,
                    "later_stage_jobs": len(later),
                    "pruned": len(pruned_later),
                    "ok": len(pruned_later) == len([n for n in later if state0[n] in _RAN]),
                }
            )

    # C2: needs: [] jobs are still pruned by an earlier-stage blocking failure.
    for event in events:
        active = pipe.active(event)
        for name, job in active.items():
            if job.needs != ():
                continue
            si = stages.index(job.stage) if job.stage in stages else -1
            earlier_gates = [
                n
                for n, j in active.items()
                if j.is_gate and j.stage in stages and stages.index(j.stage) < si
            ]
            if not earlier_gates:
                continue
            state = pipe.simulate(event, failing={earlier_gates[0]})
            checks.append(
                {
                    "claim": "C2-barrier-survives-needs-empty",
                    "event": event,
                    "job": name,
                    "failing_gate": earlier_gates[0],
                    "state": state[name],
                    "ok": state[name] == PRUNED_BARRIER,
                }
            )

    # C3: advisory gates never prune anything.
    for event in events:
        for name, job in pipe.active(event).items():
            if not job.allow_failure:
                continue
            state0 = pipe.simulate(event)
            state = pipe.simulate(event, failing={name})
            ran0 = {n for n, s in state0.items() if s in _RAN}
            ran1 = {n for n, s in state.items() if s in _RAN}
            checks.append(
                {
                    "claim": "C3-advisory-cannot-stop",
                    "event": event,
                    "gate": name,
                    "lost_jobs": sorted(ran0 - ran1 - {name}),
                    "ok": not (ran0 - ran1 - {name}),
                }
            )

    # C4: rules-restricted gates must actually be restricted.  Two falsifiable
    # properties, derived from the *declared* condition rather than from the
    # resolver, so the check cannot be tautological:
    #   C4a  a job whose only activating rule requires schedule runs on
    #        schedule and on no other event;
    #   C4b  a job gated on the default branch runs on main_push only.
    for name, job in pipe.jobs.items():
        for needle, claim in (
            ('== "schedule"', "C4a-schedule-only"),
            ("CI_DEFAULT_BRANCH", "C4b-default-branch-only"),
        ):
            if not _declares(job, needle):
                continue
            runs = {e: job.runs_on(e) for e in events}
            if claim.startswith("C4a"):
                ok = runs["schedule"] and not any(runs[e] for e in events if e != "schedule")
            else:
                ok = runs["main_push"] and not runs["mr"] and not runs["branch_push"]
            checks.append(
                {
                    "claim": claim,
                    "event": "all",
                    "job": name,
                    "runs": runs,
                    "ok": bool(ok),
                }
            )
    return checks


def _declares(job: Job, needle: str) -> bool:
    """True if an activating rule condition mentions ``needle``."""
    for rule in job.rules:
        cond = str(rule.get("if") or "")
        if cond and needle in cond and str(rule.get("when", "on_success")) != "never":
            return True
    return False


def summarize(pipe: Pipeline) -> dict[str, Any]:
    checks = claim_checks(pipe)
    receipt = pipe.receipt()
    receipt["claims"] = {
        "n_checks": len(checks),
        "n_failed": sum(1 for c in checks if not c["ok"]),
        "failed": [c for c in checks if not c["ok"]],
    }
    return receipt


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else ".gitlab-ci.yml"
    print(json.dumps(summarize(Pipeline.load(target)), indent=2, sort_keys=True)[:4000])
