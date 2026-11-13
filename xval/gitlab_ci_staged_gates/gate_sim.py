"""Gate-ladder simulator: evaluate a CI gate policy over a real test corpus.

The simulator answers three questions for a (policy, event) pair, using only
facts that are exact and host-independent:

1. *guard set*   -- which corpus items does at least one blocking gate inspect?
2. *escape set*  -- which items reach the default branch / release unguarded?
3. *ladder depth*-- how many serial rungs a failing item must traverse before a
   blocking gate can reject it (0 = rejected by the first rung, ``None`` =
   never rejected on this event).

Marker expressions are evaluated with the same boolean rules pytest uses
(``and`` / ``or`` / ``not``, left-to-right) and the result is validated against
real ``pytest --collect-only`` counts by ``run_xval.py``.

Timing is deliberately *not* a pass/fail KPI: the measurement host is shared
with other agents, so wall clock carries a co-tenant confounder (recorded in
``summary.json``).  A CPU-time cost model is provided for context only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# --------------------------------------------------------------------------- #
# Marker expression evaluation (pytest semantics for the subset in use)
# --------------------------------------------------------------------------- #
def marker_match(expr: str | None, markers: Iterable[str], funcname: str = "") -> bool:
    """Evaluate a pytest ``-m`` expression.

    Supports the subset that staging ladders actually use: ``A``, ``A and B``,
    ``A or B``, ``not A``, and combinations without parentheses.  A bare name
    also matches a pytest ``parametrize`` / ``skipif`` style marker name used as
    a bare string in the expression; markers are matched case-insensitively
    like pytest does.
    """
    if expr is None or not str(expr).strip():
        return True
    marks = {m.lower() for m in markers}
    for or_term in str(expr).split(" or "):
        if all(_eval_and_term(t, marks) for t in or_term.split(" and ")):
            return True
    return False


def _eval_and_term(term: str, marks: set[str]) -> bool:
    term = term.strip()
    negate = False
    while term.startswith("not "):
        negate = not negate
        term = term[4:].strip()
    hit = term.lower() in marks
    return (not hit) if negate else hit


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
@dataclass
class Corpus:
    items: list[dict[str, Any]]

    @classmethod
    def load(cls, path: str | Path) -> "Corpus":
        return cls(json.loads(Path(path).read_text()))

    def __len__(self) -> int:
        return len(self.items)

    def select(self, expr: str | None, scope: list[str] | None = None) -> list[dict[str, Any]]:
        out = []
        for it in self.items:
            if scope and not any(it["nodeid"].startswith(s) for s in scope):
                continue
            if marker_match(expr, it["markers"], it["nodeid"]):
                out.append(it)
        return out

    def ids(self, expr: str | None, scope: list[str] | None = None) -> set[str]:
        return {it["nodeid"] for it in self.select(expr, scope)}


# --------------------------------------------------------------------------- #
# Policy application
# --------------------------------------------------------------------------- #
@dataclass
class GateRun:
    job: str
    tier: str
    stage: str
    expr: str | None
    events: list[str]
    selected: list[str] = field(default_factory=list)
    skipped: bool = False


CORPUS_KINDS = ("pytest", "nox")


def job_marker_expr(job: dict[str, Any], event: str) -> str | None:
    """Marker expression a job applies on ``event`` (None = whole corpus)."""
    sel = job.get("selection") or {}
    expr = sel.get("marker_expr")
    marker_events = sel.get("marker_events")
    if marker_events is not None:
        # GitHub Actions encodes the `-m` filter only on the fast events; the
        # full-scope branch runs the whole corpus.
        return expr if event in marker_events else None
    if sel.get("kind") == "nox":
        # Resolve the nox session to its pytest arguments (see nox_refs).
        expr = sel.get("resolved_marker_expr", expr)
    return expr


def apply_policy(
    spec: dict[str, Any],
    event: str,
    corpus: Corpus,
    marker_rename: dict[str, str] | None = None,
    scope_override: dict[str, list[str]] | None = None,
) -> list[GateRun]:
    """Resolve every active job of ``spec`` for ``event`` onto the corpus."""
    runs: list[GateRun] = []
    for job in spec["jobs"]:
        if event not in job["events"]:
            runs.append(
                GateRun(job["id"], job["tier"], job["stage"], None, job["events"], skipped=True)
            )
            continue
        if job["selection"].get("kind") not in CORPUS_KINDS:
            # lint / build / targeted jobs do not inspect the marker corpus;
            # they are recorded but contribute no guard coverage.
            runs.append(
                GateRun(
                    job["id"],
                    job["tier"],
                    job["stage"],
                    None,
                    job["events"],
                    selected=[],
                    skipped=False,
                )
            )
            continue
        expr = job_marker_expr(job, event)
        if marker_rename and expr:
            for old, new in marker_rename.items():
                expr = expr.replace(old, new)
        scope = job["selection"].get("scope") or scope_override.get(job["id"]) or None
        runs.append(
            GateRun(
                job=job["id"],
                tier=job["tier"],
                stage=job["stage"],
                expr=expr,
                events=job["events"],
                selected=sorted(corpus.ids(expr, scope)),
            )
        )
    return runs


def guard_vector(
    runs: list[GateRun],
    corpus: Corpus,
    blocking_tiers: tuple[str, ...] = ("blocking", "conditional"),
) -> dict[str, int | None]:
    """Per corpus item: finish rung index of the first gate that can reject it.

    0 means "the first blocking gate selects this item", ``None`` means the
    item is never inspected by a gate that can fail the pipeline on this event.
    """
    gates = [r for r in runs if not r.skipped and r.tier in blocking_tiers]
    index = {nodeid: None for nodeid in (it["nodeid"] for it in corpus.items)}
    for gate_idx, gate in enumerate(gates):
        for nodeid in gate.selected:
            if index[nodeid] is None:
                index[nodeid] = gate_idx
    return index


def kpis(
    spec: dict[str, Any],
    event: str,
    corpus: Corpus,
    marker_rename: dict[str, str] | None = None,
    scope_override: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    runs = apply_policy(spec, event, corpus, marker_rename, scope_override)
    active = [r for r in runs if not r.skipped]
    blocking = [r for r in active if r.tier in ("blocking", "conditional")]
    guarded: set[str] = set()
    for r in blocking:
        guarded |= set(r.selected)
    all_ids = {it["nodeid"] for it in corpus.items}
    gv = guard_vector(runs, corpus)
    latencies = [v for v in gv.values() if v is not None]
    return {
        "policy": spec["name"],
        "event": event,
        "n_active_jobs": len(active),
        "n_skipped_jobs": len(runs) - len(active),
        "n_blocking_gates": len(blocking),
        "n_guarded": len(guarded),
        "guard_fraction": len(guarded) / max(1, len(all_ids)),
        "n_escaped": len(all_ids - guarded),
        "escape_fraction": len(all_ids - guarded) / max(1, len(all_ids)),
        "mean_rung": (sum(latencies) / len(latencies)) if latencies else None,
        "max_rung": max(latencies) if latencies else None,
        "n_tests_executed": sum(len(r.selected) for r in active) or 0,
        "per_gate": [
            {
                "job": r.job,
                "tier": r.tier,
                "expr": r.expr,
                "n_selected": len(r.selected),
                "kind": _job_kind(spec, r.job),
            }
            for r in active
        ],
        "guard_vector": gv,
    }


def _job_kind(spec: dict[str, Any], job_id: str) -> str:
    for j in spec["jobs"]:
        if j["id"] == job_id:
            return str(j["selection"].get("kind"))
    return "?"


def ladder_structure(spec: dict[str, Any]) -> dict[str, Any]:
    """Structural facts of the declared ladder (censused, not modelled)."""
    jobs = spec["jobs"]
    tiers: dict[str, int] = {}
    for j in jobs:
        tiers[j["tier"]] = tiers.get(j["tier"], 0) + 1
    declared = len(spec.get("stages") or [])
    active_stages = {j["stage"] for j in jobs}
    edges = sum(len(j["needs"]) for j in jobs)
    immediate = sum(1 for j in jobs if j.get("needs_declared") and not j["needs"])
    barrier_jobs = sum(1 for j in jobs if not j.get("needs_declared"))
    configs = sum(
        1 for j in jobs if j["selection"].get("kind") in ("pytest", "nox") and _has_tests(j)
    )
    return {
        "policy": spec["name"],
        "n_jobs": len(jobs),
        "n_stages_declared": declared,
        "n_stages_active": len({j["stage"] for j in jobs if j["stage"] != "<flat>"}),
        "empty_stages": spec.get("empty_stages", []),
        "tier_counts": tiers,
        "needs_edges": edges,
        "immediate_jobs": immediate,
        "barrier_jobs": barrier_jobs,
        "test_gate_configs": configs,
        "artifact_contracts": sum(
            len(j["artifacts"]) + len(j.get("artifact_reports") or []) for j in jobs
        ),
        "coverage_gates": sum(1 for j in jobs if j.get("coverage_gate")),
        "parallel_workers": sorted({str(j["parallelism"].get("workers")) for j in jobs}),
        "notes": spec.get("notes", []),
    }


def _has_tests(job: dict[str, Any]) -> bool:
    sel = job["selection"]
    return (
        bool(sel.get("marker_expr"))
        or sel.get("kind") == "nox"
        or any("pytest" in c for c in sel.get("commands") or [])
    )
