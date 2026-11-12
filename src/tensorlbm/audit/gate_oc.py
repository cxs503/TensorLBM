#!/usr/bin/env python3
"""m16895 · Acceptance-gate operating characteristics (verification-face falsification).

Why this module exists
----------------------
Every acceptance record in this repo is a set of *gates*: "error < 10 %",
"spurious/characteristic < 0.005", "errors decrease with resolution", ...  A gate
that has never been demonstrated to **FAIL** on a defective measurement is not
evidence — it is a decoration.  Repeated audits of this repo (m16308 "恒真判据",
m16034 "reference-convention inconsistency") found that the usual failure mode is
not a wrong number, it is a **gate that cannot fail**, or a gate whose *recorded*
threshold silently differs from the one the code compares against.

This module supplies the machine for that, domain-agnostic (no physics, no case
names — the domain gate set lives next to the case, e.g.
``cases/mcmp_stratified_poiseuille/gate_oc_run.py``):

1. **Declarative gate** — ``Gate(key, op, threshold)`` where ``op`` is one of
   ``le`` (PASS iff value <= t), ``ge`` (PASS iff value >= t), ``abs_le``
   (PASS iff |value| <= t).  Missing / non-finite metric → ``WITHHELD``, never
   ``PASS`` (a gate must not be satisfied by absence of data).
2. **Threshold triangulation** — a gate is only as good as the agreement between
   the number that is *declared* (harness), the number that is *recorded*
   (result JSON) and the number that is *compared against in the producing
   source*.  ``triangulate()`` returns every disagreement.
3. **Switch-point check** — ``switch_probe()`` evaluates the gate at
   ``t*(1±eps)`` and requires PASS below and FAIL above.  This catches hidden
   slack (``x < 2*t``) and inverted comparisons.
4. **Detection curve** — ``detection_curve()`` maps a defect magnitude to the
   verdict, giving the gate's *minimum detectable defect* and proving it is not
   vacuously permissive.
5. **Mutant discrimination** — ``discriminate()`` shows that a battery of probe
   records can tell the canonical gate apart from degenerate variants
   (threshold×2, threshold/2, flipped operator, always-PASS, always-FAIL).  If a
   mutant is indistinguishable, the probe battery is under-powered and the
   "the gate is verified" claim is unproven.

All functions are pure: no I/O, no domain constants.  ``run_report()`` assembles
the JSON-serialisable record a case script can write to disk.

Verdict vocabulary (deliberately not ``bool``): ``PASS`` / ``FAIL`` / ``WITHHELD``
/ ``ERROR`` (malformed input).  Only ``FAIL`` is a measurement of a defect;
``WITHHELD`` is an admission that the record does not answer the question.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

PASS = "PASS"
FAIL = "FAIL"
WITHHELD = "WITHHELD"
ERROR = "ERROR"

_OPS: tuple[str, ...] = ("le", "ge", "abs_le")


# ---------------------------------------------------------------------------
# Gate definition and evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Gate:
    """One declarative acceptance gate.

    Attributes
    ----------
    name:       stable identifier (used as a dict key in every report).
    key:        metric name inside a record.
    op:         ``"le"`` | ``"ge"`` | ``"abs_le"`` — see module docstring.
    threshold:  the declared acceptance threshold (SI-free, whatever the metric is).
    rationale:  one line of *why this number*; empty rationale is reported as a
                defect by ``audit_gate_hygiene()``.
    """

    name: str
    key: str
    op: str
    threshold: float
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.op not in _OPS:
            raise ValueError(f"gate {self.name!r}: op must be one of {_OPS}, got {self.op!r}")
        if not (isinstance(self.threshold, (int, float)) and math.isfinite(self.threshold)):
            raise ValueError(f"gate {self.name!r}: non-finite threshold {self.threshold!r}")


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def check(gate: Gate, value: Any) -> str:
    """Evaluate one gate against one metric value.  Returns PASS/FAIL/WITHHELD/ERROR."""
    v = _as_float(value)
    if v is None:
        return WITHHELD
    if gate.op == "le":
        return PASS if v <= gate.threshold else FAIL
    if gate.op == "ge":
        return PASS if v >= gate.threshold else FAIL
    # abs_le
    return PASS if abs(v) <= gate.threshold else FAIL


def evaluate(gates: Sequence[Gate], record: Mapping[str, Any]) -> dict[str, str]:
    """Evaluate a gate set against a flat metric record (order-preserving)."""
    if not isinstance(record, Mapping):
        return {g.name: ERROR for g in gates}
    return {g.name: check(g, record.get(g.key)) for g in gates}


def summarise(verdicts: Mapping[str, str]) -> dict[str, Any]:
    """Aggregate verdict counts; ``overall`` is FAIL if any FAIL, else PASS only if
    every gate produced PASS (any WITHHELD/ERROR drags the overall to WITHHELD)."""
    counts: dict[str, int] = {}
    for v in verdicts.values():
        counts[v] = counts.get(v, 0) + 1
    if counts.get(ERROR):
        overall = ERROR
    elif counts.get(FAIL):
        overall = FAIL
    elif counts.get(WITHHELD):
        overall = WITHHELD
    else:
        overall = PASS if verdicts else WITHHELD
    return {"overall": overall, "counts": counts}


# ---------------------------------------------------------------------------
# Threshold triangulation: declared vs recorded vs producing-source
# ---------------------------------------------------------------------------

def parse_source_constant(path: str, const: str, key: str) -> float | None:
    """Read ``const[key]`` out of a Python source file **without importing it**.

    Parsing (``ast``) rather than importing keeps the check independent of the
    producing module's heavy imports (torch, matplotlib) and — more importantly —
    independent of any code path that might mutate the constant at run time.
    Returns ``None`` when the constant or the key cannot be found (which the
    caller must treat as a verification gap, not as agreement).
    """
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if const not in targets:
            continue
        call = node.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "dict"):
            return None
        for kw in call.keywords:
            if kw.arg == key:
                try:
                    return float(ast.literal_eval(kw.value))
                except (ValueError, TypeError):
                    return None
    return None


def triangulate(
    declared: Mapping[str, float],
    recorded: Mapping[str, float] | None,
    from_source: Mapping[str, float | None],
    rel_tol: float = 1e-12,
) -> dict[str, Any]:
    """Compare the same threshold across three independent carriers.

    ``from_source`` values of ``None`` are *missing evidence* (reported in
    ``missing``), not agreement.  Returns ``{"agree": bool, "rows": [...],
    "missing": [...]}``; a row is a disagreement with all three numbers attached.
    """
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for name, dec in declared.items():
        rec = None if recorded is None else recorded.get(name)
        src = from_source.get(name)
        if rec is None or src is None:
            missing.append(name)
            continue
        ok_rec = math.isclose(float(rec), float(dec), rel_tol=rel_tol, abs_tol=0.0)
        ok_src = math.isclose(float(src), float(dec), rel_tol=rel_tol, abs_tol=0.0)
        if not (ok_rec and ok_src):
            rows.append(dict(gate=name, declared=float(dec), recorded=float(rec),
                             source=float(src), recorded_agrees=ok_rec,
                             source_agrees=ok_src))
    return {"agree": not rows and not missing, "rows": rows, "missing": missing}


# ---------------------------------------------------------------------------
# Switch-point and detection curve
# ---------------------------------------------------------------------------

def switch_probe(gate: Gate, eps: float = 1e-3) -> dict[str, Any]:
    """Verify the gate switches exactly at its declared threshold.

    Below-threshold probe must PASS, above-threshold probe must FAIL.  ``eps`` is
    relative.  Returns ``{"ok": bool, "below": verdict, "above": verdict, ...}``.
    """
    t = gate.threshold
    if t == 0.0:
        # zero threshold: perturb by an absolute epsilon instead
        below, above = -eps, +eps
    else:
        below, above = t * (1.0 - eps), t * (1.0 + eps)
    if gate.op == "ge":
        below, above = above, below   # for a >= gate the defect side is *below*
    v_below, v_above = check(gate, below), check(gate, above)
    return {
        "gate": gate.name, "threshold": t, "op": gate.op,
        "probe_below": below, "verdict_below": v_below,
        "probe_above": above, "verdict_above": v_above,
        "ok": v_below == PASS and v_above == FAIL,
    }


def detection_curve(
    gate: Gate,
    base: float,
    factors: Iterable[float] = (0.0, 0.25, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 1.5, 2.0, 5.0),
) -> list[dict[str, Any]]:
    """Map ``value = base * (1 + factor)`` to the gate verdict.

    ``factor == -1`` is skipped (would null the metric).  The returned list is the
    gate's *operating characteristic*: the first factor whose verdict is FAIL is
    the smallest relative defect the gate can see, hence the honest sensitivity
    statement ("this gate cannot distinguish a 9.9 % flow error from a perfect
    solution").
    """
    out: list[dict[str, Any]] = []
    for f in factors:
        if f <= -1.0:
            continue
        # A ``ge`` gate fails *below* its threshold, so the defect direction is
        # mirrored: factor f>0 always means "a defect of magnitude f*threshold".
        signed = -f if gate.op == "ge" else f
        value = base * (1.0 + signed)
        out.append({"factor": float(f), "value": float(value), "verdict": check(gate, value)})
    return out


def minimum_detectable(curve: Sequence[Mapping[str, Any]]) -> float | None:
    """Smallest |factor| that trips the gate, ``None`` if the gate never trips."""
    tripped = [abs(c["factor"]) for c in curve if c.get("verdict") == FAIL]
    return min(tripped) if tripped else None


# ---------------------------------------------------------------------------
# Mutants: can the probe battery tell a correct gate from a degenerate one?
# ---------------------------------------------------------------------------

def mutants(gate: Gate) -> dict[str, Gate]:
    """Degenerate variants of ``gate`` that a competent probe battery must reject."""
    t = gate.threshold
    double = Gate(f"{gate.name}__threshold_x2", gate.key, gate.op, t * 2.0 if t else 1e-6,
                  rationale="mutant: 2x threshold")
    half = Gate(f"{gate.name}__threshold_x0.5", gate.key, gate.op, t * 0.5 if t else 1e-6,
                rationale="mutant: 0.5x threshold")
    flip_op = {"le": "ge", "ge": "le", "abs_le": "ge"}[gate.op]
    flipped = Gate(f"{gate.name}__op_flipped", gate.key, flip_op, t,
                   rationale="mutant: inverted comparison")
    return {"threshold_x2": double, "threshold_x0.5": half, "op_flipped": flipped}


def discriminate(
    gate: Gate,
    probes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Count probe records on which each mutant disagrees with the canonical gate.

    A mutant with ``killed_by == 0`` means the probe battery cannot detect that
    class of error — the gate's verification is under-powered.  Also reports
    ``canonical_all_pass``: a canonical gate that PASSes every probe is *suspect*
    unless the battery was designed to contain conforming records only.
    """
    canon = [check(gate, p.get(gate.key)) for p in probes]
    out: dict[str, Any] = {
        "gate": gate.name,
        "n_probes": len(probes),
        "canonical_verdicts": canon,
        "canonical_all_pass": all(v == PASS for v in canon),
        "mutants": {},
    }
    for label, mg in mutants(gate).items():
        m = [check(mg, p.get(mg.key)) for p in probes]
        killed = [i for i, (a, b) in enumerate(zip(canon, m)) if a != b]
        out["mutants"][label] = {
            "n_differ": len(killed),
            "killed_by": killed,
            "killed": bool(killed),
        }
    out["all_mutants_killed"] = all(v["killed"] for v in out["mutants"].values())
    return out


def audit_gate_hygiene(gates: Sequence[Gate]) -> list[dict[str, Any]]:
    """Structural defects that make a gate unusable, independent of any record."""
    findings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for g in gates:
        if g.name in seen:
            findings.append({"gate": g.name, "issue": "duplicate gate name"})
        seen.add(g.name)
        if not g.rationale.strip():
            findings.append({"gate": g.name, "issue": "no rationale for the threshold"})
        if not g.key.strip():
            findings.append({"gate": g.name, "issue": "no metric key"})
        if g.op == "abs_le" and g.threshold < 0:
            findings.append({"gate": g.name, "issue": "negative threshold on |x| gate"})
        if g.threshold == 0.0 and g.op in ("le", "ge"):
            findings.append({"gate": g.name, "issue": "zero threshold makes the gate a sign test"})
    return findings


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

@dataclass
class GateSetReport:
    """Container assembled by :func:`run_report` (JSON-serialisable via ``as_dict``)."""

    gates: Sequence[Gate]
    records: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    triangulation: dict[str, Any] | None = None
    probes: Sequence[Mapping[str, Any]] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        gate_rows = []
        for g in self.gates:
            rec_verdicts = {name: check(g, rec.get(g.key))
                            for name, rec in self.records.items()}
            probe_set = [p for p in self.probes] or [{"_probe": "empty"}]
            curve = detection_curve(g, _reference_value(g, self.records))
            gate_rows.append({
                "gate": g.name, "key": g.key, "op": g.op, "threshold": g.threshold,
                "rationale": g.rationale,
                "verdicts": rec_verdicts,
                "summaries": {name: summarise({g.name: v})["overall"]
                              for name, v in rec_verdicts.items()},
                "switch_probe": switch_probe(g),
                "detection_curve": curve,
                "minimum_detectable_factor": minimum_detectable(curve),
                "discrimination": discriminate(g, probe_set),
            })
        return {
            "n_gates": len(self.gates),
            "hygiene": audit_gate_hygiene(self.gates),
            "triangulation": self.triangulation,
            "records": list(self.records),
            "gates": gate_rows,
        }


def _reference_value(gate: Gate, records: Mapping[str, Mapping[str, Any]]) -> float:
    """Sensitivity is measured relative to the *acceptance threshold*, not the
    observed value: a near-zero observed metric would make any multiplicative
    sweep meaningless (0 * anything = 0)."""
    return abs(gate.threshold) if gate.threshold else 1.0


def run_report(
    gates: Sequence[Gate],
    records: Mapping[str, Mapping[str, Any]],
    triangulation: dict[str, Any] | None = None,
    probes: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Assemble one gate-set report (the object a case script writes to disk)."""
    return GateSetReport(gates=gates, records=records, triangulation=triangulation,
                         probes=probes).as_dict()


def probe_battery(
    gate: Gate,
    factors: Sequence[float] = (-0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 10.0),
    key: str | None = None,
) -> list[dict[str, Any]]:
    """Conforming/defective probe records spanning the gate's decision boundary.

    Each probe is a one-key record ``{key: threshold*(1+factor)}``.  The battery
    must contain both conforming (|factor| small on the safe side) and defective
    probes, otherwise ``discriminate()`` cannot kill any mutant.
    """
    k = key or gate.key
    out = []
    for f in factors:
        value = gate.threshold * (1.0 + f)
        if gate.threshold == 0.0:
            value = f
        out.append({k: float(value), "_factor": float(f)})
    return out


__all__ = [
    "PASS", "FAIL", "WITHHELD", "ERROR",
    "Gate", "check", "evaluate", "summarise",
    "parse_source_constant", "triangulate",
    "switch_probe", "detection_curve", "minimum_detectable",
    "mutants", "discriminate", "audit_gate_hygiene",
    "probe_battery", "run_report", "GateSetReport",
]