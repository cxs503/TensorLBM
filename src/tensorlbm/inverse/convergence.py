"""Machine-readable convergence log for inverse-problem optimisation.

Provides a JSON schema (:data:`CONVERGENCE_SCHEMA`) and a :class:`ConvergenceLog`
recorder that captures per-iteration state (loss, parameters, gradients, learning
rate) and writes a reproducible, machine-checkable JSON file.

Schema
------
The output JSON follows ``tensorlbm/inverse/convergence/v1`` and contains:

- ``schema``: version string
- ``created_unix``: UTC timestamp
- ``problem``: problem descriptor (name, parameters, seeds)
- ``history``: per-iteration records (loss, params, gradients, lr)
- ``summary``: final state, convergence flag, wall time

This schema is designed for automated verification: a downstream checker can
read the JSON, verify that ``loss`` decreased, that ``converged`` is consistent
with the tolerance, and that the same seed reproduces the same history.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "CONVERGENCE_SCHEMA",
    "ConvergenceLog",
    "IterationRecord",
    "read_convergence_json",
    "validate_convergence_payload",
    "write_convergence_json",
]

CONVERGENCE_SCHEMA = "tensorlbm/inverse/convergence/v1"


@dataclass(frozen=True)
class IterationRecord:
    """One row of the convergence history."""

    iter: int
    phase: str  # "adam", "lbfgs_eval", "final"
    loss: float
    params: dict[str, float]
    gradients: dict[str, float] = field(default_factory=dict)
    lr: float = 0.0
    #: Free-form per-iteration diagnostics.  Scalars are the common case, but
    #: list-valued entries (e.g. a per-observation-time loss) are allowed and
    #: are written verbatim into the JSON.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "iter": self.iter,
            "phase": self.phase,
            "loss": self.loss,
            "params": self.params,
        }
        if self.gradients:
            d["gradients"] = self.gradients
        if self.lr:
            d["lr"] = self.lr
        if self.extra:
            d["extra"] = self.extra
        return d


@dataclass
class ConvergenceLog:
    """Accumulator for convergence history.

    Usage::

        log = ConvergenceLog(problem="tg_tau_recovery", seed=0)
        for it in range(iters):
            ...
            log.record(IterationRecord(...))
        log.finalise(tau_hat=..., converged=True, wall_s=...)
        log.write("artifacts/convergence.json")
    """

    problem: str
    seed: int
    history: list[IterationRecord] = field(default_factory=list)
    created_unix: float = field(default_factory=time.time)
    _finalised: bool = False
    _summary: dict[str, Any] = field(default_factory=dict)

    def record(self, rec: IterationRecord) -> None:
        self.history.append(rec)

    def finalise(self, *, wall_s: float, converged: bool, **summary_kwargs: Any) -> None:
        """Seal the log with final summary statistics."""
        self._finalised = True
        self._summary = {
            "wall_s": round(wall_s, 3),
            "converged": converged,
            "n_iterations": len(self.history),
            **summary_kwargs,
        }

    def to_payload(self) -> dict[str, Any]:
        if not self._finalised:
            raise RuntimeError("call finalise() before to_payload()")
        return {
            "schema": CONVERGENCE_SCHEMA,
            "created_unix": self.created_unix,
            "problem": self.problem,
            "seed": self.seed,
            "history": [r.to_dict() for r in self.history],
            "summary": self._summary,
        }

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_payload(), indent=2) + "\n")
        return p


def write_convergence_json(
    path: str | Path,
    *,
    problem: str,
    seed: int,
    history: Sequence[IterationRecord],
    wall_s: float,
    converged: bool,
    **summary_kwargs: Any,
) -> Path:
    """Convenience: build and write a convergence JSON in one call."""
    log = ConvergenceLog(problem=problem, seed=seed)
    for rec in history:
        log.record(rec)
    log.finalise(wall_s=wall_s, converged=converged, **summary_kwargs)
    return log.write(path)


# ---------------------------------------------------------------------------
# Machine checking of a log
# ---------------------------------------------------------------------------


def read_convergence_json(path: str | Path) -> dict[str, Any]:
    """Load a convergence JSON file produced by :meth:`ConvergenceLog.write`."""
    return json.loads(Path(path).read_text())


def validate_convergence_payload(
    payload: Mapping[str, Any],
    *,
    require_loss_decrease: bool = True,
) -> list[str]:
    """Check a convergence payload against ``tensorlbm/inverse/convergence/v1``.

    Returns a list of human-readable problems (empty list = valid).  The
    function never raises, so it can be used as a gate on artefacts produced by
    someone else's run: it checks the schema tag, the required keys, the types
    of the history rows, and the internal consistency of the summary
    (``n_iterations`` against the history length; ``converged`` against an
    actually non-increasing loss).
    """
    problems: list[str] = []

    def is_number(x: Any) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    if not isinstance(payload, Mapping):
        return [f"payload is {type(payload).__name__}, expected a mapping"]

    schema = payload.get("schema")
    if schema != CONVERGENCE_SCHEMA:
        problems.append(f"schema is {schema!r}, expected {CONVERGENCE_SCHEMA!r}")

    for key in ("created_unix", "problem", "seed", "history", "summary"):
        if key not in payload:
            problems.append(f"missing top-level key {key!r}")
    if problems:
        return problems

    if not is_number(payload["created_unix"]):
        problems.append("created_unix is not a number")
    if not isinstance(payload["problem"], str) or not payload["problem"]:
        problems.append("problem is not a non-empty string")
    if not isinstance(payload["seed"], int):
        problems.append("seed is not an int")

    history = payload["history"]
    if not isinstance(history, list) or not history:
        problems.append("history is not a non-empty list")
        history = []

    losses: list[float] = []
    for i, rec in enumerate(history):
        if not isinstance(rec, Mapping):
            problems.append(f"history[{i}] is not a mapping")
            continue
        for key in ("iter", "phase", "loss", "params"):
            if key not in rec:
                problems.append(f"history[{i}] missing {key!r}")
        if not is_number(rec.get("iter")):
            problems.append(f"history[{i}].iter is not a number")
        if not isinstance(rec.get("phase"), str):
            problems.append(f"history[{i}].phase is not a string")
        if not is_number(rec.get("loss")):
            problems.append(f"history[{i}].loss is not a number")
        else:
            losses.append(float(rec["loss"]))
        params = rec.get("params")
        if not isinstance(params, Mapping) or not all(
            is_number(v) for v in params.values()
        ):
            problems.append(f"history[{i}].params is not a mapping of numbers")

    summary = payload["summary"]
    if not isinstance(summary, Mapping):
        problems.append("summary is not a mapping")
        return problems

    for key in ("wall_s", "converged", "n_iterations"):
        if key not in summary:
            problems.append(f"summary missing {key!r}")
    if "converged" in summary and not isinstance(summary["converged"], bool):
        problems.append("summary.converged is not a bool")
    if is_number(summary.get("n_iterations")) and int(summary["n_iterations"]) != len(history):
        problems.append(
            f"summary.n_iterations={summary['n_iterations']} != len(history)={len(history)}"
        )
    if is_number(summary.get("final_loss")) and history:
        if not is_number(history[-1].get("loss")):
            problems.append("final record has no numeric loss")
    if require_loss_decrease and len(losses) >= 2:
        first = losses[0]
        final = summary.get("final_loss", losses[-1])
        if is_number(final) and final > first:
            problems.append(
                f"final_loss {final!r} exceeds the first recorded loss {first!r}"
            )
    return problems
