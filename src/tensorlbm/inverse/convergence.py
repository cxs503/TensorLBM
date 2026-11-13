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
from typing import Any, Sequence

__all__ = [
    "CONVERGENCE_SCHEMA",
    "ConvergenceLog",
    "IterationRecord",
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
    extra: dict[str, float] = field(default_factory=dict)

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
