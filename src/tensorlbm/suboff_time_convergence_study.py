"""SUBOFF bare-hull time convergence study runner (D3Q19+MRT, diagnostic only).

This runner executes the SUBOFF bare-hull D3Q19+MRT validation runner at four or
more different step counts on a **fixed grid** (48×24×24 by default).  It
collects the measured Ct candidate per time level — the mean Ct over a capture
window of the final steps — and computes relative-change indicators, but
deliberately withholds any convergence or physical-validation claim.

The study is a **diagnostic only**: it shows how the measured Ct candidate
varies across time-step counts, but does not assert that the solution has
converged in time or that the Ct values are physically validated.

This runner composes the existing cold-path validation runner
(:mod:`tensorlbm.suboff_validation_runner`) with existing solver operators.
It does **not** modify any solver hot path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any

from .suboff_validation_runner import (
    SuboffValidationConfig,
    run_suboff_d3q19_mrt_validation,
)

_SCHEMA = "suboff-time-convergence-study-r1"
_MIN_TIME_LEVELS = 4


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return float(value)


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TimeLevel:
    """One time-step-count level for the time convergence study.

    The study runs the SUBOFF bare-hull D3Q19+MRT solver for ``n_steps`` steps
    and uses the mean Ct over the final ``capture_window`` steps as the
    representative measured Ct candidate for this level.
    """

    level_id: str
    n_steps: int
    capture_window: int

    def __post_init__(self) -> None:
        if not isinstance(self.level_id, str) or not self.level_id:
            raise ValueError("level_id must be a non-empty string")
        if isinstance(self.n_steps, bool) or not isinstance(self.n_steps, int) or self.n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        if isinstance(self.capture_window, bool) or not isinstance(self.capture_window, int) or self.capture_window < 1:
            raise ValueError("capture_window must be a positive integer")
        if self.capture_window > self.n_steps:
            raise ValueError("capture_window must be <= n_steps")

    @property
    def capture_steps(self) -> tuple[int, ...]:
        """The 1-based step indices in the capture window (ascending)."""
        start = self.n_steps - self.capture_window + 1
        return tuple(range(start, self.n_steps + 1))


@dataclass(frozen=True, slots=True)
class TimeConvergenceStudyConfig:
    """Configuration for the SUBOFF time convergence study.

    All time levels share the same fixed grid, D3Q19+MRT composition, inlet
    velocity, Reynolds number, and bare-hull geometry type.  Only the number
    of time steps varies across levels.
    """

    time_levels: tuple[TimeLevel, ...]
    nx: int = 48
    ny: int = 24
    nz: int = 24
    u_in: float = 0.06
    re: float = 200.0
    hull_length: float = 24.0
    device: str = "cpu"
    lattice: str = "D3Q19"
    collision: str = "MRT"
    hull_type: str = "bare_hull"

    def __post_init__(self) -> None:
        if not isinstance(self.time_levels, tuple) or len(self.time_levels) < _MIN_TIME_LEVELS:
            raise ValueError(
                f"time convergence study requires at least {_MIN_TIME_LEVELS} time levels"
            )
        if any(not isinstance(level, TimeLevel) for level in self.time_levels):
            raise TypeError("time_levels must contain only TimeLevel instances")
        # Unique level IDs
        level_ids = [level.level_id for level in self.time_levels]
        if len(set(level_ids)) != len(level_ids):
            raise ValueError("time level level_ids must be unique")
        if self.hull_type != "bare_hull":
            raise ValueError("time convergence study supports only hull_type='bare_hull'")
        if self.lattice != "D3Q19":
            raise ValueError("time convergence study requires lattice='D3Q19'")
        if self.collision != "MRT":
            raise ValueError("time convergence study requires collision='MRT'")
        for name in ("nx", "ny", "nz"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        _finite_positive(self.u_in, "u_in")
        if not 0.0 < self.u_in < 0.15:
            raise ValueError("u_in must be in (0, 0.15)")
        _finite_positive(self.re, "re")
        _finite_positive(self.hull_length, "hull_length")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")


# ---------------------------------------------------------------------------
# Per-level execution
# ---------------------------------------------------------------------------

def _run_one_level(
    level: TimeLevel,
    config: TimeConvergenceStudyConfig,
) -> dict[str, Any]:
    """Run the validation runner at one time level and extract per-level data.

    Builds a :class:`SuboffValidationConfig` with the study's fixed grid and
    physics parameters, sets ``n_steps`` to the level's step count, runs the
    real D3Q19+MRT+bounce-back solver loop, and extracts the measured Ct
    candidate (mean Ct over the capture window).
    """
    run_config = SuboffValidationConfig(
        nx=config.nx,
        ny=config.ny,
        nz=config.nz,
        n_steps=level.n_steps,
        warmup=0,
        u_in=config.u_in,
        re=config.re,
        hull_length=config.hull_length,
        device=config.device,
        use_wall_function=False,
    )

    evidence = run_suboff_d3q19_mrt_validation(run_config)

    # Extract the measured Ct candidate: mean Ct over the capture window
    ct_series = evidence.ct_time_series
    capture_indices = [s - 1 for s in level.capture_steps]  # 0-based
    captured_cts = [ct_series[i]["ct"] for i in capture_indices]
    measured_ct = sum(captured_cts) / len(captured_cts)

    return {
        "level_id": level.level_id,
        "n_steps": level.n_steps,
        "capture_window": level.capture_window,
        "capture_steps": list(level.capture_steps),
        "status": evidence.status,
        "physical_validation": evidence.physical_validation,
        "Ct": measured_ct,
        "force_time_series": evidence.force_time_series,
        "ct_time_series": evidence.ct_time_series,
        "wetted_area": evidence.wetted_area,
        "dynamic_pressure": evidence.dynamic_pressure,
        "runtime": evidence.runtime,
        "config": evidence.config,
    }


# ---------------------------------------------------------------------------
# Convergence indicator
# ---------------------------------------------------------------------------

def _compute_convergence_indicator(ct_values: list[float]) -> dict[str, Any]:
    """Compute relative-change indicators without claiming convergence."""
    relative_changes: list[float] = []
    for i in range(len(ct_values) - 1):
        ct_prev = ct_values[i]
        ct_next = ct_values[i + 1]
        if ct_prev == 0.0:
            relative_changes.append(float("inf"))
        else:
            relative_changes.append(abs(ct_next - ct_prev) / abs(ct_prev))

    # Determine trend direction
    diffs = []
    for i in range(len(ct_values) - 1):
        diffs.append(ct_values[i + 1] - ct_values[i])
    if diffs and all(d < 0 for d in diffs):
        ct_trend = "decreasing"
    elif diffs and all(d > 0 for d in diffs):
        ct_trend = "increasing"
    else:
        ct_trend = "non_monotonic"

    finite_changes = [c for c in relative_changes if isfinite(c)]
    max_relative_change = max(finite_changes) if finite_changes else float("inf")

    return {
        "relative_ct_changes": relative_changes,
        "ct_trend": ct_trend,
        "max_relative_change": max_relative_change,
        "convergence_claim": "withheld",
        "note": (
            "Relative Ct changes across time levels are diagnostic indicators "
            "only; they do not constitute a time convergence claim or physical "
            "validation."
        ),
    }


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_suboff_time_convergence_study(
    config: TimeConvergenceStudyConfig,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the SUBOFF bare-hull time convergence study.

    Executes the D3Q19+MRT validation runner at each time level on the fixed
    grid, collects the measured Ct candidate per level, and computes
    relative-change convergence indicators.  The result is a machine-readable
    artifact with ``status='diagnostic_only'`` and ``physical_validation=False``.

    Parameters
    ----------
    config :
        Study configuration with at least four time levels.
    output_path :
        Optional path to write the JSON artifact.  When provided, the artifact
        is written to disk in addition to being returned.

    Returns
    -------
    dict
        Machine-readable convergence artifact.
    """
    if not isinstance(config, TimeConvergenceStudyConfig):
        raise TypeError("config must be a TimeConvergenceStudyConfig")

    per_level_results = [_run_one_level(level, config) for level in config.time_levels]
    ct_per_level = [result["Ct"] for result in per_level_results]
    convergence_indicator = _compute_convergence_indicator(ct_per_level)

    time_level_records = [
        {
            "level_id": level.level_id,
            "n_steps": level.n_steps,
            "capture_window": level.capture_window,
            "capture_steps": list(level.capture_steps),
        }
        for level in config.time_levels
    ]

    provenance = {
        "runner_api": (
            "tensorlbm.suboff_validation_runner.run_suboff_d3q19_mrt_validation"
        ),
        "model_identity": {
            "lattice": config.lattice,
            "collision": config.collision,
            "hull_type": config.hull_type,
            "boundary": "bounce_back",
            "wall": "static",
            "physics": "single_phase_incompressible",
        },
        "grid_shape": {"nx": config.nx, "ny": config.ny, "nz": config.nz},
        "force_method": "d3q19_momentum_exchange",
        "sample_phase": "post_stream_pre_bounce_back",
        "ct_aggregation": "mean_over_capture_window",
        "prohibition": "no_convergence_claim_or_physical_validation",
    }

    payload: dict[str, Any] = {
        "artifact_kind": "suboff_time_convergence_study",
        "schema": _SCHEMA,
        "status": "diagnostic_only",
        "physical_validation": False,
        "grid_shape": {"nx": config.nx, "ny": config.ny, "nz": config.nz},
        "time_levels": time_level_records,
        "Ct_per_level": ct_per_level,
        "convergence_indicator": convergence_indicator,
        "per_level_results": per_level_results,
        "provenance": provenance,
    }
    payload["provenance_hash"] = _canonical_hash(payload)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    return payload


__all__ = [
    "TimeConvergenceStudyConfig",
    "TimeLevel",
    "run_suboff_time_convergence_study",
]
