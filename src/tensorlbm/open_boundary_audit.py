"""Grid-normalized aggregation of open-boundary population mutations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class OpenBoundaryHistoryAudit:
    samples: int
    finite_samples: int
    reference_mass: float
    reference_momentum: float
    cumulative_mass_delta: float
    mean_absolute_mass_delta_fraction: float
    maximum_absolute_mass_delta_fraction: float
    cumulative_momentum_delta: tuple[float, float, float]
    mean_momentum_delta_fraction: float
    maximum_momentum_delta_fraction: float
    maximum_face_sum_mass_closure_fraction: float
    maximum_face_sum_momentum_closure_fraction: float
    finite: bool

    def to_dict(self) -> dict[str, float | int | bool | list[float]]:
        result = dict(vars(self))
        result["cumulative_momentum_delta"] = list(
            self.cumulative_momentum_delta,
        )
        return result


def audit_open_boundary_history(
    records: Sequence[Mapping[str, object]],
    *,
    reference_mass: float,
    reference_momentum: float,
) -> OpenBoundaryHistoryAudit:
    """Aggregate combined per-step BC deltas with grid-normalized scales.

    ``reference_mass`` is normally the initial domain population mass and
    ``reference_momentum`` its streamwise far-field momentum magnitude.  The
    audit does not impose universal thresholds; benchmark families must derive
    limits from domain and grid sensitivity before using these observations as
    acceptance gates.
    """
    if not math.isfinite(reference_mass) or reference_mass <= 0.0:
        raise ValueError("reference_mass must be finite and positive")
    if not math.isfinite(reference_momentum) or reference_momentum <= 0.0:
        raise ValueError("reference_momentum must be finite and positive")

    masses: list[float] = []
    momenta: list[tuple[float, float, float]] = []
    finite_samples = 0
    mass_closure: list[float] = []
    momentum_closure: list[float] = []
    for record in records:
        mass = float(record["mass_delta"])
        momentum_value = record["momentum_delta"]
        if not isinstance(momentum_value, (list, tuple)) or len(momentum_value) != 3:
            raise ValueError("momentum_delta must contain three components")
        momentum = tuple(float(value) for value in momentum_value)
        finite = bool(record.get("finite", True)) and all(
            math.isfinite(value) for value in (mass, *momentum)
        )
        finite_samples += int(finite)
        masses.append(mass)
        momenta.append(momentum)
        stages = record.get("stages", ())
        if isinstance(stages, (list, tuple)):
            for stage in stages:
                if not isinstance(stage, Mapping):
                    continue
                mass_closure.append(
                    abs(
                        float(
                            stage.get("face_sum_mass_closure_error", 0.0),
                        )
                    )
                )
                closure_value = stage.get(
                    "face_sum_momentum_closure_error",
                    (0.0, 0.0, 0.0),
                )
                if isinstance(closure_value, (list, tuple)) and len(closure_value) == 3:
                    momentum_closure.append(
                        math.sqrt(sum(float(value) ** 2 for value in closure_value))
                    )

    sample_count = len(records)
    cumulative_mass = sum(masses)
    cumulative_momentum = tuple(sum(momentum[axis] for momentum in momenta) for axis in range(3))
    momentum_norms = [math.sqrt(sum(value**2 for value in momentum)) for momentum in momenta]
    return OpenBoundaryHistoryAudit(
        samples=sample_count,
        finite_samples=finite_samples,
        reference_mass=reference_mass,
        reference_momentum=reference_momentum,
        cumulative_mass_delta=cumulative_mass,
        mean_absolute_mass_delta_fraction=(
            sum(abs(value) for value in masses) / sample_count / reference_mass
            if sample_count
            else 0.0
        ),
        maximum_absolute_mass_delta_fraction=(
            max((abs(value) for value in masses), default=0.0) / reference_mass
        ),
        cumulative_momentum_delta=cumulative_momentum,
        mean_momentum_delta_fraction=(
            sum(momentum_norms) / sample_count / reference_momentum if sample_count else 0.0
        ),
        maximum_momentum_delta_fraction=(max(momentum_norms, default=0.0) / reference_momentum),
        maximum_face_sum_mass_closure_fraction=(max(mass_closure, default=0.0) / reference_mass),
        maximum_face_sum_momentum_closure_fraction=(
            max(momentum_closure, default=0.0) / reference_momentum
        ),
        finite=(finite_samples == sample_count),
    )


__all__ = ["OpenBoundaryHistoryAudit", "audit_open_boundary_history"]
