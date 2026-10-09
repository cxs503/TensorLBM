"""Fail-closed temporal averaging for force sampled on subcycled AMR levels.

For a refinement depth ``d``, the finest level advances ``2**d`` uniformly
spaced substeps during one root-grid step.  Force must therefore be sampled
and averaged exactly that many times.  Keeping this rule in one small common
component prevents a solver or benchmark from silently retaining a hard-coded
denominator when another refinement level is introduced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True, slots=True)
class UniformSubcycleAverager:
    """Validate and average uniformly spaced finest-level force samples."""

    refinement_depth: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.refinement_depth, bool)
            or not isinstance(self.refinement_depth, int)
            or self.refinement_depth < 0
        ):
            raise ValueError("refinement_depth must be a non-negative integer")

    @property
    def expected_samples(self) -> int:
        return 1 << self.refinement_depth

    def mean(self, values: Iterable[float], *, observable: str = "force") -> float:
        """Return the time mean after exact count and finite-value checks."""
        samples = tuple(float(value) for value in values)
        if len(samples) != self.expected_samples:
            raise RuntimeError(
                f"{observable} requires {self.expected_samples} uniformly spaced "
                f"samples at refinement depth {self.refinement_depth}; "
                f"observed {len(samples)}",
            )
        if not all(math.isfinite(value) for value in samples):
            raise FloatingPointError(f"{observable} contains a non-finite sample")
        return math.fsum(samples) / self.expected_samples

    def provenance(self, observed_samples: int) -> dict[str, int | bool]:
        """Return explicit aggregation evidence for logs and result files."""
        return {
            "refinement_depth": self.refinement_depth,
            "expected_samples": self.expected_samples,
            "observed_samples": observed_samples,
            "uniform_sample_count_met": observed_samples == self.expected_samples,
        }


__all__ = ["UniformSubcycleAverager"]
