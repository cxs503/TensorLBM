"""CFD field-data quality checks (clean-room, per the cleanroom spec).

Checks are specific to LBM field-data products — finiteness, mass
conservation, shape/dtype conformance — unlike industrial time-series
checks (missing-value / outlier / consistency) used elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tensorlbm.data.catalog import QualityCheck
from tensorlbm.data.contracts import FieldProduct


@dataclass(frozen=True, slots=True)
class FieldQualityResult:
    checks: tuple[QualityCheck, ...]
    overall_score: int
    status: str

    @property
    def passed(self) -> bool:
        return self.status == "passed"


def _finite_check(arr: np.ndarray, name: str) -> QualityCheck:
    n_bad = int(np.isnan(arr).sum() + np.isinf(arr).sum())
    return QualityCheck(
        check_name="finiteness",
        passed=n_bad == 0,
        detail=f"{name}: {n_bad} non-finite elements of {arr.size}",
    )


def _shape_check(arr: np.ndarray, expected_shape: tuple[int, ...], name: str) -> QualityCheck:
    ok = tuple(arr.shape) == tuple(expected_shape)
    return QualityCheck(
        check_name="shape_conformance",
        passed=ok,
        detail=f"{name}: shape {arr.shape} vs expected {expected_shape}",
    )


def _mass_conservation_check(rho: np.ndarray, name: str, tol: float = 1e-6) -> QualityCheck:
    """LBM density should stay ~1.0 (incompressible); flag gross drift."""
    mean = float(rho.mean()) if rho.size else float("nan")
    drift = abs(mean - 1.0)
    return QualityCheck(
        check_name="mass_conservation",
        passed=drift <= tol,
        detail=f"{name}: |<rho>-1| = {drift:.3e} (tol {tol:.1e})",
    )


def check_field_product(
    product: FieldProduct,
    arr: np.ndarray,
    *,
    mass_field: bool = False,
    mass_tol: float = 1e-6,
) -> FieldQualityResult:
    """Run the standard quality checks for a field-data product.

    ``arr`` is the on-disk field (e.g. velocity magnitude, pressure, density).
    When ``mass_field`` is True, an extra mass-conservation check is applied
    (valid for density fields).
    """
    checks: list[QualityCheck] = [
        _finite_check(arr, product.field_name),
        _shape_check(arr, product.shape, product.field_name),
    ]
    if mass_field:
        checks.append(_mass_conservation_check(arr, product.field_name, mass_tol))
    passed = sum(1 for c in checks if c.passed)
    score = round(100 * passed / len(checks)) if checks else 0
    status = "passed" if passed == len(checks) else (
        "warning" if passed > 0 else "failed"
    )
    return FieldQualityResult(tuple(checks), score, status)


def validate_field_product(
    product: FieldProduct,
    arr: np.ndarray,
    *,
    mass_field: bool = False,
    mass_tol: float = 1e-6,
) -> Sequence[QualityCheck]:
    """Convenience: return just the check list (for catalog.record_quality)."""
    return check_field_product(
        product, arr, mass_field=mass_field, mass_tol=mass_tol,
    ).checks
