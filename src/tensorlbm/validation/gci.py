"""Roache Grid Convergence Index (GCI) machinery for verification studies.

The implementation follows Roache (1994, *J. Fluids Eng.* 116, 405) and the
unequal-refinement-ratio extension documented by Celik & Zhang (2008) and
ASME V&V-20.  Everything here is pure ``math`` so that the index can be
computed on any scalar sequence (integral forces, profile errors, friction
velocities, ...) independently of the solver backend.

Conventions
-----------
* ``grid_sizes`` are characteristic cell sizes ``h`` (or cell counts: any
  quantity that *decreases* under refinement).  Order does not matter; the
  helper sorts them from coarse to fine.
* ``values`` are the same observable evaluated on each grid.
* The refinement ratios are ``r21 = h_medium / h_fine`` and
  ``r32 = h_coarse / h_medium``.  Both must be > 1.
* The reported ``gci_fine_pct`` uses the safety factor ``Fs`` (1.25 for a
  three-grid study, 3.0 when only two grids are available) and is expressed
  as a percentage of the extrapolated value.

Failure modes are reported, never hidden: a non-monotone sequence yields
``apparent_order = None`` and ``converged = False`` with a ``reason`` string,
so downstream verdict logic can be fail-closed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

__all__ = [
    "FS_THREE_GRID",
    "FS_TWO_GRID",
    "GCIReport",
    "gci",
    "observed_order",
    "richardson_extrapolation",
]

FS_THREE_GRID = 1.25
FS_TWO_GRID = 3.0


def _sorted_study(
    grid_sizes: Sequence[float], values: Sequence[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return ``(h, phi)`` sorted coarse -> fine by decreasing ``h``."""
    if len(grid_sizes) != len(values):
        raise ValueError("grid_sizes and values must have the same length")
    if len(grid_sizes) < 2:
        raise ValueError("a grid study needs at least two grids")
    pairs = sorted(zip(grid_sizes, values, strict=True), key=lambda item: -float(item[0]))
    h = tuple(float(item[0]) for item in pairs)
    phi = tuple(float(item[1]) for item in pairs)
    if h[0] <= h[-1]:
        raise ValueError("grid sizes must differ between coarse and fine grids")
    for coarse, fine in zip(h, h[1:], strict=False):
        if not coarse > fine:
            raise ValueError("grid sizes must be distinct")
    for value in phi:
        if not math.isfinite(value):
            raise ValueError("observable values must be finite")
    return h, phi


def _initial_order(phi: tuple[float, ...], ratio: float) -> float:
    coarse, medium, fine = phi[-3], phi[-2], phi[-1]
    delta_32 = medium - coarse
    delta_21 = fine - medium
    if delta_21 == 0.0 or delta_32 == 0.0:
        raise ValueError("identical values on two grids; order is indeterminate")
    return abs(math.log(abs(delta_32 / delta_21)) / math.log(ratio))


def observed_order(
    grid_sizes: Sequence[float],
    values: Sequence[float],
    *,
    maximum_iterations: int = 200,
    tolerance: float = 1.0e-10,
) -> float:
    """Apparent order of accuracy ``p`` of the grid triple.

    For a uniform refinement ratio this is ``ln|Δ32/Δ21| / ln(r)``; for
    unequal ratios the Celik & Zhang fixed-point form is iterated to
    convergence.  A non-monotone triple (``Δ32·Δ21 < 0``) raises
    :class:`ValueError`; callers that want a fail-closed report should use
    :func:`gci` instead.
    """
    h, phi = _sorted_study(grid_sizes, values)
    if len(h) != 3:
        raise ValueError("observed_order needs exactly three grids")
    r21 = h[1] / h[2]
    r32 = h[0] / h[1]
    delta_32 = phi[1] - phi[0]
    delta_21 = phi[2] - phi[1]
    if delta_21 == 0.0 or delta_32 == 0.0:
        raise ValueError("identical values on two grids; order is indeterminate")
    if delta_21 * delta_32 < 0.0:
        raise ValueError("grid triple is not monotone; apparent order is undefined")

    log_ratio = math.log(abs(delta_32 / delta_21)) / math.log(r21)
    if abs(r21 - r32) <= 1.0e-12:
        return abs(log_ratio)

    order = abs(log_ratio)
    for _ in range(maximum_iterations):
        if order < 1.0e-8:  # guard the removable singularity at p -> 0
            break
        correction = math.log((r21**order - 1.0) / (r32**order - 1.0)) / math.log(r21)
        updated = abs(log_ratio + correction)
        if abs(updated - order) < tolerance:
            return updated
        order = updated
    return order


def richardson_extrapolation(
    h_coarse: float,
    h_fine: float,
    phi_coarse: float,
    phi_fine: float,
    order: float,
) -> float:
    """Richardson extrapolation to zero cell size."""
    ratio = h_coarse / h_fine
    if ratio <= 1.0:
        raise ValueError("coarse grid must be larger than fine grid")
    denominator = ratio**order - 1.0
    if abs(denominator) < 1.0e-14:
        raise ValueError("degenerate refinement ratio for the requested order")
    return phi_fine + (phi_fine - phi_coarse) / denominator


@dataclass(frozen=True)
class GCIReport:
    """Result of a three-grid (or two-grid) GCI analysis."""

    grid_sizes: tuple[float, ...]
    values: tuple[float, ...]
    refinement_ratios: tuple[float, ...]
    apparent_order: float | None
    extrapolated_value: float | None
    extrapolated_from_coarse_pair: float | None
    extrapolation_consistency_pct: float | None
    approximate_relative_error_pct: float | None
    gci_fine_pct: float | None
    gci_medium_pct: float | None
    asymptotic_ratio: float | None
    asymptotic_band_ok: bool | None
    safety_factor: float
    grid_count: int
    converged: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def gci(
    grid_sizes: Sequence[float],
    values: Sequence[float],
    *,
    safety_factor: float | None = None,
    asymptotic_band: tuple[float, float] = (0.8, 1.25),
    monotone_required: bool = True,
) -> GCIReport:
    """Grid Convergence Index for a 2- or 3-grid study.

    ``grid_sizes`` may be cell sizes, cell counts or any refinement metric
    provided it decreases under refinement.  ``safety_factor`` defaults to
    Roache's 1.25 (three grids) or 3.0 (two grids).
    """
    h, phi = _sorted_study(grid_sizes, values)
    count = len(h)
    ratios = tuple(h[i] / h[i + 1] for i in range(count - 1))
    default_fs = FS_THREE_GRID if count == 3 else FS_TWO_GRID
    fs = default_fs if safety_factor is None else float(safety_factor)

    def _fail(reason: str) -> GCIReport:
        return GCIReport(
            grid_sizes=h,
            values=phi,
            refinement_ratios=ratios,
            apparent_order=None,
            extrapolated_value=None,
            extrapolated_from_coarse_pair=None,
            extrapolation_consistency_pct=None,
            approximate_relative_error_pct=None,
            gci_fine_pct=None,
            gci_medium_pct=None,
            asymptotic_ratio=None,
            asymptotic_band_ok=None,
            safety_factor=fs,
            grid_count=count,
            converged=False,
            reason=reason,
        )

    if count > 3:
        return _fail("gci supports two or three grids; supply a triple")
    if count == 2:
        r = ratios[0]
        delta = phi[1] - phi[0]
        if delta == 0.0:
            return _fail("identical values on both grids")
        # Two-grid GCI needs an assumed order; Roache recommends p = 1
        # (first-order, conservative) unless a formal order is available.
        order = 1.0
        extrapolated = richardson_extrapolation(h[0], h[1], phi[0], phi[1], order)
        e_a = abs(phi[1] - extrapolated) / max(abs(extrapolated), 1.0e-30) * 100.0
        gci_fine = fs * abs(delta) / max(abs(phi[1]), 1.0e-30) / (r**order - 1.0) * 100.0
        return GCIReport(
            grid_sizes=h,
            values=phi,
            refinement_ratios=ratios,
            apparent_order=order,
            extrapolated_value=extrapolated,
            extrapolated_from_coarse_pair=None,
            extrapolation_consistency_pct=None,
            approximate_relative_error_pct=e_a,
            gci_fine_pct=gci_fine,
            gci_medium_pct=None,
            asymptotic_ratio=None,
            asymptotic_band_ok=None,
            safety_factor=fs,
            grid_count=count,
            converged=True,
            reason="two-grid estimate with assumed first-order convergence",
        )

    delta_32 = phi[1] - phi[0]
    delta_21 = phi[2] - phi[1]
    if delta_21 == 0.0 or delta_32 == 0.0:
        return _fail("identical values on two grids; order is indeterminate")
    if monotone_required and delta_21 * delta_32 < 0.0:
        return _fail("grid triple is not monotone under refinement")

    try:
        order = abs(observed_order(h, phi))
    except ValueError as exc:  # pragma: no cover - defensive
        return _fail(str(exc))

    r21, r32 = ratios[1], ratios[0]
    extrapolated_21 = richardson_extrapolation(h[1], h[2], phi[1], phi[2], order)
    extrapolated_32 = richardson_extrapolation(h[0], h[1], phi[0], phi[1], order)
    consistency = (
        abs(extrapolated_32 - extrapolated_21) / max(abs(extrapolated_21), 1.0e-30) * 100.0
    )
    e_a = abs(phi[2] - extrapolated_21) / max(abs(extrapolated_21), 1.0e-30) * 100.0

    denominator_fine = r21**order - 1.0
    denominator_medium = r32**order - 1.0
    if abs(denominator_fine) < 1.0e-14 or abs(denominator_medium) < 1.0e-14:
        return _fail("degenerate refinement ratio for the observed order")

    gci_fine = fs * abs(delta_21) / max(abs(phi[2]), 1.0e-30) / denominator_fine * 100.0
    gci_medium = fs * abs(delta_32) / max(abs(phi[1]), 1.0e-30) / denominator_medium * 100.0
    asymptotic = gci_medium / max(r21**order * gci_fine, 1.0e-30)
    band_ok = asymptotic_band[0] <= asymptotic <= asymptotic_band[1]
    reasons = []
    if not band_ok:
        reasons.append(
            "asymptotic ratio %.3f outside band [%.2f, %.2f]"
            % (asymptotic, asymptotic_band[0], asymptotic_band[1])
        )
    if order <= 0.05:
        reasons.append("apparent order %.3f is degenerate" % order)

    return GCIReport(
        grid_sizes=h,
        values=phi,
        refinement_ratios=ratios,
        apparent_order=order,
        extrapolated_value=extrapolated_21,
        extrapolated_from_coarse_pair=extrapolated_32,
        extrapolation_consistency_pct=consistency,
        approximate_relative_error_pct=e_a,
        gci_fine_pct=gci_fine,
        gci_medium_pct=gci_medium,
        asymptotic_ratio=asymptotic,
        asymptotic_band_ok=band_ok,
        safety_factor=fs,
        grid_count=count,
        converged=band_ok and order > 0.05,
        reason="; ".join(reasons) if reasons else "asymptotic range confirmed",
    )