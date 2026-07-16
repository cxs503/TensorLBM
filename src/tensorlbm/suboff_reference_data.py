"""Compiled SUBOFF reference resistance data for the accuracy-recommendation gate.

This module compiles known literature values for the DARPA SUBOFF submarine
hull resistance coefficient (Ct) into a typed, frozen data structure.  Each
entry carries an explicit source citation, applicable conditions, and a
declared uncertainty.  Values that could not be independently confirmed from
primary sources are marked ``WITHHELD_NO_REFERENCE_DATA_AVAILABLE`` and carry
no numeric Ct/Re/uncertainty.

The compiled data is consumed by
:mod:`tensorlbm.accuracy_recommendation` to construct
:class:`~tensorlbm.accuracy_recommendation.ErrorMetric` records for the
physical-accuracy admission gate.

Reference data sources
----------------------

1. **ITTC 1957 Model-Ship Correlation Line** — a semi-empirical friction
   formula: ``Cf = 0.075 / (log10(Re) - 2)^2``.  This is a well-established
   correlation line used throughout naval hydrodynamics and is already
   used in :mod:`tensorlbm.suboff_resistance`.  It provides a *frictional*
   resistance coefficient only; it does not include pressure or form drag.

2. **DARPA SUBOFF AFF-8 experimental** — tow-tank measurement of the full
   SUBOFF configuration (bare hull + sail + four stern appendages) at
   ``Re = 2.0×10^6``, ``Ct ≈ 0.0040`` (based on wetted surface area).
   This value is cited from the task specification; primary-source
   verification is pending.

3. **DARPA SUBOFF AFF-1 bare hull experimental** — WITHHELD.  Specific
   experimental Ct values for the bare hull (AFF-1) could not be
   independently confirmed from primary DARPA technical reports.

4. **Published CFD reference values** — WITHHELD.  Specific RANS/LES Ct
   values for SUBOFF AFF-1 could not be independently confirmed from
   published literature.

All non-withheld values are based on the wetted-surface-area normalization,
consistent with the ITTC-1957 convention and the SUBOFF experimental
tradition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, log10
from numbers import Real
from typing import Sequence

__all__ = [
    "SuboffReferenceDatum",
    "SUBOFF_REFERENCE_REGISTRY",
    "get_reference_data",
    "get_reference_data_by_case",
    "list_available_case_ids",
    "list_available_reference_ids",
    "compute_ittc1957_cf",
]

_WITHHELD_MARKER = "WITHHELD_NO_REFERENCE_DATA_AVAILABLE"


def compute_ittc1957_cf(re: float) -> float:
    """Compute the ITTC 1957 model-ship correlation friction coefficient.

    ``Cf = 0.075 / (log10(Re) - 2)^2``

    Parameters
    ----------
    re :
        Reynolds number (must be > 100 for formula validity).

    Returns
    -------
    float
        Frictional resistance coefficient Cf (wetted-surface basis).
    """
    if not isinstance(re, Real) or isinstance(re, bool):
        raise TypeError("re must be a real number")
    if not isfinite(re) or re <= 100.0:
        raise ValueError("re must be finite and > 100 for ITTC-1957 formula")
    return 0.075 / (log10(float(re)) - 2.0) ** 2


@dataclass(frozen=True)
class SuboffReferenceDatum:
    """One compiled SUBOFF resistance reference value from literature.

    Attributes
    ----------
    case_id :
        Identifier for the physical case (e.g. ``"SUBOFF-AFF1-bare-hull-Re1.2e7"``).
    reference_id :
        Unique identifier for this specific reference value.
    reference_source_id :
        Identifier for the source family (e.g. ``"ITTC-1957-model-ship-correlation-line"``).
    Ct_reference :
        Reference total resistance coefficient (wetted-surface basis).
        ``None`` for WITHHELD entries.
    Re :
        Reynolds number for the reference value.  ``None`` for WITHHELD entries.
    uncertainty :
        One-sigma uncertainty in Ct_reference.  ``None`` for WITHHELD entries.
    source_citation :
        Human-readable citation string.
    hull_type :
        SUBOFF hull variant: ``"bare_hull"`` (AFF-1), ``"full"`` (AFF-8), etc.
    reference_area_basis :
        Area basis for Ct normalization: ``"wetted_surface"`` or ``"cross_section"``.
    applicable_conditions :
        Description of applicable flow conditions (Re, phase, etc.).
    notes :
        Additional notes, caveats, or provenance information.
    """

    case_id: str
    reference_id: str
    reference_source_id: str
    Ct_reference: float | None
    Re: float | None
    uncertainty: float | None
    source_citation: str
    hull_type: str = "bare_hull"
    reference_area_basis: str = "wetted_surface"
    applicable_conditions: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        for name in ("case_id", "reference_id", "reference_source_id",
                     "source_citation", "hull_type", "reference_area_basis",
                     "applicable_conditions", "notes"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        if self.is_withheld:
            # WITHHELD entries must not carry numeric values.
            if self.Ct_reference is not None:
                raise ValueError(
                    "WITHHELD entry must have Ct_reference=None"
                )
            if self.Re is not None:
                raise ValueError("WITHHELD entry must have Re=None")
            if self.uncertainty is not None:
                raise ValueError("WITHHELD entry must have uncertainty=None")
        else:
            # Non-withheld entries must have valid numeric values.
            self._validate_numeric("Ct_reference", self.Ct_reference)
            self._validate_numeric("Re", self.Re)
            self._validate_numeric("uncertainty", self.uncertainty, allow_zero=True)

    @staticmethod
    def _validate_numeric(name: str, value: object,
                          allow_zero: bool = False) -> None:
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real number (not bool)")
        v = float(value)
        if not isfinite(v):
            raise ValueError(f"{name} must be finite")
        if allow_zero:
            if v < 0.0:
                raise ValueError(f"{name} must be >= 0")
        else:
            if v <= 0.0:
                raise ValueError(f"{name} must be > 0")

    @property
    def is_withheld(self) -> bool:
        """True if this entry is a WITHHELD placeholder."""
        return self.reference_source_id == _WITHHELD_MARKER

    @classmethod
    def withheld(
        cls,
        *,
        case_id: str,
        reference_id: str,
        reference_source_id: str,
        source_citation: str,
        hull_type: str = "bare_hull",
        reference_area_basis: str = "wetted_surface",
        applicable_conditions: str = "",
        notes: str = "",
    ) -> SuboffReferenceDatum:
        """Create a WITHHELD entry with no numeric value.

        The ``reference_source_id`` is forced to
        ``WITHHELD_NO_REFERENCE_DATA_AVAILABLE``.
        """
        return cls(
            case_id=case_id,
            reference_id=reference_id,
            reference_source_id=_WITHHELD_MARKER,
            Ct_reference=None,
            Re=None,
            uncertainty=None,
            source_citation=source_citation,
            hull_type=hull_type,
            reference_area_basis=reference_area_basis,
            applicable_conditions=applicable_conditions,
            notes=notes,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "case_id": self.case_id,
            "reference_id": self.reference_id,
            "reference_source_id": self.reference_source_id,
            "Ct_reference": self.Ct_reference,
            "Re": self.Re,
            "uncertainty": self.uncertainty,
            "source_citation": self.source_citation,
            "hull_type": self.hull_type,
            "reference_area_basis": self.reference_area_basis,
            "applicable_conditions": self.applicable_conditions,
            "notes": self.notes,
            "is_withheld": self.is_withheld,
        }


# ---------------------------------------------------------------------------
# Compiled reference data
# ---------------------------------------------------------------------------

# ITTC-1957 Cf at Re = 1.2e7 (SUBOFF bare-hull design Reynolds number)
# Cf = 0.075 / (log10(1.2e7) - 2)^2 = 0.075 / (7.07918 - 2)^2 ≈ 0.002908
_ITTC_CF_RE_1_2E7 = compute_ittc1957_cf(1.2e7)

# ITTC-1957 Cf at Re = 2.0e6
# Cf = 0.075 / (log10(2.0e6) - 2)^2 = 0.075 / (6.30103 - 2)^2 ≈ 0.004055
_ITTC_CF_RE_2_0E6 = compute_ittc1957_cf(2.0e6)

# ITTC-1957 Cf at Re = 1.0e7
# Cf = 0.075 / (log10(1.0e7) - 2)^2 = 0.075 / (7 - 2)^2 = 0.003000
_ITTC_CF_RE_1_0E7 = compute_ittc1957_cf(1.0e7)

# Uncertainty estimates:
# - ITTC-1957 line: ±5% (conservative for 3D axisymmetric body; the line is a
#   model-ship correlation, not a fundamental friction law).
# - SUBOFF AFF-8 experimental: ±10% (value cited from task specification,
#   primary-source verification pending; tow-tank repeatability typically
#   ±2-5%, but the larger bound accounts for citation uncertainty).
_ITTC_UNCERTAINTY_FRAC = 0.05
_AFF8_EXPERIMENTAL_UNCERTAINTY_FRAC = 0.10

SUBOFF_REFERENCE_REGISTRY: tuple[SuboffReferenceDatum, ...] = (
    # ------------------------------------------------------------------
    # 1. ITTC-1957 friction line at Re = 1.2e7 (SUBOFF bare-hull design Re)
    # ------------------------------------------------------------------
    SuboffReferenceDatum(
        case_id="SUBOFF-AFF1-bare-hull-Re1.2e7",
        reference_id="ITTC-1957-Cf-Re1.2e7",
        reference_source_id="ITTC-1957-model-ship-correlation-line",
        Ct_reference=_ITTC_CF_RE_1_2E7,
        Re=1.2e7,
        uncertainty=_ITTC_CF_RE_1_2E7 * _ITTC_UNCERTAINTY_FRAC,
        source_citation=(
            "ITTC 1957 Model-Ship Correlation Line: "
            "Cf = 0.075 / (log10(Re) - 2)^2, evaluated at Re = 1.2e7"
        ),
        hull_type="bare_hull",
        reference_area_basis="wetted_surface",
        applicable_conditions=(
            "Re=1.2e7, single-phase incompressible, deep water, "
            "turbulent boundary layer, bare hull (AFF-1)"
        ),
        notes=(
            "Frictional resistance coefficient only; does not include "
            "pressure/form drag. The ITTC-1957 line is a model-ship "
            "correlation line, not a fundamental friction law. "
            "Uncertainty ±5% (conservative for 3D axisymmetric body)."
        ),
    ),

    # ------------------------------------------------------------------
    # 2. ITTC-1957 friction line at Re = 1.0e7
    # ------------------------------------------------------------------
    SuboffReferenceDatum(
        case_id="SUBOFF-AFF1-bare-hull-Re1.0e7",
        reference_id="ITTC-1957-Cf-Re1.0e7",
        reference_source_id="ITTC-1957-model-ship-correlation-line",
        Ct_reference=_ITTC_CF_RE_1_0E7,
        Re=1.0e7,
        uncertainty=_ITTC_CF_RE_1_0E7 * _ITTC_UNCERTAINTY_FRAC,
        source_citation=(
            "ITTC 1957 Model-Ship Correlation Line: "
            "Cf = 0.075 / (log10(Re) - 2)^2, evaluated at Re = 1.0e7"
        ),
        hull_type="bare_hull",
        reference_area_basis="wetted_surface",
        applicable_conditions=(
            "Re=1.0e7, single-phase incompressible, deep water, "
            "turbulent boundary layer, bare hull (AFF-1)"
        ),
        notes=(
            "Frictional resistance coefficient only; does not include "
            "pressure/form drag. Uncertainty ±5%."
        ),
    ),

    # ------------------------------------------------------------------
    # 3. ITTC-1957 friction line at Re = 2.0e6
    # ------------------------------------------------------------------
    SuboffReferenceDatum(
        case_id="SUBOFF-AFF8-full-Re2.0e6",
        reference_id="ITTC-1957-Cf-Re2.0e6",
        reference_source_id="ITTC-1957-model-ship-correlation-line",
        Ct_reference=_ITTC_CF_RE_2_0E6,
        Re=2.0e6,
        uncertainty=_ITTC_CF_RE_2_0E6 * _ITTC_UNCERTAINTY_FRAC,
        source_citation=(
            "ITTC 1957 Model-Ship Correlation Line: "
            "Cf = 0.075 / (log10(Re) - 2)^2, evaluated at Re = 2.0e6"
        ),
        hull_type="bare_hull",
        reference_area_basis="wetted_surface",
        applicable_conditions=(
            "Re=2.0e6, single-phase incompressible, deep water, "
            "turbulent boundary layer"
        ),
        notes=(
            "Frictional resistance coefficient only; does not include "
            "pressure/form drag. Uncertainty ±5%."
        ),
    ),

    # ------------------------------------------------------------------
    # 4. SUBOFF AFF-8 full configuration experimental at Re = 2.0e6
    # ------------------------------------------------------------------
    SuboffReferenceDatum(
        case_id="SUBOFF-AFF8-full-Re2.0e6",
        reference_id="DARPA-SUBOFF-AFF8-experimental-Re2.0e6",
        reference_source_id="DARPA-SUBOFF-experimental",
        Ct_reference=0.0040,
        Re=2.0e6,
        uncertainty=0.0040 * _AFF8_EXPERIMENTAL_UNCERTAINTY_FRAC,
        source_citation=(
            "DARPA SUBOFF AFF-8 experimental tow-tank measurement, "
            "Ct ≈ 0.0040 at Re = 2.0e6 (cited from task specification; "
            "primary-source verification pending)"
        ),
        hull_type="full",
        reference_area_basis="wetted_surface",
        applicable_conditions=(
            "Re=2.0e6, single-phase incompressible, deep water, "
            "full configuration AFF-8 (bare hull + sail + 4 stern appendages)"
        ),
        notes=(
            "Total resistance coefficient (friction + pressure + form). "
            "Full configuration AFF-8. Value cited from task specification; "
            "primary DARPA technical report verification pending. "
            "Uncertainty ±10% (accounts for citation uncertainty)."
        ),
    ),

    # ------------------------------------------------------------------
    # 5. SUBOFF AFF-1 bare hull experimental — WITHHELD
    # ------------------------------------------------------------------
    SuboffReferenceDatum.withheld(
        case_id="SUBOFF-AFF1-bare-hull-Re1.2e7",
        reference_id="WITHHELD-DARPA-SUBOFF-AFF1-experimental-Re1.2e7",
        reference_source_id="DARPA-SUBOFF-experimental",
        source_citation=(
            "DARPA SUBOFF AFF-1 bare hull experimental tow-tank data; "
            "specific Ct values not independently verified from primary "
            "DARPA technical reports (DTRC/SHD-1298 series)."
        ),
        hull_type="bare_hull",
        reference_area_basis="wetted_surface",
        applicable_conditions=(
            "Re=1.2e7, single-phase incompressible, deep water, "
            "bare hull (AFF-1)"
        ),
        notes=(
            "WITHHELD: Specific experimental Ct values for SUBOFF AFF-1 "
            "bare hull could not be independently confirmed from primary "
            "DARPA technical reports. The DARPA SUBOFF program conducted "
            "tow-tank experiments at DTRC (David Taylor Research Center); "
            "results are documented in DTRC/SHD-1298 series reports and "
            "the 1992 DARPA SUBOFF Conference Proceedings. Without access "
            "to these primary sources, specific numerical Ct values are "
            "withheld pending verification."
        ),
    ),

    # ------------------------------------------------------------------
    # 6. Published CFD reference for SUBOFF AFF-1 — WITHHELD
    # ------------------------------------------------------------------
    SuboffReferenceDatum.withheld(
        case_id="SUBOFF-AFF1-bare-hull-Re1.2e7",
        reference_id="WITHHELD-CFD-RANS-AFF1-Re1.2e7",
        reference_source_id="published-CFD-validation",
        source_citation=(
            "Published RANS/LES CFD validation studies for SUBOFF AFF-1 "
            "bare hull; specific Ct values not independently verified."
        ),
        hull_type="bare_hull",
        reference_area_basis="wetted_surface",
        applicable_conditions=(
            "Re=1.2e7, single-phase incompressible, deep water, "
            "bare hull (AFF-1)"
        ),
        notes=(
            "WITHHELD: Multiple RANS and LES validation studies for "
            "SUBOFF AFF-1 have been published in the naval hydrodynamics "
            "literature (e.g., NATO AVT-183 workshop, various journal "
            "articles). Without access to these primary sources, specific "
            "numerical Ct values are withheld pending verification."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Look-up functions
# ---------------------------------------------------------------------------

def get_reference_data(reference_id: str) -> SuboffReferenceDatum | None:
    """Look up a reference datum by its ``reference_id``.

    Returns ``None`` if no matching entry exists.
    """
    for d in SUBOFF_REFERENCE_REGISTRY:
        if d.reference_id == reference_id:
            return d
    return None


def get_reference_data_by_case(case_id: str) -> list[SuboffReferenceDatum]:
    """Return all reference data for a given ``case_id``."""
    return [d for d in SUBOFF_REFERENCE_REGISTRY if d.case_id == case_id]


def list_available_case_ids() -> list[str]:
    """Return a sorted list of all unique case IDs in the registry."""
    return sorted({d.case_id for d in SUBOFF_REFERENCE_REGISTRY})


def list_available_reference_ids() -> list[str]:
    """Return a sorted list of all reference IDs in the registry."""
    return sorted(d.reference_id for d in SUBOFF_REFERENCE_REGISTRY)
