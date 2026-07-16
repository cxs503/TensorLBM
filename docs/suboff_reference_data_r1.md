# SUBOFF reference resistance-data compilation (r1)

`tensorlbm.suboff_reference_data` compiles known literature values for the
DARPA SUBOFF submarine hull resistance coefficient (Ct) into a typed, frozen
data structure.  Each entry carries an explicit source citation, applicable
conditions, and a declared uncertainty.  Values that could not be
independently confirmed from primary sources are marked
`WITHHELD_NO_REFERENCE_DATA_AVAILABLE` and carry no numeric value.

## Purpose

The compiled data is consumed by
[`tensorlbm.accuracy_recommendation`](accuracy_recommendation_evidence_gate.md)
to construct `ErrorMetric` records for the physical-accuracy admission gate.
The gate ranks candidates by the declared physical-error upper bound
(`error.value + error.uncertainty`), where the error is computed against a
reference Ct value from this registry.

## Typed data structure

```python
@dataclass(frozen=True)
class SuboffReferenceDatum:
    case_id: str               # e.g. "SUBOFF-AFF1-bare-hull-Re1.2e7"
    reference_id: str          # unique ID for this reference value
    reference_source_id: str   # source family ID
    Ct_reference: float | None # reference Ct (None for WITHHELD)
    Re: float | None           # Reynolds number (None for WITHHELD)
    uncertainty: float | None # one-sigma uncertainty (None for WITHHELD)
    source_citation: str       # human-readable citation
    hull_type: str             # "bare_hull" (AFF-1) or "full" (AFF-8)
    reference_area_basis: str  # "wetted_surface" or "cross_section"
    applicable_conditions: str # flow conditions description
    notes: str                 # caveats and provenance
```

The `is_withheld` property returns `True` when `reference_source_id` equals
`WITHHELD_NO_REFERENCE_DATA_AVAILABLE`.  WITHHELD entries enforce
`Ct_reference=None`, `Re=None`, `uncertainty=None` at construction time.

## Compiled reference data

### 1. ITTC 1957 Model-Ship Correlation Line

A semi-empirical friction formula: `Cf = 0.075 / (log10(Re) - 2)^2`.

This is a well-established correlation line used throughout naval
hydrodynamics and is already used in `tensorlbm.suboff_resistance`.  It
provides a **frictional** resistance coefficient only; it does not include
pressure or form drag.

| reference_id | Re | Cf | Uncertainty | Case |
|---|---|---|---|---|
| `ITTC-1957-Cf-Re1.2e7` | 1.2×10⁷ | 0.00291 | ±5% (0.00015) | SUBOFF-AFF1-bare-hull-Re1.2e7 |
| `ITTC-1957-Cf-Re1.0e7` | 1.0×10⁷ | 0.00300 | ±5% (0.00015) | SUBOFF-AFF1-bare-hull-Re1.0e7 |
| `ITTC-1957-Cf-Re2.0e6` | 2.0×10⁶ | 0.00405 | ±5% (0.00020) | SUBOFF-AFF8-full-Re2.0e6 |

**Source**: ITTC 1957 Model-Ship Correlation Line.

**Uncertainty rationale**: The ITTC-1957 line is a model-ship correlation
line, not a fundamental friction law.  For flat plates its uncertainty is
approximately ±2–3%; for a 3D axisymmetric body like the SUBOFF bare hull,
a conservative ±5% is used.

### 2. DARPA SUBOFF AFF-8 experimental

| reference_id | Re | Ct | Uncertainty | Case |
|---|---|---|---|---|
| `DARPA-SUBOFF-AFF8-experimental-Re2.0e6` | 2.0×10⁶ | 0.0040 | ±10% (0.0004) | SUBOFF-AFF8-full-Re2.0e6 |

**Source**: DARPA SUBOFF AFF-8 experimental tow-tank measurement, cited from
the task specification.  Primary-source verification pending.

**Uncertainty rationale**: Tow-tank repeatability is typically ±2–5%, but the
larger ±10% bound accounts for citation uncertainty (value cited from task
specification rather than independently verified from primary DARPA technical
reports).

### 3. WITHHELD entries

| reference_id | Status | Reason |
|---|---|---|
| `WITHHELD-DARPA-SUBOFF-AFF1-experimental-Re1.2e7` | WITHHELD | Specific experimental Ct values for SUBOFF AFF-1 bare hull could not be independently confirmed from primary DARPA technical reports (DTRC/SHD-1298 series). |
| `WITHHELD-CFD-RANS-AFF1-Re1.2e7` | WITHHELD | Specific CFD reference Ct values from published RANS/LES validation studies could not be independently confirmed. |

WITHHELD entries carry `Ct_reference=None`, `Re=None`, `uncertainty=None`.
They cannot be used to construct a valid `ErrorMetric` for the
accuracy-recommendation gate — this is the intended fail-closed behaviour.

## Look-up API

```python
from tensorlbm.suboff_reference_data import (
    SUBOFF_REFERENCE_REGISTRY,
    get_reference_data,
    get_reference_data_by_case,
    list_available_case_ids,
    list_available_reference_ids,
    compute_ittc1957_cf,
)

# Look up by reference_id
ref = get_reference_data("ITTC-1957-Cf-Re1.2e7")

# Look up all references for a case
refs = get_reference_data_by_case("SUBOFF-AFF1-bare-hull-Re1.2e7")

# Compute ITTC-1957 Cf at any Re
cf = compute_ittc1957_cf(1.2e7)  # 0.00291
```

## Integration with the accuracy-recommendation gate

The reference data provides the `case_id`, `reference_id`,
`reference_source_id`, `Ct_reference`, and `uncertainty` needed to construct
`PhysicalAccuracyEvidence` records for
`recommend_by_physical_accuracy()`:

```python
from tensorlbm.accuracy_recommendation import (
    ConvergenceEvidence, ErrorMetric, KPIDefinition,
    PhysicalAccuracyEvidence, recommend_by_physical_accuracy,
)
from tensorlbm.suboff_reference_data import get_reference_data

ref = get_reference_data("ITTC-1957-Cf-Re1.2e7")
ct_measured = ref.Ct_reference * 1.03  # 3% above reference
error = abs(ct_measured - ref.Ct_reference) / ref.Ct_reference

evidence = PhysicalAccuracyEvidence(
    candidate_id="D3Q19-MRT",
    case_id=ref.case_id,
    reference_id=ref.reference_id,
    reference_source_id=ref.reference_source_id,
    configuration_hash="a" * 64,
    provenance_hash="b" * 64,
    kpi=KPIDefinition("Ct_total", "1", "time_mean", "post-transient window"),
    error=ErrorMetric("absolute relative error", "reference Ct_total",
                      error, ref.uncertainty / ref.Ct_reference),
    convergence=ConvergenceEvidence(True, True, True),
)

result = recommend_by_physical_accuracy([evidence])
```

## Design principles

1. **Fail-closed**: WITHHELD entries carry no numeric value and cannot feed
   the gate's error metric.
2. **Explicit provenance**: Every entry has a `source_citation` and
   `applicable_conditions` string.
3. **Honest uncertainty**: Uncertainty bounds are declared per-entry, not
   assumed globally.
4. **No solver modification**: This module is pure data + look-up; it does
   not modify any solver hot path.
5. **Wetted-surface basis**: All non-withheld Ct values use the
   wetted-surface-area normalization, consistent with the ITTC-1957
   convention and the SUBOFF experimental tradition.
