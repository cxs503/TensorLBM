# Accuracy-recommendation physical-evidence gate

`tensorlbm.accuracy_recommendation.recommend_by_physical_accuracy()` is a pure,
typed, fail-closed admission gate for recommendations stated as **physical
accuracy**. It does not run a solver, change collision kernels, or infer
accuracy from numerical consistency.

## What is explicitly rejected

The following are not `PhysicalAccuracyEvidence`, and produce
`WITHHELD_NO_PHYSICAL_ACCURACY_EVIDENCE` with
`MISSING_TYPED_PHYSICAL_ACCURACY_EVIDENCE`:

- D3Q19/D3Q27 fixed-point or collision-contract comparisons;
- collision capability matrices and general capability claims;
- full-wet/SUBOFF runtime, force-window, or checkpoint artifacts alone.

Such records may be useful provenance or diagnostics, but they do not establish
error against a physical reference.

## Admission requirements

Every compared candidate must provide a typed `PhysicalAccuracyEvidence` with:

1. the same `case_id`, `reference_id`, and `reference_source_id` for every
   candidate;
2. grid, time, **and** domain convergence evidence;
3. an exactly matching `KPIDefinition` (name, units, aggregation, and sampling
   window);
4. an exactly matching `ErrorMetric` definition (`name` and `normalization`)
   for every candidate before error bounds are ordered;
5. valid SHA-256 `configuration_hash` and `provenance_hash` values;
6. a declared finite real-number error metric and uncertainty (not strings or
   booleans).

`ConvergenceEvidence.grid`, `.time`, and `.domain` are each required to be an
actual `bool`; truthy values such as `1` and `"true"` are rejected at typed
evidence construction rather than treated as convergence.

If any requirement is absent or incompatible, the result is withheld and lists
both human-readable `missing_requirements` and stable `reason_codes`.

## Ranking rule

Only after admission the gate ranks candidates by the declared physical-error
upper bound, `error.value + error.uncertainty` (then `candidate_id` for a
deterministic tie break). It never compares D3Q19 and D3Q27 based on a fixed
point, collision residual, or a capability declaration.

A recommendation status of `RECOMMENDED_FROM_PHYSICAL_ACCURACY_EVIDENCE` is
therefore bounded by the submitted, hash-bound evidence data; it is not a
claim that the software has independently established physical accuracy.
