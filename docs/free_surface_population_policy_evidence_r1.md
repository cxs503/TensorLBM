# Körner I→G `f` ownership policy evidence decision gate — R1

## R1 decision

`free_surface_population_policy_evidence.py` is a pure, cold-path feasibility
evaluator. It neither selects nor implements an ownership policy, and every
result is `WITHHELD_MISSING_POLICY_EVIDENCE` with `feasible == False`.

It evaluates these three **exclusive later-writer options** independently:

1. `explicit_boundary_reconstruction` — a named q-wise reconstruction operator,
   explicit I→G sources, reconstructed destinations, published boundary state,
   and replay reference.
2. `conservative_partition_transfer` — a named q-wise source/destination map,
   explicit partitions, multi-owner partition weights, momentum treatment, and
   replay reference.
3. `gas_boundary_reservoir` — a named q-wise debit to a named gas/boundary
   reservoir, published reservoir accounting and boundary state, explicit
   sources, and replay reference.

For all three, evidence must be published by the real production provenance
`production_free_surface_step_runtime_ledger` and must assert an actual `f`
population transfer. A shaped mapping is deliberately non-production.

## What is explicitly not policy proof

The evaluator never treats independent-mass debit/credit, a mass ledger,
`f_before`/`f_after` snapshots, a population sum, or a before/after residual as
proof of an ownership policy. Such records can audit arithmetic but do not name
the q-wise operator, its ownership destination, phase/boundary semantics, or
replayable implementation.

## Production audit result

The TDD test runs the actual `free_surface_lbm.free_surface_step` on the
existing D3Q19 runtime fixture, captures its production runtime ledger and
replay capture, then evaluates the extracted production evidence. Current
production publishes no actual `f` population transfer and no policy-specific
evidence payload. Therefore **all three** options report
`WITHHELD_MISSING_POLICY_EVIDENCE`; in particular each is missing
`actual_f_population_transfer`.

This does not alter `free_surface_lbm` or topology behavior. R1 is solely the
decision gate required before a future writer can choose and TDD one policy.

## Verification

```text
python -m pytest -q tests/test_free_surface_population_policy_evidence.py \
  tests/test_free_surface_population_transfer_plan.py \
  tests/test_free_surface_production_evidence.py
```
