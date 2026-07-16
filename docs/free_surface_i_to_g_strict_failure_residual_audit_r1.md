# I→G Strict-Failure Residual Cold Audit R1

## Boundary

The new cold audit is deliberately not a solver repair. It runs only when the
existing experiment requests `capture_replay_stages=True`; default solver calls
neither import the audit nor allocate its snapshots. The strict I→G gate remains
bit-exact and unchanged.

When B/C reaches its real step-3 I→G failure, the failure is raised before the
topology transaction can create phase or candidate evidence. The opt-in capture
therefore freezes the complete failed `build_i_to_g_ownership_transaction`
invocation immediately before that call, then passes independent tensor clones
to the builder. On rejection it binds the pre-call frozen invocation to the
exact exception and replays that detached invocation. It never exposes a
partial runtime ledger.

## Real B/C result

For both `B_forced_conversion_deterministic` and
`C_dam_break_style_tiny_dynamic_topology`, step 3 reproduces exactly:

```text
TopologyTransactionError:
WITHHELD: entire free_surface_step topology candidate has non-exact
I→G debit/credit closure
```

The same captured failure has 76 donors and 110 receiving cells. It is a
`STRICT_FAILURE_REPLAYED_EXACT` rejection replay, not
`AVAILABLE_REPLAYED_EXACT`: there is no topology candidate or phase sequence to
promote after the strict builder stops.

## Candidate screen

All candidates are examined detached from the same frozen prestate and are
returned as `WITHHELD_NOT_REPRESENTABLE` for these real failures:

1. `record_only`: makes no float32 state change, so it cannot equal the donor
   state delta.
2. `local_receiver_residual`: the per-donor residual fails the actual receiver
   float32 increment/state representability checks already captured by the cold
   ledger.
3. `alternative_exact_split`: a different split changes the actual legacy
   float32 link increments. With no accepted same-order candidate, it cannot
   establish exact state mutation, capacity/clamp safety, or full phase replay.

The common phase context is
`BUILDER_REJECTED_BEFORE_TOPOLOGY_PHASE_EVIDENCE`. This is not a gap hidden by a
new tolerance: full phase replay compatibility is explicitly false. No candidate
is integrated and none is generalized to B/C or to default production calls.

A future local candidate can be labeled only
`PROPOSAL_FEASIBLE_NOT_INTEGRATED` after it independently proves all of:
exact same float32 state mutation, matching donor delta, combined receiver
capacity, no clamp loss, and complete same-order phase replay. This R1 result
does not meet those conditions.

## Verification

```text
python -m py_compile src/tensorlbm/free_surface_topology_transaction.py \
  src/tensorlbm/free_surface_lbm.py \
  src/tensorlbm/free_surface_closure_experiment.py \
  src/tensorlbm/free_surface_topology_mutation_replay_contract.py \
  src/tensorlbm/free_surface_i_to_g_failed_residual_audit.py
pytest -q tests/test_free_surface_topology_mutation_replay_contract.py \
  tests/test_free_surface_i_to_g_exact_ledger.py
# 18 passed
```
