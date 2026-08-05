# Free-Surface I→G Exact Float32 D3Q19 Ledger Feasibility R1

## Scope and status

This is a cold diagnostic only.  It does not mutate solver state, does not apply
any correction, does not alter the default `free_surface_step` path, and does
not make a physical, PV, or total-inventory closure claim.

The diagnostic is implemented in
`src/tensorlbm/free_surface_i_to_g_exact_ledger.py`.  Its only public outcome labels are intentionally narrow:

- `WITHHELD_NOT_REPRESENTABLE`: the R1 overall result and method C.  The cold
  observer cannot replay and verify the complete same-order topology mutation;
  local arithmetic that happens not to fail is not evidence that C is feasible
  or integrable;
- `DIAGNOSTIC_ONLY_NOT_STATE_CONSERVATION`: a record ledger can close by
  definition, but its debit is not the solver donor state mutation.

A result never upgrades `physical_closure_claim`, which remains `False`.
Population ownership remains `WITHHELD_NO_POPULATION_TRANSFER`.

## Evidence source

Evidence was captured from the existing real D3Q19 float32 B/C topology
fixtures through the opt-in proposal at HEAD
`9f00b4f3f0c6bc98164c34d7091cb475029a90cb`:

- `B_forced_conversion_deterministic`, step 3;
- `C_dam_break_style_tiny_dynamic_topology`, step 3.

Both fixtures independently reproduce the same actual I→G candidate and are
correctly rejected by the existing strict proposal before state publication:

```text
WITHHELD: entire free_surface_step topology candidate has non-exact
I→G debit/credit closure
```

Observed candidate facts:

| field | observed value |
|---|---:|
| dtype | `torch.float32` |
| donors | 76 |
| receiving cells | 110 |
| D3Q19 link credits | 620 |
| donor state debit | -0.5198758840560913 |
| stored receiver-increment credit | +0.5198759436607361 |
| actual float32 operation residual | +5.960464477539063e-08 |
| donor versus rounded-link exact-record difference | -3.434251993894577e-09 |
| receivers with nonzero aggregation residual | 77 / 110 |
| largest receiver aggregation residual | 9.89530235528946e-10 |

The last two rows compare the exact sum of already-rounded float32 link records
with the stored float32 receiver increment.  They demonstrate that a global
ledger can hide receiver-local aggregation loss.

## Required distinction

### 1. Mathematical exact sum

Each float32 link value can be interpreted as its exact IEEE-754 rational
number.  Their sum is then deterministic.  This is a property of the records,
not of the solver state mutation.

For the B/C candidate, the exact sum of rounded link credits differs from the
exact donor field sum.  The mismatch originates at `credit = round32(mass / n)`:
for a donor with `n` legal links, `n * round32(mass / n)` generally differs from
`mass`.

### 2. Float32 operation exactness

The production-style receiver increment is a float32 scatter/reduction, while
the donor is cleared by a separate float32 state operation.  The independently
observed float32 aggregates above leave `+5.960464477539063e-08`, so equality
under `residual == 0` is false.  No tolerance is introduced.

### 3. State-mutation mass conservation

The candidate topology mutation writes receiver independent mass via an
aggregated increment and clears a converting donor to zero.  It also clears the
donor populations and does not transfer them.  A record representation cannot
be called state conservation unless it reproduces both actual float32 mutation
paths.  It cannot be called physical closure in any event.

## Representation assessment

### A — donor debit = sum of actually rounded per-link credits

The diagnostic can make this ledger exact by construction: debit is the
negative exact integer-quanta sum of the actual rounded link credits, and
credit is the corresponding positive sum.

Status: `DIAGNOSTIC_ONLY_NOT_STATE_CONSERVATION`.

This is useful evidence for what credits were recorded.  It explicitly changes
the debit meaning from *donor state delta* to *rounded link-credit total*.
Therefore it must not be presented as mass conservation.  The diagnostic always
reports the donor-versus-rounded-link residual beside A.

### B — deterministic fixed-point / integer-quanta diagnostic

Each finite binary32 value is represented exactly as a Python integer count of
units of `2**-149`.  Python integers are used deliberately to avoid int64
overflow for normal-scale float32 values.  This makes link-record aggregation
order-independent and exact for the diagnostic representation.

Status: `DIAGNOSTIC_ONLY_NOT_STATE_CONSERVATION`.

It proves only the encoded-record arithmetic.  It does not remove float32
division rounding, receiver scatter aggregation rounding, or the distinct donor
clear operation.  It never mutates the solver and is not a fixed-point solver.

### C — residual assigned to one declared local receiver link

A deterministic first legal D3Q19 receiver is selected for each donor.  The
cold diagnostic simulates adding that donor's exact rounded-link residual to
that receiver and accepts only if all conditions hold:

1. the donor is a pre-topology INTERFACE cell;
2. a legal surviving INTERFACE receiver exists;
3. the residual itself is exactly representable as float32;
4. the proposed addition survives the actual float32 receiver addition exactly;
5. the adjusted receiver increment matches the exact assigned records;
6. the full same-order topology mutation is available for replay.

This R1 observer intentionally does not receive the co-applied legacy
redistribution increment and does not replay clamp, conversion, halo, and
isolation.  It therefore withholds C even for a one-link local state:
`WITHHELD_NOT_REPRESENTABLE` with
`complete same-order topology mutation is unavailable to this cold observer`.
This is deliberate fail-closed behavior, not a failed physical calculation.
Even if a local arithmetic check did not fail, it would not make C feasible or
integrable in R1: only a complete same-order topology-mutation replay could
support that different claim.

For a three-link state with a nonzero receiver mass, C independently shows a
stronger obstruction: its residual can be visible in an increment but is
rounded away by the actual `receiver_mass + receiver_increment` float32 state
addition.  The check therefore evaluates the final detached mass state, not
merely an increment tensor.

For each independently replayed real B and C candidate, C is
`WITHHELD_NOT_REPRESENTABLE`; its first direct failure is receiver float32
increment aggregation, before state addition.  Thus C cannot repair or
represent either actual candidate without changing the actual state arithmetic.
No hidden correction is applied.

## Final R1 conclusion

R1 overall and C: `WITHHELD_NOT_REPRESENTABLE` for each real B and C float32
multi-link candidate.  C is not feasible or integrated in R1.

A and B are exact diagnostic ledgers only and disclose their donor-state delta
mismatch.  Their `DIAGNOSTIC_ONLY_NOT_STATE_CONSERVATION` status is a
sub-diagnostic label, not proposal success, not a non-WITHHELD physical
conclusion, and not state conservation.  There is no claim of physical closure,
no default-solver integration, and no global or local correction.

## Verification

```text
python -m py_compile src/tensorlbm/free_surface_i_to_g_exact_ledger.py \
  tests/test_free_surface_i_to_g_exact_ledger.py
pytest -q tests/test_free_surface_i_to_g_exact_ledger.py
# 5 passed

pytest -q tests/test_free_surface_i_to_g_ownership_closure.py \
  tests/test_free_surface_topology_transaction.py \
  tests/test_free_surface_closure_experiment.py \
  tests/test_free_surface_inventory_reconciliation.py
# 51 passed
```

The repository-wide `pytest -q` collection remains blocked by a pre-existing
unrelated import error in `tests/test_gallium_pf_energy.py`: the example imports
`benchmark_gallium_pf`, which imports unavailable
`benchmark_gallium_melting`.  This diagnostic neither changes nor bypasses that
failure.
