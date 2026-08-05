# Körner I→G actual `f` population-transfer plan / validator R1

## Scope and production finding

`free_surface_population_transfer_plan.py` is a **pure cold-path** contract. It
imports neither `free_surface_lbm` nor the topology transaction and returns no
mutation operation. The only R1 plan outcome after valid evidence is
`WITHHELD_UNSPECIFIED_TRANSFER_POLICY` with `operations == ()`.

The production topology implementation currently creates a real population
mass gap at an I→G conversion:

```python
# free_surface_topology_transaction.py, conversion block
cf = torch.where(to_gas.unsqueeze(0), torch.zeros_like(cf), cf)
```

The same block sets the donor `mass` and `fill` to zero. The optional existing
I→G closure only redistributes **independent mass** to surviving INTERFACE
receivers; it explicitly does not transfer `f`. Thus its signed independent
mass debit/credit closure is not population closure. Existing conversion
evidence records `population_before` / `population_after`, and declares
`i_to_g_population_owner_status = "WITHHELD_NO_POPULATION_TRANSFER"`.
No future writer may call that independent-mass credit an `f` transfer.

## Why copying one donor `f` is not a policy

Let `F_D = sum_q f_D[q]` and `F_R = sum_{r,q} f_R[q]` over explicit source and
destination partitions. An actual population mutation needs evidence

`ΔF = Σ(f_D_after - f_D_before) + Σ(f_R_after - f_R_before)`.

A copy without donor debit yields `ΔF = F_D` (non-conservative). A copy plus
zeroing donor can give a machine-zero `ΔF`, but does **not** establish that
copying all kinetic populations is the physically valid interface/gas boundary
operator. It may violate the intended phase/boundary reconstruction and cannot
be inferred from independent mass. R1 therefore withholds both cases unless a
future policy supplies its own validated definition.

## Required future-writer evidence

The caller must provide one `IToGPopulationTransferEvent` and
`PopulationTransferEvidence`:

- D3Q19, a non-empty event id, and explicit, non-empty source and destination
  partitions (no inferred neighbours; unique and disjoint);
- exactly every converting source is pre-`I`, post-`G`; every destination is a
  surviving pre/post-`I` owner;
- explicit `(N, 19)` pre/post `f` tensors keyed in partition order, with same
  finite dtype/device; and
- an **independent** signed scalar mass debit, credit, and residual. The
  residual must match `debit + credit`; it is never calculated from `f`.

The validator separately computes `ΔF` and checks the mass record. There is no
tolerance. A float32 zero is tagged only as machine-zero:
`exact_float32_closure_claimed` is always false. R1 makes no mathematical
exactness or physical-acceptance claim.

## Ownership options deliberately left for a later policy R2

A writer must choose and TDD one operator rather than silently copying:

1. **No actual transfer / boundary reconstruction**: remove donor populations
   and construct destination/interface populations through a separately
   specified kinetic boundary rule. This needs a global population/boundary
   ledger, not merely local `ΔF`.
2. **Conservative partition transfer**: prescribe a q-wise map from every
   donor row to every receiver row, with donor debit, receiver credit, and
   population-sum evidence. It must define multi-donor/multi-receiver weights,
   momentum treatment, and phase boundary semantics.
3. **Reservoir/boundary owner**: debit donor `f` to an explicit gas/boundary
   reservoir rather than an INTERFACE receiver. That owner and its accounting
   must be represented in the event; it cannot be magic disappearance.

None is implemented by R1. `policy="copy"` and any other non-unspecified
policy are `WITHHELD_UNSUPPORTED_TRANSFER_POLICY`.

## Verification

```text
python -m pytest -q tests/test_free_surface_population_transfer_plan.py
```

The TDD tests cover empty fail-closed plans, unpaired-copy population residual,
copy-plus-reset still withheld without an operator policy, partition/ownership
rejection, separate mass-ledger tamper detection, float32 non-exactness, and
unknown-policy rejection.
