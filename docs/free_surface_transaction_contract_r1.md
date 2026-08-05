# D3Q19 Körner Conservative I→G Transaction Contract R1

## Scope

R1 freezes a **cold, detached diagnostic contract** for a Körner I→G
transaction.  It is implemented in
`src/tensorlbm/free_surface_transaction_contract.py` and is intentionally not
imported by the solver, free-surface LBM path, topology transaction, or ledger.
It does not mutate any tensor, apply a correction, change reduction order,
replace float32 with float64, or make a physical-closure claim.

This contract is **D3Q19 only**. Any other lattice is fail-closed as
`WITHHELD_D3Q19_ONLY`.

## State vocabulary and conversion

The complete R1 state vocabulary is exactly:

| Symbol | Meaning |
|---|---|
| `G` | gas |
| `I` | interface |
| `L` | liquid |
| `S` | solid |

An R1 conversion record must contain a valid lattice cell and must be exactly
`I → G`; aliases, implicit phase inference, and other conversions are withheld.

## Required transaction input evidence

`TransactionInput` requires all of the following before diagnostic acceptance:

1. a non-empty event ID;
2. lattice `D3Q19`;
3. one or more explicit `I → G` conversion cells;
4. donor ownership for every converting cell, declaring pre-state `I`,
   independent-mass ownership, and `f` population ownership;
5. receiver ownership for every actual transfer destination, declaring `I`,
   independent-mass, and `f` ownership; its cell set must exactly match the
   transfer destination-cell set;
6. population-transfer evidence containing tuple-typed actual source and
   destination cells plus a non-empty replay reference; and
7. finite, exactly-zero roundoff residual evidence with an explicit exact
   diagnostic-closure claim.

A plan, intended transfer, independent-mass-only record, or arithmetic ledger
is not actual `f` population-transfer evidence.  Missing or incomplete actual
population transfer is always
`WITHHELD_NO_POPULATION_TRANSFER`.

## Roundoff and acceptance separation

No tolerance is used.  A nonzero or non-finite roundoff residual is
`WITHHELD_ROUNDOFF_NOT_EXACT`; it cannot be renamed or promoted to exact
closure.

The report exposes two separate booleans:

- `diagnostic_accepted`: only the detached R1 evidence schema has been proven
  complete and exact under this narrow contract;
- `physical_accepted`: always `False` in R1.

Consequently, even a complete input returns
`DIAGNOSTIC_ACCEPTED_PHYSICAL_WITHHELD`, not physical acceptance.  R1 has no
complete same-order solver mutation replay and therefore cannot establish
physical, PV, inventory, or population closure.

## Fail-closed status surface

The validator returns named withholding statuses for absent event IDs,
conversion cells, donor or receiver ownership, population transfer, roundoff
evidence, and invalid data.  It never throws a partially accepted transaction
into a production path.  It accepts no global correction and performs no dtype
promotion or float64 replacement.

## Verification

```text
python -m pytest -q tests/test_free_surface_transaction_contract.py
# 7 passed
```

The test suite was written first (RED: the new module was absent and collection
failed with `ModuleNotFoundError`), then implemented to GREEN.  Tests cover
complete D3Q19 diagnostic-only acceptance, D3Q19-only rejection, named missing
actual-`f` transfer withholding, exact receiver/destination ownership binding,
invalid transfer collection withholding, roundoff non-promotion, and missing
event or receiver ownership.
