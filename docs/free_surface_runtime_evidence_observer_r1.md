# D3Q19 Körner runtime evidence observer R1

`tensorlbm.free_surface_runtime_evidence_observer` is a cold, detached adapter
from explicit runtime-shaped I→G evidence into the already merged
`free_surface_transaction_contract`.

## Boundary

The observer imports only the transaction contract.  It does **not** import or
call the free-surface solver, topology transaction, ownership ledger, or any
mutation path.  It applies no global correction and no numerical dtype
conversion (including no float64 promotion).

## Accepted input shape

Pass either `RuntimeKornerEvidence` or a mapping with these explicit fields:

- `event_id`, `lattice`, `conversions`
- `donor_ownership`, `receiver_ownership`
- `population_transfer` containing all of
  `actual_f_population_transfer`, `source_cells`, `destination_cells`, and
  `replay_reference`
- `roundoff`

Nested conversion and ownership entries are mappings.  Cells are three integer
coordinates; state values are contract values such as `I` and `G`.

## Fail-closed semantics

Only `population_transfer` is allowed to produce
`PopulationTransferEvidence`.  Fields such as `fill`, `mass`, `flags`, or a
transfer intention are intentionally ignored: they cannot imply an actual
f-population source, destination, or replay payload.  If any actual-population
field is absent or malformed, the constructed `TransactionInput` contains no
valid transfer evidence, and the contract returns
`WITHHELD_NO_POPULATION_TRANSFER` once the other I→G prerequisites are
complete.

Malformed input never escapes as an observer exception; it returns a withheld
contract report.  A fully explicit synthetic map can return
`DIAGNOSTIC_ACCEPTED_PHYSICAL_WITHHELD`; this means only detached diagnostic
completeness.  `physical_accepted` remains `False` unconditionally.
