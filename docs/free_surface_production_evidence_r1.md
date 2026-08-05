# Körner production I→G evidence observer — R1

## Audit result

The production path is `free_surface_lbm.free_surface_step`:

1. It identifies `i_to_g = to_gas & interface_mask` only when the opt-in
   ownership closure is enabled.
2. `build_i_to_g_ownership_transaction` emits independent-mass debit/credit
   links, receiver masks, and an exact same-dtype residual.  This is explicitly
   **not** an `f` population transfer.
3. `build_topology_transaction` records sparse conversion evidence when its
   normal diagnostic capture is requested.  Published fields include conversion
   cells, mass/fill snapshots, `f_before`/`f_after` observations, I→G ownership
   links/debit/credit/residual, and
   `i_to_g_population_owner_status="WITHHELD_NO_POPULATION_TRANSFER"`.
4. Replay capture provides transaction/replay tensor payloads, but no published
   source/destination/replay record for an actual transfer of `f` populations.
   The topology code zeroes the converting donor populations and documents that
   it performs **no f transfer**.

The smallest real runnable fixture is a 5×6×7 all-GAS domain with a central
INTERFACE cell and its D3Q19 neighbor shell also INTERFACE, initialized from
`equilibrium3d`.  The R1 test runs the real `free_surface_step` through
`run_free_surface_step_with_observer` using this fixture.

## Adapter contract

`free_surface_production_evidence.py` is cold/additive:

- `run_free_surface_step_with_observer` invokes the actual production step with
  its existing optional `runtime_ledger`, `replay_capture`, and replay-stage
  capture dictionaries.
- `extract_runtime_korner_evidence` lists available result-key paths and keeps
  its provenance (`production_free_surface_step_runtime_ledger` for the real
  wrapper; mappings are explicitly `shaped_result_mapping_not_claimed_production`).
- `observe_korner_runtime_evidence` calls the existing
  `diagnose_korner_i_to_g_transaction` only for a complete future R1
  transaction. It never creates a transfer, owners, source, destination, or
  replay reference from snapshots.

## R1 result

Current production topology output lacks an explicit actual `f`-population
source, destination, and replay reference.  Therefore the observer returns:

`WITHHELD_NO_POPULATION_TRANSFER`

This is a real fail-closed result, not an assertion that no evidence can ever
be added.  `f_before` / `f_after`, ownership mass links, and generic replay
payloads are deliberately insufficient to infer a transfer.
