# CH collision → adapter-stream phase inventory and boundary stream-flux diagnostic R1

## Scope

`phase_inventory_flux` is a **cold-path, diagnostic-only** ledger over the
actual `FreeEnergyAdapterStreamLoopResult.step_states` produced by the existing
CH collision → adapter-stream loop.  It does not alter production collision or
streaming code and does not introduce pressure, wetting, or a physical flux
model.

## API

`diagnose_adapter_stream_phase_inventory_flux(result)` returns
`PhaseInventoryFluxDiagnostic` with:

* one explicit per-step `phi_integral`, `g_sum`, and changes from the preceding
  returned state;
* adapter-stream `g` boundary crossing terms (`outgoing`, `incoming`, `net`);
* `status="diagnostic_only"`, `physical=False`, and both
  `physical_phase_flux` and `collision_contribution` withheld as `None`.

The loop additionally records these boundary terms in each post-stream loop
diagnostic, derived from the real post-collision `g` passed to the adapter.

## Boundary semantics and limits

* **periodic:** outgoing exterior D3Q19 links are paired with periodic re-entry.
  The reported *net adapter-stream transfer* is structurally zero.  This says
  nothing about collision contribution or total phase conservation.
* **no_flux:** adapter link reflection has structurally zero boundary crossing.
  It likewise does not define collision contribution or physical phase flux.

No result from this R1 diagnostic is a continuum/physical phase-flux claim.

## Verification

`tests/test_phasefield_phase_inventory_flux.py` uses an identity collision
sentinel so that the periodic and no-flux ledgers can be verified against actual
loop `step_states`, including periodic transfer and reflected no-flux links.
