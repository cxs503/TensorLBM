# SUBOFF full-wet production force-window adapter R1

`tensorlbm.suboff_full_wet_production_window` is a fail-closed adapter from the
public `run_fully_wetted_flow` API to
`observe_suboff_real_state_force_window`.

## Audited public-result boundary

The R1 `FullyWettedFlowResult` public fields are `density`, `velocity`,
`force`, `reaction`, `moment`, `status`, and `evidence`.  It does **not** expose
D3Q19 `f` populations or a per-step population-state sequence.  In particular,
the final density/velocity fields and same-phase force diagnostic cannot be
used to reconstruct an observed force window without inventing state.

Accordingly, the default tiny production run is deliberately returned as:

```text
status = window_status = WITHHELD_NO_POPULATION_STATE
force_window = null
```

The artifact retains public runner status, force diagnostic metadata, evidence,
geometry identity, a reasoned provenance record, and a SHA-256
`provenance_hash`.  No full-wet, boundary, solver, or obstacle hot-path code is
changed.

## Future-compatible measured path

The adapter only invokes `observe_suboff_real_state_force_window` if a future
**public** full-wet result explicitly exports a nonempty
`population_states: Sequence[torch.Tensor]`.  Those tensors are passed directly
to the observer; they are neither synthesized nor reconstructed.  A successful
observer result remains `measured_candidate` and always has
`physical_validation: false`.

The injectable `runner` argument exists only to contract-test a public result
interface.  It is not a synthetic-state production path.

## Minimal use

```python
artifact = run_suboff_full_wet_production_window(asset, config)
assert artifact["window_status"] == "WITHHELD_NO_POPULATION_STATE"
assert artifact["force_window"] is None
```
