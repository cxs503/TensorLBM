# SUBOFF full-wet production force-window adapter R1

`tensorlbm.suboff_full_wet_production_window` is a fail-closed adapter from the
public `run_fully_wetted_flow` API to
`observe_suboff_real_state_force_window`.

## Audited public-result boundary

The R1 `FullyWettedFlowResult` public fields are `density`, `velocity`, `force`,
`reaction`, `moment`, `status`, `evidence`, and opt-in
`population_snapshots`.  By default, `population_snapshots == ()` and the
adapter remains fail-closed; it never derives `f` from density, velocity, or a
force diagnostic.

## Opt-in production population export

Set `FullyWettedFlowConfig.capture_population_steps` to a tuple of unique,
ascending one-based solver step indices, for example `(10, 20)`.  This is
explicitly opt-in, so default runs retain no population-window memory.  Each
`D3Q19PopulationSnapshot` contains its step index, the documented
`post_stream_pre_bounce_back` phase, an ownership hash of the immutable voxel
geometry snapshot, and a detached `(19, z, y, x)` float32 clone of the actual
production state immediately after `plan.step` and before the retained channel
boundary update.  No population is reconstructed or synthesized.

Snapshot records are frozen and their public `f` property returns a detached
clone, so consumer-side in-place mutation cannot alter the result record.

Accordingly, a default tiny production run is deliberately returned as:

```text
status = window_status = WITHHELD_NO_POPULATION_STATE
force_window = null
```

The artifact retains public runner status, force diagnostic metadata, evidence,
geometry identity, a reasoned provenance record, and a SHA-256
`provenance_hash`.  No full-wet, boundary, solver, or obstacle hot-path code is
changed.

## Future-compatible measured path

The adapter only invokes `observe_suboff_real_state_force_window` for a public
result with a nonempty `population_states: Sequence[torch.Tensor]` view.  The
full-wet R1 result supplies that view only from its opt-in snapshots.  Those
tensors are passed directly to the observer; they are neither synthesized nor
reconstructed.  A successful observer result remains `measured_candidate` and
always has `physical_validation: false`.

The injectable `runner` argument exists only to contract-test a public result
interface.  It is not a synthetic-state production path.

## Minimal use

```python
artifact = run_suboff_full_wet_production_window(asset, config)
assert artifact["window_status"] == "WITHHELD_NO_POPULATION_STATE"
assert artifact["force_window"] is None
```
