# SUBOFF real-state link-wise force observer R1

`tensorlbm.suboff_real_state_force` supplies a minimal observer for a caller's
actual D3Q19 population snapshots.  It is an observer only: it neither creates
an equilibrium population nor runs collision, streaming, boundaries, obstacle
updates, or population resets.

## API

```python
observe_suboff_real_state_force_window(asset, states, config=...)
```

- `asset` is a `GeometryAsset` with a static `(z, y, x)` solid mask.
- `states` is a non-empty caller-supplied sequence of floating, finite tensors,
  each exactly shaped `(19, z, y, x)` and resident on the asset's device.
- The returned `SuboffRealStateForceWindow` contains a `ForceObservation`, a
  `ResistanceForceContract`, one force for each state in the window, and their
  arithmetic mean.

Wall links are compiled from the asset.  For each solid-to-fluid link direction
`q` and fluid neighbor `(z, y, x)`, the observer reads the actual population
`f[opposite[q], z, y, x]` and adds the stationary bounce-back contribution
`-2 f c_q` to the body force.

## Provenance and scope

The observation has `sample_phase="post_stream_pre_bounce_back"`, force sign
`force_on="body"`, and explicit wall-link ownership.  A complete link ledger
allows the resistance contract to be classified as `measured_candidate`; its
`validated` field is always false.  This classification does **not** make a
physical accuracy, reference agreement, convergence, or control-volume closure
claim.

The N states are supplied by the caller, so different state windows yield
independent observations.  The observer never treats the older synthetic
`suboff_full_wet_runtime` path as a physical source.
