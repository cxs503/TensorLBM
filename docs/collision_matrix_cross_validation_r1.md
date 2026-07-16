# Collision-matrix cross-validation R1

`tensorlbm.collision_matrix_cross_validation` is a diagnostic-only, deterministic
runner for capability-matrix evidence. It has no collision hot-path changes.

## Scope

The runner traverses the audited six-cell matrix (D3Q19/D3Q27 × MRT/CM/KBC):

- executes three small CPU/`torch.float32` probes only for cells marked
  `AVAILABLE` by `advanced_collision_contract`;
- emits `SKIPPED_WITHHELD`, with the exact `WITHHELD_*` status, for CM and KBC
  cells that the public contract withholds;
- never changes an explicitly withheld cell into a numerical failure.

For each executable MRT cell it records:

1. equilibrium fixed-point residual;
2. recovered mass and momentum invariant residual after collision;
3. finite post-collision values for a deterministic non-equilibrium input.

These are bounded implementation-consistency probes, not an accuracy,
physics-validation, stability, or ranking assertion.

## Evidence format and provenance

`CollisionMatrixCrossValidationEvidence` and its nested frozen result dataclasses
are directly consumable by a future capability-matrix aggregator. The JSON
writer adds `canonical_payload_sha256`, computed from the canonical JSON payload
before adding that field. Executed cells record the contract entrypoint and SHA-256
of every exact Python source file invoked by the probe. The evidence artifact is
therefore independently traceable to exact source bytes.

Generate the tracked R1 artifact from the repository root:

```bash
PYTHONPATH=src python -c 'from tensorlbm.collision_matrix_cross_validation import write_collision_matrix_evidence; write_collision_matrix_evidence("docs/evidence/collision-matrix-cross-validation-r1.json")'
```

The runner uses fixed scalar fields, fixed perturbation coefficients, CPU, and
no random generator. Re-running unchanged source in the stated environment
produces equal result dataclasses and canonical payload hash.
