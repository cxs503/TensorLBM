# Production Field-Dataset Campaign Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Move from the verified 2×2 CPU smoke chain to a reproducible, evidence-gated multi-case field-data campaign without overstating model or CFD physical capability.

**Architecture:** Keep the LBM hot path unchanged. Add cold-path campaign tooling that produces real `FieldDataProductR2` field bytes and a `FieldDatasetR2` manifest from a fixed, explicitly supported Torch D3Q19/MRT full-wet execution path. Reuse the existing dataset materializer, trainer, and masked-holdout evaluator; do not introduce generic backend dispatch into timestep code.

**Tech Stack:** Python 3.11, PyTorch CPU float32, NumPy `.npy`, TensorLBM Runtime/Data/ML contracts, pytest.

---

## Current audited baseline

- Branch / commit: `dev/main` at `d05a626416da367c2d0b24bb74ca9e486e075070`.
- Current validated ML path:
  ```text
  verified FieldDataProductR2 bytes
  → FieldDatasetR2 split/group gate
  → CPU Torch materialization
  → train-only flow-transformer smoke training
  → val/test data-only holdout metrics
  → evidence-gated masked-token model holdout metrics
  ```
- Current model result is only a smoke-trained masked velocity-token reconstruction measurement. It is not a CFD physical validation, time prediction, resistance prediction, free-surface validation, GPU/SDAA result, or production model.
- Existing `torch_dataset_flow_training.py` provenance is the source of model/dataset binding. The formal model evaluator consumes weights as verified NPZ bytes; it must remain path-TOCTOU safe.

## Narrow next unlock

A reproducible small **full-wet static-voxel** field-data campaign with multiple source cases/trajectories, group-exclusive train/val/test assignment, actual byte/evidence verification, actual CPU training, and independently evaluated masked holdout metrics.

Excluded claims:

```text
free-surface closure
physical drag/resistance accuracy
CFD temporal forecasting
turbulence-model accuracy
GPU/SDAA performance
Paddle/MindSpore implementation
production model readiness
```

## Proposed scope

Use only the existing supported execution envelope:

```text
Torch / CPU / float32 / D3Q19 / MRT
single-phase incompressible full-wet static voxel
no forcing, no turbulence, no free surface
```

A campaign should have at least 4 independently identified source groups, with two or more snapshots in the train split and nonempty val/test splits. Make group/case/trajectory keys explicit; do not derive them from an arbitrary output filename.

---

### Task 1: Define a pure full-wet field snapshot extraction contract

**Objective:** Produce a validated `(ny, nx, 2)` velocity field only from the supported full-wet solver state, outside the timestep hot path.

**Files:**
- Create: `src/tensorlbm/data/full_wet_snapshot.py`
- Create: `tests/data/test_full_wet_snapshot.py`
- Inspect: `src/tensorlbm/solver3d.py`, `src/tensorlbm/applications/full_wet.py` (or current full-wet application boundary)

**Step 1: Write failing tests**

Cover:

```python
def test_extracts_c_contiguous_little_endian_float32_ux_uy_snapshot(): ...
def test_rejects_unsupported_lattice_collision_or_device(): ...
def test_extraction_is_cold_path_and_does_not_advance_solver_state(): ...
def test_rejects_nonfinite_velocity_or_invalid_plane_selection(): ...
```

Use a tiny known equilibrium/state fixture and assert component ordering is exactly `(u_x, u_y)`, not tensor-axis order.

**Step 2: Run RED**

```bash
PYTHONPATH=src pytest -q tests/data/test_full_wet_snapshot.py
```

Expected: import/contract failure before implementation.

**Step 3: Implement the smallest pure extractor**

- Accept an already-produced full-wet state and an explicit plane/slice descriptor.
- Derive velocity using the existing solver/macroscopic helper only; do not recreate collision or streaming logic.
- Return a detached NumPy `float32`, C-contiguous `(ny, nx, 2)` array.
- Reject unsupported model metadata, CUDA/SDAA tensors, nonfinite values, ambiguous axes, and state shapes that cannot yield an unambiguous plane.

**Step 4: Run focused GREEN**

```bash
PYTHONPATH=src pytest -q tests/data/test_full_wet_snapshot.py
```

**Step 5: Commit**

```bash
git add src/tensorlbm/data/full_wet_snapshot.py tests/data/test_full_wet_snapshot.py
git commit -m "feat(data): add full-wet field snapshot extraction"
```

---

### Task 2: Add a deterministic full-wet campaign runner that writes evidence-bound field products

**Objective:** Turn declared small full-wet case specifications into actual `.npy` field bytes plus `FieldDataProductR2` objects.

**Files:**
- Create: `src/tensorlbm/data/full_wet_campaign.py`
- Create: `tests/data/test_full_wet_campaign.py`
- Reuse: `src/tensorlbm/data/field_r2.py`, `src/tensorlbm/runtime/evidence.py`, `src/tensorlbm/applications/full_wet.py`

**Step 1: Write failing tests**

Cover one small deterministic case first:

```python
def test_campaign_writes_real_npy_then_constructs_field_product_from_same_bytes(tmp_path): ...
def test_manifest_hash_and_size_match_the_written_npy_bytes(tmp_path): ...
def test_runtime_evidence_failure_prevents_field_product_publication(tmp_path): ...
def test_second_run_with_same_case_id_does_not_silently_overwrite(tmp_path): ...
def test_campaign_rejects_free_surface_turbulence_forcing_or_non_cpu_request(tmp_path): ...
```

**Step 2: Run RED**

```bash
PYTHONPATH=src pytest -q tests/data/test_full_wet_campaign.py
```

**Step 3: Implement**

- Add a frozen `FullWetCampaignCase` with explicit identifiers:
  ```text
  case_id, group_id, source_case_id, source_trajectory_id,
  lattice, collision, dtype, device, grid, steps, sampling_step, plane
  ```
- Enforce the supported envelope before allocation/run.
- Run the existing prebound full-wet execution plan; retain no new logic in its timestep loop.
- Persist `.npy` atomically to a supplied controlled output directory.
- Construct `BlobRef`, `ArrayManifestR2`, `FieldDataProductR2`, and required Runtime evidence from the exact written bytes.
- Make output paths new-only and delete partial files on run/evidence failure.

**Step 4: Run focused GREEN**

```bash
PYTHONPATH=src pytest -q tests/data/test_full_wet_campaign.py tests/data/test_field_r2.py tests/runtime/test_evidence_contract.py
```

**Step 5: Commit**

```bash
git add src/tensorlbm/data/full_wet_campaign.py tests/data/test_full_wet_campaign.py
git commit -m "feat(data): add evidence-gated full-wet field campaign"
```

---

### Task 3: Materialize a checked-in-small or reproducible four-group campaign fixture

**Objective:** Prove that multiple real field products can be assembled into a group-safe dataset without synthetic data masquerading as CFD output.

**Files:**
- Create: `tests/data/test_full_wet_campaign_dataset.py`
- Modify: `src/tensorlbm/data/full_wet_campaign.py` only if Task 2 cannot express multiple cases cleanly
- Do not commit generated binary training artifacts unless repository policy explicitly permits it.

**Step 1: Write failing test**

Define four named cases with distinct group/case/trajectory identities, for example:

```text
train-a / group-a / trajectory-a
train-b / group-b / trajectory-b
val-c   / group-c / trajectory-c
test-d  / group-d / trajectory-d
```

Test:

```python
def test_four_real_campaign_outputs_build_group_safe_field_dataset(tmp_path): ...
def test_same_group_split_across_train_and_test_is_rejected(tmp_path): ...
def test_campaign_dataset_fingerprint_changes_when_one_real_blob_changes(tmp_path): ...
```

**Step 2: Run RED**

```bash
PYTHONPATH=src pytest -q tests/data/test_full_wet_campaign_dataset.py
```

**Step 3: Implement only fixture/orchestration support**

- Generate actual solver-derived data in `tmp_path` inside tests.
- Construct `FieldSampleRefR2` and `FieldDatasetR2` using `FieldDataProductR2` outputs from Task 2.
- Preserve exact split declarations and assert group/case/trajectory exclusivity.

**Step 4: Run focused GREEN**

```bash
PYTHONPATH=src pytest -q tests/data/test_full_wet_campaign_dataset.py tests/data/test_field_dataset_r2.py
```

**Step 5: Commit**

```bash
git add tests/data/test_full_wet_campaign_dataset.py
git commit -m "test(data): cover real multi-group full-wet campaign dataset"
```

---

### Task 4: Run the existing multi-snapshot train-only orchestrator on real campaign products

**Objective:** Connect actual campaign-generated fields to the existing CPU smoke trainer; do not create another trainer.

**Files:**
- Create: `tests/ml/test_full_wet_campaign_training.py`
- Reuse unchanged: `src/tensorlbm/ml/torch_dataset_flow_training.py`, `src/tensorlbm/ml/torch_dataset_materialize.py`

**Step 1: Write failing integration test**

Test the full chain:

```python
def test_real_full_wet_campaign_trains_on_train_split_only_and_writes_bound_provenance(tmp_path): ...
```

Assertions:

- four real campaign products exist;
- only two train snapshots enter the trainer;
- val/test blobs are evidence-validated but are not trainer inputs;
- emitted provenance matches campaign dataset fingerprint, split IDs, identities and blob hashes;
- artifact status remains smoke-only.

**Step 2: Run RED**

```bash
PYTHONPATH=src pytest -q tests/ml/test_full_wet_campaign_training.py
```

**Step 3: Add no new training feature unless a narrow adapter is needed**

If needed, add only a pure helper under `tests/` or a cold-path campaign-to-dataset adapter. Do not alter `train_flow_transformer_self_supervised`, the execution plan step, collision, or stream loop.

**Step 4: Run GREEN**

```bash
PYTHONPATH=src pytest -q tests/ml/test_full_wet_campaign_training.py tests/ml/test_torch_dataset_flow_training.py
```

**Step 5: Commit**

```bash
git add tests/ml/test_full_wet_campaign_training.py
git commit -m "test(ml): train smoke model from real full-wet campaign data"
```

---

### Task 5: Evaluate the resulting model on real campaign holdout fields

**Objective:** Use the existing model evaluator to report only masked holdout reconstruction metrics on the real campaign `test` split.

**Files:**
- Create: `tests/ml/test_full_wet_campaign_model_holdout.py`
- Reuse unchanged: `src/tensorlbm/ml/torch_flow_transformer_holdout_evaluation.py`

**Step 1: Write failing integration test**

```python
def test_real_full_wet_campaign_test_split_has_verified_masked_holdout_record(tmp_path): ...
```

Require:

- actual campaign train → weights/metadata/provenance;
- `split="test"` only;
- deterministic, nonempty mask;
- record hashes bind to the same bytes consumed;
- test product identity/blob hash appears in record;
- record is explicitly smoke-trained and `physical_truth_evaluation=False`.

**Step 2: Run RED**

```bash
PYTHONPATH=src pytest -q tests/ml/test_full_wet_campaign_model_holdout.py
```

**Step 3: Minimal integration implementation**

Prefer no production change. If a reusable cold-path factory is genuinely needed, create it under `src/tensorlbm/data/` with an explicit contract and focused unit test.

**Step 4: Run GREEN**

```bash
PYTHONPATH=src pytest -q tests/ml/test_full_wet_campaign_model_holdout.py tests/ml/test_torch_flow_transformer_holdout_evaluation.py
```

**Step 5: Commit**

```bash
git add tests/ml/test_full_wet_campaign_model_holdout.py
git commit -m "test(ml): evaluate real full-wet campaign holdout"
```

---

### Task 6: Add a narrow campaign report generator, not a production claim generator

**Objective:** Emit a human-readable JSON report whose wording cannot be mistaken for CFD validation.

**Files:**
- Create: `src/tensorlbm/ml/campaign_report.py`
- Create: `tests/ml/test_campaign_report.py`

**Step 1: Write failing test**

```python
def test_report_carries_dataset_artifact_and_masked_holdout_evidence(): ...
def test_report_labels_smoke_and_nonphysical_claim_boundaries(): ...
def test_report_refuses_records_with_mismatched_dataset_fingerprint(): ...
```

**Step 2: Implement**

Report must include exact hashes, split, model metric semantics, and an explicit claims block:

```json
{
  "validated": ["CPU Torch masked holdout reconstruction on declared full-wet field products"],
  "not_validated": ["CFD physical accuracy", "free-surface", "resistance", "temporal forecasting", "GPU/SDAA"]
}
```

Do not report percentage accuracy, drag coefficient error, or physical-validation PASS.

**Step 3: Validate**

```bash
PYTHONPATH=src pytest -q tests/ml/test_campaign_report.py
```

**Step 4: Commit**

```bash
git add src/tensorlbm/ml/campaign_report.py tests/ml/test_campaign_report.py
git commit -m "feat(ml): add bounded field campaign report"
```

---

## Final verification sequence

Run from a fresh detached worktree at the latest `dev/main` after each candidate has passed independent clean-overlay review:

```bash
PYTHONPATH=src pytest -q \
  tests/data/test_full_wet_snapshot.py \
  tests/data/test_full_wet_campaign.py \
  tests/data/test_full_wet_campaign_dataset.py \
  tests/data/test_field_r2.py \
  tests/data/test_field_dataset_r2.py \
  tests/runtime/test_evidence_contract.py \
  tests/ml/test_full_wet_campaign_training.py \
  tests/ml/test_full_wet_campaign_model_holdout.py \
  tests/ml/test_torch_dataset_flow_training.py \
  tests/ml/test_torch_flow_transformer_holdout_evaluation.py
PYTHONPATH=src python -m compileall -q src/tensorlbm/data src/tensorlbm/ml
git diff --check
```

For every writer candidate:

```text
parent admission on exact worktree
→ independent detached clean-overlay review with source/test SHA capture
→ fresh latest-main replay
→ focused tests + compileall + diff check
→ exact-path commit
```

## Risks and decisions

1. **Snapshot semantics:** A 2-D plane extracted from a 3-D full-wet state is a data representation decision, not evidence of a physically complete 3-D surrogate. The plane/axis convention must be recorded in field metadata/lineage.
2. **Case diversity:** Different initialization parameters or voxel geometries must be real declared solver inputs. Do not call byte-noised copies different CFD cases.
3. **Runtime cost:** Keep CI fixtures very small and deterministic. A larger campaign runner can be a separately launched artifact-producing job only after the tiny campaign contract is validated.
4. **Physical validation:** This campaign does not resolve free-surface topology/mass blockers or SUBOFF control-volume closure. Keep those as separate numerical workstreams.
5. **Full test collection:** If existing unrelated collection errors occur, preserve their evidence and report focused results honestly; do not hide them with broad ignores.
