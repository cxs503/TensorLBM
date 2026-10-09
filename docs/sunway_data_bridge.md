# Sunway Data Bridge (SWLBM → TensorLBM R2 products)

The Sunway line runs SWLBM, an independent C codebase on SW26010 many-core
nodes — it never executes torch. It joins the platform at the **data/contract
layer**: its outputs are converted into the same PASS-gated
`FieldDataProductR2` + `FieldDataCatalog` records that torch solvers produce,
so dataset building, quality gating, lineage, and downstream AI training
treat Sunway runs as first-class citizens.

Reference implementation: `src/tensorlbm/data/sunway_bridge.py`
(schema token `sunway.swlbm.bridge.v1`).

## 1. Observed SWLBM output layout (psn002)

Run directory (real example, cluster psn002):

```
/home/export/online3/swbxyh/hydro/swlbm/suboff_geshan_ok/test/result/
├── force_history_pid6032_t1787010921.csv   # per-run force time series
└── force_history_Q19_BGK_5k.csv            # second run in the same dir
```

Field (lattice) dumps are also produced for post-processing cases; their
capture is pending a live sample and is deliberately left as a stub this wave
(see §5).

## 2. `force_history` CSV contract

Header (exact order, one row per force-sampling phase; sampling is every
`100` lattice steps in observed runs):

| Column | Type | Meaning |
|---|---|---|
| `step` | int ≥ 0 | lattice step of the sample |
| `F_x`, `F_y`, `F_z` | float | hydrodynamic force on the body, lattice units. SWLBM samples in the post-stream / pre-bounce-back phase with the Ladd wet-node rule `F_α = 2·Σ_q c_{q,α}·f[q, x_solid]` |
| `C_D`, `C_L`, `C_S` | float | drag / lateral / side force coefficients (non-dimensionalised by SWLBM's own reference dynamic pressure) |
| `Re` | float | Reynolds number of the run (constant per column) |
| `C_F_ITTC` | float | ITTC-1957 friction coefficient `0.075/(log10(Re)−2)²` used as the reference for skin-friction checks |
| `wet_nodes` | int ≥ 0 | count of wet (fluid-side) boundary nodes — a mesh-resolution fingerprint |

Bridge validation is fail-closed: a missing column, a non-numeric cell, a
negative `step`/`wet_nodes`, or any non-finite float refuses the conversion
(the product must be PASS-gated, mirroring `solver_export`).

## 3. Mapping to `FieldDataProductR2`

`convert_swlbm_csv(catalog, csv_path, metadata, *, blob_root=None) ->
product_id` (`"{run_id}:forces"`):

| R2 array | Role | Shape/dtype | Axes | Units |
|---|---|---|---|---|
| `force_lattice` | FEATURE | `(N, 3)` float32 | `timestep` (SAMPLE) × `component` (`F_x,F_y,F_z`) | `lattice_force` |
| `force_coefficients` | TARGET | `(N, 3)` float32 | `timestep` × `component` (`C_D,C_L,C_S`) | `dimensionless` |
| `wet_nodes` | AUXILIARY | `(N,)` int32 | `timestep` | `1` |

* **Run manifest** — `model_identity = {solver: "SWLBM", case, lattice,
  platform: "sunway"}`; metrics (exact-decimal JSON evidence, verified by
  `verify_metric_evidence`): `rows`, `step_last`, `re`, `cf_ittc`,
  `cd_last`, `cl_last`, `cs_last`, `cd_mean_tail` (mean over the last 10% of
  samples — the statistically-steady drag estimate), `f_x_last`,
  `wet_nodes_last`.
* **Artifacts** — `swlbm:metrics` (application/json evidence) and
  `swlbm:source_csv` (the raw CSV bytes, `text/csv`, sha256-bound); the
  product's `source_artifact_id` is the CSV artifact.
* **Required metadata** — `run_id`, `case`, `code_sha` (40-hex SWLBM source
  revision). Optional: `lattice`, `queue`, `sunway_host`, `re`, `u_in`,
  `nu`, `tau`, …  All scalars become catalog metadata rows
  (`source="sunway_bridge"`), so training-side queries can select by cluster
  queue, host, or physics.
* **Lineage** — `sunway_bridge: {schema, csv_path, csv_sha256, rows,
  columns, module}` plus a catalog `derived_from` record
  `run:{run_id} → {product_id}`.
* **Quality** — finiteness + shape checks over the materialised NPY blobs
  via `check_field_product` (mass-conservation is N/A for force series).
* Blobs land under `blob_root` (default `<csv dir>/blobs/<csv stem>/`),
  verified byte-for-byte with `validate_for_use` **before** catalog
  registration (fail closed, same as solver exports).

`load_product(catalog, product_id)` / `load_product_arrays(...)` work on the
result unchanged (the catalog stores the `product_json` row).

## 4. Execution/transport notes for LSF (learned live on psn002)

* Submit from a working directory **not** on the "fs base" home filesystem
  (e.g. `/home/export/online3/...`); bsub refuses `fs base` CWDs and output
  paths.
* Queue `q_sw_share` (Sunway nodes); `q_x86_share` was closed for submission.
* SWA LSF supports **no** `-W` (walltime) and **no** `-e` (stderr file);
  stderr merges into `-o`.
* Compute nodes have a minimal userland: no `/bin/bash`, use `/bin/sh`
  (`TENSORLBM_HPC_LSF_SHELL=sh`); scripts must stay POSIX.
* The batch script must be visible on execution nodes → stage it under
  `TENSORLBM_HPC_LOG_DIR` pointed at the shared filesystem.
* `bjobs -l` prints `Status<DONE>` etc. — the stable parse across LSF
  flavours (short-format column order differs on SWA).

## 5. Field dumps — specified TODO

`convert_swlbm_field_dump()` currently raises `NotImplementedError` by
design. Planned mapping (implement when a dump sample is captured):

| SWLBM dump | R2 array | Notes |
|---|---|---|
| velocity field `(u, v, w)` per node | `velocity` FEATURE float32, axes spatial × component | axis order must be recorded from the dump header; TensorLBM canonical layout is `(nz, ny, nx)` with streamwise `x` |
| density `rho` | `rho` TARGET float32 | PASS gate reuses the solver_export density-drift check (|⟨rho⟩−1| ≤ mass_tol) |
| solid/geometry mask | `solid_mask` MASK int32 | wet/dry from the same Ladd wet-node set that produces `wet_nodes` |

Interface contract already fixed:
`convert_swlbm_field_dump(catalog, dump_path, metadata, *, blob_root=None)
-> product_id` with `product_id = "{run_id}:step{N:06d}"`.
