# Cylinder-flow AI dataset: first reproducible stage

This tutorial establishes a real-data contract between the TensorLBM cylinder-flow
solver and downstream AI training. It does not claim that a surrogate has been
trained or validated; that requires running the solver and training/evaluation
jobs on a compute environment.

## 1. Generate a real trajectory

Install TensorLBM in the target environment and run the existing solver example:

```bash
pip install -e .
python examples/cylinder_flow.py \
  --nx 160 --ny 64 --u-in 0.06 --re 20 --radius 6 \
  --n-steps 400 --output-interval 20 \
  --output-root outputs/cylinder_ai --run-name re20_smoke --overwrite
```

This is a short smoke run for checking output generation, not a claim of a
converged benchmark. For a publishable Re=20 validation, follow the established
[cylinder Re=20 benchmark tutorial](cylinder_re20_st.md), run its full protocol,
and preserve its parameter and validation evidence. For unsteady wake learning,
use a separate Re=100 trajectory long enough to observe multiple shedding cycles;
do not treat the short smoke run as adequate training data.

## 2. Output contract

At each diagnostic/output step, the run directory contains both the existing
visualization and a numerical archive:

- `flow_step_XXXXXX.png`: speed and vorticity visualization.
- `fields_step_XXXXXX.npz`: numerical fields, loadable with `numpy.load(..., allow_pickle=False)`.
- `forces.csv`: sampled drag/lift coefficients.
- `run_metadata.json`: solver configuration, derived parameters, runtime and diagnostics.
- solver checkpoints, as already produced by the solver.

Each NPZ archive has these arrays:

| Key | Shape | Meaning |
| --- | --- | --- |
| `step` | scalar | Integer solver step |
| `ux`, `uy` | `(ny, nx)` | Velocity components in lattice units |
| `rho` | `(ny, nx)` | Density |
| `speed` | `(ny, nx)` | Velocity magnitude |
| `vorticity` | `(ny, nx)` | Existing grid-difference vorticity convention |
| `solid_mask` | `(ny, nx)` | Boolean cylinder mask; true means solid |

The arrays are numeric and stored without pickle. Do not label density as pressure:
pressure recovery requires an explicitly documented equation of state and
validity assumptions. The present output contract does not add a pressure field.

## 3. Initial quality checks

Before training, reject a run if any of these checks fail:

1. Every NPZ archive opens with `allow_pickle=False`.
2. All field arrays have the configured `(ny, nx)` shape and expected dtype.
3. `ux`, `uy`, `rho`, `speed`, and `vorticity` contain only finite values.
4. Snapshot steps are strictly increasing and agree with the configured output cadence,
   allowing the final step when it is not a cadence multiple.
5. The solid mask is identical across snapshots of a fixed-geometry trajectory.
6. `forces.csv` and metadata exist, and diagnostic values are finite.
7. The run configuration and code revision are retained with the dataset manifest.

## 4. Training split rules

Treat a complete solver trajectory as the split unit. Never randomly split adjacent
frames from one trajectory across train and test: that leaks nearly identical states.
For generalization claims, reserve complete runs and preferably unseen Reynolds
numbers or geometries for validation/test. Record run IDs in a versioned manifest.

Normalize velocity consistently (for example by inlet speed), retain the scale in
the manifest, and exclude solid cells from field losses or handle them through an
explicit mask channel. The current masked-reconstruction transformer tokenizes each
grid cell and defaults to 4096 tokens, so a 160 x 64 grid cannot be passed to that
architecture unchanged. Use a smaller grid for the first model smoke test or implement
patching/coarsening with a documented error study before scaling up.

## 5. What this stage does not establish

Generating snapshots is only the data-ingestion stage. It does not establish:
- physical validity of a particular simulation;
- adequate duration or convergence of a wake trajectory;
- model forecast accuracy;
- generalization to unseen Reynolds numbers;
- faster or cheaper end-to-end inference than TensorLBM.

The next stage should implement a trajectory-aware dataset loader and baseline
forecast task, then report relative field error, vorticity error, force-history
error where applicable, physical diagnostics, wall-clock cost, and an explicit
comparison against persistence or another simple baseline.

## 6. Stage 2 implementation status

Stage 2 and Stage 3 are now implemented and benchmark-validated on CPU:

- **Quality gates, spin-up trimming (`trim_trajectory_spinup`), multi-frame/multi-step sliding windows, trajectory-level split manifest, and R2 catalog bridge**:
  `src/tensorlbm/ai/cylinder_multimodal_data.py`
- **Patchified multimodal flow-force transformer (`history_steps` temporal diffs + `pushforward_steps` unrolled autoregressive training) & persistence evaluator**:
  `src/tensorlbm/ai/cylinder_multimodal_model.py`
- **Runnable multi-Re & variable-radius CLI**:
  `examples/cylinder_multimodal_train.py` (`--min-step`, `--history-steps`, `--pushforward-steps`, `--train-radii`, `--val-radii`, `--test-radii`)
- **Architecture & Stage-2/Stage-3 benchmark specification**:
  `docs/plans/multimodal-cylinder-training-spec.md`

