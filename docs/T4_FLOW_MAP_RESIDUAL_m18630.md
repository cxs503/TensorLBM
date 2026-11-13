# T4 physics-AI loop: residual flow map over the exact viscous prior (m18630, re-open of #2003)

**Repo**: TensorLBM · **Branch**: agent/m18630 · **Anchor**: GAP-LBM-T4-PHYSICS-AI-LOOP
**Feature re-delivered**: `bench_t4_physics_ai_loop.py`, `src/tensorlbm/ai/flow_field_loop.py`,
`tests/test_flow_field_loop.py`, `tools/t4_report_from_results.py`, `tools/t4_send_report.py`
(all already on `main`, delivered by m15932/m15970).  This round audits the *numbers* of that
feature, finds that the AI step loses to its own physics baselines, and fixes the
parameterisation.

## 1 Case and parameters

Decaying multi-mode Taylor-Green vortex in a periodic box, D2Q9 BGK, lattice units, produced by
`tensorlbm.solver` primitives.  Corpus = `bench_t4_medium.py` (n=64, 2000 steps, sample every
200 steps, u0=0.05, Re = u0/(k1 nu) ∈ {50,100,200} train, **75 holdout**, seeds 0/1, 8 runs /
88 snapshots / sha256 manifest).  The analytic single-mode Taylor-Green solution is used **only**
as a ruler inside `verify_producer` (field error 0.296 %/0.310 %, decay-rate 0.124 %/0.712 % →
`verified=True`).

## 2 Method

* API: `FlowMapConfig(residual=True)` (new default), `flow_map_step()` (new single inference
  path), `load_flow_map()` (new; reloads the parameterisation flag with the weights).
* Parameterisation: `u(t+dt) = P_nu[dt] u(t) + FNO(u(t), nu)` where `P_nu[dt]` is the exact
  discrete viscous prior `exp(-nu|k|^2 dt)` in Fourier space.  Direct parameterisation
  `u(t+dt) = FNO(u)` remains available (`residual=False`).
* Why: the cheap priors move the target into the reach of the small training budgets used here
  (target energy `MSE(y-Py)/MSE(y) = 0.0685/0.3354 = 0.20`), and give the surrogate a
  **structural floor**: at `FNO→0` the model *is* the prior.
* Benchmarks: `bench_t4_medium.py` budget raised 12→60 epochs, width 12→16, modes 6→8;
  `bench_t4_physics_ai_loop.py` 12→60 epochs, width 12→20, layers 2→3, modes 6→10.
  `evaluate_holdout`/`_predict_rollout`/`feedback_acceleration` now all call `flow_map_step`, so
  the conditioning channel, residual base and units cannot drift between call sites (the m15970
  unit mismatch had to be fixed in three copies of the same snippet).

## 3 Results (program-injected, no hand-typed numbers)

Holdout = unseen Re=75, one-step rel-L2 `||u_pred-u_LBM||/||u_LBM||`:

| run | BEFORE direct, 12 ep (main) | residual, 120 ep (this work) | direct, 120 ep | viscous prior | persistence |
|---|---|---|---|---|---|
| tgv_n64_re75_s0 | 0.9314 | **0.1424** | 0.1886 | 0.3639 | 0.3857 |
| tgv_n64_re75_s1 | 0.9291 | **0.1453** | 0.1920 | 0.3863 | 0.4084 |
| mean | 0.9303 | **0.1439** | 0.1903 | 0.3751 | 0.3970 |

Rollout (autoregressive, 10 samples from snapshot 0) and handback, BEFORE vs physics baselines:

| metric | direct FNO (main) | viscous prior | persistence |
|---|---|---|---|
| one-step rel-L2 | 0.9303 | 0.3751 | 0.3970 |
| rollout final rel-L2 | 1.0005 | 1.2826 | 1.7308 |
| handback rel-L2 (k=2,4) | 0.997 / 1.000 | — | — |

Training diagnostics that explain the collapse: direct/12 ep ended at val MSE 0.2371, i.e.
between the zero-output predictor (0.3354) and identity (0.0745) — the network had learned to
shrink the field, not to advance it.

## 4 Figures

No figure is shipped with this note: the residual campaign that produced the table was run as a
measurement script against the existing medium dataset, and the shared overlay filesystem hit
100 % full (≈44 KB free) before the matplotlib stage of `bench_t4_medium.py` could be rerun.
`bench_t4_medium.py` still writes fig1–fig4 when it runs; the missing-figure reason is therefore
infrastructure, not absence of data (all numbers above are in the tables).

## 5 Caveats / limitations

* The 120-epoch numbers come from the same medium corpus and the same code path
  (`load_dataset`, `viscous_prior_step`, `FNO2d`, `flow_map_step`), driven by a measurement
  script; the updated `bench_t4_medium.py` (60 ep, width 16) was **not** re-run end-to-end
  because the filesystem filled up.  Treat `0.1439` as a measured prototype value, not as a
  benchmark artifact.
* Rollout/healing for the residual model is not measured yet (only one-step is).  The
  before/after rollout table above is therefore one-sided.
* The surrogate operates on a decaying TGV family at Re ≤ 400, n ≤ 128; nothing here says
  anything about wall-bounded or high-Re flows.
* `predicted` handback errors ≈ 1.0 for the direct model mean "the surrogate state is useless to
  the solver", not "healing failed" — the healing branch requires `heal_steps % dt == 0` and was
  not exercised at k=2/4 in the medium run.

## 6 Error assessment (explicit)

| quantity | reference | error | < 3 %? |
|---|---|---|---|
| producer field (Re=50) | exact NS Taylor-Green | 0.296 % | **yes** |
| producer decay rate (Re=50) | 2 nu k^2 | 0.124 % | **yes** |
| FNO one-step, holdout Re=75 (residual) | pure-LBM field | 14.24 / 14.53 % | **no** |
| FNO one-step, holdout Re=75 (direct, main) | pure-LBM field | 93.03 % | **no** |
| viscous prior one-step (reference floor) | pure-LBM field | 37.51 % | **no** |

**Answer: no key surrogate quantity reaches the 3 % line.**  The surrogate now clears the 15 %
acceptance line (14.24 % / 14.53 %) and beats both physics baselines by 2.6×, which is the
capability claim made here; the 3 % claim is not made and would need a finer dt (more samples
per decay), a bigger budget, and a rollout study.

## 7 Tests

`tests/test_flow_field_loop.py`: 15 passed (11 pre-existing + 4 new), including
`test_residual_flow_map_has_the_viscous_prior_as_a_floor` (untrained residual model ≤ prior ×
1.02, same net in direct mode > 2× worse),
`test_flow_map_step_reduces_to_the_prior_when_the_residual_is_zero`,
`test_train_flow_map_persists_the_parameterisation`,
`test_evaluate_holdout_scores_through_the_single_inference_path`.
