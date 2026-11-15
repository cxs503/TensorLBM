# XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM — status (round 2, mission 21640)

Cross-validation of the **two-component Shan-Chen (MCMP) D2Q9 model with EDM forcing**
on a layered plane-Poiseuille flow, following the *sailfish* `two-component-sc-layered-poiseuille-edm`
recipe: two components, one distribution function each, own relaxation times, cross-species
repulsion, exact-difference forcing, driven by a common body force.

## 1. What round 1 (m21558) got wrong — root cause

The m21594 audit measured `u_peak_err = 177.78 %`, `eps_inf = 253.30 %` and found that the
same solver with `G = 0` and a single component still returned `u_max / (g t) = 1.000`.
That is the signature of **no viscous braking at all**, and the cause is in the old
`stream_bounce_back_y`:

```python
streamed = torch.roll(f[i], shifts=(0, -cx), dims=(0, 1))   # only x was rolled
if cy > 0: streamed[1, :] = f[OPPOSITE[i], 1, :]            # then the wall rows were overwritten
```

the vertical directions were never advected (and the overwrite discarded the wall
values anyway), so momentum could only accumulate: `u = g t`, flat top-hat profile,
zero shear. The interface-position mismatch flagged by the audit (`nus=[ν_A, ν_B]`
mapped onto the wrong layer, 22.36 % self-error of the reference) was a second,
independent bookkeeping defect.

## 2. Implementation (files)

| role | path |
|---|---|
| model / solver | `src/tensorlbm/verification/layered_poiseuille/two_component_sc.py` |
| exact reference | `src/tensorlbm/verification/layered_poiseuille/analytic.py` (`solve_layered`, 2N×2N LU, independent of the solver) |
| metrics | `src/tensorlbm/verification/layered_poiseuille/metrics.py` (`profile_errors`, `layer_fluxes`, `peak_metrics`) |
| **entry point** | `src/tensorlbm/verification/layered_poiseuille/run_two_component_sc_validation.py` (`python -m ...`, case matrix, gates, JSON + PNG) |
| **production kernels (shared, integrated)** | `src/tensorlbm/multiphase/__init__.py`: `sc_two_component_force`, `collide_edm_common` (new) + `psi_exp`, `stream_bounce_back` (existing, reused) |
| tests | `tests/test_two_component_sc_layered_poiseuille.py` (11 physical assertions) |
| run artifact | `artifacts/xval_two_component_sc_layered_poiseuille_edm/{results.json,sha256.txt,two_component_sc_layered.png}` |

The verification solver no longer re-implements the lattice operators: it drives the
**shipped** `tensorlbm.multiphase` kernels (`sc_two_component_force`,
`collide_edm_common`, `psi_exp`, `stream_bounce_back`), so a defect in the production SC
kernel now shows up as a benchmark failure here.

Model settings that matter (all validated below):

* force: `F_σ = -G ψ_σ Σ_i w_i ψ_σ'(x+c_i) c_i`, `G = 2.5 > 0` repulsive, `ψ = ρ0(1-e^{-ρ/ρ0})`, `ρ0 = 1`;
* collision: EDM at the **common, viscosity-weighted** velocity
  (`u* = Σ ρ_σ u_σ/τ_σ / Σ ρ_σ/τ_σ`, Martys & Chen 1996), force shift `F_σ/ρ_σ` per component;
* walls: halfway bounce-back, `wall_mode="mirror"` (neutral wetting). The `"dry"` variant
  (`ψ_wall = 0`) leaves a spurious minority-species adsorption layer within ~3 cells of both
  walls and raises `eps_inf` from 6.8 % to 14.4 % — measured, not assumed;
* the interaction strength is **ramped** from 0 over 2000 steps (`g_ramp_steps`): starting
  the sharp (ρ ≈ 2) demixing transient at full `G` diverges (NaN within 200 steps);
* initial densities are the measured coexistence pair (`ρ_A = 2.12`, `ρ_B = 2.36`), not 2/2.

## 3. Results (`artifacts/.../results.json`, sha256 in `sha256.txt`, H = 32, nx = 1 periodic)

| case | KPI | got | ref | error | owner gate 15 % |
|---|---|---|---|---|---|
| `control_single` (G=0, one component, ν=0.1) | `eps_inf` | — | exact Poiseuille | **0.25 %** | PASS (self target 3 %) |
| `control_mixture` (G=0, two components, equal τ) | `eps_inf` | — | exact Poiseuille | **0.37 %** | PASS |
| `layered_r167` (ν=0.100 bottom / 0.1667 top) | `u_peak_err` | 6.30e-2 | 6.79e-2 | **7.4 %** | PASS |
| | `Q_err` | — | — | **8.2 %** | PASS |
| | `eps_inf` | — | — | **8.8 %** | PASS |
| | `eps_l2_rel` | — | — | **8.2 %** | PASS |
| | `y_iface` | 19.21 | 16.0 (nominal) | 10.0 % of H | **FAIL (5 % H sub-gate)** |
| | `ν_fit` lower/upper | 0.1088 / 0.1757 | 0.1000 / 0.1667 | **+8.8 % / +5.4 %** | PASS |
| `layered_static` (g=0) | spurious / driven peak | 0.0 | — | **0.0 %** | PASS |

**Post-hoc diagnostic (disclosed):** the layer interface is *not* free — each component's mass
is conserved, so the layer thicknesses follow from the coexistence densities and the interface
settles 3.2 cells above the nominal `H/2` (`y_iface = 19.21`). Re-evaluating the exact reference
at the *measured* interface gives `u_peak_err = 9.7 %`, `eps_inf = 10.4 %`, `eps_l2_rel = 9.3 %`,
`Q_err = 8.9 %` (stored as `metrics_iface_matched`). Both comparisons stay inside the 15 %
owner gate; none reaches the 3 % publication target — the deviation is dominated by the
diffuse (≈1 cell) viscosity transition at the interface, which a sharp two-layer reference
cannot represent pointwise.

## 4. Not run / limitations

* `layered_r233` (ν ratio 2.33) and `layered_r167_h48` (H = 48 resolution check) are in
  `CASE_MATRIX["full"]` but were **not run** in this round (local budget). The `full` matrix
  is one command away (`--matrix full`).
* Grid convergence is therefore **not** established — no claim of second-order convergence.
* The interface position is a mass-controlled degree of freedom; the 5 %·H sub-gate is
  *not* met with a nominal-`H/2` reference. Reported as FAIL, not silently dropped.
* Two kpi caveats: `layered_static` has no driven reference so `eps_l2_rel` is `nan`
  (the gate there is the spurious-current ratio); `nx = 1` is admissible because the
  problem is homogeneous in x (periodic), which was checked at `nx = 2` and `nx = 8`.

## 5. Reproduction

```bash
cd TensorLBM
PYTHONPATH=src python -m tensorlbm.verification.layered_poiseuille.run_two_component_sc_validation \
    --matrix core --out artifacts/xval_two_component_sc_layered_poiseuille_edm
PYTHONPATH=src python -m pytest tests/test_two_component_sc_layered_poiseuille.py -q
```

Per case (each < 120 s, deterministic — no random seeds involved):

```bash
PYTHONPATH=src python -m tensorlbm.verification.layered_poiseuille.run_two_component_sc_validation \
    --case layered_r167 --no-figure --out artifacts/xval_two_component_sc_layered_poiseuille_edm
```


## Round 3 (m21732): benchmark layer, real entry, production callers

The round-2 delivery had implementation, tests and a JSON artifact but no
reusable ref-vs-got layer, no single entry the machines could point at, and no
call site outside the validation script itself.  Round 3 adds exactly that:

* `src/tensorlbm/verification/layered_poiseuille/benchmark.py` -- shared scoring
  layer: drives `run_case_spec` and reshapes the result into an explicit
  `ref`/`got`/`err`/`threshold`/`pass` table (`kpi_table`), with CSV/markdown
  writers and a sha256'd JSON payload.  Layer-curvature (`nu_*`) KPIs are
  reported as *skipped with a reason* when `H < 32` (fit window <= 5 points)
  instead of being silently dropped.
* `xval/two_component_sc_layered_poiseuille/run_benchmark.py` -- CLI entry that
  writes `benchmark.json` / `benchmark.csv` / `benchmark.md` /
  `benchmark.sha256` and exits non-zero when a gate fails.
* production callers (grep-able, not tests):
  `evidence/paper_benchmarks/run_all_benchmarks.py` (`BENCHMARKS` entry),
  `src/tensorlbm/verification/xval/cases.py` (`XValCase` in `ALL_CASES`),
  `.github/workflows/multiphase-benchmarks.yml` (CI gate step).
* `xval/.../make_report_artifacts.py` regenerates the tables in
  `docs/XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM-BENCHMARK.md` and the
  figures from the run payloads, so no number is hand-copied.

Round-3 artifact provenance (honest): `benchmark.json` comes from a light-matrix
run (`H = 22`, 3000 steps, 4 cases) in this worktree;
`results.json` records `control_single` / `control_mixture` re-run in this
worktree and the round-2 core records for `layered_r167` / `layered_static`
(identical code path, deterministic run).  The `core`/`full` matrix re-run
(adding `layered_r233` and the `H = 48` resolution check, previously not run)
was submitted to the compute gate as jobs `cj3671` (validation `--matrix full`)
and `cj3672` (benchmark `--matrix core`) and had not been dispatched when this
round closed; `--matrix full` remains one command.
