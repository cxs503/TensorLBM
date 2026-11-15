# xval: two-component Shan-Chen layered Poiseuille (EDM)

Anchor: `XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM`.

This directory holds the **entry point** and **run artifacts** of the anchor.  The
physics lives in
`src/tensorlbm/verification/layered_poiseuille/{two_component_sc,analytic,metrics}.py`;
the reference-vs-computed scoring lives in
`src/tensorlbm/verification/layered_poiseuille/benchmark.py`.

## What is validated

Two-component Shan-Chen (MCMP, one distribution per component, `psi = rho0(1-e^{-rho/rho0})`,
`G = 2.5 > 0` repulsive) in a plane channel with periodic `x` and halfway bounce-back
walls. Each component keeps its own `tau`, so the mixture has a viscosity contrast;
both are driven by the same body-force acceleration. The steady state must be the
exact piecewise-quadratic layered solution

    nu_j u_j'' = -g,  u(0) = u(H) = 0,  u and nu u' continuous at the interface

computed independently by `analytic.solve_layered` (2N x 2N LU solve, no lattice).

## Entry points

```bash
cd TensorLBM

# 1) ref-vs-got benchmark table (benchmark.json / .csv / .md / .sha256)
PYTHONPATH=src python xval/two_component_sc_layered_poiseuille/run_benchmark.py \
    --matrix core --out artifacts/xval_two_component_sc_layered_poiseuille_edm

# 2) pre-registered case matrix incl. profiles, gate verdicts and the figure
PYTHONPATH=src python -m tensorlbm.verification.layered_poiseuille.run_two_component_sc_validation \
    --matrix full --out artifacts/xval_two_component_sc_layered_poiseuille_edm

# 3) tests (build blocks, defect regressions, gates, callers)
PYTHONPATH=src python -m pytest tests/test_two_component_sc_layered_poiseuille.py -q
```

`--matrix light` is the cheap CI matrix (`H = 22`, 3000 steps); `core` (default,
`H = 32`, 12000 steps) and `full` (adds the `H = 48` resolution check) are the
pre-registered matrices. Both scripts exit non-zero if a gate fails.

## Production callers

The same scoring layer is called from two production code paths (not tests):

| caller | call |
|---|---|
| `evidence/paper_benchmarks/run_all_benchmarks.py` | `BENCHMARKS` entry `two_component_sc_layered_poiseuille` → `benchmark_single_case("layered_r167_light")` |
| `src/tensorlbm/verification/xval/cases.py` | `XValCase("two_component_sc_layered_poiseuille")` registered in `ALL_CASES` / `register_default_cases` |

## Gates (pre-registered, owner standard)

* control (one component, `G = 0`): `eps_inf <= 3 %` vs exact single-fluid Poiseuille
  — this is the detector for the defect that sank the first delivery (`u_max = g t`,
  i.e. no viscous braking at all);
* layered: `u_peak_err`, `Q_err`, `eps_l2_rel`, `eps_inf <= 15 %`, interface position
  `<= 5 % H`;
* static (`g = 0`): spurious current `<= 15 %` of the driven peak.

## Artifacts

| file | content |
|---|---|
| `artifacts/.../benchmark.json` | payload: environment, `repro_cmd`, KPI table (`ref`/`got`/`err`/`gate`), per-case profiles |
| `artifacts/.../benchmark.csv` | the KPI table, flat |
| `artifacts/.../benchmark.md` | paper-style markdown table + repro command |
| `artifacts/.../results.json` | pre-registered case matrix: metrics, gates, profiles, reference profiles |
| `artifacts/.../two_component_sc_layered.png` | velocity profiles, demixing profile, pointwise deviation |
| `artifacts/.../sha256.txt`, `benchmark.sha256` | artifact identity |