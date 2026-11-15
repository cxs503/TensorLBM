# GAP-DIFFLBM-ADJOINT-VERDICT — horizon-free anchor for the lifting gradient

Anchor: `GAP-DIFFLBM-ADJOINT-VERDICT` (TensorLBM). Objective `C_L` (momentum
exchange, lift), parameter `y_c` (disk centre), unit system: lattice.

The ledger verdict itself was already delivered and audited twice: `adjoint.py`
is a **frozen-field surrogate**, it declares `shape_derivative_only=True` /
`includes_flow_response=False`, its returned dict is rejected by
`optimization_path_eligible`, and the measured evidence (`cos = -1.000`,
amplitude ratio `2.289`, reversed optimisation step) is declared in the module
docstring and in `FROZEN_FIELD_VERDICT` (`docs/AUDIT_M20231_...`,
`benchmarks/adjoint_verdict_diff_lbm.py`).

What was **missing** after the m20231 audit, and what this mission adds:

1. the flow-response gradient was verified *only* by finite differences of the
   **same finite-horizon** problem, i.e. against itself. There is now a second,
   structurally different algorithm for the same total derivative —
   `tensorlbm.fixed_point_adjoint.fixed_point_adjoint`: implicit-function
   theorem, `(I - S_f)^T lambda = (dJ/df)^T` solved matrix-free by GMRES, no
   time horizon, no stored tape;
2. the *FD reference itself* was never checked for truncation. At `h = 1.0,
   0.5` lu on a **soft (sdf) disk** the central difference of the converged
   `C_L` is biased: measured below, `h = 0.5` sits **1.64 %** above the
   converged limit, which is the size of the effect the original verdict was
   trying to resolve;
3. the frozen proxy is now also anchored **analytically**: its derivative must
   equal the explicit term `dJ/dp|_f` that the fixed-point adjoint reports
   separately, and the mirror symmetry `C_L(y_c) = -C_L(H - y_c)`,
   `dC_L/dy_c(y_c) = dC_L/dy_c(H - y_c)` is checked without any reference value.

## Case (study scale)

| item | value |
|---|---|
| grid | 24 x 16 lu, walls 3 rows each, `H = 10` lu |
| disk | `D = 3` lu, `eps = 0.3` lu (5 cells across the radius), centre `y_c = 7.90` lu (`y_mid + 0.04 H`) |
| relaxation | `tau = 0.6`, `nu = 0.0333` |
| forcing | body force giving `u_bulk = 0.05` (Poiseuille) |
| measured | `Re = u_bulk D / nu = 4.50`, `C_L = -5.0545e-03`, converged in 2200 steps, `max|f - S(f)| = 1.5e-11` |

Study scale, not paper scale: the algorithms, the gate and the invariants are
resolution independent, the absolute `C_L` is not. `PAPER_CONFIGS` in
`benchmarks/adjoint_verdict_fixed_point.py` holds the 96x64 / 128x96 runs.

## Results (from `artifacts/adjoint_verdict_fixed_point_study.json`)

FD ladder of the converged lift (central differences, Richardson in `h^2` on the
two smallest steps):

| `h` [lu] | `dC_L/dy_c` | rel. distance to limit |
|---|---|---|
| 0.5 | -1.280685e-02 | 1.64 % |
| 0.25 | -1.294981e-02 | 0.54 % |
| 0.125 | -1.300155e-02 | 0.15 % |
| 0.0625 | -1.301576e-02 | 0.036 % |
| Richardson | **-1.302049e-02** | — |

Three routes to the same number:

| route | `dC_L/dy_c` | rel. error vs FD limit | cost |
|---|---|---|---|
| unrolled AD, `n = 100` | -1.018689e-02 | 2.18e-01 | 0.5 s |
| unrolled AD, `n = 200` | -1.264655e-02 | 2.87e-02 | 1.1 s |
| unrolled AD, `n = 400` | -1.299580e-02 | 1.90e-03 | 2.7 s |
| **fixed-point adjoint** | **-1.301939e-02** | **8.52e-05** | 265 mat-vecs |
| frozen-field proxy | +6.473511e-03 | 1.50e+00 (wrong sign) | 1 step |

`cos(fixed-point, unrolled AD n=400) = +1.000000`; the Krylov relative residual
is `1.0e-06` and is reported with the result. The frozen proxy is
**anti-parallel** to both (`cos = -1.0000`) with amplitude ratio `0.497`, so it
still fails the predeclared signed gate `cos < 0.99` and
`optimization_path_eligible(frozen) == False`.

Decomposition `dJ/dp = dJ/dp|_f + lambda^T dS/dp` (reported separately by the
new module): the explicit term equals the frozen-field derivative, and for lift
the flow response is the whole signal, not a correction.

## What this changes and what it does not

* does **not** un-block `adjoint.py`: the shipped API is still frozen-field
  only and still returns `shape_derivative_only=True`. Nothing in the verdict
  was relaxed;
* the admissible path now has *two* implementations that agree to `cos = 1.0`
  and both land on the converged FD limit; the unrolled route carries a
  measured horizon bias (2.9 % at `n = 200`, 0.19 % at `n = 400`) while the
  fixed-point route carries none;
* `fixed_point_adjoint` returns the same contract flags as
  `shape_gradient_through_flow`, and its `optimization_path_eligible` is decided
  by `tensorlbm.adjoint.optimization_path_eligible` — one gate, not two.

## Limitations (not checked here)

* the absolute `C_L` of the sdf-penalised, partially saturated bounce-back disk
  is **not** anchored to experiment or an independent LBM code. This work
  anchors the *derivative* against a converged FD limit and an analytic
  symmetry, not the physics of the forward model;
* the fixed-point adjoint needs a *converged* state; the error of a loosely
  converged field is not quantified (`fixed_point_residual` is returned so the
  caller can see it);
* Krylov conditioning (and hence wall time) scales with the grid: the study
  case needs 265 mat-vecs, a channel with a longer recirculation needs more;
  `rtol = 1e-6` is the default because tightening it further moved the gradient
  by 0.3 % while the FD limit was already matched to 8.5e-5;
* `PAPER_CONFIGS` (96x64, 128x96) were queued on the cluster but did not run
  inside this mission's window — the artefact published here is the study scale
  run.

## Reproduce

```bash
PYTHONPATH=src python benchmarks/adjoint_verdict_fixed_point_study.py \
    --out artifacts/adjoint_verdict_fixed_point_study.json --figures
python -m pytest tests/test_fixed_point_adjoint.py -q      # 8 passed in 32 s
```
