# GAP-PP-3-3DMCMP-SPURIOUS-SCALING — round 1 status (honest, no production table)

**Verdict of round 1: PARTIAL / honest-FAIL on the acceptance criterion.**
The *instrument* (protocol, runner, statistics, falsification gates) is built,
tested and validated against synthetic ground truth; the **production `u_max`
matrix over ≥3 resolutions × ≥3 density ratios was not completed** because of
control-plane compute budget, not because of a physics or code blocker.

```
fail_stage      : production_sweep_not_completed
failure_reason  : 3-D D3Q19 float64 droplet marches at R = n/8 need ~2 000-5 000
                  steps per case; the shared build box was at load ≈ 270/128
                  (≈3x oversubscribed), measured 1.5-3 s/step at n = 64, i.e.
                  ~1-2 h wall for the 36-case matrix. Two cluster submissions
                  (cj1078, TensorLBM stage_dir = worktree, 16 cores) stayed
                  "queued" (287 jobs ahead) for 9 min and were killed to re-plan.
                  Two background local campaigns were started and killed by the
                  scheduler-side process-group reaper while the protocol was
                  still being finalised (width rule / control design changed
                  mid-flight); the third attempt had <3 min of wall clock left.
tried_paths     : (1) cluster submit (stage_dir) x1 -> queued, no dispatch;
                  (2) local background campaign, 36 cases, 5000 steps, 30 jobs
                      -> 6 min in: 0 cases finished, box oversubscribed;
                  (3) reduced ladder 24/32/40, 15 cases, 1500 steps, 18 jobs
                      -> 2 min in: 0 cases finished at cutoff.
what_is_missing : the u_max table itself -> p, sigma_p, C1/C2/C3 verdict.
next_round      : unchanged protocol in README.md; submit
                  `python benchmarks/mcmp3d_spurious_scaling/run.py --out
                   artifacts/mcmp3d_spurious_scaling --jobs 30 --threads 2
                   --steps 5000 --check-every 250` to the 2xloong64 cluster with
                  a 7200 s timeout (expect ≈ 60-90 min wall) — the harness,
                  the JSON schema, the fit and every gate are already done, so
                  the next round needs *only* machine time, no new method.
```

## What is delivered and verified in this round

| item | state | evidence |
|---|---|---|
| pre-registered protocol (grids, couplings, steady criterion, controls, fit) | **done** | `benchmarks/mcmp3d_spurious_scaling/README.md` |
| campaign runner → single JSON artifact, per-check history, no hand-typed numbers | **done** | `benchmarks/mcmp3d_spurious_scaling/run.py` |
| fits (per-grid OLS, pooled fixed-effects, measured-Δρ robustness, resolution axis), tables, figures, `verdict.json` | **done**, validated on synthetic data with known exponents 1.0/2.0/3.0 | `benchmarks/mcmp3d_spurious_scaling/fit_report.py`, `tests/test_mcmp3d_spurious_scaling.py` |
| falsification gates that can actually fail (Z vanishing intercept, P plateau, M mass budget) + negative tests for each | **done** | 12 tests pass: `pytest tests/test_mcmp3d_spurious_scaling.py -q` |
| coexistence window of the model EOS (Maxwell) | **done, real numbers** | `results` table below |
| production `u_max` matrix | **NOT DONE** | see `fail_stage` above |

## Real measurements obtained while piloting (not the acceptance table)

These are genuine solver runs (D3Q19, float64, τ = 1, `difference` force form),
recorded here because they already settle two design questions:

| run | n | G_eff | Δρ (Maxwell) | form / forcing | u_max | note |
|---|---|---|---|---|---|---|
| `z_zero` | 32 | −4.0 | 1.735e−03 | difference / EDM | **3.25e−06** | no interface: u_max collapses ≈ 4 orders of magnitude below the Δρ ≈ 1.8 cases (≈1e−2) → the fitted law must have a vanishing intercept. Control Z is a working instrument check. |
| `wb_psi2` | 32 | −5.0 | 1.776 | psi2 / EDM | **1.52e−01** | the "curl-free" `psi2` force does *not* suppress the spurious current; it also inflates the interface (R_eq = 6.02 for R = 4) and collapses the Laplace pressure (dp = 1.20e−03, σ = 3.62e−03). |

**Consequence (protocol change, stated as a deviation):** the `psi2` arm was
pre-registered as a *gate* control ("well-balanced force must collapse u_max").
The measurement rejects that hypothesis — `psi2` is not an equivalent force
discretisation of the same physical model here — so it was demoted to an
**exploratory, non-gating** arm, and the falsification duty moved to
Z (vanishing intercept) + P (u_max plateau over the last three checks) +
M (mass conservation to round-off). This is recorded as a **rejected
hypothesis** in the capability ledger, not as a passing test.

Also corrected before any production number was produced: the initial tanh
width is a **constant** `W_init = 2.0` lattice units, because the *equilibrium*
interface thickness is fixed by the model (≈ constant in lattice units at fixed
`G_eff`); the ladder therefore refines box/radius at fixed interface resolution.

## Model EOS window measured in this round (Maxwell equal-area, `psi_exp`)

| G_eff | ρ_l | ρ_v | Δρ | ρ_l/ρ_v |
|---|---|---|---|---|
| −4.0 | 0.67771 | 0.675977 | 1.735e−03 | 1.003 |
| −4.4 | 1.39029 | 0.283848 | 1.106 | 4.90 |
| −4.7 | 1.67796 | 0.203222 | 1.475 | 8.26 |
| −5.0 | 1.92887 | 0.152895 | 1.776 | 12.62 |
| −5.3 | 2.15848 | 0.118235 | 2.040 | 18.26 |
| −5.6 | 2.37367 | 0.093071 | 2.281 | 25.50 |

Couplings beyond |G_eff| ≳ 5.8 return no Maxwell pair (`converged=False`), which
caps the abscissa lever arm at ≈ 2.1× — the reason the protocol uses 5 levels.

## Round-2 entry point

Nothing in the code needs to change. Only machine time:

```bash
python benchmarks/mcmp3d_spurious_scaling/run.py \
    --out artifacts/mcmp3d_spurious_scaling --jobs 30 --threads 2 \
    --steps 5000 --check-every 250 --steady-tol 1e-3
python benchmarks/mcmp3d_spurious_scaling/fit_report.py \
    --results artifacts/mcmp3d_spurious_scaling/results.json
```

`fit_report.py` emits `verdict.json` with C1/C2/C3/Z/P/M evaluated; if C2 fails
on real data the same artifact carries the residual structure needed to decide
whether the power-law *ansatz* (rather than the measurement) is what fails.