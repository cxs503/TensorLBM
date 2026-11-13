# PREREG — XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR (mission 18237)

Written **before** any number of this round was produced. Nothing below is
tuned after seeing results; deviations are reported in §7 of
`docs/XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR.md`.

## 1. Decision this study must support

Round 1 (`docs/XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER.md`, mission 17843)
established that the OSS *mechanism* (`graph_unroll` + in-graph `tau` + BPTT)
reproduces exactly, but that the **selection**: which sub-grid model and, more
importantly, *which objective* the gradient is taken of — is the part that
decides whether the calibrated model is physically right. Round 2 therefore
asks a decision question:

> **Decision D**: on the Taylor–Green-vortex anchor at `Re = 500`, which
> (closure, objective) pair should be *selected* by a BPTT gradient pipeline so
> that the selected model reproduces the fine-grid (128², filtered) reference
> within the owner's 15 % bar — and does the selection pipeline pick it?

* **Alternative A1 (closure hypothesis)**: the choice of closure
  (literature Smagorinsky `Cs` vs the OSS `Sc` proxy) dominates the outcome.
* **Alternative A2 (objective hypothesis, round-1 lead)**: the choice of
  objective observable dominates; a pointwise field-matching objective selects
  a model that is physically wrong even when the optimum is exact, whereas an
  integral objective (kinetic-energy / enstrophy decay) selects a model that
  transfers to the hold-out time window.

D is decided by *regret* on a held-out observable (§4), not by narrative.

## 2. Hypothesis

H2: with a held-out terminal observable, the regret of the BPTT selection made
with an **integral** objective is < 15 pp and smaller than the regret of the
pointwise-field objective, i.e. A2 is supported and A1 is not.

Falsifier: if the pointwise-selected and integral-selected models have the same
hold-out error within 2 pp, A2 is refuted (closure or nothing else dominates).

## 3. Workload (fixed, no per-arm tuning)

| quantity | symbol | value |
|---|---|---|
| domain | — | `2π × 2π`, periodic |
| reference velocity | `U0` | 0.1 (lattice) |
| mach | `Ma = U0/c_s`, `c_s=1/√3` | 0.1732 |
| integral length | `L` | 1.0 |
| physical viscosity | `ν = U0 L / Re`, Re = 500 | 2.0e-4 |
| grid | `Δx = 2π/N` | N = 128 (ref), N = 32 (LES), N = 256 (conv.) |
| ref relaxation | `τ(128) = 0.5+3ν/Δx²` | 0.749007 |
| conv relaxation | `τ(256)` | 1.496029 |
| coarse relaxation | `τ0(32)` | 0.515563 |
| LES filter | — | 4×4 box (128→32) |
| time window | `T` | 200 steps; snapshots 50, 100, 150, 200 |
| train snapshots | — | **50, 100, 150** (early window) |
| hold-out snapshot | — | **200** (never seen by any objective) |
| initial condition | — | 2D TGV (analytic, no random seed) |
| closures | — | `smagorinsky` (`Cs`), `oss_proxy` (`Sc`) |
| objectives | — | `field`, `energy`, `enstrophy` (all on train snapshots) |
| θ range | — | `[1e-3, 1.0]` (clamped) |
| BPTT | — | Adam on `log10 θ`, `lr = 0.1`, 20 iter, starts `0.02` and `0.17` |
| derivative-free control | — | 25-point sweep `θ∈[0,0.6]` + golden section on `[0,0.6]` |

## 4. Metrics and pre-declared thresholds

Judged quantities (each answered explicitly, PASS/FAIL):

| id | quantity | definition | threshold |
|---|---|---|---|
| P1 | gradient correctness | `|∂J/∂θ|_autograd` vs central FD (`h=1e-3`), relative | **< 15 %** |
| P2 | selection agreement | BPTT-selected θ vs derivative-free sweep argmin, relative (if argmin > 0.02; otherwise loss agreement `ΔJ/J < 15 %` and reported as bound-hit) | **< 15 %** |
| P3 | physical accuracy of the selected model | hold-out `ke_err = |K(200)−K_ref(200)|/K_ref(200)` | **< 15 %** |
| P4 | regret of the selection | `ke_err(selected) − min_{combo,θ} ke_err` in percentage points (`pp`) | **≤ 15 pp** |
| P5 | reproducibility | spread of BPTT θ over the two start points, relative | reported, ≤ 15 % expected |

* **Selection rule (pre-declared)**: the *selected* combo is the one with the
  smallest **training-window** loss `J` among the six, i.e. exactly what a
  practitioner would do with no access to the reference. Regret (P4) is then
  measured on the hold-out. `ke_err` is used as the primary physical judge
  because round 1 showed the pointwise field is decorrelated by the
  enstrophy cascade (error floor ≈ 50–90 %), so a pointwise judge cannot
  distinguish model quality; `field_err`, `z_err` and `spec_err` are reported
  alongside for completeness, never used to override the primary judge.
* The `energy` objective is trained on t = 50/100/150 and judged at t = 200,
  so P3 is an *extrapolation* metric for it too — no objective is judged on a
  snapshot it saw.

## 5. Controls, confounders, repetitions

* **Control 1** — derivative-free optimum (`sweep_loss` + `golden_minimize`)
  shares no autograd code with the BPTT path: distinguishes "optimiser wrong"
  from "objective wrong".
* **Control 2** — central finite difference of the same loss: distinguishes a
  wrong gradient from a wrong minimum.
* **Control 3** — `θ = 0` (no sub-grid model) is always in the sweep: the
  selector must be *able* to select "no model". Round 1's KE numbers imply
  that this may well be the physical optimum; that is a legitimate outcome and
  is what P4 measures.
* **Confounders held fixed**: identical lattice, identical `f0` (sampled from
  the same analytic field), identical `tau0`, identical train/hold-out
  snapshots, identical optimiser hyper-parameters, CPU float64, no random
  seeds anywhere (deterministic initial guess, deterministic Adam).
* **Repetitions**: BPTT is run from two fixed start points (0.02 = a
  deliberately "too small" start, 0.17 = the OSS default); the reference
  resolution is checked 128² → 256² for the judged integral; the training
  window is checked {50,100,150} (primary) vs {50,100,150,200} (secondary,
  in-sample for the energy objective — reported separately, never used for P3/P4).

## 6. Analysis, uncertainty, stopping rules

* Aggregation: 1 number per (combo, start) for θ and `J`; hold-out metrics are
  evaluated once per selected θ.
* No outlier removal, no missing-data imputation (a failed stage is reported as
  a failure, `results.json` keeps the partial record).
* Stopping rule: fixed iteration counts (20 Adam steps, 25-point sweep, 40
  golden-section iterations); the study stops when all 6 combos × 2 starts are
  recorded. No early stopping on the hold-out (that would leak it).
* Environment: repo worktree `/root/tp-agent/work/m18237`, `PYTHONPATH=src`,
  CPU float64, torch autograd; wall-clock per stage < 100 s (shell cap).

## 7. Interpretation limits

Correlation ≠ causation: this is a two-closure, one-flow, one-Re study. A
"selected" constant is only identified *for this flow and grid*; the claim
under test is about the *pipeline*, not about the physical value of `Cs`.