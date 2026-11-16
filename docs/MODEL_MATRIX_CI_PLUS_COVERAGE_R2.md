# Model matrix CI: coverage extension + gate detection power (round 2)

Anchor: `XVAL-LBM-MODEL-MATRIX-CI-PLUS-COVERAGE`. Round 1 built the matrix
(registry key = model, analytic cases, coverage accounting, exit-code gate).
Round 2 closes the two things that made round 1's green verdict weak:

1. **The denominator was hand-picked.** 14 collision operators ship in this
   repository as user-selectable models but only 9 were matrix keys.  Five
   more are now registered and *gated* (`d3q19/cumulant`, `d3q27/cumulant-`
   `legacy`, `d3q27/cumulant-geier`, `d3q19/cascaded`, `d3q27/cascaded-impl2`).
2. **The gate only saw one channel.** `shear_decay` measures the shear
   viscosity; the bulk (trace) eigenvalue `s_b` was *unverified* in every
   operator that exposes one (`s_e` in MRT, `s_bulk` in cascaded, `omega_b` in
   cumulant).  A mis-wired bulk rate was invisible.  A new gated case,
   `acoustic_decay`, measures sound attenuation `S = 4nu/3 + nu_b` and
   therefore sees the bulk channel.  A mutation study quantifies what the gate
   can and cannot catch.

## 1. Coverage after round 2 (machine-generated)

* registry entries: **14 models** (was 9); declared cells: **47** (was 21)
* executed / declared: **47/47 = 100.0%**, missing cells: none
* gate failures: **none**, verdict **PASS**
* guard pipeline (construct + one step): 14/14 `GUARD_PASS` (`guard.json`)

| model | lattice | family | enforcement | bulk eigenvalue source |
|---|---|---|---|---|
| `d3q19/bgk` | D3Q19 | single-relaxation-time | gate | `None` = 1/tau (single-rate) |
| `d3q19/cascaded` | D3Q19 | cascaded-central-moment | gate | `s_bulk` = 1.0 |
| `d3q19/cumulant` | D3Q19 | cumulant | gate | `omega_b` = 1.0 |
| `d3q19/mrt` | D3Q19 | multi-relaxation-time | gate | `s_e` = 1.19 |
| `d3q19/mrt-smag` | D3Q19 | mrt+les | guard | `None` = 1/tau (single-rate) |
| `d3q19/rlbm` | D3Q19 | regularized | gate | `None` = 1/tau (single-rate) |
| `d3q19/trt` | D3Q19 | two-relaxation-time | gate | `None` = 1/tau (single-rate) |
| `d3q27/cascaded` | D3Q27 | cascaded-central-moment | gate | `s_bulk` = 1.0 |
| `d3q27/cascaded-impl2` | D3Q27 | cascaded-central-moment (independent 2nd implementation) | gate | `s_bulk` = 1.0 |
| `d3q27/cascaded-les` | D3Q27 | cascaded+les | guard | `None` = 1/tau (single-rate) |
| `d3q27/cumulant-geier` | D3Q27 | cumulant (Geier 2015) | gate | `omega_b` = 1.0 |
| `d3q27/cumulant-legacy` | D3Q27 | cumulant (central-moment legacy variant) | gate | `omega_b` = 1.0 |
| `d3q27/kbc` | D3Q27 | entropic-kbc | gate | `None` = 1/tau (single-rate) |
| `d3q27/kbc-les` | D3Q27 | entropic-kbc+les | guard | `None` = 1/tau (single-rate) |

## 2. Bulk channel: reference vs measured (`artifacts/model_matrix_plus/BENCHMARK_REF_VS_GOT.md`)

`S = 2*gamma/k^2` is measured from the envelope decay of a right-going
acoustic wave; the reference is the Chapman-Enskog formula
`4nu/3 + (2/9)(1/s_b - 1/2)`.  `s_b_implied` is obtained by inverting the
measurement -- it is a *check of the declaration*, not an input.

| model | tau | S_ref | S_meas | error % | s_b declared | s_b implied | fit r2 |
|---|---|---|---|---|---|---|---|
| `d3q19/bgk` | 0.60 | 0.06667 | 0.06670 | +0.04 | 1.6667 | 1.6663 | 0.99960 |
| `d3q19/bgk` | 0.80 | 0.20000 | 0.20093 | +0.46 | 1.2500 | 1.2435 | 0.99723 |
| `d3q19/cascaded` | 0.60 | 0.15556 | 0.15437 | -0.77 | 1.0000 | 1.0054 | 0.99834 |
| `d3q19/cascaded` | 0.80 | 0.24444 | 0.24638 | +0.79 | 1.0000 | 0.9914 | 0.99624 |
| `d3q19/cumulant` | 0.60 | 0.15556 | 0.15528 | -0.18 | 1.0000 | 1.0013 | 0.99834 |
| `d3q19/cumulant` | 0.80 | 0.24444 | 0.24661 | +0.88 | 1.0000 | 0.9904 | 0.99624 |
| `d3q19/mrt` | 0.60 | 0.12007 | 0.12009 | +0.01 | 1.1900 | 1.1899 | 0.99895 |
| `d3q19/mrt` | 0.80 | 0.20896 | 0.21042 | +0.70 | 1.1900 | 1.1808 | 0.99715 |
| `d3q19/rlbm` | 0.60 | 0.06667 | 0.06670 | +0.04 | 1.6667 | 1.6663 | 0.99960 |
| `d3q19/rlbm` | 0.80 | 0.20000 | 0.20093 | +0.46 | 1.2500 | 1.2435 | 0.99723 |
| `d3q19/trt` | 0.60 | 0.06667 | 0.06670 | +0.04 | 1.6667 | 1.6663 | 0.99960 |
| `d3q19/trt` | 0.80 | 0.20000 | 0.20093 | +0.46 | 1.2500 | 1.2435 | 0.99723 |
| `d3q27/cascaded` | 0.60 | 0.15556 | 0.15497 | -0.37 | 1.0000 | 1.0026 | 0.99834 |
| `d3q27/cascaded` | 0.80 | 0.24444 | 0.24653 | +0.85 | 1.0000 | 0.9907 | 0.99624 |
| `d3q27/cascaded-impl2` | 0.60 | 0.15556 | 0.15437 | -0.77 | 1.0000 | 1.0054 | 0.99834 |
| `d3q27/cascaded-impl2` | 0.80 | 0.24444 | 0.24638 | +0.79 | 1.0000 | 0.9914 | 0.99624 |
| `d3q27/cumulant-geier` | 0.60 | 0.15556 | 0.15497 | -0.37 | 1.0000 | 1.0026 | 0.99834 |
| `d3q27/cumulant-geier` | 0.80 | 0.24444 | 0.24653 | +0.85 | 1.0000 | 0.9907 | 0.99624 |
| `d3q27/cumulant-legacy` | 0.60 | 0.15556 | 0.15437 | -0.77 | 1.0000 | 1.0054 | 0.99834 |
| `d3q27/cumulant-legacy` | 0.80 | 0.24444 | 0.24638 | +0.79 | 1.0000 | 0.9914 | 0.99624 |
| `d3q27/kbc` | 0.60 | 0.06667 | 0.06670 | +0.04 | 1.6667 | 1.6663 | 0.99960 |
| `d3q27/kbc` | 0.80 | 0.20000 | 0.20093 | +0.46 | 1.2500 | 1.2435 | 0.99723 |

**Is the error < 3 %? Yes** for all 22 bulk-channel cells; worst 0.88 %.
The lowest resolutions (nx=12/16) are the hard ones: at nx=24/32 the same
cells move to 0.03-0.4 % (round-1 style convergence sweep, `pilot` logs in
the commit message body).

Cross-checks that the declaration is not a tautology:

| family | declared s_b | implied s_b (tau=0.6) | implied s_b (tau=0.8) | comment |
|---|---|---|---|---|
| BGK / TRT / regularized / KBC (single-rate) | 1/tau = 1.6667 / 1.2500 | 1.6663 | 1.2435 | recovers the BGK relation nu_b = 2nu/3 |
| MRT (`s_e=1.19`) | 1.19 | 1.1899 | 1.1808 | independent trace rate is real |
| cascaded / cumulant (`s_bulk`/`omega_b`=1.0) | 1.0 | 1.0013-1.0054 | 0.9904-0.9914 | declaration confirmed |

## 3. Detection power of the gate (`artifacts/model_matrix_plus/mutation/MUTATION_DETECTION.md`)

A green gate is evidence only if it can go red. Each mutation wraps the real
production operator (no numerics reimplemented) and the reference stays the
*unmutated* analytic value.

| model | mutation | smallest detected magnitude | detected by |
|---|---|---|---|
| `d3q19/bgk` | bulk_only | n/a | - |
| `d3q19/bgk` | mass_leak | 1e-09 | shear_decay, acoustic_decay |
| `d3q19/bgk` | momentum_leak | 1e-10 | shear_decay, acoustic_decay |
| `d3q19/bgk` | nu_scale | 0.02 | shear_decay |
| `d3q19/cascaded` | bulk_only | 0.02 | acoustic_decay |
| `d3q19/cascaded` | mass_leak | 1e-09 | shear_decay, acoustic_decay |
| `d3q19/cascaded` | momentum_leak | 1e-10 | shear_decay, acoustic_decay |
| `d3q19/cascaded` | nu_scale | 0.02 | shear_decay |
| `d3q19/mrt` | bulk_only | 0.05 | acoustic_decay |
| `d3q19/mrt` | mass_leak | 1e-09 | shear_decay, acoustic_decay |
| `d3q19/mrt` | momentum_leak | 1e-10 | shear_decay, acoustic_decay |
| `d3q19/mrt` | nu_scale | 0.02 | shear_decay |
| `d3q27/kbc` | bulk_only | n/a | - |
| `d3q27/kbc` | mass_leak | 1e-09 | shear_decay, acoustic_decay |
| `d3q27/kbc` | momentum_leak | 1e-10 | shear_decay, acoustic_decay |
| `d3q27/kbc` | nu_scale | 0.02 | shear_decay |

Reading of the numbers:

* `nu_scale` (shear-viscosity defect) is caught from **+2 %** on every model,
  i.e. inside the owner's 15 % acceptance with a factor 7 of headroom. The
  2 % (not 3 %) floor is the +1.2 % discretisation bias of the fitted shear
  rate at res=16 eating into the tolerance.
* `bulk_only` (trace eigenvalue corrupted, `s_b -> (1+d) s_b`) is **caught by
  `acoustic_decay` at d = 2 % (cascaded) and 5 % (MRT)** and is **invisible to
  the round-1 gate** (`shear_decay` stays PASS by construction, because a
  solenoidal shear wave never excites the trace moment). This is the concrete
  defect class round 2 buys.
* `mass_leak` is caught at `eps = 1e-9` per step (mass-drift bound 1e-8 over
  ~200 steps), `momentum_leak` at `1e-10` per step -- both far below any
  physical effect, so those two are effectively absolute.
* **Limitation, stated honestly:** single-rate operators (BGK, TRT,
  regularized, KBC) *cannot* express a bulk-only defect -- one eigenvalue
  drives both channels -- so `bulk_only` is reported `n/a` for them. Their
  bulk channel is verified only in the sense that its value is pinned to the
  shear channel by construction (and the measurement confirms 1/tau).

## 4. What this does *not* prove

* `acoustic_decay` verifies the trace eigenvalue at small Mach and at
  `k = 2 pi/nx`. It says nothing about bulk viscosity in the presence of
  walls, forcing, or strong compression.
* A `PASS` still says nothing about the LES (`guard`) models: they have no
  molecular reference for either channel and are counted separately, never as
  gate passes.
* The five newly registered operators are verified on the analytic cases
  only. Their *flow-level* accuracy (cylinder drag, backward-facing step, ...)
  is covered by other anchors, not by this matrix.

## 5. Reproduce

```
python -m tensorlbm.model_matrix list                     # 14 keys
python -m tensorlbm.model_matrix guard --out artifacts/model_matrix_plus
python -m tensorlbm.model_matrix run --res 16 --out artifacts/model_matrix_plus
python -c 'from tensorlbm.model_matrix.mutation import main; main([])'
python -m pytest tests/test_model_matrix.py tests/test_model_matrix_plus_coverage.py -q
```

## 6. Files

| file | role |
|---|---|
| `src/tensorlbm/model_matrix/registry.py` | +5 registry keys, `bulk_param` declaration, `bulk_viscosity()` |
| `src/tensorlbm/model_matrix/cases.py` | + `acoustic_decay` case (gated, 3 %) |
| `src/tensorlbm/model_matrix/mutation.py` | mutation / detection-power study |
| `src/tensorlbm/model_matrix/coverage.py` | benchmark table renders the acoustic quantity |
| `tests/test_model_matrix_plus_coverage.py` | 21 falsifiable tests (incl. 3 negative controls) |
| `artifacts/model_matrix_plus/` | run record, coverage, ref-vs-got, guard, mutation evidence, figures |

