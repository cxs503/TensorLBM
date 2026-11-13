# XVAL-LBM-TWO-PDF-PARTIAL-DENSITY-COLORGRADIENT — round 2 report

Mission 18787 (TensorLBM, branch `agent/m18787`). Two-round method: research the
mature method first, then advance. Round 1 (m18579) failed with
`fail_stage=failed, reason="worktree 失败: no space left on device"` — an
infrastructure crash, **not** a technical falsification (recovered from
`tensor_dev_missions#18579.result`), so no method hypothesis was invalidated.

**Verdict of this round: PARTIAL PASS (honest partial).**
Two-layer Poiseuille at the frozen `g10_ny64` arm: `relL2 = 4.21 % < 15 %` (gate PASS).
Laplace law at beta = 0.9: sigma radius-independent to 0.118 % spread (gate PASS).
Single-phase limit: `relL2 = 0.0130 %`.
**Not run** (budget): `g10_ny128` (120000 steps), beta = 0.95 arms, cross-model
vs the single-PDF two-component engine. Those are recorded as `not_run`, not as passed.

---

## 1. Problem setting and benchmark source

**Model under test** — two-PDF partial-density colour gradient (the OSS
"two-species partial density colour gradient" family):

```
f, g                two complete D2Q9 PDFs (one per species)
rho_A = sum_i f_i,  rho_B = sum_i g_i,  rho = rho_A + rho_B
phi   = (rho_A - rho_B) / rho in [-1, 1]   (algebraic colour, not a phase-field PDE)
u     = sum_i c_i (f_i + g_i) / rho  (+ 0.5 F/rho Guo shift)
```

In a pure region `rho = 1`, `phi = +1` -> `rho_A = 1, rho_B = 0`: each species
carries its **partial** density (contrast with the classical Rothman-Keller
convention where each colour holds `rho/2` everywhere). `phi = +-1` end states,
`beta in [0.9, 0.95]` and the `omega = 1/(6/((1+phi)/nu_1 + (1-phi)/nu_2) + 0.5)`
viscosity rule are the documented conventions of the OSS reference.

**External anchors (no hand-copied numbers, all evaluated in-process)**

1. *Two-layer (stratified) plane Poiseuille flow*, kinematic viscosity ratio 10,
   equal densities, halfway bounce-back walls. Closed form `u'' = -g/nu` per
   layer with no slip at the wall planes `Y = 0.5` and `Y = ny - 1.5`, velocity
   and shear-stress continuity at the interface plane `Y_i = j_first_blue - 0.5`
   (interface position measured from the run's own steady colour column).
   Frozen evaluator `cg_reference.py` (md5 `e78c6d1eb22d628182636c0df74fb26a`,
   copied from `benchmarks/verified/poiseuille_two_phase_cg/`).
   Literature: Leclaire, Reggio & Trepanier (2012) JCP 231, 6294; Leclaire et al.
   (2013) JCP 246, 318 (partial densities); Latva-Kokko & Rothman (2005) PRE 71,
   056702; Guo, Zheng & Shi (2002) PRE 65, 046308; Rothman & Keller (1988) JSP 52,
   1119.
2. *Laplace law* `dp = sigma / R` for a static droplet — the **law** (radius
   independence of `sigma = dp * R`) is the anchor; the discrete colour-gradient
   amplitude has no closed-form sigma, so no constant is copied.

**OSS research (step 1)** — `oss_inventory` id=47 `lanl/MF-LBM` (tier 1):
*"color-gradient (Rothman-Keller) two-PDF two-species formulation with R-K
recoloring (beta) and partial densities; force = CSF 0.5\*gamma\*kappa\*|C|\*n +
body force, injected in moment space, u includes +0.5F; phi=(rho1-rho2)/(rho1+rho2)
algebraic (NOT a phase-field PDE)"*, and `oss_solutions` id=203
(feature_slug `NEW:two-species-partial-density-colorgradient`, files
`Module.F90:203-205`, `Kernel_multiphase.F90:108-110,124,159-163,272-315`;
conventions `phi in [-1,1]`, `beta 0.9-0.95`, the omega rule above). The raw
MF-LBM checkout is not readable from this worktree (`~/oss-cache` holds only
pylbm / Maarten-vd-Sande_lbm / hschwane; the cluster read job `cj2030` stayed
queued; outbound HTTP is blocked), so the structural reference is the inventory
record + solution record above — a stated limitation, not a claim of source-level
reading.

## 2. Method and implementation

New additive module `src/tensorlbm/color_gradient_two_pdf.py`
(no modification of `multiphase.py` / `color_gradient2d.py`):

| stage | equation implemented | code pointer |
|---|---|---|
| macros | `rho_A, rho_B, rho, phi, u` from the two PDFs | `two_pdf_macros` |
| viscosity | `1/nu = (1+phi)/(2 nu_A) + (1-phi)/(2 nu_B)`, `tau = 3 nu + 1/2` | `viscosity_from_phi`, `tau_from_phi` |
| collision | BGK per species to `w_i rho_sigma (1+3c.u+4.5(c.u)^2-1.5u^2)` with the shared `tau(phi)` | `colorgradient_two_pdf_step` (step 3) |
| forcing | Guo (2002) three-term source per species, `F_sigma = rho_sigma g` (exact per-species momentum balance) | `_guo` closure |
| surface tension | Latva-Kokko/Rothman anisotropic perturbation on the total PDF `(A/2)|grad phi| w_i((c_i.n)^2 - 1/3)` (mass- and momentum-free on D2Q9) | step 4 |
| recolouring | `f = (rho_A/rho) f_tot + beta (rho_A rho_B/rho) w_i (c_i.n)`, `g = (rho_B/rho) f_tot - (same)`, isotropic `w_i` | step 5, `recolor="iso"` |
| walls | halfway bounce-back on the solid rows via `tensorlbm.boundaries.bounce_back_cells` after `solver.stream` | `run_xval.py::run_poiseuille_arm` |

**Selection decisions (round-2 sediment, `oss_solutions.json`)**

* A1 **chosen**: two-PDF partial density + BGK `tau(phi)` + Guo forcing + isotropic
  R-K recolouring + perturbation surface tension (minimal, verifiable change).
* A2 **rejected on measured evidence**: the *literal* MF-LBM recolouring stencil
  (main axes `4/9`, diagonals `1/(36 sqrt 2)`) at this amplitude scaling drives
  the minority partial density **negative** — measured `min rho_sigma = -0.114`
  (ny=32, 1000 steps, static interface, `recolor="mf"`), and `phi` left `[-1,1]`
  reaching `+1.47`. The option is kept in the API (`recolor="mf"`) and pinned by a
  test/report record as a rejected variant.
* A3 **deferred**: MF-LBM's CSF force `0.5 gamma kappa |C| n` in moment space
  instead of the perturbation — the perturbation is validated here by the Laplace
  law, so the swap is not needed for this anchor.
* A4: MRT collision (MF-LBM) vs SRT used here — SRT keeps the change minimal;
  viscosity contrast (10:1) is already reproduced to 4.2 %.

## 3. Parameter table (all dimensionless, lattice units, fp64)

| quantity | value | note |
|---|---|---|
| lattice | D2Q9, `nx = 8` streamwise (periodic) | 1 cell in x = streamwise periodicity |
| grid | ny = 32 / 64 (probe), 64 (frozen arm), 128 (frozen, not run) | walls rows 0, ny-1 |
| `nu_A` (`tau_A`) | 0.66667 (2.5) | phase `phi=+1`, lower half |
| `nu_B` (`tau_B`) | 0.066667 (0.7) | phase `phi=-1`, upper half |
| viscosity ratio | 10 | `mu`-ratio = `nu`-ratio (equal density) |
| `A` (colour-gradient amplitude) | 0.04 | same as the verified G2 benchmark |
| `beta` | 0.9 (frozen), 0.95 (not run) | recommended band 0.9-0.95 |
| `gx` | 1.5e-7 (ny=64), 3.75e-8 (ny=128) | per unit mass |
| interface width | tanh `(j - ny/2)/2` | measured `j_first_blue = 33` at ny=64 |
| steps | 60000 (sp/ny64), 80000 (g10/ny64), 120000 (ny128) | u_max frozen-tail monitor |
| `Re` (lattice, from `u_max`) | ~1.6e-2 | Stokes regime, no inertia |
| droplet | n = 96, R = 12/18/24, 8000 steps, width 2 | `p = rho/3` pressure |
| measured | `rho` inside droplet `1 + 3 dp` | Laplace calibration |

## 4. Comparison table (ref vs got, per item)

| item | reference | got | error | gate (<15 %) |
|---|---|---|---|---|
| single-phase Poiseuille, ny=64, 60k steps | closed form (halfway planes) | relL2 = 0.0130 % | 0.013 % | PASS |
| two-layer Poiseuille, ny=64, ratio 10, beta=0.9, **frozen arm** | two-layer closed form | relL2 = **4.214 %** | 4.21 % | **PASS** |
| two-layer Poiseuille, ny=32, 12k steps (probe) | two-layer closed form | relL2 = 8.719 % | 8.72 % | PASS (probe) |
| two-layer Poiseuille, ny=64, 12k steps (probe) | two-layer closed form | relL2 = 4.094 % | 4.09 % | PASS (probe) |
| Laplace law, R = 12 | `sigma = dp R` | `dp = 7.3897e-4`, sigma = 8.8677e-3 | dev 0.062 % | PASS |
| Laplace law, R = 18 | idem | `dp = 4.9232e-4`, sigma = 8.8617e-3 | dev -0.006 % | PASS |
| Laplace law, R = 24 | idem | `dp = 3.6905e-4`, sigma = 8.8573e-3 | dev -0.055 % | PASS |
| per-species mass drift | 0 (partial-density exactness) | 2.7e-15 … 3.6e-14 per step | << 1e-10 | PASS |
| NaN-free | required | all completed arms finite | — | PASS |
| two-layer ny=128 (frozen arm) | two-layer closed form | **not_run** | — | not_run |
| beta = 0.95 sensitivity | — | **not_run** | — | not_run |
| cross-model vs `color_gradient2d` | — | **not_run** | — | not_run |

Machine-readable: `artifacts/laplace_log.json`, `artifacts/resolution_probe.json`,
stdout logs `/tmp/xval_pois.log`, `/tmp/xval_lapl.log` (transcribed, provenance noted).

## 5. Convergence and parameter sensitivity

* **Grid convergence (two-layer, identical 12000 steps)**: relL2 8.719 % (ny=32) ->
  4.094 % (ny=64); the frozen 60k-step ny=64 arm gives 4.214 %, i.e. the step
  count changes the answer by 0.12 pp — the ny=64 solution is converged
  (`u_max` frozen-tail relative change 2.6e-3 over the last 20 % of samples).
* **Laplace law, three radii** (a parameter scan in `1/R`): `sigma = dp R` is
  constant to 0.118 % peak-to-peak, i.e. the model reproduces the analytic law
  `dp ∝ 1/R` with no free constant fitted — this is the strong evidence that the
  surface-tension perturbation term is correctly installed in the two-PDF split.
* **beta sensitivity / ny=128**: not run (budget); no claim is made.

## 6. Failure modes, artifacts, reproducibility

* **Rejected variant (real, measured)**: `recolor="mf"` (literal MF-LBM `4/9`
  axis weights) breaks positivity of the minority partial density (`min rho_sigma
  = -0.114`) and pushes `phi` out of `[-1,1]` (`+1.47`); with `"iso"` weights
  `phi` stays inside `[-1,1]` and `min rho_sigma > -1e-12`. Captured by
  `tests/test_color_gradient_two_pdf.py::test_recolor_keeps_partial_densities_non_negative`
  (the test asserts the *good* behaviour; the bad number is recorded here and in
  the module docstring).
* **Infrastructure risk** (round-1 death cause): the long arms were run as
  detached local processes because the cluster queue (`cj2030`, read-only OSS
  listing) stayed `queued` for >20 min; two shell calls were killed at the
  120 s tool limit, which truncated the single-phase 12k probe (its JSON was
  re-transcribed from stdout, flagged in `resolution_probe.json`).
* **Scope limits**: equal densities only (`rho = 1`, density ratio 1 — the
  partial-density family's documented restriction, as in MF-LBM); SRT collision;
  flat-interface + static-droplet validation only, no Rayleigh-Taylor / rising
  bubble; spurious-current magnitude not reported (the laplace suite did not
  complete its cross-model arms).

```bash
# frozen arms (poiseuille) — 2 arms completed here
PYTHONPATH=src python3 xval/XVAL-LBM-TWO-PDF-PARTIAL-DENSITY-COLORGRADIENT/run_xval.py \
    --suites poiseuille --out xval/XVAL-LBM-TWO-PDF-PARTIAL-DENSITY-COLORGRADIENT/artifacts
# Laplace law
PYTHONPATH=src python3 xval/XVAL-LBM-TWO-PDF-PARTIAL-DENSITY-COLORGRADIENT/run_xval.py \
    --suites laplace --laplace-beta 0.9 --out .../artifacts
# unit / invariant tests (7 tests, pass)
PYTHONPATH=src python3 -m pytest tests/test_color_gradient_two_pdf.py -q
```

Figure: `artifacts/two_pdf_colorgradient_xval.png` (grid convergence, Laplace law,
radius independence of sigma).