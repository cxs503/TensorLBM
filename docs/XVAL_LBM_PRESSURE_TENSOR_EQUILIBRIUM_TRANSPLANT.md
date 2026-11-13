# XVAL round 2: pressure-tensor equilibrium, acceptance matrix (sailfish transplant)

Anchor `XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT` (feature_ledger#435), repo TensorLBM, mission
m18648, round 2. Gates declared before any run in
`xval/XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT/predeclared.json`; owner band `< 15 %` per KPI.

Round 1 (m17879) merged the semantics transplant and passed two arms but left the curvature/resolution
series unrun and never measured two equilibrium invariants. Round 2 (this document) adds exactly those,
plus one pre-existing environment failure that has to be recorded because it also killed m18604.

## 1. Problem setting and references

The transplanted model (`src/tensorlbm/phasefield/free_energy_pressure.py`) carries the free-energy
pressure tensor **inside** the equilibrium, so the second moment of the populations *is* the pressure
tensor and mechanical equilibrium becomes directly measurable:

* flat interface: `d P_NN / dx = 0` and `P_TT = P_NN - kappa phi'^2`
* circular drop:  `p_in - p_out = sigma / R` (Laplace)

| what | where |
|---|---|
| fluid + order-parameter equilibrium (`pb`, `mu`, `f_0`, `g_0`) | `sailfish-team/sailfish` `sym_equilibrium.py:26-48`, `:50-70` |
| anisotropic `wxx/wyy/wxy` = 1/3, -1/6, -1/24 | `lb_binary.py:185-268` |
| nine-point isotropic grad/lap | `templates/finite_difference_optimized.mako:46-48` |
| pressure tensor = second moment of the populations | Wagner, *Int. J. Mod. Phys. B* **17**, 193 (2003) |
| `sigma = integral (P_NN - P_TT) dn` | Rowlinson & Widom, *J. Chem. Phys.* **76**, 1140 (1982) |
| pressure-tensor free-energy LBM, `dP_NN/dx = 0` | Pooley, Furtado & Carr, *Phys. Rev. E* **78**, 056709 (2008) |

Analytic anchors re-derived from `f = A(-phi^2/2 + phi^4/4) + kappa|grad phi|^2/2` inside the module:
`xi = sqrt(2 kappa / A)`, `sigma = (2/3) sqrt(2 kappa A) = (2/3) A xi`, `phi = tanh((x-x0)/xi)`.

## 2. Method

Nothing about the physics is re-implemented in round 2. The new work is measurement + matrix:

* `fit_interface_width` (new, in the module) — recovers `xi` from the *local* identity
  `P_NN - P_TT = kappa sech^4((x-x0)/xi)/xi^2`. The amplitude is eliminated by linear least squares, so
  the search is one-dimensional (coarse scan + golden section). It never uses `sqrt(2 kappa / A)` or the
  analytic `tanh`, hence it can falsify wrong `wxx/wyy` coefficients.
* `shear_free` KPI (new) — `max|P_xy| / <P_NN>` on a resting flat interface; the off-diagonal pair
  (`wxy`, `e_x e_y / 4`) had never been exercised at run time.
* `benchmarks/xval_pressure_tensor_equilibrium/run_matrix.py` — runs the declared arms, applies the
  predeclared gates per KPI, writes `artifacts/xval_pressure_tensor_equilibrium/results.json`.
* `benchmarks/xval_pressure_tensor_equilibrium/make_figures.py` — figures from the JSON only.
* `tests/test_pressure_tensor_equilibrium_xval.py` — pins the instruments (fit recovery, bias bound,
  shear-free, Rowlinson-Widom integral, gate logic).

Arms reuse `benchmarks/xval_free_energy_binary_equilibrium/run_xval.py`; the only edits there are the two
new KPI fields, the three new arms and a guard so a diverged (NaN) field reports `xi_fit = nan` instead of
crashing the matrix.

## 3. Parameters

`A = 0.1`, `Gamma = 0.2`, `tau_f = 1.0` (`nu = 1/6`), `tau_phi = 1.5` (`M = Gamma(tau_phi-1/2)cs^2 =
1/15`), D2Q9, float64, periodic. `kappa = A xi^2 / 2`.

| arm | kind | xi | kappa | domain | steps | R/xi |
|---|---|---|---|---|---|---|
| flat_xi2 | band (2 interfaces) | 2 | 0.2 | 64x8, band 16 | 4000 | -- |
| flat_xi3 | band | 3 | 0.45 | 64x8, band 16 | 4000 | -- |
| dropR16 | drop | 2 | 0.2 | 96x96, R=16 | 8000 | 8 |

## 4. Results (all numbers from `results.json`, none hand-typed)

| KPI | reference | got | error | gate | verdict |
|---|---|---|---|---|---|
| closure `max_ab|sum f^eq cc - (P+rho uu)|/max|P|` (flat_xi2 / flat_xi3 / dropR16) | 0 | 4.65e-16 / 3.10e-16 / 4.57e-16 | machine precision | 1e-12 | PASS |
| `sigma` from `int(P_NN - P_TT)dx` (flat_xi2) | 0.133333 | 0.127927 | **4.05 %** | 15 % | PASS |
| `sigma` (flat_xi3) | 0.200000 | 0.195981 | **2.01 %** | 15 % | PASS |
| `P_NN` uniformity (flat_xi2 / flat_xi3) | 0 | 7.24e-5 / 1.48e-5 | -- | 1 % | PASS |
| `shear_free` `max|P_xy|/<P_NN>` (flat_xi2 / flat_xi3) | 0 | 0.0 / 9.7e-18 | -- | 5 % | PASS |
| `xi` from the sech^4 fit (flat_xi3) | 3 | 3.26423 | **8.81 %** | 15 % | PASS |
| `xi` from the sech^4 fit (flat_xi2) | 2 | 2.38284 | **19.14 %** | 15 % | **FAIL (honest)** |
| Laplace `p_in - p_out` (dropR16) | 8.3333e-3 (`sigma/R`) | 8.3181e-3 | **0.18 %** | 15 % | PASS |
| spurious `max|u|` (flat / drop) | 0 | 6.4e-6 / 5.18e-4 | -- | 1e-2 | PASS |

Totals from `results.json`: **14 pass / 1 fail**. The failure is real and is reported as such:
the interface width recovered from the stress profile is 19 % high at `xi/h = 2`. The instrument is not
at fault — on an *exact* `sech^4` stress profile the same fit returns `xi` to 1e-3 (test
`test_fit_recovers_xi_from_analytic_stress`), and on the nine-point **discrete** gradient of a `tanh`
band it returns +9.9 % at `xi/h = 2`, +4.4 % at `xi/h = 3`, +2.5 % at `xi/h = 4`
(`test_fit_on_discrete_gradient_is_biased_high_but_bounded`). The measured run sits above that discrete
bound, i.e. **`xi/h = 2` is not enough resolution to read the width back to 15 %**; `xi/h = 3` is.

## 5. Convergence / sensitivity

* Resolution series (`xi/h = 2 -> 3`): `sigma` error 4.05 % -> 2.01 %, `P_NN` residual 7.24e-5 -> 1.48e-5,
  `xi_fit` error 19.14 % -> 8.81 %. All three shrink with resolution, consistent with a discretisation
  (not a coefficient) error. No rate is claimed: two points cannot establish one.
* **Not obtained:** the `xi/h = 4` point. At `kappa = 0.8` the explicit Cahn-Hilliard + pressure-tensor
  scheme goes NaN by step ~1000 with `Gamma = 0.2` **and** with `Gamma = 0.05`
  (`flat_xi4`, `flat_xi4_g05` — both arms are kept in the source as the documented stability boundary,
  excluded from the default matrix). This is a scheme stability limit that scales like `kappa * Gamma`,
  consistent with round 1's screening, not a transplant defect (the closure identity still holds to 1e-16
  before blow-up).
* Curvature series: only `R/xi = 8` completed inside one local short-run slot (83 s). `dropR24` and
  `dropR32` were declared and are **not run** — reported, not glossed.

## 6. Failure modes and artefacts

* Reproduce: `python benchmarks/xval_pressure_tensor_equilibrium/run_matrix.py --threads 6`
  (arms cached per file; `--force` re-runs), then
  `python benchmarks/xval_pressure_tensor_equilibrium/make_figures.py`.
  Tests: `python -m pytest tests/test_pressure_tensor_equilibrium_xval.py -q`.
* Artefacts: `artifacts/xval_pressure_tensor_equilibrium/{flat_xi2,flat_xi3,dropR16,results}.json`;
  `docs/figures/xval_pt_{pressure_profile,sigma_resolution,laplace}.png`.
* **Environment failure (blocking, now worked around):** the host filesystem was 100 % full
  (`/` 892G, 28 KB free) at mission start, which is the same `No space left on device` that failed
  m18604 at worktree creation. 2.5 GB were reclaimed by deleting two two-day-old OpenFOAM source trees in
  `/tmp` that no process referenced (`/tmp/of_src`, `/tmp/OpenFOAM-10`). Any mission needing more than a
  few MB of artefacts will fail again until the fleet-wide disk debt (23 GB under
  `/root/.hermes/marine-control`) is reaped by a mechanism, not by hand.
* Limitation: the comparison arm is the semantics transplant, not a run of sailfish (CUDA/pyopencl, not
  buildable here) — unchanged from round 1 and declared.
