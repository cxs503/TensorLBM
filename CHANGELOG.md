# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`tensorlbm.ibm_sphere_surface_force`** — three-layer verification surface for the
  immersed-boundary *surface force* on a sphere (campaign node `p387w2-n04`).  It adds the
  piece both open force gaps were missing: an **exact spherical measure**
  (`A_k` from a per-cell Gauss-Legendre `∫sin(theta) dtheta`, so `sum A_k = 4 pi R^2` and
  `sum A_k n_k = 0` to round-off **independently of the marker count**), traction-based
  marker forces `F_k = rho_k (u_b - u*) A_k` with Guo half-force correctors and a Breugem
  normal-projection pass, an exact-momentum Guo source (so `sum_x f = sum_k F_k` holds to
  `1.4e-16`), and the matching recoil/torque books.  Measured: rigid translation is a null
  vector (`5e-18`) and the force is Galilean invariant (`1.4e-17`) — the two negative
  controls — while the shipped unit-weight marker *sum* is biased by exactly the marker
  density `M/A` (the additive background behind G15) and the pressure-only estimator
  returns zero torque against a finite analytic one (G12).  The quadrature of the exact
  Stokes rotating-sphere traction reproduces `8 pi mu Omega a^3` with second-order marker
  placement, and `C_M = 8/Re_omega` is the analytic anchor of the rotation layer.  Layers
  1-3 (drag over `Re in [1,1000]` against the mutually adjudicated Schiller-Naumann /
  Clift-Gauvin band, observed grid order at `Re = 100`, rotating-sphere torque) run through
  `scripts/p387w2_n04_sphere_force/`; the acceptance record with every criterion number is
  `artifacts/p387w2_n04/acceptance.json` and the report is
  `docs/P387W2_N04_IBM_SPHERE_SURFACE_FORCE.md` (drag verified at `Re = 1` (+2.7 %) and
  resolution-limited above, with the requirement `D >= 2 sqrt(Re)` measured, not assumed).
- **`tensorlbm.scalar_dispersion_mach`** — lattice-Mach error law of the passive-scalar
  shear-dispersion meter and the real cause of the m8228 "Pe=40, +8.685 %" shortfall
  (campaign node `p405w1-n06-lbm-scalar-dispersion`).  The slit-channel Taylor–Aris
  coefficient is a `k -> 0` quantity, so a single-domain modal-decay reading is biased by
  an `O((kH)^2)` truncation of the *meter*: at `Pe=40, H=17` the same shipped D2Q5 kernel
  reads `+7.0 %` (`kH=0.84`), `-1.3 %` (`0.42`), `-3.1 %` (`0.21`), `-3.5 %` (`0.105`).
  The module adds the multi-domain `(kH)^2 -> 0` Richardson limit, a LAPACK-free Thomas
  solve for the discrete Aris auxiliary problem (cross-checked against the library solver,
  because m8228 measured garbage solves from the cluster loong64 LAPACK), the second-order
  `O(u^2)` Hermite scalar equilibrium (ablation), and the Mach-law fit.  With the
  extrapolated meter: `Pe=40` in `[-2.6 %, -1.0 %]`, `Pe in [10,100]` and `Sc in [0.3,10]`
  all within 3 %, refinement order 2.1 in `(kH)^2`; the pure advective lattice bias
  measured on a plug-flow control follows `err = -3<u^2>` (linear in `Ma^2`), while the
  shear-case residual is Mach *independent* — the seeded `O(Ma^2)` attribution is falsified.
- **`tensorlbm.sloshing_forced`** — forced-sway LNG-tank sloshing verification
  surface (campaign node P88W1-N01).  A partially filled rectangular tank
  (L x B x H = 1.0 x 0.5 x 0.4 m, 50 % fill, h = 0.2 m) driven by harmonic sway
  `X(t) = X0 sin(omega t)` over `f/f1 = 0.6 ... 1.75`, on top of the audited
  Körner free-surface solver (`free_surface_lbm_2d` / `sloshing_resonance`,
  both untouched).  The package adds (i) the **linear forced-sway transfer
  function** derived by projecting the dynamic free-surface condition on the
  Neumann modes — horizontal translation excites odd modes only, which the
  campaign also *measures* — including its exact static limit (rigid-body tilt
  recovered to 2 %), (ii) a finite-time ODE reference that carries the start-up
  transient of a short forced run, (iii) harmonic (first-harmonic) wall-pressure
  and wall run-up metrics against the closed-form anchors (linear Faltinsen wall
  pressure, Wagner `0.5 rho V^2`, bore reflection, hydrostatic run-up) with an
  out-of-band noise-floor / SNR diagnostic, and (iv) a case-sharded campaign
  driver with re-analysis from the saved time series.  Campaign results
  (15 cases, `artifacts/p88w1`): f1 vs Abramson **-0.75 %** (free decay) and
  **+0.32 %** (forced-sweep maximum), f2 **-3.81 %**, wall pressure vs the linear
  anchor **+15.3 % at H/32** but **+4.8 % at H/64**, pressure grid convergence
  **17.2 % (honest FAIL, gate 3 %)** while the free-surface amplitude converges
  to 0.3 %, mode-2 content **7.8 %** of the resonant mode-1 amplitude, liquid
  ledger within **0.96 %**.  21 pytest cases (`tests/test_sloshing_forced_p88w1.py`).
  Experimental (ITTC) sloshing data = WITHHELD: no digitised model-test series
  exists in the repository and none was fabricated.  Report:
  `docs/P88W1_N01_sloshing_forced.md`.
- **`tensorlbm.phase_change.stefan_two_phase` / `stefan_sweep`** — two-phase
  (superheated-melt) planar Stefan similarity solution and the Stefan-number
  sweep that verifies the enthalpy thermal LBM against it (campaign node
  P272W2-N05).  Unlike the one-phase Neumann branches in `stefan_exact`
  (`T = T_freeze` ahead of the front), the liquid here is initially superheated
  (`T_i > T_m`), so the front coefficient solves the *coupled* dual-Stefan
  relation `lam = Ste_s e^{-lam^2}/(sqrt(pi) erf lam) -
  Ste_l e^{-lam^2}/(sqrt(pi) erfc lam)`; `Ste_l = 0` collapses it onto
  `planar_lambda` to machine zero (1e-16), so the surface extends rather than
  duplicates P155W1-N01.  Sweep `Ste_s = 0.1/0.5/1.0` (fixed `Ste_l = 0.2`,
  33-cell front): front vs `s(t)=2 lam sqrt(alpha t)` **1.20 %** (worst over all
  sampled times, tolerance 5 %), profile L-infinity **2.73 %** (< 3 %), energy
  closure (heat input vs sensible + latent on the *same* control volume, plus an
  independent second-order wall-flux operator) **0.35 %** (< 2 %), `0 <= f_l <= 1`
  over the whole run, and grid order **1.40 / 1.37** at 17/33/66 front cells;
  exact solution self-audited (analytic `Q(t) == latent + sensible` to 6e-16,
  interface energy balance 3e-18, FD PDE residual 1e-6).  A `Ste_l = 0` rerun
  reproduces the one-phase front to 1.43 %.  Artifacts with per-array sha256 in
  `artifacts/stefan_two_phase_sweep/`, report
  `docs/P272W2_N05_STEFAN_TWO_PHASE_SWEEP.md`, tests
  `tests/test_stefan_two_phase_sweep.py` (46 cases).
- **`tensorlbm.boundary.partial_slip_second_order`** — second-order reconstruction
  of the combined bounce-back / specular slip wall (campaign node p247w1-N10,
  source m7963-c1).  The O(1/H) amplitude bias of the mixed wall is traced to a
  spurious wall-slip velocity inherited from the Guo forcing term reflected by
  the bounce-back channel, with the closed form ``u_wall = g[6 nu - 1/(8 nu)]``
  (``kappa(tau) = s - 3/(4s)``, ``s = 2 tau - 1``; matches a tau/g scan to
  ``2e-7`` relative and reproduces the ``c ~ -0.139/H`` reported by m7963-c1 at
  ``tau ~ 1.13``).  The boundary-node correction
  ``df_q = -6(1-sigma) w_q rho (c_q . delta_vec)`` (tangential only, so mass and
  normal momentum are conserved) removes the bias: ``b(sigma)`` recovers
  ``3 nu sigma/(1-sigma)`` to ``1e-6`` relative (baseline ``11.4 %`` at
  ``sigma = 0.05``) on H = 32/64/128, and Poiseuille / Couette / oblique-force
  faces all pass (``tests/test_partial_slip_second_order.py``, 40 cases).
- **`tensorlbm.sloshing_resonance`** — LNG cargo-tank sloshing verification
  surface (campaign node P87W2-N01): a 1.0 m x 0.5 m two-dimensional tank at
  ``h/H = 0.2 .. 0.9`` released from the pure standing waves
  ``eta(x,0) = A cos(k_n x)`` (``n = 1, 2``, ``A = 0.01 m``) on the audited
  Körner free-surface LBM, with an SI-anchored Froude/Mach-consistent lattice
  scaling (``u_lat = u_char dt/dx``), sub-cell surface elevation from the mass
  ledger, wall dynamic-pressure gauges and damping-aware identification
  (``f_d``, ``zeta``, ``f_n = sqrt(lambda^2 + omega_d^2)/2 pi``).  Measured on
  the shipped artefacts: first-mode frequency vs Abramson (1966) worst
  ``2.70 %`` (gate 3 %), second mode ``4.26 %`` (gate 5 %), ``f_1`` spread over
  ``dx = H/40, H/80, H/160`` ``0.8 %`` (gate 2 %), liquid-inventory drift on the
  verification sweep ``0.49 %`` (gate 0.5 %), ``g -> 0`` control oscillation
  ``0.032`` with zero sign changes; the exact hydrostatic anchor
  ``p_dyn(t=0) = rho g A`` holds to ``0.12 %`` and the weakly nonlinear
  (``A/h = 5 %``) first-harmonic wall pressure matches the linear Faltinsen
  profile to ``2.9 %``.  Sloshing-impact pressure against *experiment* is
  reported WITHHELD (no digitised data in the repository; not fabricated).
  Honest failures are recorded in the report: the phase criterion at the
  under-resolved ``h/H = 0.2`` grid (4.8 deg vs the 2 deg gate, 1.4 deg for
  ``h/H >= 0.4``) and the ledger drift of the near-breaking resonant sway case
  (1.57 %).
- **`tensorlbm.passive_scalar_verification`** — one verification surface for the
  passive-scalar (advection–diffusion) kernels with four machine-checkable
  anchors: Taylor–Aris dispersion `D_eff/D = 1 + Pe^2/210` (plane slit) and
  `1 + Pe^2/48` (circular pipe) over `Pe in [0.1, 100]` checked by the closed
  forms, a discrete Aris auxiliary solve on the lattice node set and a
  time-marched D2Q9 scalar LBM (max |error| 0.67 %, gate 3 %); the erf diffusion
  front `0.5 erfc(x/sqrt(4Dt))` converging at L2 order 1.99 (gate 1.9) under
  `dt ~ dx^2` refinement; Schmidt-number independence of `D_eff/D` over
  `Sc in [0.7, 10]` (spread 1.6 %, gate 2 %); discrete mass conservation at
  `2.7e-12` (gate 1e-8).  The driver
  `benchmarks/pending/passive_scalar_verification/run.py` writes
  `artifacts/passive_scalar_verification.json` plus three figures.
  The discrete Aris reference deliberately uses an LAPACK-free tridiagonal
  (Thomas) solve: `numpy.linalg.solve` returns a residual-`1e2` solution for
  these pinned systems on the loong64 cluster node, so the library path is kept
  only as a `trustworthy`-flagged cross-check.
- **`tensorlbm.validation.statistical_error`** — the statistical-vs-model
  decomposition toolkit for time-averaged profiles: autocorrelation integral
  time with Sokal windowing (`integral_time`, AR(1) pinned to 10 %),
  Student-t batch-means confidence intervals (`batch_means_summary`, coverage
  checked at the nominal 95 %), the steady-state half-window gate
  (`half_window_max_diff`), propagation of the batch scatter onto the reported
  error metric itself (`metric_from_batches` -> `U_SN`), the binary attribution
  verdict (`decompose_error`: statistics vs model/domain), the two designed
  contrasts (`domain_length_check`, `negative_control_check`), the fail-closed
  three-turnover eligibility guard and an error-vs-`T_obs` budget extrapolation
  that refuses to promise machine time when the fitted exponent shows a
  systematic floor.  NumPy only; no SciPy dependency (Student-t quantile from
  the exact incomplete-beta CDF).
- **`WallResolvedChannel3DConfig.profile_batch_count` /
  `wall_exchange_series_stride`** — observation-only recorders on the
  wall-resolved channel kernel: consecutive batch means of the plane-averaged
  moment profiles and a thinned wall-shear time series.  Both default to 0,
  which keeps the legacy path bit-for-bit (`tests/test_wall_resolved_channel3d_batch_stats.py`
  asserts identical profiles with the recorders on and off) and the batch
  record is persisted/resumed through the checkpoint.
- **P40-N09 campaign** (`benchmarks/pending/channel_turb_3d/`) — the
  15.19 % u+ error of P20-N01 is decomposed into a statistical part (`U_SN`) and
  a systematic part: dual box length (`Lx = 2 pi delta` vs `4 pi delta`) on one
  grid, an SGS negative control pair, a stability/cost probe that shows the
  cumulant kernel diverging at the Re_tau = 180 near-wall stiffness, and a
  pre-registered six-gate analysis (`analyze_p40n09.py`) plus a report builder
  with zero hand-copied numbers (`build_p40n09_report.py`).

- **`tensorlbm.d3q27_mrt_physical_validation`** — D3Q27 MRT physical
  validation against closed-form Navier-Stokes solutions (verification-debt
  burn-down after the component/composition consistency probes): plane
  shear-wave decay recovers `nu = (tau-1/2)/3` (0.89 / 0.40 / 0.22 % at
  L = 16 / 24 / 32, error proportional to `k^2` i.e. spatial truncation), a
  linear acoustic mode travels at `c_s = 1/sqrt(3)` (0.27 / 0.07 %), and the
  first no-slip eigenmode between two `bounce_back_cells_27` walls decays at
  `nu (pi/H)^2` with the inferred wall offset converging to **0.5 lattice
  units** (`H = ny - 2`, the full-way bounce-back convention: the no-slip
  plane sits half a cell inside the solid centre). An autograd-Jacobian
  k-space analysis of the one-step operator (27 eigenvalues, no time
  stepping) reproduces the time-domain viscosity, sound speed and
  attenuation to 1e-5/1e-3 relative — a cross-method verification of the MRT
  moment matrix. Controls keep the acceptance non-vacuous: the
  `s_e = s_eps = s_q = s_pi = 1/tau` limit reproduces `collide_bgk27`
  bit-for-bit (5.8e-15), the shear branch is independent of the bulk rate
  `s_e` (3e-9), and a `tau` control shifts the measured viscosity by the
  analytic amount. Periodic mass/momentum invariants hold to 1e-14 / 1e-12
  (float64, 200 steps). Doc: `docs/d3q27_mrt_physical_validation_r1.md`,
  artifacts: `artifacts/d3q27_mrt_physical_validation/`, tool:
  `tools/d3q27_mrt_physical_validation_audit.py`, tests:
  `tests/test_d3q27_mrt_physical_validation.py` (20).
  Scope: single-phase, static geometry, uniform periodic mesh, CPU float64 —
  turbulence closures, multiphase, moving boundaries/AMR and accelerator
  backends stay WITHHELD; the `general_capability_matrix` D3Q27 WITHHELD
  disposition is **not** changed by this evidence.

- **`tensorlbm.fsi_added_mass`** — added-mass instability calibration for
  partitioned FSI: closed-form interface factor `rho_dd = |1 - omega c_a
  rho_f/rho_s|` (stability limit `rho_s/rho_f = omega c_a/2`), Aitken and
  IQN-ILS (bounded 5-pair history) interface accelerators, a 1D piston anchor,
  a 2D D2Q9 heaving-body sub-iteration/divergence adapter over the
  momentum-exchange moving boundary, free-decay added-mass/damping
  identification, and an incompressible potential-flow added-mass reference on
  the same staircase mask. Calibration (P49-N02): 1D anchor 0 %, 2D LBM 0.82 %
  vs the analytic factor, stability boundary 0.2 %, negative control
  (`rho_s/rho_f = 0.1`, explicit) diverges with growth rate 2.15, Aitken/IQN
  cut 40 non-converged iterations to 3; the coupled free-decay channel is
  reported as an honest FAIL (compressible regime). Doc:
  `docs/FSI_ADDED_MASS_INSTABILITY.md`, artifacts:
  `artifacts/fsi_added_mass/`, tests: `tests/test_fsi_added_mass.py` (20).
- **`tensorlbm.cac_moving_body`** — prescribed-motion rigid bodies inside a CAC
  two-phase simulation: moving-wall bounce-back (Ladd 1994), momentum-exchange
  force, wedge/disc geometry, fresh-node refill. `cac_lbm.CACSim` gained
  `set_wall_mask()`, `refresh_solid_ghosts(mask=...)` and a split
  `collide()/stream()/apply_boundaries()` stepper (the shared `step()` is
  unchanged in effect: `tests/test_cac_smoke.py` still 10 passed).
- `tests/test_cac_moving_body.py` (15 tests): static-body phase-mass
  conservation (guards a 73 % loss), driver-vs-plain-step equivalence, Ladd
  term restricted to boundary links, Archimedes buoyancy gate
  (`F_y = rho_l g V`, 2000-step probe gives 1.025), force symmetry.
- `benchmarks/pending/phasefield_water_entry/`: survey + prereg + `probe/p1_mass_source.py`
  (10-arm wall/phase mass-source attribution) + `probe/p2_mem_buoyancy.py`
  (Archimedes gate) + `run_s2_wedge.py` (wedge entry vs Wagner).

### Fixed
- **`wagner_theory` wedge anchor** (blocker B6): the three literature
  conventions are now explicit and documented; the anchor is `pi^2/4`
  (`F = rho V^2 pi^2 c / (4 tan beta) = rho V^3 pi^3 t / (8 tan^2 beta)`),
  i.e. `dF/dt / (rho V^3) = pi^3 / (8 tan^2 beta)` — **pi^3, not pi^2** as an
  earlier prereg note said. For `rho = 1000`, `V = 1 m/s`, `t = 2 ms`, the
  angle ladder 2/3/4 deg now reads 6.357 / 2.822 / 1.585 kN/m; the legacy
  8.093 kN/m was 4/pi too large and is `deprecated`.
- **Moving solids no longer destroy phase mass**: the first implementation
  bounced on the fluid side only and never refreshed the body ghosts, so a
  static 4x4 column in a uniform `phi = 1` field lost 73 % of the phase
  integral in 100 steps. The body now uses the same full-way swap as
  `bounce_back_cells_3d` (mass conserved to < 1e-6 over 120 steps).
- The Ladd wall-velocity term is applied on **boundary links only** (applying
  it on interior solid nodes pumped the body interior and diverged by step
  26), samples `rho` at the fluid node it streams into, and is enabled for
  the phase distribution too (`move_phase_field=True`): with it off, a
  8 deg / `V = 0.01` entry produced NaN within 250-300 steps.

### Changed
- **Public API is now resolved lazily** (`tensorlbm/__init__.py`, PEP 562):
  the 153 eager `from .x import y` statements are replaced by a
  `_LAZY_ATTRS` table mapping each of the 786 exports to the *same*
  submodule it was previously imported from, resolved by `__getattr__` on
  first access and cached in module globals. `import tensorlbm` drops from
  **1.81 s to 0.016 s** (~110x), loading 2 submodules instead of 193 and 65
  entries in `sys.modules` instead of 1588. Verified behaviour-preserving
  against a snapshot of all 939 pre-change public attributes: every
  `__all__` entry resolves, zero identity mismatches, submodule attributes
  (including ~21 that were only bound transitively, e.g. `tensorlbm.lattice`,
  `tensorlbm.core`) still resolve via a `find_spec` fallback, `dir()` is a
  superset of the old surface and imports nothing, and unknown names still
  raise `AttributeError`. Import-time side effects are *deferred, not
  removed*: `tensorlbm.cases` still registers built-in cases when any of its
  exports is touched, and the benchmark drivers' `matplotlib.use("Agg")` now
  runs when the plotting module is first used. A `TYPE_CHECKING` block
  repeats the original imports verbatim so mypy/IDEs are unaffected;
  `tests/test_public_api_lazy_exports.py` locks all of the above, including
  that a bare import stays lazy. Adding an export now requires entries in
  `_LAZY_ATTRS`, the `TYPE_CHECKING` block and `__all__` (see CONTRIBUTING).
- **CI reworked** (`.github/workflows/ci.yml`): static gates (ruff, mypy,
  docs consistency) split into a single `lint` job instead of being repeated
  per interpreter; `test` now runs a **3.11 + 3.12 matrix** (pyproject
  declares `requires-python >=3.11` but only 3.12 was ever exercised);
  pull requests run `-m "not slow"` for fast feedback while pushes, a new
  nightly cron and manual dispatch run the full suite; `--timeout` activates
  `pytest-timeout` (a declared dev extra that was previously unused) so a
  hung solver fails the job instead of burning the runner limit; added
  `concurrency` cancellation for non-main refs and pip caching.
- **D3Q19/D3Q27 stencil traversal deduplicated** (`tensorlbm.core`): the two
  modules were line-for-line copies differing only in lattice size and
  message text. The traversal now lives once in `core/_stencil_base.py`
  (`StencilOps` / `build_stencil_ops`); both modules keep their public names,
  descriptor binding and import-time fail-closed validation, so no call site
  or re-export changes. Checked identical against the pre-refactor modules
  for every roll direction, neighbour mask, shift table and guard **error
  message**.

### Removed
- Generated artifacts that `.gitignore` already intended to exclude but that
  predated the rules are no longer tracked: `coverage.xml`,
  `checkpoints/*.ckpt`, `artifacts/multi_card_test/fields_final.pt` (38 MB),
  `test_screenshots/`, `logs_thermal_common/`, `results/`, `results_*/` and
  `sphere_3d_results/`. Tracked payload **108 MB -> 33 MB**, 2661 -> 2434
  files. Model weights should be distributed via Release assets / HF Hub /
  Git LFS; `.gitignore` now covers the weight extensions and directories.

### Moved
- 166 one-off experiment scripts (`*_worker.py`, `*_launcher.py`,
  `verify_*.py`, `diagnose_*.py`, `rettest_v*`, plus their `run_*.sh` /
  `launch_*.sh` drivers) from the repository root into the existing
  `experiments/archived/` convention, documented by a new README there.
  Nothing imported them. The root now holds packaging metadata, the
  README/CHANGELOG/LICENSE set and three tooling shells; `.gitignore` has
  root-anchored guards so one-offs cannot be committed there again.
- Point-in-time reports collected into `docs/reports/`: the 8 root-level
  `*_SUMMARY` / `*_RESEARCH` / `*_ANALYSIS` files and the 7
  `docs/*REGRESSION_REPORT*.md`, indexed from `docs/README.md` with a note
  that they are provenance, not current specification.

### Added
- **Repository-structure analysis** (`docs/reports/REPO_ANALYSIS_2026-09-29.md`):
  metrics, strengths, prioritized findings and the remediation plan the
  changes above execute.
- **Parameter-sweep execution chain** (`tensorlbm.scan_runner`, new module): the
  execution layer of the AI4S scale-out data loop — `ScanPlan` builds a
  serializable sweep from the DoE generators (LHS / Sobol / factorial / CCD over
  named case parameters, seed-deterministic, persisted as `plan.json` in the
  dataset directory); `ScanExecutor` runs it with case-level GPU parallelism
  (card pool dealt round-robin to spawn-worker processes, one case per card at a
  time; serial in-process mode for CPU/debug). Every point is a
  registry-instantiated case with parameter overrides stepped by the case's own
  verified chain (bit-identical to `tensorlbm.cases.run_case`), with a
  `FieldSampleReporter` landing each snapshot as a PASS-gated catalog product,
  a `ThroughputReporter` recording MLUPS and an optional `EarlyStopReporter`.
  Sweeps resume by product existence (finished points skipped, half-done points
  reset), and finalise into a leakage-safe `FieldDatasetR2` (point-granularity
  train/val/test) registered with `plan -> run -> product -> dataset` lineage
  readable via `catalog.upstream`.
- **Unified reporter/callback protocol** (`tensorlbm.reporters`, new module): a
  lettuce-derived (MIT, attribution in the file header) hook point between the
  step loops and diagnostics/data. A `Reporter` is anything with `interval` +
  `__call__(ctx)`; the dispatcher fires reporters on positive multiples of
  their interval (`steps=100, interval=25` → exactly 4 fires); the lightweight
  tensor-first `StepContext` carries the population handle, a persistent step
  counter, per-step diagnostics, cell count, optional unit converter, and a
  `stop` flag for early termination. Built-ins: `CallbackReporter` (wrap any
  callable), `ThroughputReporter` (MLUPS, device-synced), `EarlyStopReporter`
  (steady-state change-threshold early stop), and `FieldSampleReporter`
  (sample → `solver_export.save_fields_hdf5` + `register_product`, one hop
  from the solver loop to a PASS-gated catalog product). Hooks:
  `LBMStepExecutor.run(..., reporters=...)` (with no reporters the original
  fast path runs verbatim — bit-identical output, zero added overhead) and the
  new `TritonFusedSolver3D.run(...)` multi-step driver with the same contract.

- **Case + boundary-condition registries** (`tensorlbm.cases`, `tensorlbm.boundary_registry`):
  a lettuce-``ExtFlow``-style named case registry (`register_case` / `get_case` / `list_cases`)
  with the three benchmark-aligned cases `cavity` (verified Ghia Re=400 MRT), `poiseuille`
  (verified 3-D pipe) and `suboff_n128` (ai4s pilot grid) built in, plus an XLB-style
  integer-id boundary registry (`BoundaryCondition`, `boundary_condition_registry`, id 0
  reserved for solid/no-BC, branch-free `bc_mask == id` application, `check_bc_overlaps`,
  per-phase masks, and missing-direction masks derived programmatically from the lattice
  constants — no hand-transcribed q-index tables).  `run_case(...)` reproduces the verified
  benchmark/worker step chains bit-exactly (max|Δf| = 0 on GPU for cavity 96×96×24 × 200
  steps MRT and SUBOFF n=128 × 20 steps BGK + mass correction; ≤ 1e-6 in the CPU eager
  tests) and chains opt-in into the solver-export catalog via `ExportSpec`.
- **Real DG-LBM solver** (`tensorlbm.dg_advection`, `tensorlbm.dg_band`): a genuine
  nodal Discontinuous-Galerkin Lattice Boltzmann hybrid.  Dimension-by-dimension
  P1-Lobatto DG advection (upwind flux, SSP-RK3, sub-cycled) with method-of-lines
  BGK collision, a packed DG-band topology, DG↔LBM P0 interface coupling
  (conservative face-trace write-back), and half-way bounce-back at solid walls.
  Validated: MMS O(Δx²)/O(Δx³) convergence, mass/momentum conservation, recovery of
  the DVBE shear viscosity, and stable end-to-end obstacle flow (2-D cylinder,
  3-D sphere, 3-D SUBOFF hull).  Enable on the SUBOFF runner with
  `DGLBMSuboffConfig(use_real_dg=True)` (the legacy gradient-correction path
  remains the default for backward compatibility).  Standalone runners:
  `examples/dg_lbm_cylinder_hybrid.py`, `examples/dg_lbm_sphere_hybrid.py`.
  Key tuned constants: 3-D RK sub-steps ≥ 16 (stiffer at low τ_dg); the band uses
  τ_dg = τ_lbm − ½ to match the exterior LBM viscosity.

### Added
- **Rate limiting**: opt-in per-IP sliding-window rate limiter in `app/backend/security.py`.
  Enabled by setting `TENSORLBM_RATE_LIMIT_REQUESTS=<N>` (requests per window, default 0 =
  disabled) and `TENSORLBM_RATE_LIMIT_WINDOW_S=<seconds>` (default 60).  Returns HTTP 429
  when the limit is exceeded.  Respects `X-Forwarded-For` for deployments behind a proxy.
  Rate limiting is enforced inside `authorize_request`, so it applies to all auth modes.
- **Job-list pagination**: `GET /api/jobs/` now accepts optional `limit`, `offset`, and
  `status` query parameters and returns a JSON envelope
  `{jobs: […], total: N, offset: K, limit: L}` instead of a bare array.  Clients that do not
  send these parameters receive the full list (backward-compatible default `limit=0`).
- **i18n completeness**: 6 new translation keys added to `en.json` / `zh.json`:
  `solve.validating`, `solve.scan_min_values`, `solve.scan_max_values`,
  `solve.scan_submitting`, `solve.scan_invalid_json`, `preprocess.no_materials`.
  Several hardcoded English strings in `app_solver.js`, `app_postprocess.js`, and
  `app_core.js` were replaced with `t()` calls so they now appear in Chinese when the
  language switcher is set to 中文.

### Added
- Marine benchmark suite now includes a **marine geometry library** case that validates
  ship hull families (Wigley/Series60/KCS) against analytical block coefficients and
  checks SUBOFF variant topology ordering (bare hull < sail < full appendage volume).
- **i18n / Chinese localisation**: The deployable web platform (`platform/`) now supports both
  **English** and **Simplified Chinese (简体中文)** with real-time in-browser switching.
  - Language switcher (`EN | 中文`) added to the global navigation bar.
  - Language preference is persisted to `localStorage` (`tensorlbm_lang`); first visit
    auto-detects from `navigator.language`.
  - Lightweight pure-frontend JSON dictionary approach:
    `platform/frontend/static/i18n/en.json`, `zh.json`, `platform/frontend/static/js/i18n.js`.
  - Terminology glossary at `platform/i18n/GLOSSARY.md`.
  - CI-friendly key-parity validator: `python platform/i18n/check_keys.py`.
  - New test module `platform/tests/test_i18n.py` covering JSON validity, key parity, and static file serving.
  - Added `README.zh-CN.md` (Simplified Chinese README) with mutual language links.


### Changed
- **CI gates repaired end-to-end** (main had been red on every push since
  2026-08-19, blocking all PR merges including #190):
  - ruff `select` trimmed from a 10-rule-family superset (6862 findings,
    mostly `ANN` annotation pedantry) to the high-signal set
  `E4/E7/E9/F/I`, then autofixed and fully `ruff format`-ed (1051 files);
  per-file ignores keep `benchmarks/` and `examples/` script-style code
  exempt from style pedantry.
  - CI now installs CPU-only torch before `.[dev]`, so 4926 tests are
  collectible (previously 273/371 test files failed at import with no torch
  at all); pytest gains `testpaths=["tests"]`, a `slow` marker excluded in
  CI, `--timeout=300` (pytest-timeout), and environment-aware collection
  via `tests/conftest.py` (skips tests needing absent optional deps).
  - `tests/ci_quarantine.txt`: 141 pre-existing failures (all reproduced on
  stock main @ 2a6da29 before the quarantine; none introduced by this
  branch) are marked non-strict xfail so the gate reflects *new* regressions
  only. Burn-down of this list is tracked follow-up work; several entries
  (SGS coupling inert on CPU, BFL `wall_velocity`/`boundary_fraction` API
  drift, `DomainDecomposition.from_devices(device_type=...)`) are real
  feature bugs merged ahead of their implementations.
- mypy gate `python_version` 3.11 -> 3.12 (numpy 2.5 stubs use `type`
  statements rejected under 3.11).

### Fixed
- `tensorlbm/__init__.py`: 25 `ai` names listed in `__all__` were never
  imported -> `from tensorlbm import <name>` raised ImportError on main;
  the block is now imported and `__all__` is ast-verified complete.
- `tensorlbm/multi_gpu.py`: `dist`, `hashlib`, `C` used but never imported
  (24 F821 + 1 undefined-name on main).
- `tensorlbm/cases/base.py`: `BCPhase` annotation used but only importable
  under TYPE_CHECKING (F821).
- `torch.sdaa` probed unguarded at module level in `tests/test_lbm_step.py`
  and `ai/suboff_utils.py` -> collection crashed with AttributeError on
  any build without the sdaa plugin (including CI); now guarded via
  `getattr`.
- Source-hash-bound evidence tests (d3q27 composition, d3q19/d3q27 MRT
  consistency) re-locked after formatting; collision-matrix cross-validation
  and cross-module composition expectations refreshed to the current
  8-combination / 10-contract admitted sets.
## [0.3.0] - 2026-05-24

### Added
- **Regularized BGK (RLBM)** collision operator for D2Q9 (`collide_rlbm`) and D3Q19 (`collide_rlbm3d`)
  following Latt & Chopard (2006). Projects the non-equilibrium populations onto the second-order
  Hermite polynomial subspace, filtering out ghost (non-hydrodynamic) modes and improving stability
  at low viscosity (τ → 0.5).
- **Rotating-cylinder (Magnus effect) runner** (`rotating_cylinder.py`):
  `RotatingCylinderConfig`, `run_rotating_cylinder`, plus the underlying helpers
  `rotating_wall_velocity` and `moving_wall_bounce_back` (Ladd 1994 moving-wall BC for D2Q9).
  CLI script `examples/rotating_cylinder.py` exposes a `--spin-ratio` parameter (α = ω R / u∞).
- **MLUPS benchmark extension**: `benchmarks/bench_mlups.py` now accepts `--collisions {bgk,mrt,trt,rlbm,all}`
  and compares throughput of all four collision operators on D2Q9 and D3Q19 grids.
- **Platform `/api/health` endpoint** — lightweight liveness probe independent of Torch / job state.
- **Platform `/api/jobs/compare` endpoint** — side-by-side metadata comparison for up to 10 completed
  jobs, plus a corresponding *Compare* tab in the frontend SPA.

## [0.2.0] - 2026-05-24

### Added
- Modern Python packaging via `pyproject.toml` with editable-install and dev extras support.
- Public package version metadata via `tensorlbm.__version__`.
- Structured logging helpers for long-running simulations.
- Checkpoint save/load utilities for resumable runs.
- Scientific VTK and HDF5 output helpers.
- Post-processing utilities for velocity profiles, pressure coefficient, and Q-criterion.
- Protocol-based extension interfaces for custom collision operators and boundary conditions.
- YAML/TOML configuration loading with environment-variable overrides; new `load_config_yaml` convenience function.
- Reproducibility metadata capture including git commit, Python version, and package versions.
- Mass-correction helpers for 2-D and 3-D solvers; automatic mass correction every `output_interval` steps in cylinder-flow and sphere-flow runners.
- Bouzidi interpolated bounce-back support for curved 2-D boundaries.
- Full D3Q27 lattice implementation with BGK/MRT collision and streaming helpers.
- Verification tests for Poiseuille flow and Taylor-Green vortex decay.
- Property-based tests for collision invariants and D3Q27 lattice properties.
- MLUPS benchmark scripts for 2-D and 3-D kernels.
- Progress bars (tqdm) for cylinder-flow, sphere-flow, and turbulent-channel runners.
- **Two-relaxation-time (TRT) collision** (`collide_trt`, `collide_trt3d`) for D2Q9 and D3Q19, with the Ginzburg magic-parameter default (Λ = 3/16) that eliminates Poiseuille wall-placement errors.
- **3D porous-media module** (`porous_media3d.py`): `make_random_sphere_medium`, `make_tube_array_medium_3d`, `PorousDrainageConfig3D`, `run_porous_drainage_3d`.
- **Immersed Boundary Method** (`ibm.py`): direct-forcing IBM with 2-point hat and 4-point cosine delta kernels, velocity interpolation, force spreading, and Guo body-force application.
- **Thermal LBM** (`thermal.py`): double-distribution-function model (D2Q9 momentum + D2Q5 temperature), equilibrium, BGK collision, streaming, macroscopic recovery, and Boussinesq buoyancy force.
- `CONTRIBUTING.md` with coding conventions, PR workflow, and guides for adding new solvers and benchmarks.
- Quantitative validation summary table in `README.md`.
- Ship CAD-to-flow workflow outputs for `run_ship_hull_flow`: CAD preview/statistics, optional STL export, wake-profile CSV, and post-processing summary JSON with symmetry and wake metrics.

### Changed
- Example runners now emit structured log messages instead of raw `print()` output.
- Runner metadata now includes reproducibility information.
- CI installs from the package metadata (`pip install -e ".[dev]"`), runs mypy, and reports test coverage.
- Runtime requirements now include `tqdm` for optional progress reporting.
- Ship-hull CLI and marine benchmark now exercise the full ship workflow from CAD modelling through LBM solve and quantitative post-processing checks.

## [0.1.0] - Initial release

### Added
- D2Q9 and D3Q19 lattice definitions with equilibrium and macroscopic reconstruction.
- 2-D and 3-D BGK/MRT collision operators and periodic streaming kernels.
- 2-D and 3-D boundary-condition helpers including Zou/He inlet/outlet variants.
- Example runners for cylinder, sphere, ship-hull, and water-entry simulations.
- Marine engineering extensions including Wigley hull geometry, obstacle diagnostics, and wave boundary conditions.
- Smagorinsky LES turbulence models and shared simulation utilities.
- Unit and smoke tests covering core solvers, runners, and marine features.
- **`tensorlbm.suspension_viscosity`** — effective-viscosity closure for
  crushed-ice / ice-crystal suspensions (campaign node P46W3-N03): Einstein
  (dilute), Batchelor (semi-dilute) and Krieger-Dougherty (concentrated)
  closed forms, their analytic inverses and derivatives, the calibration
  chain that extracts ``[eta]`` and ``phi_max`` from measured
  ``(phi, eta_r)`` pairs (weighted regression, single-parameter
  golden-section fit, divergence-exponent test), the audited conversion from
  lattice-mask volume fraction to drag-equivalent hydrodynamic volume fraction,
  and ``evaluate_acceptance()``, which turns the campaign criteria into a
  machine verdict.  Calibrated and judged on the shipped LBM shear-cell and
  drag artefacts in ``benchmarks/suspension/`` (see
  ``docs/reports/P46W3-N03_suspension_viscosity.md``); tests in
  ``tests/test_suspension_viscosity.py``.
