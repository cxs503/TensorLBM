# Static contact-angle benchmark — `tests/benchmarks/wetting_contact_angle/`

**What it is.** A reproducible instrument that answers one question: *given the
prescribed equilibrium contact angle ``theta_eq``, what angle does the library's
SCMP wetting wall force actually produce after relaxation?* It sweeps a
pre-registered angle ladder over two grid resolutions and evaluates
``|theta_measured - theta_prescribed| < 5 deg`` (``ACCEPT_DEG`` in
`wetting_bench.py`) for two competing prescription laws — the analytic Young
relation and a measured calibration used as a coarse→fine hold-out.

**Anchor / consumer.** `GAP-LBM-MCMP-POISEUILLE` needs a wetting criterion (a
wall that can be set to a known contact angle) before layered two-phase channel
flows can be trusted. This benchmark is that criterion for the single-component
(SCMP) path, and it registers the missing half (MCMP) as a gap — see §4.

**Verdict (from `results/REPORT.md`, generated — no number is typed here).**
Both laws fail the 5° criterion on the current wall force. The report carries
the per-case table, the estimator control, the grid-sensitivity pair, the
metastability probe and the root-cause list. This is an **honest FAIL**: the
criterion was not relaxed, no case was dropped, and the failing numbers are the
deliverable (they tell the downstream node that the knob must be calibrated
per-resolution and that a unique static angle does not exist at the 10° level).

## 1. Method in one page

| ingredient | choice | why |
|---|---|---|
| initial state | *exact* circular segment of the prescribed angle (`cap_init`) | starting from a 90° blob leaves the measured angle drifting for >10^4 steps |
| wall force | SCMP pseudopotential wall, ``psi_w = f_w * psi(rho_l)`` | the only wetting knob the library exposes |
| prescription A | Young relation ``cos(theta) = (psi_w-psi_v)/(psi_l-psi_v)`` | analytic, zero fitted parameters — the law the library documents |
| prescription B | piecewise-linear inverse of the *measured* ``theta(f_w)`` curve, fitted on the coarse grid, verified on the fine grid | a hold-out: it cannot pass by construction |
| measurement | liquid-content area `A` + apex height `h` → ``theta = g^-1(A/h^2)`` (primary), whole-cap circle fit + near-wall band fit (cross-checks) | integral observables survive the one-cell wall depletion layer; the three-way spread is the uncertainty floor |
| acceptance | ``|Δθ| < 5°``; convergence gate ``drift < 1°`` over the last 25 % | pre-registered before the sweep |
| angle convention | **physical** (inside the liquid) | the library's contour estimators return ``180 - θ_physical``; converted in `contact_angle.py` and pinned by a test |

## 2. Files

| file | role |
|---|---|
| `wetting_bench.py` | prescriptions (`wall_fraction_for_angle`, `calibration_curve`, `invert_calibration`), one relaxed case (`run_case`), grids, provenance |
| `contact_angle.py` | the three estimators + the analytic cap generator used as a control |
| `run_case_cli.py` | one case per OS process (the sweep runs cases in parallel processes) |
| `run_all.sh` | stage fan-out (`a`, `b1`, `b2`) with `xargs -P` |
| `assemble.py` | per-case JSON → artifact (`results/contact_angle_sweep.json`) + hold-out plan |
| `run_metastability.py` | stage C: same ``f_w``, two initial caps |
| `make_report.py` | figures + `results/REPORT.md` (tables injected from the JSON) |
| `test_wetting_contact_angle.py` | estimator self-validation, convention/limit pins, smoke relaxation, artifact gate, zero-hand-copy gate |
| `results/` | artifact, per-case records, figures, generated report |

Reproduce (minutes; each case is one 2-D drop):

```bash
export PYTHONPATH=$PWD/src
PAR=5 THREADS=6 bash tests/benchmarks/wetting_contact_angle/run_all.sh a
PAR=5 THREADS=6 bash tests/benchmarks/wetting_contact_angle/run_all.sh b1
python tests/benchmarks/wetting_contact_angle/assemble.py --plan
bash  tests/benchmarks/wetting_contact_angle/run_all.sh b2
python tests/benchmarks/wetting_contact_angle/run_metastability.py --fw 0.52
python tests/benchmarks/wetting_contact_angle/assemble.py
python tests/benchmarks/wetting_contact_angle/make_report.py
python -m pytest tests/benchmarks/wetting_contact_angle -q
```

## 3. Consumption chain (library practice → this harness)

Each row is a concrete piece of the source repository this harness reuses; the
line numbers are the ones the sweep ran against
(`git rev-parse HEAD` is recorded in the artifact's `meta.git_rev`).

| # | source practice (repo + file:line) | what it does there | where this harness consumes it |
|---|---|---|---|
| 1 | `TensorLBM src/tensorlbm/multiphase.py:335` `_apply_wall_psi` + `:651` `sc_single_component_force` | substitutes the wall pseudopotential into the SC force sum on solid cells — i.e. defines the wetting wall force | the knob under test: `wetting_bench.py:201` `run_case` installs ``wall_fraction`` on the stepper; `wetting_bench.py:158` `wall_fraction_for_angle` converts a prescribed angle to it |
| 2 | `TensorLBM src/tensorlbm/multiphase.py:732` `collide_sc_single_component` | SCMP collision + interaction force + forcing scheme (`edm`) | every relaxation step: `run_case` drives `WallStepper.run` (see row 4), whose `_raw_step` calls this kernel |
| 3 | `TensorLBM src/tensorlbm/multiphase.py:366` `_sc_neighbor_weighted_sum` | the backward-gather ψ stencil that the wall substitution feeds | implicitly the object of rows 1/2; the *reason* the effective wall strength differs from ``psi_w`` and the reason the Young law fails (see `results/REPORT.md` §6) |
| 4 | `TensorLBM src/tensorlbm/contact_line/engine.py:269` `WallStepper` (half-way bounce-back + wall ψ + `run`) | packages a wall-bounded SCMP stepper with a uniform/gradient wall fraction | `wetting_bench.py:201` `run_case` uses it as the time integrator and reads `spurious_current`, `rho_mid`, `psi_l` from it |
| 5 | `TensorLBM src/tensorlbm/contact_line/engine.py:185` `sessile_drop_state` (with `height`) | tanh-profile drop on the wall; ``height`` places the circle centre | `wetting_bench.py:184` `cap_init` passes ``height = -R cos(theta)`` so the initial state is the *exact* segment (physical convention) |
| 6 | `TensorLBM src/tensorlbm/contact_line/theta_fit.py:125/187/248/304` `interface_points` / `circle_fit` / `contact_angle_from_circle` / `angle_estimate` | level-set extraction and the two contour angle estimators | `contact_angle.py:185` `measure_contact_angle` calls them for the ``circle`` and ``band`` cross-checks and converts their supplement convention (`contact_angle.py:74` `LIB_IS_SUPPLEMENT`) |
| 7 | `TensorLBM src/tensorlbm/wetting/metrics.py:187` `spherical_cap_inversion` | the existing 2-D cap inversion (mask-based area + contact width) on a phase field | not called directly: its mask-based contact width returns NaN on the wall depletion layer, so `contact_angle.py:119` `cap_inversion` replaces the width observable with the **apex height** (an integral that does not touch the wall layer) and is validated against the same analytic caps |
| 8 | `TensorLBM src/tensorlbm/contact_line/engine.py:83` `RHO_L_DEFAULT`/`RHO_V_DEFAULT` (+ `psi`) | coexistence pair the SCMP kernels are calibrated for | `wetting_bench.py:145`–`:174` evaluate ``psi_l = psi(rho_l)``, ``psi_v = psi(rho_v)`` from the library itself, so the Young law and the reachable-angle bound carry no fitted parameter |

## 4. Scope, gaps, and limitations

* **MCMP wetting wall force is NOT_SUPPORTED** by the library:
  `src/tensorlbm/multiphase.py:495` `sc_two_component_force` accepts only
  `solid_mask` (which zeroes solid densities → a *dry*, fully non-wetting wall);
  there is no wall-density / wall-ψ parameter on the two-component path
  (`:538` `collide_sc_two_component` likewise). The benchmark therefore covers
  SCMP only; adding an MCMP wall-density knob is the next step for the
  `GAP-LBM-MCMP-POISEUILLE` wetting criterion.
* **Angles beyond ``theta_max`` of the Young law are unrepresentable** with a
  non-negative wall pseudopotential (``psi_w >= 0``): the reachable bound is
  evaluated by `reachable_angle_limit_deg` and the 110° ladder point is reported
  as `OUT_OF_RANGE` rather than silently clamped.
* **Strong wetting is unmeasurable with this estimator** (``f_w >= 0.65`` on the
  coarse grid): the drop degenerates into a film and the area+apex inversion
  returns NaN (`NO_INTERFACE`); the case is recorded as such, not dropped.
* **The settled angle is history-dependent** (metastability probe, stage C), so
  every measured angle must be read as "the angle that survives relaxation from
  this initial cap"; the report states the measured split.
* **The three contour estimators disagree by up to ~25°** in the mid-range
  (`estimator_spread` per case): the drop is not a clean circular cap there
  (wall depletion layer + foot), which is precisely why the primary estimator is
  an integral one and why the spread is published instead of hidden.

## 5. Zero hand-copy policy

`results/contact_angle_sweep.json` is the single source of numbers;
`results/REPORT.md` and the three figures are generated from it by
`make_report.py`. `test_report_tables_are_generated_not_typed` re-renders every
table from the JSON and asserts it appears **verbatim** in the report, and
`test_artifact_status_matches_criterion` re-derives every `PASS/FAIL` from the
recorded ``theta_measured``/``theta_eq`` pair.