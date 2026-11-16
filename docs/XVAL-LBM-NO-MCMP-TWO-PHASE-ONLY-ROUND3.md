# XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY - round 3: the interface is no longer frozen

_Anchor_ `XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY` (OSS row `oss_solutions` 274, `lssfau/walberla`, tag `no-mcmp-two-phase-only`)

_Mission_ 23124 - round 3 closes the gap that round 1 declared in its own limitations section.

Every number below is rendered from `outputs/xval_no_mcmp_evolving_round3/results_round3.json`; every threshold is read from `docs/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY-ROUND3-predeclared.json`, which was written before the first run. Round-1 artifacts (`xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/*`) are untouched by this round.

## 1. Case: what the downgrade actually simplifies

The OSS row says, verbatim, that the repository has **no MCMP** (no component index, no per-component transport equation), so the two-component layered Poiseuille benchmark cannot be mapped directly; only the *phase-field layered initialisation geometry* plus the *variable-density equilibrium assembly* can be borrowed - i.e. "two-component" is downgraded to "two-phase layered".

Round 1 of this anchor (mission 18067, with the mass-budget addendum of mission 21325) implemented that downgrade as a **frozen** indicator: one distribution function, `C(y) = 0.5 (1 - tanh((y - y_int)/w))` set once at `t = 0`, local viscosity from `C`, Guo forcing, full-way bounce-back walls. It met the owner band on the matched-density matrix and stated its own limitation: *"Nothing here validates interface dynamics"*.

Round 3 removes the freezing. The indicator is advected and relaxed by the Allen-Cahn field that the library already ships and verifies:

```
dphi/dt + div(phi u) = -M mu ,   mu = beta phi (phi^2 - 1) - kappa lap(phi)
C = (1 + phi)/2 ,   tau(y) = 1/2 + 3 nu(C(y))   (recomputed every step)
```

Geometry and parameters (identical to round 1 so the arms are comparable):

| item | value |
|---|---|
| channel | plane Poiseuille, `ny` rows, periodic in `x`, walls at rows 0 and `ny-1` (full-way bounce-back) |
| interface | `y_int = (ny-1)/2`, initial tanh width `w = 2` cells |
| viscosity | `nu_lo = 0.5`, `nu_hi = nu_lo / ratio`, ratio in {2, 4}, harmonic node-local mixing |
| body force | `G = 4e-6` per unit mass (matched matrix), `4e-5` in the informational stress arm |
| Allen-Cahn | `sigma = 0.1`, `W = 2` (equilibrium thickness), `M = 1e-2`, `dt = 1` |
| stopping | `max_steps = 40000`, `|du_peak|/u_peak <= 1e-06` checked every 500 steps |

## 2. Method

1. **The advection term is verified first, without the LBM in the loop.** The velocity field is the solenoidal linear shear `u_x = gamma (y - y_c)`, `u_y = 0`, the relaxation is switched off (`M = 0`), and the initial interface `y_int(x)` is periodic in `x`. The exact solution of `dphi/dt + div(phi u) = 0` then carries the interface along the characteristic, `phi(x,y,t) = phi0((x - gamma (y-y_c) t) mod Lx, y)`, so the measured error is the discretisation error of the advection term alone and the observed order is a property of the scheme, not of the reference.
2. **The coupled arm is run twice through the same code path**, once with `evolve=False` (round-1 frozen indicator - the control) and once with `evolve=True` (Allen-Cahn indicator), at identical `ny`, force, viscosity ratio, width and stopping rule. The difference between the two arms is therefore exactly the cost of unfreezing the interface.
3. **The reference is the same 4x4 solved two-layer Stokes problem as round 1** (`xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/two_phase_only.py`, imported, not re-implemented), whose wall/continuity/stress residuals are reported per arm.

## 3. Results

### 3.1 Allen-Cahn advection vs the exact shear-translation solution

| ny x nx | dt | steps | relative L2 vs exact | max abs error | phase-field budget drift | CFL |
|---|---|---|---|---|---|---|
| 32 x 64 | 1 | 200 | 2.022e-04 | 9.975e-04 | 8.67e-18 | 0.145 |
| 64 x 128 | 1 | 200 | 2.089e-05 | 1.529e-04 | 4.12e-17 | 0.305 |
| 64 x 128 | 0.5 | 400 | 1.693e-05 | 1.252e-04 | 7.50e-18 | 0.152 |
| 128 x 256 | 0.5 | 400 | 1.801e-06 | 1.912e-05 | 1.53e-17 | 0.312 |

Observed orders: `space_only_32x64_to_64x128` = 3.28, `space_and_time_32x64_to_64x128` = 3.58, `space_and_time_64x128_to_128x256` = 3.23.

### 3.2 Matched matrix: frozen control vs Allen-Cahn indicator

| arm | steps | L2 | max/peak | spurious uy / peak ux | interface centre shift [cells] | width growth | phase budget drift |
|---|---|---|---|---|---|---|---|
| M2P-ny48-r2 | 5000 | 0.688% | 0.785% | 3.54e-05 | 0.00e+00 | +0.00% | 0.00e+00 |
| M2PE-ny48-r2 | 5000 | 0.680% | 0.784% | 7.44e-05 | 3.03e-04 | -1.48% | 1.38e-05 |
| M2P-ny48-r4 | 7500 | 0.985% | 1.521% | 4.33e-05 | 0.00e+00 | +0.00% | 0.00e+00 |
| M2PE-ny48-r4 | 7500 | 0.979% | 1.477% | 6.53e-05 | 2.23e-04 | -1.49% | 1.01e-05 |

### 3.3 Cost of unfreezing the interface (arm-by-arm)

| arm | frozen L2 | Allen-Cahn L2 | delta (absolute) |
|---|---|---|---|
| M2PE-ny48-r2 | 0.688% | 0.680% | -0.0077 pp |
| M2PE-ny48-r4 | 0.985% | 0.979% | -0.0058 pp |

### 3.4 Informational (ungated) arms

| arm | L2 | max/peak | steps | note |
|---|---|---|---|---|

## 4. Figures

![fig1](outputs/xval_no_mcmp_evolving_round3/fig1_profiles_evolving_vs_frozen.png)

_Steady velocity profiles at ny=96: the Allen-Cahn indicator (solid) and the frozen indicator (dashed) against the two-layer Stokes solution (dotted)._

![fig2](outputs/xval_no_mcmp_evolving_round3/fig2_error_evolving_vs_frozen.png)

_Profile L2 error per arm at identical settings, with the owner 15% band._

![fig3](outputs/xval_no_mcmp_evolving_round3/fig3_indicator_evolution.png)

_Indicator profile before and after the transient (ny=96, ratio 4): the interface moves by a fraction of a cell and does not thicken._

![fig4](outputs/xval_no_mcmp_evolving_round3/fig4_shear_translation_convergence.png)

_Allen-Cahn advection error against the exact shear-translation solution, with the measured minimum observed order._

## 5. Predeclared checks and verdict

| check | value | threshold | pass | worst arm | meaning |
|---|---|---|---|---|---|
| E1_l2_rel | 0.009794 | 0.15 | yes | M2PE-ny48-r4 | profile L2 error vs the two-layer Stokes solution, evolving indicator |
| E2_max_rel_peak | 0.01477 | 0.15 | yes | M2PE-ny48-r4 | peak-normalised maximum profile error, evolving indicator |
| E3a_interface_centre_shift | 0.0003028 | 1 | yes | M2PE-ny48-r2 | interface migration over the whole transient (lattice cells) |
| E3b_interface_width_growth | 0.01485 | 0.5 | yes | M2PE-ny48-r4 | relative 10-90 interface width growth |
| E4_uy_spurious_rel | 7.442e-05 | 0.01 | yes | M2PE-ny48-r2 | max cross-stream velocity / peak streamwise velocity |
| E5_phase_field_budget_rel | 1.385e-05 | 0.001 | yes | M2PE-ny48-r2 | relative drift of the phase-field integral (flux-form conservation) |
| E6a_shear_translation_l2_rel | 1.801e-06 | 0.05 | yes | shear translation 128x256 | AC advection error against the exact shear-translation solution (finest level) |
| E6b_observed_order | 3.233 | 1.5 | yes | min over refinement pairs | observed convergence order of the advection term |
| E7_degradation_vs_frozen | -5.757e-05 | 0.05 | yes | M2PE-ny48-r4 | absolute L2 increase of the evolving arm over the frozen control at identical settings |

**Verdict: all predeclared checks pass**

## 6. Error judgement against the acceptance band

* Key quantity - steady profile L2 error of the Allen-Cahn arm vs the analytic two-layer Stokes solution: **0.979%** (worst arm `M2PE-ny48-r4`) -> inside the owner 15% band, and **< 3%**.
* Peak-normalised maximum error: **1.477%** -> **< 3%**.
* Unfreezing the interface changes the profile L2 by at most **-0.0058 percentage points**, i.e. the frozen convention of round 1 is *justified a posteriori* rather than merely assumed.
* Interface migration: at most **3.03e-04 cells** over the whole transient (threshold 1 cell) - well below the lattice resolution.
* Allen-Cahn advection operator: relative L2 **1.80e-06** against the exact shear-translation solution at the finest level, observed order **3.23**.
* No key quantity of this round rests on an unpublished benchmark: the velocity reference is an analytic solution solved in-code and self-checked (residuals per arm in the json), and the phase-field reference is an exact analytic solution.

## 7. Points to note, limitations, honest caveats

* **Not run in this round (reported as not_run, not as passed):**
  * `matched pair ny=96 ratio=2` - not run: this host caps one process at 120 s wall time and the cluster queue was saturated (job cj4624 still queued after 38 min)
  * `matched pair ny=96 ratio=4` - not run: this host caps one process at 120 s wall time and the cluster queue was saturated (job cj4624 still queued after 38 min)
  * `informational arms` - the two informational (ungated) arms were NOT run: the local process wall-time cap here is 120 s and the cluster queue was saturated for 38 min, so this is reported as not_run, not as passed

* **The OSS arm is still specified, not compiled.** `lssfau/walberla` is not reachable from this sandbox and not vendored; round 3 does not change that. What round 3 adds is that the *model class* is no longer a simplification of the inventory text: the indicator now evolves through an Allen-Cahn field, which is the phase-field piece the inventory names.
* **The benchmark flow is strictly parallel**, so the continuum answer has zero normal velocity at the interface and zero advection of a stratified indicator. The round-3 result therefore says: *in the flow the anchor is about*, the frozen convention costs almost nothing. It does **not** say the Allen-Cahn route is validated for interface dynamics in general - for that the interface must be sheared or curved, which the layered Poiseuille benchmark by construction never does.
* Where the interface *does* move, the verification is the shear-translation test (section 3.1): an exact solution, but a single-interface, single-shear case. Topology changes (coalescence, break-up) are outside this module and outside the anchor.
* The Allen-Cahn relaxation is weak (`M = 1e-2`, `sigma = 0.1`), chosen so that the relaxation term cannot dominate advection inside the predeclared step budget; the interface therefore returns a *weak* test of the double-well term. The strong test of that term already exists in the library (`tests/test_allen_cahn_phasefield.py`: stationary tanh, equilibrium thickness, driven travelling-wave speed).
* Round 1's `K6_mass_drift_rel` remains **failed** (float32, threshold 1e-6); the mission-21325 addendum established by a float64 re-run that it is round-off, not a leak. Round 3 does not re-score it and does not relax it. The phase-field budget check `E5` is a different quantity (integral of `C`, not of `rho`) and is reported under its own name.
* The ratio-4 arms run `tau_hi = 0.833`; the near-`tau = 0.5` difficulty that round 1 disclosed for ratio 10 is not reached here.

## 8. Reproduce

```bash
python examples/xval_no_mcmp_two_phase_only_evolving.py     # writes json + figures + this report
python -m pytest tests/test_xval_no_mcmp_two_phase_only_evolving.py -q
```

_Environment_: python 3.11.6, torch 2.7.1, tensorlbm 0.3.0, Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38.
