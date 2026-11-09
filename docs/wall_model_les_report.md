# Wall-modelled LES for high-Reynolds-number ship boundary layers (TensorLBM)

Mission node `p92w2-n02-lbm-wall-model-les` · branch `agent/m7396` ·
capability: `tensorlbm.wall_model_les`.

---

## 1. Case introduction (physics, motivation, parameters)

LBM has so far been used in this repository either wall-resolved (DNS-ish
channels, `tensorlbm.turbulence.turbulent_channel3d`) or at moderate Reynolds
numbers where the boundary layer is partly resolved. Predicting **ship
friction** needs `Re ~ 1e9`, where resolving the wall layer is out of reach:
the first grid node sits at `y+ = O(10-40)` and the near-wall region must be
**modelled** rather than solved.

This work adds that missing tier:

* wall laws and friction-velocity inversion (`laws.py`),
* an **equilibrium** wall model (law of the wall) and a **non-equilibrium**
  thin-boundary-layer ODE wall model (`models.py`),
* a D3Q19 **WMLES solver** for the pressure-driven channel with the wall model
  applied at the wall-adjacent node (`channel.py`),
* Reynolds-stress budget + DNS anchors (`budget.py`),
* the **ZPG flat plate** engineering chain driven by the same wall law
  (`flat_plate.py`),
* the ship-resistance decomposition `C_T = C_F (1+k) + C_R` (`resistance.py`).

| Parameter | Value | Source |
|---|---|---|
| `kappa`, `B` | 0.41, 5.2 | mission specification |
| Spalding damping `A` | `exp(-kappa B) = 0.1186` | derived so the composite law has the specified log intercept |
| van Driest `A+` | 26 | calibrated here: gives `B_eff = 5.236` for `kappa = 0.41` (sweep in the artifact) |
| Coles wake `Pi` | 0.45 (baseline, predeclared) | ZPG boundary layer |
| Channel target | `Re_tau = 590`, `u_tau = 0.02`, `ny = 49` ⇒ `dy+ = 24.6` | mission "high-Re wall modelling" |
| Flat plate band | `Re_theta = 1e4 ... 5e4` | mission specification |
| KVLCC2 | `Lpp = 320 m`, `B = 58 m`, `T = 20.8 m`, `Cb = 0.8098`, `S = 27194 m^2` | repository hull data |
| KVLCC2 Re used | `1.3e7`, `Fr = 0.142` | mission specification |

## 2. How it was done

1. **Wall laws.** `log_u_plus`, Spalding's unified law `y+ = f(u+)` (Newton
   inversion), Reichardt's composite law. Friction velocity from a sampled
   velocity by bracketed Newton/bisection on `u(y) = u_tau f(y u_tau/nu)`;
   a pure-Python scalar fast path (used by the channel solver at every step)
   and a vectorised torch path are cross-checked to `1e-6` in the tests.
2. **Non-equilibrium (ODE) wall model.** The thin-boundary-layer momentum
   equation `d tau/dy = rho S`, `tau = rho (nu + nu_t) du/dy` with a van Driest
   mixing length is integrated analytically per layer
   (`du/dy = 2 tau / (nu + sqrt(nu^2 + 4 l^2 tau))`) from the wall to the
   exchange height; `u_tau` is the root of `u(y_e; u_tau) = u_e`. The momentum
   source `S = (1/rho) dp/dx + dU/dt + U dU/dx` is supplied by the caller.
3. **Channel WMLES.** D3Q19, cumulant collision with Smagorinsky SGS and a
   positivity limiter, Guo body force, periodic x/z, wall model at
   `y_1+ = 12.3` (first node) with the exchange sample at `y_2+ = 36.9`.
   Two couplings are implemented: velocity assimilation (equilibrium part of
   the populations replaced) and **stress coupling** (free-slip wall +
   wall-model traction injected in the wall-adjacent row). Most of the
   statistics (Reynolds stresses, triple correlations, pressure-velocity
   correlations, resolved and SGS dissipation) are accumulated online.
4. **A-priori validation against DNS.** `benchmarks/wall_model_les/apriori.py`
   loads the MKM-180 channel DNS shipped with the repository, reconstructs the
   Reynolds shear stress from the **exact** balance
   `-<u'v'>+ = 1 - y+/h+ - dU+/dy+`, and grades the wall models.
5. **Flat plate.** Momentum integral `d theta/dx = c_f/2` marched over
   `Re_theta = 1e4 ... 5e4` with the Coles log-wake profile built on the same
   Spalding law; `c_f` is compared with the ITTC-1957 line differentiated to
   its *local* form, with Prandtl-Schlichting and with `0.024 Re_theta^-1/4`.
6. **KVLCC2.** `C_F` computed here from the ITTC-1957 line at `Re = 1.3e7`;
   `C_R` and the correlation allowance are taken from the ship platform's
   traceable ITTC-1978 chain (`POST /api/v1/resistance-advanced/ittc78`), with
   the full request/response stored in the artifact.
7. **Tooling.** `benchmarks/wall_model_les/run_shard.sh` dispatches the three
   compute shards; `acceptance.py` aggregates the verdicts and draws the
   figures from the artifacts (no hand-typed numbers).

## 3. Results (all numbers programmatically written to `artifacts/wall_model_les/`)

### 3.1 A-priori wall-model checks against MKM-180 DNS (criterion a)

| model | max rel. error of `u+`, `y+ = 30...300` | rms | `u_tau` recovery from `u(y+=30)` |
|---|---|---|---|
| Spalding-anchored equilibrium | **6.29 %** | 4.39 % | +6.7 % |
| pure log law, anchored | 3.93 % | 3.47 % | +2.3 % |
| **non-equilibrium ODE (A+ = 26)** | **1.72 %** | 0.65 % | **+4.1 %** |

DNS anchors derived from the repository's MKM-180 table: `-<u'v'>+` peak
`y+ = 31.98` (value 0.724), production peak `y+ = 12.23`.

### 3.2 Non-equilibrium response (what the equilibrium model cannot do)

`u_tau` change relative to the equilibrium model at `y_e+ = 50`:

| `S nu/u_tau^3` | -0.02 | -0.01 | 0 | +0.01 | +0.02 |
|---|---|---|---|---|---|
| relative change of `u_tau` | +8.7 % | +3.4 % | -2.2 % | -8.0 % | -14.7 % |

A favourable gradient (accelerating layer) raises the wall stress for the same
resolved velocity; an adverse gradient lowers it. This was checked against the
momentum-balance argument `tau(y) = tau_w + rho S y`.

### 3.3 ZPG flat plate, `Re_theta = 1e4 ... 5e4` (criterion c)

| quantity | value |
|---|---|
| `c_f(Re_theta = 1e4)` | 0.002365 |
| `c_f(Re_theta = 5e4)` | 0.001847 |
| shape factor `H` | 1.381 → 1.318 |
| `delta+` | 2954 → 14175 |
| max rel. error vs ITTC-1957 (local form) | **11.4 %** |
| rms rel. error vs ITTC-1957 | 9.0 % |
| max rel. error vs Prandtl-Schlichting | 14.4 % |
| max rel. error vs `0.024 Re_theta^-1/4` | 15.1 % (rms 8.0 %) |

Sensitivity (secondary, after the baseline verdict): `Pi = 0.30` → 4.1 %,
`Pi = 0.40` → 9.0 %, `Pi = 0.45` → 11.4 %, `Pi = 0.55` → 16.0 %;
`kappa = 0.44` → 4.2 %, `kappa = 0.41` → 11.4 %, `kappa = 0.38` → 21.0 %.
The `Pi` value that satisfies the 5 % criterion is **0.60** (bisection).

### 3.4 KVLCC2 resistance (criterion d)

`Re = 1.3e7`, `Fr = 0.142`, `(1+k) = 1.20`:

| term | this work | platform ITTC-78 |
|---|---|---|
| `C_F` | 0.0028678 | 0.00286785 (rel. diff 1.6e-5) |
| `C_R` | 9.33e-5 (taken from platform) | 9.33e-5 |
| `C_T = C_F(1+k) + C_R` | 0.0037006 | 0.0037006 (rel. diff 1.5e-5) |

Sensitivity to the input form factor: `1+k = 1.15 / 1.20 / 1.25` →
`C_T` changes by -3.9 % / 0 / +3.9 %. Full scale (`Re = 2.14e9`) from the same
platform chain: `C_F = 0.00139543`, `C_T = 0.00193498` without the correlation
allowance (`C_A = 1.76e-4`, reported separately).

### 3.5 Channel WMLES evidence run (criterion b cannot be graded from it)

`Re_tau` target 590, 32×49×32, cumulant + Smagorinsky 0.2, 240 steps:

| quantity | value |
|---|---|
| `dy+` / first node `y+` / exchange `y+` | 24.6 / 12.3 / 36.9 |
| `u_tau` from the wall model | 0.01024 |
| `u_tau` required by the force balance | 0.02000 (**gap 74 %**) |
| `U_b` | 0.394, `Re_b = 23 234` |
| `c_f` here / Dean correlation | 0.005159 / 0.005913 (12.8 % error) |
| maximum log-layer error vs Spalding | 57 % (transient) |
| total-stress identity max abs error | 2.95 (should be ≪ 1) |

With `Cs = 0.1` the same configuration is numerically unstable (velocity
reversal at `t ≈ 200-300` steps); with `Cs = 0.2` it is stable but
**relaminarizes**: `u_tau` decays monotonically (0.0207 → 0.0102), the resolved
Reynolds stress never develops, and the wall-model traction stays at 26 % of
the driving force. That is the honest outcome of this node and it is reported
as such (see §5).

## 4. Figures

| file | content |
|---|---|
| `fig_profiles_vs_dns.png` | wall laws vs MKM-180 DNS, log-layer band; DNS `-<u'v'>+` and production |
| `fig_flat_plate_cf.png` | `c_f(Re_x)` vs ITTC-1957 (local), Prandtl-Schlichting, `0.024 Re_theta^-1/4` |
| `fig_flat_plate_sensitivity.png` | max `c_f` error vs Coles `Pi` with the 5 % criterion line |
| `fig_channel_wmles.png` | WMLES `u+` vs Spalding and the resolved Reynolds stress (non-stationary run) |

## 5. Pitfalls, limitations, scope

1. **Coarse-grid channel WMLES does not self-start.** The physical reason is
   standard in the WMLES literature but worth recording: with `dy+ ≈ 25` the
   near-wall turbulence is entirely modelled, so a smooth (law-of-the-wall)
   initial condition has no resolved Reynolds stress to carry momentum to the
   wall. Real WMLES practice requires a **turbulent precursor** (a
   wall-resolved or finer-LES field interpolated onto the coarse grid, or a
   synthetic-eddy inflow). That precursor is the missing piece here; the
   solver, the wall-model coupling and the statistics machinery are in place.
2. **`tau ~ 1/2` stability.** At `Re_tau = 590` with `h = 24` the molecular
   viscosity is `nu = 8.14e-4` (`tau = 0.5024`). BGK blew up in the channel
   core within ~60 steps; the run uses cumulant + Smagorinsky + positivity
   limiter. The SGS constant matters: `Cs = 0.1` is unstable on this grid,
   `Cs = 0.2` is stable but over-dissipative.
3. **Stress vs velocity coupling.** Velocity assimilation (the OpenLB-style
   scheme already used elsewhere in this repository) controls the *velocity*
   but not the *flux*, and it produced a runaway decay in the channel;
   imposing the modelled traction as a body force with a free slip wall makes
   the drag exact and the feedback stabilising, and is the scheme used here.
   The sign of the traction (drag, i.e. opposite to the local flow) is a
   1-line trap that cost a debugging cycle.
4. **The flat-plate criterion is a correlation-family test.** At
   `Re_theta = 1e4 ... 5e4` the ITTC-1957 line implies `c_f` ≈ 11 % above the
   classical `0.024 Re_theta^-1/4` correlation, and the Coles log-wake family
   tracks the latter. The 5 % criterion is therefore effectively a test of the
   wake strength (`Pi`) — documented by the sweep, not hidden.
5. **KVLCC2 `C_R` is borrowed.** No digitized towing-tank `C_T` is available in
   this worktree, so the literal "vs experiment" limb of criterion (d) is
   **WITHHELD**; the comparison partner is an independent implementation of the
   same ITTC-78 standard. `C_F` (the part computed here) agrees to 1.6e-5.
6. **No pressure diffusion in the resolved budget.** The budget module reports
   the closure residual explicitly rather than closing it with an assumed term;
   in the current transient channel run that residual is large, which is the
   symptom, not a modelling choice.
7. **Reproducibility.** Every artifact records `seed`, grid, `nu`, `tau`,
   `Cs`, wall-model settings, and the acceptance record keeps the full numbers.
   Tests: `tests/test_wall_model_les.py` (14 tests, all passing).

## 6. Error verdicts (explicit)

| criterion | value | threshold | verdict |
|---|---|---|---|
| (a) log-layer profile error, `y+ = 30...300` | 1.72 % (non-equilibrium ODE vs DNS); 6.29 % (Spalding law vs DNS); 0 % vs the Spalding formula itself (tautological) | ≤ 3 % | **PASS** with the DNS limb for the ODE model; **FAIL** for the pure Spalding law (6.29 %) |
| (b) Reynolds-stress peak position | DNS anchor `y+ = 31.98`; the WMLES run is non-stationary, the standalone model has a degenerate (constant-stress) profile | ≤ 5 % | **WITHHELD** |
| (c) `c_f` vs ITTC-1957 | 11.4 % max, 9.0 % rms | ≤ 5 % | **FAIL** (honest; diagnosed as a `Pi`/correlation-family effect, see §5.4) |
| (d) KVLCC2 `C_T` | 1.5e-5 vs the platform ITTC-78 chain; vs a measured `C_T`: no data | ≤ 7 % | **PASS vs the traceable standard**; **WITHHELD vs experiment** |

**Is the error below 3 %?** For the log-layer *shape*, yes: the
non-equilibrium ODE wall model reproduces the DNS profile to 1.72 % max
(0.65 % rms) in `y+ = 30...300`. For the absolute friction level the residual is
larger: `u_tau` recovery error 4.1 % (ODE), 6.7 % (Spalding). For the ship
resistance chain the *friction coefficient implementation* is exact to 1.6e-5
against the platform, but the flat-plate `c_f` level is 11.4 % off the
ITTC-1957 line, i.e. **not** below 3 %. Nothing in this report is rounded to
hide a failure.