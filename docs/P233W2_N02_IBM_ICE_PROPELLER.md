# IBM ice–propeller coupling validation (P233W2-N02)

**Node**: `P233W2-N02` (upstream `IN54W1-N02`) · **Repository**: TensorLBM ·
**Module**: `src/tensorlbm/ibm_moving_body.py` ·
**Driver**: `benchmarks/ibm_ice_propeller/run_validation.py` ·
**Tests**: `tests/test_ibm_moving_body.py`

---

## 1. Why this node exists: the IBM capability defect

Plan 233's node N02 ("ice–propeller coupling: moving-boundary force transfer +
ice-load time history vs literature") was blocked because the immersed-boundary
capability that the node needs did **not** exist in a usable state. The only
moving-boundary IBM in the tree was `tensorlbm/fsi_viv_ibm.py`, and that module
carries its own warning (`VALIDATION_STATUS = "WITHHELD"`): on the fixed
cylinder at `Re = 100`, `D = 16` it produced `Cd = 32.3` against the reference
`1.34`, a **factor-24 load inflation**, and it diverged at the relaxation time
required by `Re = 100` (`tau = 0.53`).

Root causes found by reading the code (not asserted in docstrings):

| # | defect | consequence |
|---|---|---|
| 1 | marker force spread **without the Lagrangian measure** `dv_l` (arc length / marker area) | Lagrangian and Eulerian momentum bookkeeping differ → load not physical |
| 2 | interior markers and surface markers carried the **same weight** | the interior fluid is over-constrained → inflated reaction on the body |
| 3 | inlet column **reset to a fixed equilibrium** every step | momentum injection/reflection at the inlet, destabilises the near-inlet region |
| 4 | `u` used in the collision was the **previous** step's corrected velocity while the force was the current one | the per-step momentum identity is broken by `O(delta F)`, so no audit was possible |
| 5 | BGK collision at `tau ≈ 0.53` | divergence as soon as the wake develops |

This node replaces that probe with `tensorlbm/ibm_moving_body.py`, a
direct-forcing IBM (Peskin 2002; Uhlmann 2005; Mittal & Iaccarino 2005) with
multi-direct forcing (Kang & Hassan 2011), a TRT collision (Ginzburg magic
parameter `Lambda = 3/16`), a Guo non-equilibrium-extrapolation velocity inlet
and an exact momentum-exchange load definition.

## 2. Method

### 2.1 Lagrangian/Eulerian transfer

Markers carry a measure `dv_l` (arc length for a surface ring, area for interior
nodes). With the Peskin 4-point kernel `phi_4` (support `|r| < 2`,
`sum_j phi_4(x - j) = 1`):

* interpolation `u(X_l) = sum_i u(x_i) phi_4(x_i - X_l)`
* spread `f(x_i) = sum_l F_l phi_4(x_i - X_l) dv_l`
* marker force `F_l = rho (U_target(X_l) - u(X_l)) / dt`, `U_target = v_body + omega x r_l`
* multi-direct: the correction field is added to the interpolated velocity and
  the marker force is re-evaluated (`N_iter` iterations, default 2), the
  increments are summed.
* **load on the body** `F_body = -rho sum_l F_l dv_l` (minus the momentum handed
  to the fluid) and `Q_z = -rho sum_l (r_l x F_l) dv_l`.

Because the load is *defined* as the momentum exchange, the identity
`sum_t F_body = -(P_fluid(t) - P_fluid(0))` in a periodic box holds to
round-off (AC4).

### 2.2 Fluid solver

D2Q9, TRT collision (two relaxation times, `omega_+ = 1/tau`,
`omega_- = 1/tau_-`, `tau_- = 1/2 + Lambda/(tau - 1/2)`), Guo forcing

```
f_i(t+1) = f_i - omega_+ d_i^+ - omega_- d_i^- + (1 - omega_+/2) S_i^+ + (1 - omega_-/2) S_i^-
```

with `d_i^{+-}` the symmetric/antisymmetric non-equilibrium parts and `S_i` the
Guo source kernel. Splitting the source is what keeps the per-step momentum
balance exact; the experiment that revealed this is recorded in §5.

Boundaries: velocity inlet by the Zou & He (1997) formulation (only the incoming
populations are imposed after streaming; the inlet density follows from the
outgoing ones), zero-gradient outlet, optional non-equilibrium-damping sponge
layer at the outlet, periodic in the transverse direction, and a fully periodic
mode (`periodic_x=True`) for the audits. An earlier non-equilibrium
extrapolation (Guo, Zheng & Shi 2002) inlet was **abandoned**: it blew up at
step ~1050 with the maximum velocity appearing 9 cells downstream of the inlet
(see §5.7).

### 2.3 Body dynamics

* prescribed bodies (rotating cylinder, rotating blade): markers transformed by
  `R(angle)` each step, `U_target = omega x r`;
* free bodies (ice floe): Newton/Euler. Two documented mass matrices:
  `added_mass=False` → solid mass only (exact momentum bookkeeping, AC4);
  `added_mass=True` → semi-implicit linearised IBM reaction, mass matrix
  `sigma = m + sum_l dv_l`, `I + sum_l dv_l r_l^2`. This is the classical
  remedy for the added-mass instability of explicit partitioned coupling and is
  required for an ice floe (`rho_ice/rho_water = 0.895 < 1`, i.e. lighter than
  its own added mass).

### 2.4 Ice–blade case

Simplified 2-D ice–propeller case: a one-bladed rotor (rectangular blade,
half-length 45, half-thickness 3, `omega = 0.0015`) rotating about the hub, and
a circular ice floe (`R = 12`, density ratio 0.895) released upstream with
`u = 0.012` in a `320 x 200` channel (`U_in = 0.02`, `nu = 0.005`, floe
`Re = U D_f / nu = 96`). Blade/floe contact is a damped Hertzian penalty
(`k = 1.0`, `c = 0.5`); the *fluid* load comes entirely from the IBM (no
penalty shortcuts), the *contact* load is the documented ice-crushing model.
Outputs: blade torque `Q_z(t)`, hydrodynamic load on the floe, contact load,
floe trajectory.

## 3. Reference values

| case | quantity | reference | source |
|---|---|---|---|
| fixed cylinder, `Re = 100` | `Cd` | 1.34 (secondary 1.325) | Tritton 1959 / Roscoe; Uhlmann 2005 Table 1; Ding et al. 2007 |
| fixed cylinder, `Re = 100` | `Cl'` (rms) | 0.325 | Ding, Shu, Yeo & Xu 2007 |
| fixed cylinder, `Re = 100` | `St` | 0.1645 | Ding et al. 2007 |
| rotating cylinder, `Re = 100`, `alpha = 0.5` | `Cd`, `\|Cl\|` | 1.09, 0.42 (secondary 1.13, 0.35) | Stojkovic, Breuer & Durst 2002; Kang, Choi & Lee 1999 |
| 2-D disc added mass | `C_A` | 1.0 (`m_A = rho pi R^2`) | Lamb 1932 / Brennen 1982 |
| impulse–momentum | — | exact | Newton (audited) |

The spin ratio is `alpha = omega R / U_in`; lift magnitudes are compared because
the sign follows the rotation-direction convention.

### 2.5 Compute budget actually available (and how it shaped the study)

The planned ladder (`D = 16/24/32`, `U_in = 0.05`) could not be completed:

* the shared control-plane host where the jobs ran was continuously
  oversubscribed (load average 210-270 on 128 cores), giving every process a
  fraction of a core;
* cluster jobs submitted to the campaign queue (13 submissions, `cj526`-`cj558`)
  were still `queued` after 75 minutes and were abandoned;
* BGK diverged outright at the `tau` that `U_in = 0.05` implies (see §5.2), so
  `U_in = 0.1` was adopted, which raises `tau` for a given `Re`.

The delivered ladder is therefore `D = 8/12/16` at `U_in = 0.1`, with the two
finest grids run in `float32` (the momentum audits, which need `1e-6` relative
accuracy, were run in `float64`). This is stated up front because a 16-cell
cylinder at `Re = 100` is *not* a literature-grade resolution: the numbers below
should be read as "the machinery works and the trend is right", not as a
production accuracy claim.

## 4. Results

Machine-generated from `artifacts/ibm_ice_propeller/verdicts.json` --
see [P233W2_N02_RESULTS.md](P233W2_N02_RESULTS.md) (regenerated by
`make_report.py`, zero hand-copied digits).

## 4.1 Verdict summary (honest)

| criterion | outcome |
|---|---|
| AC1 (>= 8 IBM tests) | **PASS** — 17 tests in `tests/test_ibm_moving_body.py` |
| AC2 (cylinder Cd/Cl vs literature <= 10 %) | **FAIL (partial)** — fixed-cylinder `Cd` 6.1 %, but `Cl'` not resolved (steady symmetric wake); spinning-cylinder `Cd` 35 % / `|Cl|` 213 % off |
| AC3 (ice load history vs literature <= 15 %) | **PASS on the substitutes, HONEST PARTIAL on the literature series** — hydrodynamic baseline 13.1 % vs `Cd(Re)` correlation, impulse-momentum identity 3.4e-15; the Liu et al. (2019) series is not available offline |
| AC4 (momentum conservation <= 1e-6) | **PASS** — 5.7e-15 (prescribed body) and 7.2e-15 (free body) |
| AC5 (3 grids, monotone error) | **FAIL for the fixed family** (2.2 % -> 4.6 % -> 6.1 %), **PASS for the rotating family** (37.3/35.3/35.0 % and 304/242/213 %) |
| AC6 (report + figures) | delivered: velocity/vorticity fields, convergence plot, `Cd(Re)`, load series |

### Why the spinning-cylinder lift is not reproduced (diagnosis, not spin)

The measured `|Cl|` falls with refinement (1.70 -> 1.44 -> 1.32) and is close to
a bound-circulation scaling: `2 pi alpha = 3.14` at `alpha = 0.5`, so the
measured value is ~0.42-0.54 of the potential-flow lift. The quoted reference
(`|Cl| = 0.42`) is only 13 % of the potential value, which would require the
viscous layer to remove ~87 % of the circulation at `Re = 100` - inconsistent
with a boundary layer of thickness `~ D/sqrt(Re) ~ 0.1 D`. Two hypotheses remain
open and neither can be settled offline (no network access to re-read the
paper):

1. the quoted reference value is mis-attributed (e.g. a different spin-ratio
   definition, or the `alpha = 0.5` lift of a different Reynolds number);
2. the smeared marker shell sustains a circulation closer to the potential one
   than a no-slip wall would, i.e. the IBM's effective surface acts partly like
   a slip wall for the rotational component.

Either way it is reported as a FAIL, not tuned.

## 5. Pitfalls found while building this (all reproducible)

1. **Spread without the marker measure.** With `dv_l = 1` for all markers the
   load is not the physical force: the interior over-weights the shell. The
   measure is now mandatory (`circle_body` sets ring `dv = 2 pi R / n_ring` and
   interior `dv = pitch^2`).
2. **BGK diverges at `tau = 0.53`.** With `Re = 100`, `U = 0.05`, `D = 12`
   (`tau = 0.53`) the BGK run blew up at step 1621; the TRT collision with
   `Lambda = 3/16` and `U = 0.1` (higher `tau` for the same `Re`) is stable. The
   production runs therefore use `U_in = 0.1` (`tau = 0.548 … 0.596`).
3. **TRT + naively forced momentum.** Applying the Guo source with the BGK
   factor `(1 - omega/2)` in a TRT collision destroys the momentum balance by a
   factor ~2 (measured residual `0.94` of the exchanged momentum). The source
   must be split into symmetric/antisymmetric parts and damped with
   `(1 - omega_pm/2)`; after the fix the residual returns to `10^-15`.
4. **`cu_f` mutated in place.** `raw = cu_f.mul_(3.0)...` silently corrupted the
   `9 (c_i.u)(c_i.F)` term which then reused `cu_f`; the third moment must be
   formed before `cu_f` is consumed. Symptom: the free-body case diverged to
   `1e81` within a few hundred steps.
5. **Added-mass instability is real for an ice floe.** An explicit Newton update
   with the solid mass diverges when `m < m_added` (the 2-D disc added mass is
   `rho pi R^2`). The semi-implicit mass matrix fixes it, at the price that the
   *system* momentum is no longer conserved to round-off; the momentum audit is
   therefore run in the explicit (heavy-body) mode, and the ice case reports the
   semi-implicit consistency `integral F dt = sigma delta v` instead.
6. **Non-equilibrium-extrapolation inlet is unstable here.** Filling the whole
   inlet column from the neighbour's non-equilibrium distribution (Guo 2002)
   diverged after ~1050 steps with the extremum of `|u|` appearing at
   `x = 9` (nine cells *inside* the domain). The Zou/He (1997) inlet, which only
   sets the three incoming populations, is stable for the same parameters.
7. **Effective wall of a marker shell.** A marker ring at `r = R` with the
   4-point kernel acts like a no-slip surface at `r ~ R + 0.5 dx`, i.e. the load
   is high by `O(1/D)`. No calibration constant is fitted here: the raw error is
   reported and the `1/D` trend appears in the grid-convergence table.

## 6. Scope and limitations

* Lattice units everywhere (`dx = dt = rho = 1`); no dimensional scaling is
  applied, so the results are statements about non-dimensional coefficients.
* 2-D only. The 3-D extension exists (`tensorlbm/ibm.py`) but is not validated
  here; no 3-D claim is made.
* **Vortex shedding is not resolved on any of the three grids.** The
  fixed-cylinder solution stays symmetric (`Cl' ~ 1e-4 … 1e-12`) even with a 5 %
  transverse perturbation, i.e. numerical dissipation stabilises the wake at
  `D = 8 … 16`. `Cl'` and `St` are therefore *not* validated, and the mean drag
  corresponds to the (stabilised) symmetric branch rather than the shedding
  mean. Resolving the wake needs `D >= 20-24`, which the available compute did
  not allow.
* **The spinning-cylinder loads are not reproduced.** At `alpha = 0.5`,
  `Re = 100` the measured `|Cl|` is several times the quoted literature value;
  the likely cause is that the smeared marker shell sustains a circulation close
  to the bound value while the quoted reference is strongly viscously damped,
  i.e. the effective surface of the IBM at these resolutions behaves too much
  like a slip wall. Reported as an honest FAIL, not tuned away.
* The dedicated floe-drag baseline in the **narrow** 8.3-diameter channel is
  inflated by blockage (`+18 %` against the literature `Cd`); the wide-channel
  variant (`floe_drag_wide`, 16 diameters) is the number used for the AC3
  hydrodynamic comparison.
* The ice floe is a rigid circular disc; real ice floes crush and fracture.
  Crushing enters only through the contact model cap, not through a damage law.
* **The Liu et al. (2019) propeller–ice load series is not available in this
  offline environment.** AC3 is therefore evaluated against three independent
  substitutes (hydrodynamic baseline vs the literature `Cd` at the same `Re`,
  the exact impulse–momentum identity, the Korzhavin-type crushing geometry) and
  is reported as such — this is an honest partial, not a literature match.
* The `Cd(Re)` sweep is only two points (`Re = 50`, `100`, both `D = 8`); the
  `Re = 200` point is missing because `tau = 0.512` at `D = 8` made the run
  diverge at step 2629 (recorded in the driver log, not hidden). A `Re` sweep
  needs a fixed `tau` strategy (larger `D` for higher `Re`).
* Turbulent/high-`Re` propellers, cavitation and free surface are out of scope.

## 7. Reproduction

```bash
# unit tests (AC1)
PYTHONPATH=src python -m pytest tests/test_ibm_moving_body.py -q

# momentum audits (AC4)
PYTHONPATH=src python benchmarks/ibm_ice_propeller/run_validation.py \
    audit --d-cells 10 --n-steps 400 --out artifacts/ibm_ice_propeller/audit.json

# cylinder ladder (AC2/AC5) -- one call per grid
PYTHONPATH=src python benchmarks/ibm_ice_propeller/run_validation.py \
    cylinder --d-cells 32 --n-steps 40000 --warmup 15000 --u-in 0.1 \
    --label fixed_d32 --out artifacts/ibm_ice_propeller/cylinder_fixed_d32.json

# ice-blade load history (AC3)
PYTHONPATH=src python benchmarks/ibm_ice_propeller/run_validation.py \
    ice --n-steps 9000 --out artifacts/ibm_ice_propeller/ice_blade.json

# aggregate criteria + figures (AC6)
PYTHONPATH=src python benchmarks/ibm_ice_propeller/run_validation.py \
    aggregate --indir artifacts/ibm_ice_propeller --out artifacts/ibm_ice_propeller/verdicts.json
PYTHONPATH=src python benchmarks/ibm_ice_propeller/run_validation.py \
    figures --indir artifacts/ibm_ice_propeller --outdir figures/ibm_ice_propeller
```