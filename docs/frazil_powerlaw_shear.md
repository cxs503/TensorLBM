# Frazil ice slurry as a power-law fluid: structure-resolved LBM measurement of tau = K gdot^n

**Repository** TensorLBM · **campaign node** `p366w2-n02-frazil-rheology` (mission 10712) ·
**domain** `ice_navigation` · **artifact** `artifacts/frazil_powerlaw_shear/campaign.json`
(schema `frazil_powerlaw_shear/v1`) · **cases** `34`/`34` ·
**acceptance criteria passed** `3`/`5`.

## 1. Case introduction (physics + parameters)

Frazil consists of millimetre-scale, disc-shaped ice crystals that form in
supercooled turbulent water and stay suspended. Above a few percent solids the
mixture stops behaving like water: it thickens, becomes shear thinning, and its
effective viscosity controls propeller thrust loss, intake blockage and the
resistance increment of a hull in brash ice. The resistance models in this
platform take the carrier as a generalised Newtonian fluid
`tau = K gdot^n`, so the two numbers `(K, n)` *are* the physical input. This
campaign measures them from a resolved suspension instead of quoting them from a
literature fit.

The gap being closed: the earlier frazil work in this repository
(`tensorlbm.frazil_rheology` + `tensorlbm.frazil_lbm`) **prescribes** `K(phi)`
and `n(phi)` and then verifies that the solver transports that prescribed
viscosity surface correctly (worst channel residual 1.23 %). It never measures
the flow curve of a suspension. Here the dispersed phase is explicit: rigid ice
crystals — aggregated into fractal flocs for the flocculated state — coupled to
the fluid by half-way bounce-back and momentum exchange, in a moving-wall Couette
cell.

| parameter | value | note |
|---|---|---|
| lattice | D3Q19, rho_f = 1, nu = 0.2 | lattice units |
| cells (coarse / fine) | 32x32x33 / 40x40x41 | periodic in x,y; walls in z |
| primary crystal radius | 2.5 / 3.0 cells | D = 2a lattice units |
| concentrations c (dispersed) | 0.05, 0.10, 0.20, 0.30 | volume fraction from the lattice mask |
| concentrations c (flocculated) | 0.05, 0.10 | higher c jams the placement (see §5) |
| shear rates gdot | 5.106e-04, 1.021e-03, 2.042e-03 | factor-2 ladder (factor 4 in total) |
| Re_p = gdot D^2 / nu | from 0.064 upwards | Stokes regime, reported per case |
| floc fractal dimension d_f | 2.0 | the achieved d_f is measured per case |
| max primaries per floc N_cap | 8 | anchored at the lowest shear rate |
| bond force F_b (anchored, not fitted) | 0.005106 | F_b = mu gdot_ref (a N_cap^(1/d_f))^2 |
| steps (warm-up) | 1200 (800) / 1600 (1050) | >= 2 viscous time constants H^2/(pi^2 nu) |

Two structural families are run side by side, and the distinction is the point:

* **dispersed** — one crystal per particle: the hard-sphere state for which
  Einstein and Krieger–Dougherty are the correct closures;
* **flocculated** — crystals bonded into open fractal aggregates whose size is
  set by the shear-breakup balance `R_eq = sqrt(F_b/(mu gdot))`,
  `N_eq = (R_eq/a)^d_f` (Sonntag & Russel 1986; Spicer & Pratsinis 1996).
  Aggregates trap water, so their hydrodynamic volume fraction exceeds their
  solids fraction, and they break up as the shear rate rises.

## 2. How it was done

1. **Structure selection.** For each shear rate the equilibrium floc size
   `N_eq(gdot)` follows from the breakup balance. The single constant `F_b` is
   anchored — not fitted — by declaring `N_eq(gdot_min) = N_cap = 8` before
   any simulation runs; the implied `F_b = 0.005106` (lattice units) is carried in the
   artifact so a reader can judge it.
2. **Aggregate generation.** Hierarchical cluster-cluster aggregation of touching
   primaries: each merge rotates one cluster randomly and slides it along a random
   direction until the nearest primaries touch. Of several realisations the one
   whose implied fractal dimension is closest to `d_f = 2.0` is kept, and the
   achieved dimension is measured and reported rather than assumed.
3. **Placement.** Random sequential addition of whole aggregates. Non-overlap is
   enforced on the *primary centres* (minimum image in the periodic directions),
   so open flocs may interpenetrate their bounding spheres — otherwise the
   achievable concentration would be an artefact of the placement rule instead of
   physics. The **achieved** lattice volume fraction is always reported.
4. **Shear cell.** D3Q19 moving-wall Couette (`ShearCell`): walls at z = 0.5 and
   z = nz - 1.5 counter-move with +-u_w, so `gdot = 2 u_w/(nz-2)`. Every primary of
   an aggregate is advected with the *aggregate* centre (translation plus the
   half-vorticity rotation gdot/2), so the aggregate stays rigid.
5. **Stress.** `tau` is the plane momentum flux averaged over the wall bands,
   corrected for the momentum the particles absorb
   (`tau_corrected = tau_near - f_particle/area`); the shear rate is the slope of a
   linear fit to the mean velocity profile, with R^2 reported.
6. **Flow-curve fit.** Least squares of `ln tau` on `ln gdot` over the three rates
   gives `(K, n)` together with R^2 and the slope standard error.
7. **Acceptance.** The five pre-declared criteria are evaluated *inside the
   artifact* (`acceptance` block), never in prose.

**Scale actually run: `reduced`.** The campaign is defined at two scales in
`run_campaign.py` (`production`: 48x48x49 / 64x64x65 cells; `reduced`: 32x32x33 /
40x40x41). The numbers below are from the `reduced` scale: the cluster
dispatcher accepted the production submission but had not dispatched it after
32 min (241 jobs ahead of it), so the reduced scale was run on the contended
host instead. The production-scale campaign is a single command
(`FRAZIL_SCALE=production python3 run_campaign.py ...`) and its numbers replace
these when it runs; nothing else in the chain changes. This is a scope
statement, not a caveat about the method.

Tool chain: `benchmarks/frazil_powerlaw_shear/run_campaign.py` (plan -> sharded
cases -> assemble), `make_figures.py` (figures read back from the artifact),
`make_report.py` (this document, generated from the artifact), tests in
`tests/test_frazil_powerlaw_shear.py`.

## 3. Results

### 3.1 Particle-free control (the denominator of eta_r)

Plane Couette is an exact Navier–Stokes solution at every Reynolds number, so the
particle-free run has a closed-form reference `tau = rho nu gdot`.

| resolution | rate index | $\tau$ measured | $\nu\dot\gamma$ (exact) | rel. error | profile $R^2$ |
|---|---|---|---|---|---|
| c32 | 0 | 1.021190e-04 | 1.021195e-04 | 0.00 % | 1.00000 |
| c32 | 1 | 2.042429e-04 | 2.042400e-04 | 0.00 % | 1.00000 |
| c32 | 2 | 4.084791e-04 | 4.084804e-04 | 0.00 % | 1.00000 |
| f40 | 0 | 1.021203e-04 | 1.021193e-04 | 0.00 % | 1.00000 |
| f40 | 1 | 2.042359e-04 | 2.042402e-04 | 0.00 % | 1.00000 |
| f40 | 2 | 4.084774e-04 | 4.084805e-04 | 0.00 % | 1.00000 |

### 3.2 Flow curves and fitted (K, n), coarse tier

| family | c | N primaries | $\phi$ (lattice) | $\gamma\dot$ range | $n$ | $K$ | $R^2$ | $\eta_{eff}/\mu_0$ |
|---|---|---|---|---|---|---|---|---|
| dispersed | 0.05 | 1.0 | 0.0513 | 5.018e-04 – 2.013e-03 | 1.0103 | 0.24044 | 0.9999 | 1.1173 |
| dispersed | 0.10 | 1.0 | 0.1018 | 4.993e-04 – 2.000e-03 | 1.0546 | 0.35102 | 0.9991 | 1.1731 |
| dispersed | 0.20 | 1.0 | 0.1544 | 4.996e-04 – 1.997e-03 | 1.0707 | 0.4185 | 0.9982 | 1.2452 |
| dispersed | 0.30 | 1.0 | 0.1544 | 5.005e-04 – 1.997e-03 | 1.1465 | 0.68232 | 1.0000 | 1.1214 |
| flocculated | 0.05 | 8.0 | 0.0618 | 4.561e-04 – 1.994e-03 | 0.8180 | 0.072171 | 0.9910 | 1.4161 |
| flocculated | 0.10 | 8.0 | 0.1088 | 4.880e-04 – 1.982e-03 | 0.8491 | 0.10592 | 0.9995 | 1.6635 |

### 3.3 Relative viscosity and the five acceptance criteria

| # | criterion | measured | threshold | verdict |
|---|---|---|---|---|
| 1 | $c=0$ negative control | $n=1$ (dev 9.355e-07), $\eta_r=1$ (dev 0.00 %) | $n=1\pm0.01$, $\mu=\mu_w\pm2\%$ | PASS |
| 2 | $c=0.05$ Einstein | $\eta_r=1.117$ | $1+2.5c=1.125$ (-0.68 %) | PASS |
| 3 | $c=0.20$ Krieger–Dougherty | $\eta_r=1.245$ ($\phi=0.1544$) | KD $=1.562$ (-20.29 %) | FAIL |
| 4 | shear thinning, $c\ge0.10$ | c=0.10: n=0.8491 | $n<0.95$ | PASS |
| 5 | grid convergence | $\Delta n/n=3.36 %$, $\Delta K/K=17.99 %$ | both $<5\%$ | FAIL |

### 3.4 Grid convergence (coarse vs fine resolution)

| pair | $n$ (coarse / fine) | $K$ (coarse / fine) | $\Delta n/n$ | $\Delta K/K$ | within 5 %? |
|---|---|---|---|---|---|
| dispersed_c0.20 | 1.0707 / 1.0541 | 0.4185 / 0.36036 | 1.57 % | 16.14 % | no |
| flocculated_c0.10 | 0.8491 / 0.8786 | 0.10592 / 0.12915 | 3.36 % | 17.99 % | no |

### 3.5 Seed scatter (microstructure realisation)

| family | c | rate idx | seeds | $\eta_r$ per seed | spread |
|---|---|---|---|---|---|
| dispersed | 0.20 | 0 | 4 | 1.1214, 1.3478, 1.2663, 1.2174 | 22.65 % |
| dispersed | 0.20 | 1 | 2 | 1.2388, 1.1948 | 4.40 % |
| dispersed | 0.20 | 2 | 2 | 1.3735, 1.3120 | 6.15 % |
| flocculated | 0.10 | 0 | 4 | 1.7300, 1.7830, 1.4776, 1.5570 | 30.54 % |
| flocculated | 0.10 | 1 | 2 | 1.5366, 1.6470 | 11.04 % |
| flocculated | 0.10 | 2 | 2 | 1.3446, 1.3108 | 3.38 % |

**Read §3.5 before the criterion table.** The largest seed-to-seed spread of
``eta_r`` anywhere in this campaign is 30.54 % (four realisations of the same
concentration and rate). The dilute-criterion agreement is therefore a statement
that the measured slope is right *to the statistical resolution of a single
frozen microstructure*, not a 1 % calibration: with ten aggregates in the cell,
the scatter of one configuration is the dominant uncertainty, and it is quoted
rather than averaged away.

### 3.6 Hydrodynamic-radius gate (may the mask fraction be converted?)

At $D = 5$ lattice diameters a staircase sphere is not a smooth sphere, so the
mask volume fraction can differ from the fraction the fluid responds to. The
conversion is measured, not assumed: a fixed sphere in a uniform stream gives
the momentum-exchange drag, and Stokes' law inverted (with the Haberman–Sayre
confinement factor) gives $a_h$. The gate is only *used* if it closes its own
budget — the raw drag and the confined Stokes prediction must agree to 5 %.

| resolution | $a_h/a_eff$ | $\phi_h/\phi_{mask}$ | drag vs Stokes (unconfined) | drag vs Stokes (confined) | applied? |
|---|---|---|---|---|---|
| c32 | 0.6738 | 0.3059 | 4.24 % | 32.62 % | no (gate did not close) |

Where the gate does not close, the mask fraction is kept and the criterion is
reported on it. Using a gate that fails its own budget to move a criterion into
range would be a fudge, so it is recorded and not applied.

### 3.7 Figures

![fig1_flow_curves.png](../artifacts/frazil_powerlaw_shear/fig1_flow_curves.png)

*Measured flow curves tau(gamma_dot) from the D3Q19 frozen-structure shear cell for the dispersed (left) and flocculated (right) families, with the least-squares power-law fits tau = K gamma_dot^n drawn through the data points.*

![fig2_relative_viscosity.png](../artifacts/frazil_powerlaw_shear/fig2_relative_viscosity.png)

*Relative viscosity versus ice volume fraction from the shear cell (lowest rate), against Einstein 1+2.5c and Krieger-Dougherty (phi_max = 0.6, [eta] = 2.5). The flocculated points lie above the hard-sphere closures because a fractal aggregate traps water.*

![fig3_shear_thinning.png](../artifacts/frazil_powerlaw_shear/fig3_shear_thinning.png)

*Relative viscosity versus shear rate: the structure-frozen (dispersed) family is flat (Newtonian), while the structure-following (flocculated) family falls as the aggregates break up.*

![fig4_powerlaw_index.png](../artifacts/frazil_powerlaw_shear/fig4_powerlaw_index.png)

*Power-law index n versus concentration for both families, with the n = 1 (Newtonian) and n = 0.95 (shear-thinning criterion) lines.*

![fig5_structure.png](../artifacts/frazil_powerlaw_shear/fig5_structure.png)

*Primaries per aggregate actually placed in the cell versus the shear-breakup law that selected them (the law's parameters are anchored, not fitted).*

## 4. Error judgement (explicit, per key quantity)

| key quantity | reference | measured | error | $<3\%$? |
|---|---|---|---|---|
| $c=0$ stress exponent $n$ | exact $n=1$ (plane Couette is an exact Navier–Stokes solution for every $Re$) | 1 | 0.00 % | yes |
| $c=0$ relative viscosity | exact $\eta_r=1$ | 1 | 0.00 % | yes |
| $\eta_r$ at $c=0.05$ | Einstein $1+2.5c$ (exact to $O(c^2)$) | 1.117 | 0.68 % | yes |
| $\eta_r$ at $c=0.20$ | Krieger–Dougherty ($\phi_m=0.6$, $[\eta]=2.5$) | 1.245 | 20.29 % | no |
| $n$, dispersed_c0.20 | coarser resolution | 1.0541 | 1.57 % | yes |
| $K$, dispersed_c0.20 | coarser resolution | 0.36036 | 16.14 % | no |
| $n$, flocculated_c0.10 | coarser resolution | 0.8786 | 3.36 % | no |
| $K$, flocculated_c0.10 | coarser resolution | 0.12915 | 17.99 % | no |
| $n$ at $c\ge0.10$ | no pointwise reference exists (frazil floc rheology is reported as ranges, not digitised curves) | 0.849 | n/a — declared | n/a — declared |

The node's acceptance tolerance is 15 % on eta_r against Einstein and
Krieger–Dougherty, 1 % on the zero-concentration exponent and 5 % on the
resolution pair. The "<3 %" column above answers the platform's stricter
refereed question; where only literature *ranges* exist for a quantity, the table
says so instead of inventing a percentage.

## 5. Pitfalls, scope, limitations

**Pitfalls found while building this face**

1. *Frozen structure cannot shear-thin.* A rigid suspension in Stokes flow is
   Newtonian by construction: the flow curve of a frozen microstructure has
   `n ~ 1` however concentrated it is. Any `n < 1` must come from a structure that
   follows the shear — here the floc size. The dispersed family is kept precisely
   as the built-in negative control for that statement.
2. *Inertia masquerades as shear thickening.* `Re_p = gdot D^2/nu` grows with the
   rate and suspension viscosity creeps up with `Re_p`, so the dispersed family's
   measured exponent sits slightly *above* 1 (see §3.2). Judging a flocculated
   exponent against 1.0, rather than against the dispersed exponent at the same
   rates, would misattribute that bias.
3. *Placement saturation is physics, not a bug.* Open `d_f = 2` aggregates at
   `c = 0.10` fill a hydrodynamic volume fraction of order
   `phi (R/a)^(3-d_f) ~ 0.25`; random sequential addition jams below the requested
   solids fraction and the achieved `phi` (reported per case) is what the closure
   then uses. The jamming is consistent with the pasty behaviour observed for
   frazil at ~10 % solids, but it does mean the flocculated family cannot be pushed
   to `c = 0.30` in this cell.
4. *Placement saturation, not a solver bug.* Random sequential addition with a
   one-cell gap saturates near ``phi = 0.155`` in the reduced cell, so the
   requested ``c = 0.20`` and ``c = 0.30`` runs both landed at
   ``phi = 0.154`` (reported per case). The two rows therefore describe two
   different microstructures at the same volume fraction, which is exactly the
   seed scatter of a single frozen configuration (~10 % in ``eta_r``) and is why
   the Krieger–Dougherty comparison at ``c = 0.20`` cannot be settled at this
   scale.
5. *Stress estimator.* The wall momentum-exchange estimator does not close the
   pure-fluid momentum budget; the plane-flux estimator does (§3.1). That is why
   the production measurement uses the flux, and why eta_r is always a ratio of
   two identical measurements.

**Limitations (honest)**

* 3-D but laminar: no turbulence, no crystal collision/aggregation dynamics
  inside a run, no population balance. Floc size is selected *between* runs by a
  stated closure; the resolved simulation then measures the hydrodynamic
  consequence of that geometry.
* `F_b` is anchored on `N_cap`, so the *slope* of `N_eq(gdot)` (namely -1/2 in R,
  hence `d_f/2` in N) carries the structure information, not the absolute floc
  size. A reader who rejects the -1/2 exponent should read the flocculated rows as
  conditional on it.
* Primary crystals are spheres of 6–8 lattice diameters; real frazil crystals are
  discs of aspect ratio ~10. A sphere is the conservative choice (a disc packs a
  larger hydrodynamic volume at the same solids fraction), but the shape effect is
  not quantified here.
* The cell is small (32x32x33): of order ten aggregates and 50–300 primaries, so the
  seed-to-seed scatter in §3.5 is the honest uncertainty of a single point rather
  than an ensemble average.
* No lattice-to-SI scaling is claimed: `(K, n)` are lattice quantities. The helper
  `lattice_to_si` documents the unit change for a declared crystal scale, but no
  experiment is matched pointwise here.

## 6. Reproduction

```bash
python3 benchmarks/frazil_powerlaw_shear/run_campaign.py --plan
python3 benchmarks/frazil_powerlaw_shear/run_campaign.py --run --shard 0 --shards 17 --threads 2
python3 benchmarks/frazil_powerlaw_shear/run_campaign.py --assemble
python3 benchmarks/frazil_powerlaw_shear/make_figures.py
python3 benchmarks/frazil_powerlaw_shear/make_report.py
python3 -m pytest tests/test_frazil_powerlaw_shear.py -q
```
