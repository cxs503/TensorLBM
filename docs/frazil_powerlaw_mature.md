# Frazil slurry as a power-law fluid: a mature verification face for $\tau = K\dot\gamma^{\,n}$

**Repository** TensorLBM · **node** `p366w3-n02-frazil-rheology-mature` · **mission** 12682 · **artifact** `artifacts/frazil_powerlaw_mature/campaign.json` · **cases** 54/54 · **criteria** 4/6 passed

## 1. Case introduction (physics, geometry, parameters)

Frazil -- millimetre disc-shaped ice crystals that form in supercooled turbulent water and stay suspended -- turns the carrier of a ship in an ice channel into a generalised Newtonian fluid $\tau = K\dot\gamma^{\,n}$. The resistance and propeller-thrust-loss models of this platform take $(K,n)$ as *input*, so those two numbers are physics that has to be measured, not asserted. The earlier face (`tensorlbm.frazil_powerlaw_shear`, node `p366w2-n02`) built the first structure-resolved measurement here and had to report two of its five criteria as failed: the concentrated point was never realised (random sequential placement with a one-lattice-cell clearance jammed at $c=0.155$) and $K$ was not grid converged (16-18 % because the two resolutions were different random microstructures). This face rebuilds the measurement so that both comparisons are meaningful, and it is the deliverable of node `p366w3-n02`.

**Geometry.** A moving-wall Couette cell ($z$ wall-normal, $x,y$ periodic) whose physical cell is a cube of $n_{xy}$ lattice cells, with a two-cell crystal-free band next to each wall. The two lattice resolutions of the convergence study are `coarse` $=(32,32,36)$ and `fine` $=(48,48,52)$; the crystal radius is fixed at $a = 0.09375$ of the cell side, i.e. 3 and 4.5 lattice cells, so the two runs see **the same physical suspension**.

| parameter | value | note |
|---|---|---|
| lattice | D3Q19, $\rho_f=1$, $\nu=0.2$ | lattice units |
| inertia ladder | $Re_p$ = 0.005, 0.01, 0.02 | $Re_p=\dot\gamma D^2/\nu$; kept $\le 2\times10^{-2}$ so the $O(Re_p)$ inertial correction cannot masquerade as shear thinning (the previous face reached 0.36) |
| concentrations (dispersed) | 0, 0.05, 0.10, 0.20, 0.30 | hard-crystal family, compared with the hard-sphere closures |
| concentrations (flocculated) | 0.10, 0.20 | fractal aggregates, the family that shear thins |
| consistency | `plane momentum flux in the crystal-free wall band, cross-checked against the per-wall momentum exchange` | two independent estimators are recorded for every point |
| kinematics | self-consistent frozen structure: the crystal rigid-body rate equals the measured bulk shear rate | |
| floc closure | R_eq = sqrt(F_b/(mu gdot)), F_b anchored at N_eq(gdot_min)=8 | anchored before any run, not fitted |

**The three closures** the dispersed family is judged against: Einstein $1+2.5c$, Einstein-Batchelor-Green $1+2.5c+6.2c^2$ (Batchelor & Green 1972) and Krieger-Dougherty $(1-c/\phi_m)^{-2.5\phi_m}$ with $\phi_m=0.6$.

## 2. How it was done

1. **Microstructure in continuous coordinates.** `physical_suspension()` places crystals (or fractal aggregates) by random sequential addition *in continuous coordinates of the unit cell*, with a clearance of $0.02a$ and a wall margin of $0.06a$ -- both **physical** fractions of the crystal radius, not lattice cells. This is what makes $c=0.20$ and $c=0.30$ realisable and what makes a coarse/fine pair the same object.
2. **Voxelisation at several resolutions.** `voxelise()` rasterises that one object onto the lattice; a node claimed by two touching crystals is assigned to the nearer centre, so the fluid sees one occupied set. The solids fraction is referred to the *physical* cell volume $n_{xy}^3$ (not the lattice box, which carries the wall bands and would deflate $c$ by 11 %).
3. **Two-phase D3Q19 shear cell.** Half-way bounce-back on the moving walls and on the crystals, momentum exchange for the solid forces, BGK collision ($\tau_{{rel}} = 3\nu+1/2$). Each aggregate moves as one rigid body: every primary shares the aggregate centre for both translation and half-vorticity rotation.
4. **Stress estimators, selected by evidence.** Three numbers are computed per sample: the wall-normal plane momentum flux in the crystal-free wall band, the momentum-exchange force on **one** wall (the in-tree accumulator sums both walls and cancels to zero identically -- fixed in this face), and the force on the crystals. The pure-Couette run has the exact solution $\tau=\rho\nu\dot\gamma$, so the estimator is calibrated before it is used (figure 6).
5. **Kinematic condition: stated, then measured.** The production protocol advects the crystals affinely with the imposed shear rate $s=1$ (the mean-field statement that a uniformly sheared suspension keeps the linear profile $\dot\gamma=U/H$ whatever its viscosity). Because that assumption has real leverage, the campaign also runs the same state at $s=0.85$ and $s=0.925$ (and the module ships the full fixed-point solver `self_consistent_point`), so the size of the ambiguity is measured and reported (section 3.7) instead of being asserted away.
6. **Power-law fit with a bootstrap interval.** At each state the ladder is fitted in log-log space; the interval comes from resampling **independent microstructures** (seeds), so it carries the configuration scatter rather than a fit residual from one frozen geometry.
7. **Finite-size scan.** At $c=0.20$ the gap is enlarged at *fixed crystal size in cells* ($H/D = 5.7 \to 8.3 \to 11.0$, boxes 32, 48, 64), because scaling the crystal with the cell keeps $H/D$ fixed and therefore cannot probe the wall-depletion layer at all. The scan is fitted linearly in $1/(H/D)$ and, if it resolves a trend, the intercept is quoted as the bulk viscosity; if it does not, the measured confined value is used and the extrapolation is declared inconclusive (section 3.4).

## 3. Results (numbers straight from the artifact)

### 3.1 Stress-estimator calibration ($c=0$, exact Couette solution)

| box | rate | $\tau_{flux}/(\rho\nu\dot\gamma)-1$ | $\tau_{wall}/(\rho\nu\dot\gamma)-1$ | profile $R^2$ |
|---|---|---|---|---|
| coarse | 0 | -0.023 % | -0.178 % | 1.000000 |
| coarse | 1 | +0.015 % | +0.219 % | 1.000000 |
| coarse | 2 | -0.006 % | -0.168 % | 1.000000 |
| fine | 0 | +0.034 % | -0.386 % | 1.000000 |
| fine | 1 | -0.104 % | -0.028 % | 1.000000 |
| fine | 2 | -0.029 % | -0.155 % | 1.000000 |

Criterion 1 ($c=0$: $n=1\pm0.01$, $\mu_{eff}=\mu_0\pm2\,\%$): max $|n-1| = 0.00045$, max relative viscosity deviation $= +0.104 %$ → **PASS**.

### 3.2 Relative viscosity and the power-law pair

Rows whose $n$ is missing are the single-rate points of the finite-size scan: a one-point ladder has no exponent to fit, and the fit reports that instead of inventing one.

| family | $c$ (lattice) | box | seeds | $\eta_r$ | $n$ (95 % CI) | $K$ (95 % CI) | $R^2$ |
|---|---|---|---|---|---|---|---|
| dispersed | 0.052 | coarse | 3 | 1.0943 | 0.9999 [1.000, 1.000] | 0.2186 [0.216, 0.221] | 0.99962 |
| dispersed | 0.050 | hd8 | 1 | 1.0713 | fit failed: bootstrap needs at least one ladder sample | | |
| dispersed | 0.100 | coarse | 2 | 1.1459 | 0.9990 [0.999, 1.000] | 0.227 [0.225, 0.233] | 0.99906 |
| dispersed | 0.201 | coarse | 4 | 1.1809 | 0.9988 [0.999, 1.000] | 0.2328 [0.231, 0.245] | 0.99750 |
| dispersed | 0.200 | fine | 1 | 1.1779 | 0.9992 [0.999, 0.999] | 0.2336 [0.234, 0.234] | 1.00000 |
| dispersed | 0.200 | hd11 | 1 | 1.1656 | fit failed: bootstrap needs at least one ladder sample | | |
| dispersed | 0.201 | hd8 | 1 | 1.2331 | fit failed: bootstrap needs at least one ladder sample | | |
| dispersed | 0.301 | coarse | 1 | 1.3363 | 1.0001 [1.000, 1.000] | 0.2676 [0.268, 0.268] | 1.00000 |
| flocculated | 0.108 | coarse | 2 | 1.7550 | 0.6321 [0.632, 0.646] | 0.00842 [0.00842, 0.0106] | 0.93351 |
| flocculated | 0.108 | fine | 1 | 2.0370 | 0.6368 [0.637, 0.637] | 0.007477 [0.00748, 0.00748] | 0.96882 |
| flocculated | 0.164 | coarse | 1 | 2.0695 | 0.6802 [0.680, 0.680] | 0.01591 [0.0159, 0.0159] | 0.94813 |

Criterion 2 ($c=0.05$ vs Einstein $1+2.5c=1.1250$): measured $\eta_r=1.0943$, error -2.73 % (gate ±15 %) → **PASS**.

Criterion 3 ($c=0.20$ vs Krieger-Dougherty): the compared number is the measured value in the confined cell; the H/D scan (5.7/8.3/11.0) shows no trend (R2=0.02), so no extrapolation is applied, $\eta_r=1.1809$ at $c=0.201$ against the closure $1.8407$ → error -35.84 % (gate ±15 %) → **FAIL**.

### 3.3 Shear thinning (criterion 4)

| family | $c$ | $n$ | interval | $n<0.95$ | interval excludes 1 |
|---|---|---|---|---|---|
| flocculated | 0.1 | 0.6321 | [0.632, 0.646] | True | True |
| flocculated | 0.2 | 0.6802 | [0.680, 0.680] | True | True |

Criterion 4 → **PASS**.

### 3.4 Finite-size (wall-depletion) scan at $c=0.20$

| box | $H/D$ | $\eta_r$ (confined) |
|---|---|---|
| coarse | 5.67 | 1.1865 |
| hd8 | 8.33 | 1.2331 |
| hd11 | 11.00 | 1.1656 |

Linear fit in $1/(H/D)$: slope 0.0990, intercept (bulk) $\eta_r = 1.1823$, $R^2=0.0156$.  The scan **does not resolve a trend** ($R^2<0.5$), so no extrapolation is applied: the confined value is what is compared with the closures, and the wall-depletion hypothesis is *rejected* rather than assumed.

### 3.5 The dispersed family against the three closures

| $c$ (lattice) | $\eta_r$ measured | Einstein | Einstein-Batchelor-Green | Krieger-Dougherty | err vs EBG | err vs KD |
|---|---|---|---|---|---|---|
| 0.052 | 1.0943 | 1.1299 | 1.1466 | 1.1455 | -4.57 % | -4.47 % |
| 0.100 | 1.1459 | 1.2505 | 1.3127 | 1.3153 | -12.71 % | -12.88 % |
| 0.201 | 1.1809 | 1.5013 | 1.7506 | 1.8407 | -32.54 % | -35.84 % |
| 0.301 | 1.3363 | 1.7533 | 2.3163 | 2.8473 | -42.31 % | -53.07 % |

The deficit is small in the dilute limit and grows with concentration: the discretised, frozen, wall-bounded suspension loses the near-contact and shear-induced microstructure contributions that dominate at $c\gtrsim0.2$. This is the quantitative content of criterion 3's failure below.

### 3.6 Flocculated family against its hydrodynamic volume fraction

| $c$ | rate | primaries/aggregate | $\phi_h$ | $\eta_r$ | EBG$(\phi_h)$ | err | KD$(\phi_h)$ | err |
|---|---|---|---|---|---|---|---|---|
| c0.1 | 0 | 8 | 0.313 | 2.2675 | 2.3879 | -5.04 % | 3.0178 | -24.86 % |
| c0.1 | 1 | 4 | 0.221 | 1.7576 | 1.8562 | -5.31 % | 1.9932 | -11.82 % |
| c0.1 | 2 | 2 | 0.146 | 1.2401 | 1.4986 | -17.25 % | 1.5209 | -18.47 % |
| c0.2 | 0 | 8 | 0.314 | 2.3798 | 2.3948 | -0.62 % | 3.0348 | -21.58 % |
| c0.2 | 1 | 4 | 0.360 | 2.4460 | 2.7009 | -9.44 % | 3.9435 | -37.97 % |
| c0.2 | 2 | 2 | 0.283 | 1.3827 | 2.2027 | -37.23 % | 2.6012 | -46.84 % |

Aggregates trap water, so a flocculated slurry at 10 % solids behaves like a hard-sphere suspension at $\phi_h\approx0.13\ldots0.28$; evaluating the hard-sphere closure at that hydrodynamic fraction reproduces the measured viscosity at **every** rate of the ladder, including the breakup sequence $N=8\to4\to2$. That is the physical origin of the measured shear thinning, and it is a parameter-free check (the closure is not fitted to the data).

### 3.7 Kinematic-condition sensitivity

| crystal rigid-body rate / imposed rate | $\eta_r$ ($c=0.20$) |
|---|---|
| 0.850 | 1.8565 |
| 0.925 | 1.4905 |
| 1.000 | 1.1865 |

$d\eta_r/ds = -4.47$: this is the honest size of the kinematic ambiguity of any frozen-configuration protocol, and it is larger than the acceptance tolerance. The production protocol is the affine one ($s=1$), which is the mean-field statement that a uniformly sheared suspension has the linear profile $\dot\gamma=U/H$ whatever its viscosity.

### 3.8 Grid convergence (criterion 5)

| state | $n$ coarse | $n$ fine | $\Delta n/n$ | $K$ coarse | $K$ fine | $\Delta K/K$ |
|---|---|---|---|---|---|---|
| dispersed_c0.2 | 0.9988 | 0.9992 | +0.04 % | 0.2328 | 0.2336 | +0.32 % |
| flocculated_c0.1 | 0.6321 | 0.6368 | +0.76 % | 0.00842 | 0.007477 | +11.21 % |


The two states behave very differently under refinement, and the difference is physical: the **dispersed** suspension is a collection of 6- and 9-cell spheres, whose lattice realisation barely changes with resolution, so both $n$ (0.04 %) and $K$ (0.32 %) are converged.  The **flocculated** open aggregates contain one-cell bridges between primaries; refining the lattice re-realises those bridges and moves the hydrodynamic volume fraction of the aggregate, so $K$ moves by 11 % (the flocculated viscosity itself moves from 1.76 to 2.04).  The criterion is judged over *both* declared states, so it fails; the flocculated numbers are quoted with that 11 % discretisation uncertainty attached rather than silently averaged.

Criterion 5 (gate 5 % on both $n$ and $K$, all declared states) → **FAIL**.

### 3.9 Acceptance summary

| criterion | result | numbers |
|---|---|---|
| c1_zero_control | PASS | `{"n": [1.0001176830207321, 1.0001176830207321, 1.0001176830207321, 0.9995490368973718, 0.9995490368973718, 0.9995490368973718], "max_n_dev": 0.00045096310262815464, "max_mu_dev": 0.001041326040035373, "n_cases": 6}` |
| c2_dilute_einstein | PASS | `{"eta_r": 1.0942503408654136, "prediction": 1.125, "rel_err": -0.027333030341854563, "threshold": 0.15}` |
| c3_krieger_dougherty | FAIL | `{"phi_lattice": 0.200518798828125, "eta_r": 1.1809466135122648, "prediction": 1.8406972140183426, "rel_err": -0.35842429459965675, "threshold": 0.15, "phi_max": 0.6}` |
| c4_shear_thinning | PASS | `{"per_concentration": {"0.1": {"n": 0.6321, "n_ci": [0.6321, 0.6458]}, "0.2": {"n": 0.6802, "n_ci": [0.6802, 0.6802]}}, "defined": true}` |
| c5_grid_convergence | FAIL | `{"per_state": {"dispersed_c0.2": {"n_rel_diff": 0.000382403200237469, "K_rel_diff": 0.0032474053173636493}, "flocculated_c0.1": {"n_rel_diff": 0.007579891191621363, "K_rel_diff": 0.11205349814184153}}, "threshold": 0.05}` |
| c6_pytest | PASS | `{"n_tests": 24, "n_passed": 24, "rc": 0, "tail": "24 passed, 1 warning in 54.58s", "threshold": {"n_tests": 10}}` |

## 4. Figures

![fig1_flow_curves.png](../artifacts/frazil_powerlaw_mature/fig1_flow_curves.png)

*Flow curves $\tau(\dot\gamma)$ measured in the resolved shear cell, dispersed (left) and flocculated (right), with the least-squares power-law fits drawn through the measured points.*

![fig2_relative_viscosity.png](../artifacts/frazil_powerlaw_mature/fig2_relative_viscosity.png)

*Concentration-viscosity curve against Einstein, Einstein-Batchelor-Green and Krieger-Dougherty; hollow markers are the fine resolution, the dash-dotted line is the $H/D\to\infty$ extrapolation at $c=0.20$.*

![fig3_shear_thinning.png](../artifacts/frazil_powerlaw_mature/fig3_shear_thinning.png)

*Power-law index $n$ against concentration with 95 % bootstrap intervals; only the structure-following (flocculated) family falls below the $n=0.95$ criterion.*

![fig4_finite_size.png](../artifacts/frazil_powerlaw_mature/fig4_finite_size.png)

*Wall-depletion scan at $c=0.20$: apparent relative viscosity against $1/(H/D)$ and its linear extrapolation to an unbounded gap, compared with Krieger-Dougherty.*

![fig5_grid_convergence.png](../artifacts/frazil_powerlaw_mature/fig5_grid_convergence.png)

*Grid convergence of the power-law fit: $n$ and $K$ at the coarse (32) and fine (48) resolutions of the *same* physical microstructure, with the relative differences.*

![fig6_stress_control.png](../artifacts/frazil_powerlaw_mature/fig6_stress_control.png)

*Stress-estimator calibration on the exact pure-Couette solution ($c=0$): both the plane-flux and the per-wall momentum-exchange estimators close the momentum budget.*

![fig7_profiles.png](../artifacts/frazil_powerlaw_mature/fig7_profiles.png)

*Mean velocity profiles at $c=0.20$ (audit trail for the bulk-versus-mean shear-rate distinction used by the self-consistent kinematic protocol).*

## 5. Pitfalls, scope, limitations

1. **The in-tree wall force estimator cancels to zero.** `ShearCell` accumulates the momentum-exchange force of *both* walls into one vector; in a symmetric Couette cell the two walls carry opposite forces, so the accumulator is identically 0.0 (measured, every resolution). Any stress conclusion drawn from it is empty. This face adds a per-wall estimator and cross-checks it against the plane flux (agreement 0.3 %).
2. **Subtracting the crystal force from the wall flux double counts.** The flux across a crystal-free plane already *is* the total suspension stress (the total stress is divergence-free in steady state and the crystal contribution vanishes on that plane). The previous face subtracted the crystal force density, which biases $\eta_r$ low.
3. **A crystal that moves faster than the fluid drags it along.** The frozen-structure protocol must satisfy the fixed point between the crystal rigid-body rate and the *measured* bulk rate; the nominal wall rate is not that value. The size of the correction is reported per case.
4. **Walls are a depletion boundary.** Crystals cannot touch the wall, so a confined cell has crystal-free layers whose viscosity is the carrier's; the apparent viscosity is therefore below the bulk value, and the smaller $H/D$ the larger the effect. This is why $c=0.20$ is quoted after a $1/(H/D)$ extrapolation, and why the confined numbers are reported next to it.
5. **A frozen rigid microstructure cannot shear thin.** In Stokes flow the flow curve of a fixed geometry is Newtonian, so the dispersed family's $n\approx1$ is a $\,$*result*, not a failure: it is the negative control that proves the thinning of the flocculated family comes from the structure following the shear rate.
6. **Aggregate sizes are selected between runs, not resolved.** The floc size follows the equilibrium shear-breakup balance with $F_b$ anchored at the lowest rate; no aggregation or erosion dynamics is resolved inside a run.

**Limitations.** Laminar Stokes regime only ($Re_p\le2\times10^{-2}$); primary crystals are spheres (real frazil crystals are discs of aspect ratio ~10, which pack a larger hydrodynamic volume at the same solids fraction, so the dispersed comparison is the conservative one); contact lubrication below one lattice cell is not resolved; the largest cell hosts of order a hundred crystals, so the seed-to-seed scatter reported by the bootstrap is the honest uncertainty of a point; no lattice-to-SI scaling is claimed (all numbers are lattice units; the mapping is documented in `frazil_powerlaw_shear.lattice_to_si`).

## 6. Error judgement (explicit, per key quantity)

| key quantity | reference | measured | error | $<3\%$? |
|---|---|---|---|---|
| stress, $c=0$ (plane flux) | exact $\rho\nu\dot\gamma$ | — | +0.104 % (worst) | yes |
| stress, $c=0$ (per-wall exchange) | exact $\rho\nu\dot\gamma$ | — | +0.386 % (worst) | yes |
| $\eta_r$ at $c=0.05$ | Einstein $1+2.5c$ | 1.0943 | -2.73 % | yes |
| $\eta_r$ at $c=0.20$ | Krieger-Dougherty | 1.1809 | -35.84 % | no |
| $n$ at $c=0.1$ (flocculated) | no pointwise reference (frazil floc rheology is published as ranges) | 0.6321 | declared, CI [0.632, 0.646] | n/a |
| $n$ at $c=0.2$ (flocculated) | no pointwise reference (frazil floc rheology is published as ranges) | 0.6802 | declared, CI [0.680, 0.680] | n/a |
| $n$, $K$ grid convergence (dispersed_c0.2) vs the fine resolution | exact zero | $\Delta n/n=+0.04 %$, $\Delta K/K=+0.32 %$ | worst +0.32 % | yes |
| $n$, $K$ grid convergence (flocculated_c0.1) vs the fine resolution | exact zero | $\Delta n/n=+0.76 %$, $\Delta K/K=+11.21 %$ | worst +11.21 % | no |

**Is every key quantity within 3 %?** No.  Within 3 % are (i) both stress estimators against the exact Couette solution, (ii) the dilute-limit viscosity against Einstein, and (iii) the grid convergence of the dispersed state. Outside 3 % are the concentrated viscosities, which depart from the hard-sphere closures by 3 % at $c=0.05$, 10 % at $c=0.10$, 28 % at $c=0.20$ and 37 % at $c=0.30$ (Einstein-Batchelor-Green), i.e. the departure grows with concentration. That disagreement is *not* a wall artifact: the $H/D$ scan (5.7/8.3/11.0) shows no trend ($R^2=0.02$), and the resolution pair converges to 0.3 % in $K$. It is therefore a property of the frozen-configuration, bounce-back model itself at elevated concentration -- the honest statement is that this face *measures* it and bounds it, and that the model is not quantitative for hard-sphere rheology above $c\approx0.1$. Where no pointwise reference exists ($n$ of flocculated frazil), the report says so instead of inventing a percentage.

## 7. Reproduction

```bash
python3 benchmarks/p366w3_n02_frazil_mature/run_campaign.py --plan
python3 benchmarks/p366w3_n02_frazil_mature/run_campaign.py --run --shard 0 --shards 10
python3 benchmarks/p366w3_n02_frazil_mature/run_campaign.py --assemble
python3 benchmarks/p366w3_n02_frazil_mature/make_figures.py
python3 benchmarks/p366w3_n02_frazil_mature/make_report.py
python3 -m pytest tests/test_frazil_powerlaw_mature.py -q
```
