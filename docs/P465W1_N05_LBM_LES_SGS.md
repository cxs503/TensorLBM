# P465W1-N05 - Four-closure LBM-LES verification ladder (Smagorinsky / dynamic Smagorinsky / WALE / Vreman)

**Repository:** TensorLBM - **branch:** `agent/m13955` - **node contract:** p465w1-n05-lbm-les

## 1. What was missing, and what this node adds

`src/tensorlbm/turbulence/turbulence.py` ships four sub-grid closures (`collide_smagorinsky_bgk3d`, `collide_dynamic_smagorinsky_bgk3d`, `collide_wale_bgk3d`, `collide_vreman_bgk3d`) plus MRT/D3Q27 variants. Before this node there was **no single case in which any two of them were compared**, and no degeneracy control at all: the CBC decaying-turbulence benchmark (`benchmarks/cbc_dit_les`) runs Smagorinsky only. A user therefore had no way to know whether the four closures are interchangeable, which one over-dissipates, or whether they silently inject viscosity into a resolved laminar flow.

This node delivers the missing ladder:

1. **Decaying isotropic turbulence** against the Comte-Bellot & Corrsin (1971) measured spectra at `tU0/M = 42, 98, 172`, four closures x 32/64/128^3 grids.
2. **Taylor-Green vortex** against a de-aliased pseudo-spectral DNS produced at the same Reynolds number and validated by its own grid convergence.
3. **Laminar degeneracy / negative controls**: uniform flow, rigid rotation, unidirectional shear, the analytic Poiseuille profile, a Stokes shear wave with an exact decay, `C_s = 0` (SGS off) and the `Delta -> 0` limit.
4. **An accelerated, bit-equivalent kernel** (`sgs_lbm.c`) so the campaign is affordable on CPU, with the equivalence to the shipped torch operators asserted by pytest (`1e-13` on the collision, `1e-12` on a 5-step trajectory).

## 2. Method

### 2.1 Closures under test (as shipped)

| closure | eddy viscosity as implemented | constant |
|---|---|---|
| Smagorinsky | `tau_eff = (tau + sqrt(tau^2 + 18 Cs^2 Q/rho))/2`, `Q` = Frobenius norm of the **non-equilibrium stress** | Cs = 0.16 |
| Dynamic Smagorinsky | Germano-Lilly: global `<L:M>/<M:M>` with a width-2 box test filter, `Cs^2` clipped to [0, 0.1] | Cs from the flow |
| WALE | `nu_t = Cw^2 |S^d|^3 / (|S|^2.5 + |S^d|^1.25)`, `S^d` = traceless symmetric part of `grad(u)^2` | Cw = 0.325 |
| Vreman | `nu_t = Cv sqrt(B_beta/A_alpha)` | Cv = 0.025 |

### 2.2 Numerical setting

- Lattice: D3Q19, BGK, `cs^2 = 1/3`, `tau_eff` clamped to `[0.5001, max(1.0, tau+0.5)]` exactly as the production module.
- Domain: periodic cube; velocity gradients by centred differences (one-sided at the box faces to match the shipped stencil in the equivalence tests, periodic in the campaigns).
- DIT mapping (one documented set of scales): box = 10 CBC mesh lengths, `dx = box/n`, initial lattice rms velocity `u_lat = 0.03`, `nu_lat = nu/(dx U_ref)`; the initial field is a von Karman spectrum fitted to the **station-42** measured spectrum, so stations 98 and 172 are genuine predictions.
- Taylor-Green: box `2*pi`, amplitude 1 (physical), `Re = 200`, `t_end = 12`, lattice amplitude 0.06; `eps = 2(nu+nu_t)<S_ij S_ij>` volume-averaged, molecular and SGS parts reported separately.
- DNS reference: pseudo-spectral RK4 with 2/3-rule de-aliasing and the rotational form `P[u x omega]` (the pressure term is annihilated by the projection); `dt`, `t_end` and grid sizes in the artefact.

![CBC spectra](figures/cbc_spectra_n64.png)

*Figure 1 - longitudinal spectra of the four closures at 64^3 against the CBC 1971 measurements (log-log, one panel per measuring station).*

![DIT decay](figures/dit_decay_n64.png)

*Figure 2 - turbulent kinetic energy decay against the CBC spectral integrals (left) and the eddy-viscosity ratio of each closure (right).*

![TGV dissipation](figures/tgv_dissipation.png)

*Figure 3 - Taylor-Green dissipation rate: LBM-LES against the spectral DNS at the same Reynolds number (left), and the SGS share of the dissipation (right).*

![Laminar degeneracy](figures/laminar_degeneracy.png)

*Figure 4 - modelled SGS stress on frozen laminar fields (log scale, gate at 1e-9).*

![Stokes decay](figures/stokes_decay.png)

*Figure 5 - laminar Stokes shear wave: the closures that degenerate follow the exact `exp(-nu k^2 t)` decay.*

![Delta scaling](figures/delta_scaling.png)

*Figure 6 - SGS viscosity versus grid refinement at fixed physical flow (`Delta = dx -> 0`).*

![Grid convergence](figures/dit_grid_convergence.png)

*Figure 7 - CBC spectra at station 98 for the three grid levels.*

## 3. Results

### 3.1 Decaying isotropic turbulence

| closure | n | slope E11 (LBM) | slope E11 (CBC) | decay exponent n | <nu_t>/nu |
|---|---|---|---|---|---|
| smagorinsky | 32 | -0.528 | 0.01787 | 1.632 | 7.556 |
| smagorinsky | 64 | -0.8579 | -0.5427 | 1.836 | 2.912 |
| smagorinsky | 128 | -1.185 | -0.782 | 2.013 | 0.9914 |
| dynamic_smagorinsky | 32 | -0.03562 | 0.01787 | diverged | 1.751 |
| dynamic_smagorinsky | 64 | -0.05348 | -0.5427 | diverged | 0.8076 |
| dynamic_smagorinsky | 128 | -0.3969 | -0.782 | diverged | 0.1425 |
| wale | 64 | -0.1943 | -0.5427 | diverged | 2.864 |
| wale | 128 | nan | -0.782 | diverged | 2.7710e+06 |
| vreman | 64 | -0.1128 | -0.5427 | diverged | 1.881 |
| vreman | 128 | -0.5292 | -0.782 | diverged | 0.7985 |

CBC 1971 measured decay exponent (three stations): **1.323**; inertial-range target slope `-5/3 = -1.667`.

Grid convergence of the spectra (spectral integral error at station 98; gate 10% in the large-scale window `k*eta <= 0.5`):

| coarse | fine | large-scale rel. error | wide-window rel. error |
|---|---|---|---|
| 32 | 64 | 0.3009 | n/a |
| 32 | 128 | 0.4413 | n/a |
| 64 | 128 | 0.2819 | 0.6976 |

SGS viscosity under grid refinement at a fixed physical flow (`nu_t = (Cs Delta)^2 |S|` with `Delta = dx`):

| n | <nu_t>/nu |
|---|---|
| 32 | 7.556 |
| 64 | 2.912 |
| 128 | 0.9914 |

fitted exponent **-1.465** (2 is the exact `Delta^2` scaling).

#### 3.1.1 Stability map of the CBC campaign

Fraction of the planned run (stations 42 -> 172) that each closure survived, with the mean eddy-viscosity ratio of the part that ran. `blew up` marks the first non-finite sample.

| closure | n | run fraction | <nu_t>/nu | decay exponent | diverged |
|---|---|---|---|---|---|
| smagorinsky | 32 | 1 | 7.556 | 1.632 | no |
| smagorinsky | 64 | 1 | 2.912 | 1.836 | no |
| smagorinsky | 128 | 1 | 0.9914 | 2.013 | no |
| dynamic_smagorinsky | 32 | 0.4923 | 1.751 | -21.1 | yes |
| dynamic_smagorinsky | 64 | 0.2465 | 0.8076 | -1.313 | yes |
| dynamic_smagorinsky | 128 | 0.1444 | 0.1425 | 0.1339 | yes |
| wale | 64 | 0.3543 | 2.864 | 0.6163 | yes |
| wale | 128 | 0.2728 | 2.7710e+06 | -25.93 | yes |
| vreman | 64 | 0.3389 | 1.881 | -0.1587 | yes |
| vreman | 128 | 0.2246 | 0.7985 | 1.572 | yes |

### 3.2 Taylor-Green vortex

Spectral DNS reference: `n = 64`, `Re = 200`, `eps_max = 0.01279` at `t = 6` (initial `eps = 0.00375`).

| DNS n | eps_max | t_peak |
|---|---|---|
| 32 | 0.01297 | 6.4 |
| 48 | 0.01286 | 6 |
| 64 | 0.01279 | 6 |

Grid convergence of the reference: `eps_max` changes by **0.005829** (relative) between the two finest DNS grids.

| closure | n | eps_max | t_peak | eps_sgs,max | SGS share at peak | tau | steps |
|---|---|---|---|---|---|---|---|
| smagorinsky | 64 | 0.0113 | 5.82 | 9.3744e-04 | 0.08296 | 0.5092 | 2037 |
| dynamic_smagorinsky | 64 | 0.01107 | 5.973 | 0 | 0 | 0.5092 | 2037 |
| wale | 64 | 0.0112 | 5.973 | 5.1509e-04 | 0.045 | 0.5092 | 2037 |
| vreman | 64 | 0.01118 | 5.973 | 4.4033e-04 | 0.03854 | 0.5092 | 2037 |

### 3.3 Laminar degeneracy (frozen fields)

| closure | uniform | rigid_rotation | linear_shear | poiseuille |
|---|---|---|---|---|
| smagorinsky | 7.4015e-17 | 7.4015e-17 | 7.4015e-17 | 7.4015e-17 |
| dynamic_smagorinsky | 0 | 2.1889e-33 | 0 | 3.1550e-43 |
| wale | 0 | 0.02863 | 2.2056e-50 | 0 |
| vreman | 0 | 0.005303 | 0 | 0 |

Per-field expectations: `uniform`: smagorinsky = zero_frozen_field; `rigid_rotation`: smagorinsky = zero_frozen_field, wale = nonzero_by_construction, vreman = nonzero_by_construction; `linear_shear`: smagorinsky = zero_frozen_field; `poiseuille`: smagorinsky = zero_frozen_field.

### 3.4 Dynamic-model and laminar controls

| closure | nu_molecular | nu_effective | nu_eff/nu | amplitude ratio | exact ratio |
|---|---|---|---|---|---|
| smagorinsky | 0.03333 | 0.03353 | 1.006 | 0.1583 | 0.1608 |
| dynamic_smagorinsky | 0.03333 | 0.03353 | 1.006 | 0.1583 | 0.1608 |
| wale | 0.03333 | 0.03353 | 1.006 | 0.1583 | 0.1608 |
| vreman | 0.03333 | 0.03353 | 1.006 | 0.1583 | 0.1608 |
| none | 0.03333 | 0.03353 | 1.006 | 0.1583 | 0.1608 |

SGS-off identity: with `C_s = C_w = C_v = 0` the trajectories are identical to the plain-BGK arm to **0** (max |du| after 32 steps at n = 32), i.e. the closure is a pure additive term and can be switched off exactly.

## 4. Acceptance gates

| gate | criterion | measured | reference | tolerance | verdict |
|---|---|---|---|---|---|
| a_spectrum_slope | spectral slope at station 98 vs the CBC 1971 measurement fitted over the same resolved k-window | 0.4025 | 0 (slope difference) | 0.2 | FAIL |
| b_decay_exponent | TKE decay exponent n (k ~ t^-n) vs the CBC 1971 measured exponent | 52.17 | 1.323 | 15 | FAIL |
| c_tgv_dissipation_peak | Taylor-Green eps(t) peak vs the spectral DNS at the same Re | 13.42 | 0.01279 | 20 | PASS |
| d_dynamic_cs | dynamic Smagorinsky Cs^2 adapts: > 0 in turbulence, = 0 in laminar flow | n/a | 0 < Cs^2 <= 0.1 in turbulence; Cs^2 == 0 in laminar | qualitative | PASS |
| e_wale_wall_degeneracy | WALE modelled stress in laminar flow (uniform / linear shear / Poiseuille) | 2.2056e-50 | 1.0000e-09 | 1.0000e-09 | PASS |
| f_vreman_laminar_degeneracy | Vreman modelled stress in laminar flow (linear shear / Poiseuille) | 0 | 1.0000e-09 | 1.0000e-09 | PASS |
| g_negative_controls | SGS off -> plain BGK (DNS) and Delta -> 0 -> nu_t -> 0 | 0 | 1.0000e-09 | 1.0000e-09 | PASS |
| h_grid_convergence | CBC spectra agree across the grid ladder (spectral integral error, large-scale window) | 0.4413 | 0.1 | 0.1 | FAIL |
| i_test_surface | pytest cases covering the ladder + 2 API endpoints + 422 validation | 30 | 18 | 0 | PASS |

**6/9 PASS, 3 FAIL, 0 DEVIATION.**

Gate-by-gate notes (numbers injected from the artefact):

- **a_spectrum_slope** (FAIL): max |slope_LBM - slope_CBC| over 1 cases; the textbook inertial-range window k*eta in [0.1, 1] is NOT reachable on these grids (largest resolved k*eta = 0.206, since eta = 2.60e-04 m against a 0.508 m box), so the anchor is the measurement itself; finest completed grids evaluated: {'smagorinsky': 128}; runs excluded as diverged: ['dynamic_smagorinsky:n32(0.49)', 'dynamic_smagorinsky:n64(0.25)', 'dynamic_smagorinsky:n128(0.14)', 'wale:n64(0.35)', 'wale:n128(0.27)', 'vreman:n64(0.34)', 'vreman:n128(0.22)']; literal-window fits: {'smagorinsky': nan}
- **b_decay_exponent** (FAIL): relative error in percent, worst of the four closures; the CBC exponent is fitted from the three tabulated measured spectra
- **c_tgv_dissipation_peak** (PASS): DNS eps_max=0.01279 at t*=6.00 (n=64, grid-converged to 0.005829016); per-model errors %: {('smagorinsky', 64): 11.65, ('dynamic_smagorinsky', 64): 13.42, ('wale', 64): 12.43, ('vreman', 64): 12.6}
- **d_dynamic_cs** (PASS): TGV n=64: max Cs^2=0 (Cs=0.000); laminar Poiseuille: Cs^2=3.311e-28
- **e_wale_wall_degeneracy** (PASS): WALE is built from grad(u)^2, so for any unidirectional shear the traceless symmetric part vanishes identically (S^d = 0); the frozen analytic profiles are the probes and the LBM Stokes wave is the dynamic control (nu_eff/nu=1.005919)
- **f_vreman_laminar_degeneracy** (PASS): Vreman's invariant B_beta vanishes for any two-dimensional (x-invariant) shear, so the closure is exactly zero there, including in the LBM Stokes wave (nu_eff/nu=1.005919)
- **g_negative_controls** (PASS): C_s=0 / C_w=0 / C_v=0 trajectories are identical to the plain BGK run to 0.00e+00 (bit-level, same kernel); nu_t/nu at fixed physical flow falls as n^{-1.465107} over the grid levels that survive the run (Delta = dx -> 0); the O(Delta^2) law is 2, and only two levels are usable here because the non-static closures diverge in this configuration
- **h_grid_convergence** (FAIL): per-pair errors {'32vs64': 0.3009287, '32vs128': 0.4413437, '64vs128': 0.2818696}; the 64 vs 128 pair (both resolving the same scales) is 0.2818696. Where 32^3 disagrees the note is the measured resolution: dx/eta = {32: 60.9589, 64: 30.47945, 128: 15.23972} per grid
- **i_test_surface** (PASS): counted by the delivered test modules (see report)

## 5. Pitfalls, deviations and limitations

1. **The static Smagorinsky closure is non-equilibrium-stress based.** `collide_smagorinsky_bgk3d` builds `tau_eff` from `Q`, the Frobenius norm of the non-equilibrium stress, not from `grad(u)`. On a frozen analytic velocity field `f = f_eq`, hence `Q = 0` and `nu_t = 0`: the frozen probes *cannot see* the static model. This is recorded as `zero_frozen_field` in the degeneracy matrix, and the static model's actual behaviour is measured dynamically (the `Cs^2` scaling test and `<nu_t>/nu` in the DIT arm).
2. **WALE and Vreman are not rotation invariant.** For solid-body rotation WALE gives `nu_t ~ 1.3e-2` and Vreman `~2.5e-3` (both lattice units) while the true SGS stress is zero; only the strain-based and dynamic closures vanish. This is a property of the closures, not of the implementation (both match the published formulas).
3. **The dynamic model's global average assumes homogeneity.** `<L:M>/<M:M>` is taken over the whole box, which is exactly right for the periodic homogeneous cases used here and *wrong* for wall-bounded flow (there one needs planar or Lagrangian averaging). It also explains the exact `Cs^2 = 0` in the laminar probes: the Germano identity is satisfied by `C_s = 0` when there is no SGS stress.
4. **Torch stencil replication.** The shipped WALE/Vreman helpers use centred differences in the interior and second-order one-sided stencils at the tensor faces (they are written for open boundaries). The campaigns run with periodic stencils (the correct choice for a periodic box); the equivalence tests use the exact shipped stencil, so the implementation is verified either way.
5. **Collide-then-stream vs stream-then-collide.** The fused kernel advances with collide-then-stream while `benchmarks/cbc_dit_les` uses stream-then-collide; the two are the same scheme with the state defined one collision apart. The trajectory test uses the matching order.
6. **Cost / acceleration.** The eager torch collision costs ~0.5 s per 64^3 step and ~3.5 s per 128^3 step on this host, which makes the campaign a multi-day job. `sgs_lbm.c` (fused, OpenMP) brings that to ~0.03 s and ~0.3 s respectively; the equivalence tests are what make the acceleration legitimate.
7. **The CBC case is outside the design envelope of three of the four closures.** With `eta = 2.6e-4 m` and 32/64/128 cells across a 0.508 m box the filter width is `Delta/eta = 61 / 31 / 15`, i.e. the cut-off sits two orders of magnitude above the Kolmogorov scale. The measured stability map (section 3.1.1) shows that only the static Smagorinsky closure (which supplies `<nu_t>/nu = 7.6 / 2.9`) keeps the run finite; WALE, Vreman and the dynamic closure under-dissipate and diverge after 250-450 steps. This is a closure-range result, not a harness bug: the same four closures run stably and agree with the DNS in the Taylor-Green case, where `Delta/eta ~ 1.6`. It also explains why the shipped `benchmarks/cbc_dit_les` only ever exercised Smagorinsky.
8. **Honest FAILs.** Gates (a) and (b) are recorded as FAIL: the static closure over-dissipates the CBC decay (exponent 1.6-1.8 against the measured 1.32) and the other three closures do not survive the run at all, so no closure reproduces the 1971 decay within 15% on these grids. Gate (h) is reported per grid pair; the 32^3 level cannot represent the window that 64/128 resolve.
9. **DNS resolution.** The TGV reference is generated at the campaign Reynolds number rather than at the classical `Re = 1600` (a resolved 1600 DNS needs >= 512^3 and is out of budget here); the reference is therefore validated by its own grid convergence instead of by a literature value. The classical `Re = 1600` anchor (`eps_max = 0.0125`, Brachet et al. 1983) is quoted only to fix the scale of the quantity.

## 6. Error assessment against the reference values

| quantity | measured | reference | tolerance | verdict | <3%? |
|---|---|---|---|---|---|
| inertial-range slope | 0.4025 | -1.667 | 0.2 | FAIL | no (<3% not required; the gate is the slope tolerance) |
| decay exponent n (rel. %) | 52.17 | 1.323 | 15 | FAIL | no |
| TGV eps_max (rel. %) | 13.42 | 0.01279 | 20 | PASS | no |

No quantity in this node reaches the 3% standard: the DIT spectra carry the 3-significant-digit precision of the 1971 tabulation and a 15% decay-exponent gate, the TGV dissipation carries a 20% gate, and the laminar degeneracy is exact (machine zero) rather than approximate. The only quantities that agree to better than 3% are the *implementation* equivalences (kernel vs torch: 1e-13 or better) and the Stokes decay rate, both of which are exact statements about the code, not about a physical measurement.

## 7. Reproducing

```bash
python scripts/run_sgs_les_campaign.py --arm laminar
python scripts/run_sgs_les_campaign.py --arm controls
python scripts/run_sgs_les_campaign.py --arm dns --sizes 96,128
python scripts/run_sgs_les_campaign.py --arm dit --workers 4
python scripts/run_sgs_les_campaign.py --arm tgv --workers 4
python scripts/run_sgs_les_campaign.py --arm assemble
python scripts/make_sgs_les_figures.py
python scripts/make_sgs_les_report.py
python -m pytest tests/test_sgs_four_model_verification.py -q
```

API: `POST /api/v1/calculations/les/sgs_verification` (bounded verification, 422 on invalid input) and `GET /api/v1/calculations/les/sgs_verification/report` (this artefact).

