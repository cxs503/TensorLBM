# P190-N01 — LBM free-surface fill-transport Zalesak slotted-cylinder rotation benchmark

**Campaign node:** P190-N01 · **Repository:** TensorLBM · **Library module:** `src/tensorlbm/free_surface_zalesak_lbm.py`
**Driver:** `benchmarks/zalesak_rotation/run_zalesak_lbm_campaign.py` · **Figures:** `benchmarks/zalesak_rotation/make_lbm_figures.py` · **Tests:** `tests/test_free_surface_zalesak_lbm.py`
**Artefacts:** `artifacts/zalesak_p190_n01/zalesak_lbm_all.json` (all numbers below are injected from it), `artifacts/zalesak_p190_n01/figures/`
**Runtime:** Python 3.11.6, torch 2.7.1, float64, Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38, threads=6

---

## 1. Case introduction

### 1.1 Physical background

Zalesak's (1979) slotted cylinder is the standard benchmark of an interface-capturing operator. A slotted cylinder sits off-centre in a periodic unit box and is carried by the prescribed, analytically divergence-free rigid rotation

```
u(x) = Omega * ( -(y - y0),  x - x0 )        (2-D, D2Q9)
u(x) = Omega * ( -(z - z0), 0, x - x0 )      (3-D, D3Q19, x-z plane)
```

After one revolution `t = 2*pi/Omega` the exact solution is the initial condition again, so every deviation is a discretisation error of the *fill-transport operator* — the part of a free-surface LBM that owns the GAS/LIQUID/INTERFACE bookkeeping. This node grades TensorLBM's lattice-population (fill-fraction) transport path on that benchmark, complementing P189-N06, which graded the finite-volume face-flux VOF operator.

Two different transport paths now exist in the repository and are compared here on identical geometry, identical velocity fields and identical metrics:

| path | module | transport of the liquid fraction |
|---|---|---|
| lattice populations (this node) | `tensorlbm.free_surface_zalesak_lbm` | D2Q9 (9) / D3Q19 (19) population fluxes, BGK relaxation, free-surface bound ledger |
| finite-volume VOF (P189-N06) | `tensorlbm.free_surface_zalesak` | 4 (2-D) / 6 (3-D) face fluxes, MUSCL + van-Leer, FCT-limited compression |

The velocity field is prescribed rather than solved because the benchmark is deliberately kinematic (Zalesak 1979; Rider & Kothe 1998), and because a free liquid blob spun at zero gravity cannot hold solid-body rotation at all (that would need `p < p_atm` inside the blob), so a fully coupled run has no known exact solution to grade against. The prescribed field is discretely divergence-free (criterion f), hence every volume error below is an interface-operator error and not a compressibility artefact.

### 1.2 Geometry and parameters

| parameter | 2-D (D2Q9) | 3-D (D3Q19) |
|---|---|---|
| body | slotted cylinder | slotted sphere (slab-faced slot) |
| centre | (0.5, 0.75) | (0.5, 0.5, 0.75) |
| radius | 0.15 | 0.15 |
| slot width / depth | 0.05 / 0.25 | 0.05 / 0.25 |
| rotation centre | (0.5, 0.5) | (x,z) = (0.5, 0.5) |
| domain | unit square, periodic | unit cube, periodic |
| peak lattice speed | 0.1 cells/step | 0.1 cells/step |
| analytic liquid volume | 0.058232 | -- |
| discrete volume N=64 / N=128 | 0.058090 / 0.058043 | -- |

The discrete initial volume differs from the analytic volume by -2.435e-01 % at N=64 and -3.254e-01 % at N=128: that is the sub-cell supersampling error of the geometry, i.e. a property of the initial condition, not of the transport operator.

### 1.3 Transport operator (summary)

The state is `(phi, g)`: the liquid fraction `phi` (0 = gas, 1 = liquid, sub-cell values live in INTERFACE cells) and its `Q` lattice populations `g` with `sum_q g_q = phi`. One step is a scalar-lattice BGK update

```
g_q^eq = w_q phi (1 + 3 c_q . u)                     equilibrium of the fill level
g_q^post = g_q + (g_q^eq - g_q)/tau                  BGK, nu = c_s^2 (tau - 1/2)
phi^{n+1}(i) = sum_q g_q^post(i - c_q)               D2Q9 / D3Q19 pull streaming
```

followed by the free-surface **mass ledger**: a cell that overfills hands its surplus to neighbouring cells with spare capacity and a cell that over-empties draws the deficit from neighbours that still carry fill (the continuous analogue of the Körner (2005) excess-mass redistribution on I→L / I→G conversion). Because every population is a *single-valued* link flux, the summed update telescopes and the total liquid volume is conserved to round-off; the ledger is built so that it cannot silently discard mass either.

## 2. How it was done

| item | setting |
|---|---|
| lattice | D2Q9 (2-D), D3Q19 (3-D), exact double-precision weights |
| collision | BGK, `tau = 1/2 + nu/c_s^2`, default `tau = 0.504` (nu = 1.33e-3) |
| streaming | periodic pull (`torch.roll` on the stencil shifts) |
| bound control | conservative free-surface mass ledger (proposed), Zalesak capacity limiter (`fct`), none (`high`) |
| velocity | analytic rigid rotation, exactly divergence-free on the lattice |
| steps per revolution | `round(2*pi*R_max/(u_max*dx))`, `omega = 2*pi/(steps*dx)` so the run covers exactly 2*pi with no end-of-run phase error |
| metrics | volume drift, L1 shape error, Jaccard overlap, Hausdorff distance, 10-90 % interface thickness, interface band width in cells |
| reference | in-repo finite-volume VOF (MUSCL + van-Leer + compression, `c_comp = 0.5`) |

Design of the comparison (controlled variables): the geometry, the velocity field, the step count, the metric definitions and the machine are identical for every scheme; the *only* changed variable is the fill-transport operator. Thresholds were fixed before the production runs (they are the node's acceptance criteria, §3.5).

## 3. Results

### 3.1 Scheme comparison, one revolution, N=64

| variant | tau | abs(dV)/V | L1 error | overlap | Hausdorff [cells] | thickness [cells] | steps |
|---|---|---|---|---|---|---|---|
| vof_muscl_cc0.5 | -- | 2.389e-16 | 1.198e-02 | 0.8196 | 2.00 | 1.85 | 1555 |
| lbm_tau0.504 | 0.504 | 1.614e-13 | 3.801e-02 | 0.7854 | 2.24 | 7.36 | 1555 |
| fct_tau0.504 | 0.504 | 8.135e-14 | 3.775e-02 | 0.6614 | 2.83 | 7.61 | 1555 |
| low_tau1.0 | 1 | 8.206e-14 | 1.065e-01 | 0.0000 | -- | -- | 1555 |
| high_tau0.5 | 0.5 | 4.223e+45 | 2.187e+61 | 0.0613 | 46.87 | 0.00 | 400 |

Reading of the table:

* `lbm_tau0.504` is the proposed operator (tau = 1/2 + 4e-3, free-surface mass ledger): volume is conserved to round-off and the fill level stays inside [0, 1] at every step.
* `low_tau1.0` is the pure equilibrium-flux scheme (tau = 1): it conserves mass exactly but its built-in lattice diffusion is O(1) per step and it destroys the shape within one revolution — the numerical-diffusion floor of the lattice transport.
* `high_tau0.5` is the zero-diffusivity point without any bound control: the scheme is only marginally stable on a discontinuous fill field and diverges (L1 error 2.187e+61). This is the honest negative control of the node: an unlimited tau = 1/2 lattice transport cannot be used as a free-surface operator.
* `fct_tau0.504` applies the Zalesak (1979) capacity limiter to the anti-diffusive link flux: it matches the proposed operator on the L1 error (3.775e-02 vs 3.801e-02) but its overlap is worse (0.661 vs 0.785), and its bound proof assumes the equilibrium-flux base state is inside [0, 1], which it is not at a jump, so a small leak remains (6.000e-03).

### 3.2 Relaxation-rate scan (N=64)

| tau | nu = c_s^2(tau-1/2) | abs(dV)/V | L1 error | overlap | thickness [cells] | fill range |
|---|---|---|---|---|---|---|
| 0.502 | 6.667e-04 | 1.620e-13 | 6.063e-02 | 0.3333 | 10.80 | [+0.000, +0.641] |
| 0.503 | 1.000e-03 | 1.625e-13 | 4.561e-02 | 0.7354 | 8.41 | [+0.000, +0.759] |
| 0.504 | 1.333e-03 | 1.614e-13 | 3.801e-02 | 0.7854 | 7.36 | [+0.000, +0.827] |
| 0.505 | 1.667e-03 | 1.610e-13 | 3.825e-02 | 0.7240 | 10.07 | [+0.000, +0.819] |
| 0.506 | 2.000e-03 | 1.607e-13 | 3.965e-02 | 0.6858 | 9.67 | [+0.000, +0.796] |
| 0.51 | 3.333e-03 | 1.584e-13 | 4.563e-02 | 0.5768 | 10.91 | [+0.001, +0.710] |

The scan locates the usable band of the lattice transport: below `tau ~ 0.502` the ledger has to absorb an O(1e-2) oscillation per step and the shape smears; above `tau ~ 0.506` the added diffusivity dominates. The band is narrow because the scheme sits on the stability boundary of the scalar BGK collision — this is the central numerical finding of the node.

### 3.3 Grid convergence (2-D)

| N | steps | abs(dV)/V (LBM) | L1 (LBM) | overlap (LBM) | thickness (LBM) | L1 (VOF ref) | overlap (VOF) | thickness (VOF) |
|---|---|---|---|---|---|---|---|---|
| 64 | 1555 | 1.614e-13 | 3.801e-02 | 0.7854 | 7.36 | 1.198e-02 | 0.8196 | 1.85 |
| 128 | 3186 | 3.561e-13 | 2.758e-02 | 0.9165 | 7.36 | 4.776e-03 | 0.9304 | 1.81 |
| 256 | 6395 | 6.986e-13 | 1.898e-02 | 0.9612 | 9.14 | 2.344e-03 | 0.9632 | 1.87 |

Observed order of the LBM L1 shape error: p(64→128) = 0.46, p(128→256) = 0.54; Roache GCI of the fine grid = 5.24 %. For the VOF reference: p(64→128) = 1.33, p(128→256) = 1.03, GCI = 0.28 %.

Note the two competing effects at play. The lattice operator's numerical diffusivity is fixed in *lattice* units (`nu = c_s^2(tau-1/2)`), so at fixed peak lattice speed the diffusion length in cells grows like `sqrt(N)` while the physical interface thickness it corresponds to shrinks like `1/sqrt(N)`. The interface-thickness column therefore *grows* with N while the L1 shape error decreases — the scheme is convergent in the error norm but it is not a sharp-interface scheme at these relaxation rates.

### 3.4 D2Q9 versus D3Q19 and the 3-D body

| quantity | value |
|---|---|
| relative L1 deviation of the cross-section (D3Q19 - D2Q9) | 4.333e-01 |
| max abs difference of the cross-section | 1.971e-01 |
| relative volume difference | 1.081e-15 |
| L1 error, D3Q19 cross-section | 6.176e-02 |
| L1 error, D2Q9 | 4.538e-02 |
| relative difference of the L1 error | 3.611e-01 |
| max out-of-plane (y) variation, D3Q19 run | 0.000e+00 |

The 3-D run is started from *the same* 2-D field promoted to 3-D, so the comparison isolates the lattice: the body is exactly y-invariant and the flow has no out-of-plane component, hence the D3Q19 solution must stay y-invariant — and it does, to 0.000e+00 — but its cross-section still differs from the D2Q9 solution at the 4.333e-01 relative-L1 level (overlap 0.186 versus 0.660).

Two effects are entangled here and the design of this stage separates them only partially, which is itself a result:

* **Consistency.** The out-of-plane symmetry is preserved to round-off, so the D3Q19 operator does not leak out of the rotation plane — that part of the kernel-equivalence hypothesis holds.
* **Unequal numerical diffusion.** At fixed relaxation rate the two lattices are *not* at the same point of the diffusion/accuracy trade-off: the D3Q19 cross-section thickens to 14.62 cells against 9.31 cells for D2Q9 at the same tau and the same step count. Since the interface halo carries a large fraction of the liquid volume, such a difference shows up as a large *field-level* deviation (4.333e-01) while the *shape-level* L1 error differs by 3.611e-01 (both are already large: 4.538e-02 vs 6.176e-02).

So the pre-registered fixed-tau comparison fails criterion (e), and the honest reading is that the failure measures unequal numerical diffusion rather than a loss of consistency: a tau-matched comparison (tau chosen per lattice to equalise the interface thickness) is the fair operator test and is listed in the roadmap. The kernel-equivalence expectation recorded in the module docstring at design time is falsified and the docstring updated.

Genuine 3-D slotted sphere (slab-faced slot, N=40): volume drift 1.093e-13, L1 error 1.632e-02, overlap 0.0000, Hausdorff -- cells, thickness -- cells, 978 steps, fill range [+0.000, 0.368].

Note: the fill maximum of this run stays below 0.5, i.e. the sphere is smeared to the point where the `phi >= 0.5` set is empty. The overlap and Hausdorff metrics are then undefined (printed as `--`), and the L1 error is the only meaningful number of that run. This is the same numerical-diffusion mechanism as above, amplified by the coarse N and the D3Q19 lattice.

### 3.5 Acceptance criteria

| criterion | measured | threshold | verdict | note |
|---|---|---|---|---|
| volume relative change after 2pi rotation (N=128) < 0.5% | 3.561e-13 | 0.005 | PASS | single-valued link fluxes; conservation is a structural property |
| 10-90% interface thickness (N=128) < 3 cells | 7.361e+00 | 3 | FAIL | gradient-based thickness, mean over interface cells |
| Jaccard overlap with the initial shape (N=128) > 0.95 | 9.165e-01 | 0.95 | FAIL | threshold phi >= 0.5 |
| observed order of the L1 shape error > 0.5 | 5.390e-01 | 0.5 | PASS | the volume error itself is at round-off at every N, so its order is undefined (see d_volume_note); the shape-error order is used |
| volume-error convergence order | -- | 0.5 | n/a | volume drift is 1e-15..1e-13 relative at all N -> no resolvable convergence order; reported instead of a meaningless fit |
| D3Q19 vs D2Q9 cross-section relative L1 deviation < 5% | 4.333e-01 | 0.05 | FAIL | extruded (y-invariant) body, slice y = 0.5 |
| max abs(div u) of the prescribed field < 1e-10 | 0.000e+00 | 1e-10 | PASS | face divergence of the analytic rigid rotation |

Summary: **a_mass_conservation = PASS**, **b_interface_sharpness = FAIL**, **c_shape_fidelity = FAIL**, **d_convergence_order = PASS**, **e_3d_vs_2d = FAIL**, **f_divergence = PASS**.

## 4. Figures

**fig01_fields.png** — Initial fill fraction, the proposed LBM transport after one revolution, the VOF reference, and the pointwise error of the LBM run.

![fig01_fields.png](figures/fig01_fields.png)

**fig02_convergence.png** — L1 shape error, Jaccard overlap and interface thickness versus grid resolution for the LBM transport and the VOF reference.

![fig02_convergence.png](figures/fig02_convergence.png)

**fig03_mass_and_tau.png** — Relative volume drift per step (left) and the relaxation-rate scan at N=64 (right).

![fig03_mass_and_tau.png](figures/fig03_mass_and_tau.png)

**fig04_3d_cross_section.png** — D2Q9 solution, the matching D3Q19 extruded cross-section and their pointwise difference.

![fig04_3d_cross_section.png](figures/fig04_3d_cross_section.png)

## 5. Notes, pitfalls and limits

1. **tau = 1/2 is not usable.** The zero-diffusivity point of the scalar BGK collision is only marginally stable on a discontinuous fill field; the unlimited run diverges and the ledger has to do O(1e-2) work per step, which smears the shape. Only the narrow band tau in [0.502, 0.506] is usable at these parameters, and it costs roughly 5–8 cells of interface thickness at N=64.
2. **The equilibrium flux is the numerical-diffusion floor.** With tau = 1 the scheme reproduces `nu = c_s^2/3` per step and the shape does not survive one revolution. Any accuracy claim for a fill-fraction lattice transport therefore requires a genuinely anti-diffusive mechanism, which is exactly what the FCT / compression literature adds.
3. **The FCT limiter's bound proof has a hole at a jump.** It assumes the low-order (equilibrium-flux) state lies in [0, 1]; at a jump it can overshoot by O(Ma), so a small bound leak survives. The leak is bounded and conservative, but it is not round-off.
4. **Ledger versus limiter.** The ledger bounds the state without touching the fluxes, so it cannot recover accuracy — it converts oscillation into a diffusive halo. The limiter bounds the fluxes, which preserves sharpness but needs a monotone base state that the lattice equilibrium does not provide.
5. **Geometry error is separate from transport error.** The supersampled initial fill differs from the analytic volume by O(h); the mass-conservation metric uses the *discrete* initial volume as reference, so the two are not confounded.
6. **Scope.** Only the fill-transport operator is graded. Momentum, the anti-bounce-back gas-pressure reconstruction and the coupled free-surface dynamics are out of scope: the velocity field is prescribed, exactly as in Zalesak's original benchmark.

## 6. Error assessment against the benchmark

The exact solution after one revolution is the initial condition, so the benchmark error of every metric is known without any reference simulation. Asking the node's question — *is every key quantity within 3 %?*:

| quantity | measured | exact value | < 3 %? | comment |
|---|---|---|---|---|
| liquid volume (mass conservation) | 3.561e-13 | 0 % | YES | single-valued link fluxes; round-off level |
| interface position (Hausdorff distance) | 3.16 cells | 0 cells | NO | 3 % of the 0.15 body radius = 0.0045 physical = 0.58 cells at N=128 |
| shape (Jaccard overlap) | 0.9165 | 1.0 | NO | overlap deficit is the fraction of misclassified interface cells |
| L1 shape error relative to the liquid volume | +4.736e+01 % | 0 % | NO | the diffusive halo is thin but wide: this ratio over-states the visual error, the Jaccard deficit is the fairer shape number |
| interface thickness | 7.36 cells | 0 cells | NO | the material interface is a jump |

So: **mass conservation is exact (round-off), but the shape error of the LBM fill transport is not within 3 %** at N=128. For comparison the same driver run on the repository's VOF operator reaches an L1 error of 4.776e-03 with an interface thickness of 1.81 cells at the same resolution. The lattice-population path is therefore the mass-conservative but less accurate of the two; the honest verdict of this node is FAIL on the sharpness and shape-fidelity criteria and PASS on mass conservation, divergence-freedom and (marginally) on convergence order.

## 7. Roadmap (open gaps)

* monotone base state for the FCT limiter (upwind lattice flux instead of the equilibrium flux) so the limiter can carry the full anti-diffusive flux;
* lattice-aligned interface compression with its own stability analysis (the naive per-link compression tested here is unstable at the CFL used);
* resolution-consistent relaxation rate `tau = 1/2 + c*dx` so that the diffusion length in cells stays fixed under refinement;
* coupling to the momentum solver and the ABB gas-pressure reconstruction (the Körner model proper), which the kinematic benchmark deliberately excludes.

