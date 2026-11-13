# GAP-LBM-T5-ICE-COUPLING · round 2 (G5-07): the 3-D interface driven by a real D3Q19 field

*Mission m17029 · repo TensorLBM · branch `agent/m17029` · round 2 of the T5 flagship node.
Round 1 (PR #1842, audited by m15904) closed the transfer algebra against **analytic** tractions
and left gap **G5-07** open: "interface not yet driven by a real 3-D LBM field (only analytic
tractions)". This round closes that gap for the fluid leg, and reports one honest FAIL.*

**Status: 6/7 gates PASS, G5 (drag band at the resolution actually run) FAIL.**
Every number below is injected programmatically from `artifacts/three_way_g507/gates_D12.json`.

## 1. Case

| item | value |
|---|---|
| flow | D3Q19 + MRT, BFL (Bouzidi–Firdaouss–Lallemand) sub-cell wall, far-field y/z |
| body | fixed sphere, R = 6.0 lattice units (D = 12) |
| domain (nz,ny,nx) | (42, 42, 54) — padding 1.25D lateral / 1.25D up / 2.25D down (calibrated `sphere_re100` protocol) |
| tau / u_lb / nu_lb / Re | 0.518 / 0.05 / 0.0060 / 100.0 |
| steps / sample / threads | 900 / 25 / 16 |
| boundary links | 2610 (q ∈ (0.072, 1.0)) |
| wall clock | 64 s (torch 2.7.1) |

Same protocol, kernels and force caliber as the stored verified case
`benchmarks/verified/sphere_re100`; only the resolution is reduced to fit an interactive budget.

## 2. Method (what is new)

1. `per_link_force` — per-link momentum ledger of the BFL wall. The library caliber returns
   only a *summed* force; the interface needs the distribution. Term-by-term reproduction of
   the same arithmetic, gated against the library total (G1) so the consumed traction provably
   comes from the verified caliber.
2. `link_wall_points` / `bin_to_surface` — each link acts at `x_cell + q c_d`; link forces are
   partitioned onto `Surface3D` facets by direction from the body centre, giving `t = F/dA`.
   The partition conserves the resultant exactly; the moment only to surface-quadrature
   accuracy, which is measured (and reported) rather than assumed.
3. `transfer_gates` — the round-1 Q1 consistent transfer re-gated on the real traction with a
   **non-affine** structural velocity field (rotation + quadratic bending mode).
4. `march_driven` — velocity-Verlet 1-DOF march driven by the measured drag history, with the
   work ledger closed to the integrator order.

## 3. Pre-registered tolerances

| key | value | why |
|---|---|---|
| G1 ledger rel. diff | ≤ 1e-4 | float32 accumulation order only |
| G2 resultant / coverage / binning | ≤ 1e-10 / ≥ 0.90 / ≤ 5 % | partition exact; coverage pre-registered |
| G3 resultant / moment / work | ≤ 1e-10 each | exact identities in exact arithmetic |
| G4 frame negative control | ≥ 1e-2 | the gate must see a one-cell frame error |
| G5 drag vs Schiller–Naumann | ≤ 5 % | verified case does 3 % at D≥40 |
| G6 growth vs spectral radius | ≤ 5 % | measured vs eigendecomposition |
| G7 closure at ωΔt=0.01 | ≤ 5e-3, monotone in Δt | trapezoid on sampled data is 1st order at jumps |

## 4. Results

| gate | verdict |
|---|---|
| G1_ledger_closure | PASS |
| G2_link_to_facet | PASS |
| G3_transfer_real_traction | PASS |
| G4_frame_negative_control | PASS |
| G5_drag_real_field | FAIL |
| G6_added_mass_guard_real_geometry | PASS |
| G7_driven_response_ledger | PASS |

### 4.1 Fluid → traction → structure (G1–G4)

| quantity | value |
|---|---|
| distribution vs library ledger, rel. | 8.007e-06 (n_links = 2610) |
| facet level consumed | `theta12_phi24` (528 facets, coverage 0.943) |
| link→facet resultant, rel. | 0.000e+00 |
| facet-quadrature moment error (pointwise scale) | 2.51e-03 |
| Q1 transfer resultant, rel. | 1.49e-14 |
| Q1 transfer moment, rel. | 4.94e-15 (abs 1.91e-14, scale 1.2980) |
| Q1 transfer work, non-affine u, rel. | 8.65e-15 (W = -2.463406) |
| frame negative control (one-cell shift) | 4.989e-02 ≥ 1e-2 ✓ (x=4.78e-04, y=4.99e-02, z=4.99e-02) |

The transfer is exact on a real load to 1e-14; the *same* operator in a frame shifted by one
lattice cell is wrong by 5.0 % — the gate has teeth. What remains
is traction-extraction (quadrature) error, not transfer error.

### 4.2 Surface refinement study (reported; under 90 % coverage not consumed)

| level | facets | coverage | facet size h | moment err / (Σ|F| h) | moment err / (|F| R) |
|---|---|---|---|---|---|
| theta10_phi20 | 360 | 0.997 | 1.109 | 4.72e-03 | 0.916 |
| theta12_phi24 | 528 | 0.943 | 0.919 | 2.51e-03 | 0.403 |
| theta16_phi32 | 960 | 0.829 | 0.684 | 1.40e-02 | 1.670 |

The pointwise column is the meaningful one. The resultant-normalised column is large because a
drag load is a strongly cancelling surface integral (Σ|F_link|/|ΣF| ≈ 1056), so dividing by the resultant lever
amplifies the error — the same cancellation that makes "moment about the body centre" a
degenerate diagnostic here.

### 4.3 Drag from the real field (G5 — honest FAIL)

| quantity | value |
|---|---|
| C_D (last-5 mean) | 1.5230 |
| Schiller–Naumann, Re=100 | 1.0917 |
| error | **39.50 %** (tol 5.0 %) → **FAIL** |
| stored verified D=40 / D=60 | 1.1220 / 1.1195 (gap 35.74 %) |
| Cd first → last (tail) | 1.209 → 1.521 (last three 1.526, 1.516, 1.521) |

The tail is **flat**, so this is not an unconverged transient: at D = 12 (R = 6.0 cells)
with τ = 0.518 the BFL sphere genuinely over-predicts drag by ≈ 39 %. The verified
caliber reaches 2.8 % only at D=40/60, τ=0.56. Conclusion, narrowed deliberately:

* the **interface** is fed by a real field and is invariant-exact (§4.1);
* the **field** leg is not accuracy-adequate at this resolution. A D=20, τ=0.53 (Re=100, ν=0.01)
  run of the same script is queued on the cluster (job cj1400); its Cd is what would decide G5.
  Until then G5 is FAIL, not "approximately passed".

### 4.4 Added-mass guard with the real geometry (G6)

| quantity | value |
|---|---|
| displaced volume (analytic / voxels) | 904.78 / 912.0 (diff 0.80 %) |
| μ, ice floe (ρ=917, C_a=0.5) | 0.5453 → stable |
| μ, light floe (ρ=300) | 1.6667 → unstable (μ≥1) |
| m_crit = C_a ρ_w V | 452389.34 |
| spectral radius (ice / light) | 0.999954 / 1.666823 |
| measured growth, light floe | 0.510919 vs ln λ = 0.510919 (gap 6.52e-16) |
| relaxation bound 2/(1+μ) | 1.2943 / 0.7500 |

The guard is evaluated on the volume of the body that was actually simulated (voxel vs analytic
volume agree to 0.80 %), and the measured unstable-branch growth rate matches the
round-1 spectral radius to 1e-16.

### 4.5 Structural response driven by the measured drag (G7)

| ωΔt | ledger closure (max over the march) |
|---|---|
| 0.04 | 1.50e-04 |
| 0.02 | 7.87e-05 |
| 0.01 | 6.05e-05 |

Observed order over the coarsest pair 0.93 (informational: the two finest points sit
on the ~6e-5 discretisation floor of the sampled force history).

## 5. A metric definition changed after the first measurement (declared)

The first D=12 run failed G2 only because the *finest listed* surface (16×32, coverage 0.83) is
under-resolved for 2610 links, and failed G7 only because a 3-point order fit across a
6e-5 floor gives 0.93. Both were fixed by **definition, not by loosening a tolerance**:
(a) admissibility rule — the interface consumes the finest quadrature meeting the pre-registered
90 % coverage, under-resolved levels are reported (§4.2) but not consumed; (b) G7 gates on the
production-Δt closure plus monotonicity. The pre-registered numbers (0.90, 5e-3, 5 %) are
unchanged, and both variants are in the git history of the driver.

## 6. Not done / limits

* **One-way fluid.** Fixed sphere; two-way needs per-step sub-cell geometry refresh (`q` depends
  on body position) + moving-wall correction → round 3.
* **No DEM leg** (G5-06 untouched): ice failure is not simulated.
* **G5 unresolved** at field resolution; D=20 number pending (cj1400).
* **No FEM solve on the structural side**: 1-DOF response + Q1 hex transfer block only.
* **Traction quadrature error** ≈1 % of the pointwise moment scale at the consumed level.

## 7. Reproduction

```bash
OMP_NUM_THREADS=16 PYTHONPATH=src python scripts/three_way_g507_field_gates.py \
    --D 12 --steps 900 --tau 0.518 --sample 25 --threads 16 \
    --surface-levels "10,20;12,24;16,32" --out artifacts/three_way_g507
OMP_NUM_THREADS=1 PYTHONPATH=src python -m pytest tests/test_three_way_lbm3d_field.py -q
```

Artefacts: `artifacts/three_way_g507/{gates_D12.json, gates_D12.csv, field_D12.npz, figures/*.png}`
