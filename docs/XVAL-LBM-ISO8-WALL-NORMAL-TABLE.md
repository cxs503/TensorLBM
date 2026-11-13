# XVAL-LBM-ISO8-WALL-NORMAL-TABLE — cross-validation against lanl/MF-LBM

**Anchor** `XVAL-LBM-ISO8-WALL-NORMAL-TABLE` (round 1, task board, OSS cross-validation) · repo **TensorLBM** · branch `agent/m17865` · 2026-11-13

## ① Problem and the reference

Curvature-dependent boundary conditions (geometric wetting / contact angle, Laplace-pressure closures,
second-order slip) need the **wall orientation on a staircase boundary**. Averaging axis directions is
anisotropic, so the normal has to come from an isotropic discrete gradient of a *smoothed* solid indicator.

`lanl/MF-LBM` (BSD-3-Clause, upstream commit `11be324a97fe6331b2271102da370460715355e8`) does exactly this
once per geometry in `preprocessing/3.wall_boundary_preprocess/3d_wall_boundary_process.f90`:
solid fraction smoothed by **4** D3Q27 convolutions with `w=(8/27, 2/27, 1/54, 1/216)` (kernel sum
1.000000000000 → mass conserving), then an **ISO8** gradient
(7 weights `0.0888889, 0.047619, 0.0190476, 0.00992063, 0.0031746, 0.0015873, 0.000198413` over **34 central pairs**, 5×5×5 reach)
at every fluid node touching the solid; the resulting `(snx,sny,snz)` is consumed by `Phase_gradient.F90`
to impose the contact angle.

The Fortran source is vendored in `xval/XVAL-LBM-ISO8-WALL-NORMAL-TABLE/oss_src/`
(sha256 `d5ff4f9ed42f984f224e75d9debd913348fe0e40b2e4050cc21f98a0f8d394d4`); every constant used by the transplant is **parsed out of it**
(`scripts/xval_iso8_wall_normal_table.py extract`) and `tests/test_wall_normal_iso8.py` re-parses it, so the
transplant cannot drift from upstream.

## ② Method

1. Stencil extraction (zero hand-copy): ISO8 weights, the 34 difference pairs **per gradient component**,
   the D3Q27 smoothing kernel, the 18-direction half set, `w_equ`, loop constants (4 iterations, ghost 6,
   overlap 4). Parse-time invariant: the y/z pair sets are the xy/xz axis permutations of the x set.
2. OSS arm: the vendored file is compiled (`gfortran -O2 -fopenmp -ffree-line-length-none -finit-integer=0
   -finit-real=zero`) and run on the **byte-identical** wall array; its binary table
   (packed records: 96 B solid, 36 B fluid) is parsed.
3. TensorLBM arm: `src/tensorlbm/boundary/wall_normal_iso8.py`, a vectorised NumPy transplant of the same
   pipeline (pad ghost+overlap → mark solid/fluid boundary nodes → 4× D3Q27 smoothing → ISO8 gradient →
   normalise → tables in the upstream Fortran listing order, `la_weight`/`neighbor_list` included).
4. Comparison: row-by-row table identity, then both tables against the analytic normal, then competing
   stencils on the same smoothed field.
5. Predeclared design in `xval/XVAL-LBM-ISO8-WALL-NORMAL-TABLE/predeclared.json`
   (K1 ≤1e-12, K2 ≤15 % owner band, K3 isotropy, K4 curvature ≤15 %; cases fixed before the run).

## ③ Parameters (lattice units; this is a preprocessing capability, so no Re/Ma — no flow is integrated)

| case | domain | geometry | fluid-boundary nodes | physical nodes (K2) |
|---|---|---|---|---|
| sphere_n48_R10_offcenter | 48³ | sphere R=10, centre (24.3,23.6,24.1) | 1815 | 1815 |
| sphere_n64_R14_offcenter | 64³ | sphere R=14, centre (32.2,31.5,32.4) | 3450 | 3450 |
| plane_n40_alpha26.565 | 40³ | plane, normal (2,1,0) | 3216 | 2360 |
| plane_n40_alpha45 | 40³ | plane, normal (1,1,0) | 3024 | 2200 |

Fixed: float64, padding `ghost=6` + `overlap=4` (= 10, upstream), ISO8 gain on analytic linear fields
**0.880952** in all 8 directions, smoothing iterations 4 (upstream).

## ④ Comparison (ref vs got, ✓/✗)

### K1 — exactness against the compiled OSS binary (predeclared ≤ 1e-12)

| case | OSS / ours nodes | node table identical | max abs dn | max d-angle [deg] | max d-la_weight |
|---|---|---|---|---|---|
| sphere_n48_R10_offcenter | 1815 / 1815 | True | 8.88e-16 | 1.21e-06 | 0 |
| sphere_n64_R14_offcenter | 3450 / 3450 | True | 8.88e-16 | 1.21e-06 | 0 |
| plane_n40_alpha26.565 | 3216 / 3216 | True | 6.66e-16 | 0 | 0 |
| plane_n40_alpha45 | 3024 / 3024 | True | 7.77e-16 | 0 | 0 |

✓ **PASS** — agreement at the float64 round-off floor (8.9e-16), including off-centre geometry; node sets,
18-entry neighbour lists and `la_weight` identical. Cost: OSS 0.07–0.13 s vs ours 0.17–0.86 s.

### K2 — accuracy vs the analytic normal (owner band 15 %; chordal relative error 2 sin(dθ/2))

| case | mean rel [%] | max rel [%] | mean d-angle [deg] | p95 d-angle [deg] |
|---|---|---|---|---|
| sphere_n48_R10_offcenter | 3.43 | 11.7 | 1.97 | 4.12 |
| sphere_n64_R14_offcenter | 3.29 | 13.3 | 1.89 | 4.12 |
| plane_n40_alpha26.565 | 2.47 | 32.8 | 1.42 | 10.5 |
| plane_n40_alpha45 | 3.33 | 43.6 | 1.92 | 13.7 |

✓ **PASS on the mean** (2.5–3.4 % ≤ 15 %), ✗ **honest FAIL on individual staircase corner nodes**
(max 11.7–43.6 %). No node trimmed; the worst case is the 45° plane (densest concave corners).

| case | arm | mean rel [%] | max rel [%] | degenerate nodes |
|---|---|---|---|---|
| sphere_n48_R10_offcenter | iso8_smooth4 | 3.43 | 11.7 | 0 |
| sphere_n48_R10_offcenter | central_smooth4 | 4.31 | 13.5 | 0 |
| sphere_n48_R10_offcenter | iso8_nosmooth | 12.6 | 51.2 | 0 |
| sphere_n48_R10_offcenter | central_nosmooth | 36.5 | 86.4 | 702 |
| sphere_n64_R14_offcenter | iso8_smooth4 | 3.29 | 13.3 | 0 |
| sphere_n64_R14_offcenter | central_smooth4 | 4.26 | 15.7 | 0 |
| sphere_n64_R14_offcenter | iso8_nosmooth | 12.7 | 48.4 | 0 |
| sphere_n64_R14_offcenter | central_nosmooth | 36.5 | 84.3 | 1314 |
| plane_n40_alpha26.565 | iso8_smooth4 | 2.47 | 32.8 | 0 |
| plane_n40_alpha26.565 | central_smooth4 | 3.21 | 33.1 | 0 |
| plane_n40_alpha26.565 | iso8_nosmooth | 12.3 | 44.4 | 0 |
| plane_n40_alpha26.565 | central_nosmooth | 39.3 | 46 | 760 |
| plane_n40_alpha45 | iso8_smooth4 | 3.33 | 43.6 | 0 |
| plane_n40_alpha45 | central_smooth4 | 3.25 | 45.2 | 0 |
| plane_n40_alpha45 | iso8_nosmooth | 2.12 | 48.7 | 0 |
| plane_n40_alpha45 | central_nosmooth | 5.47 | 76.5 | 1080 |

✓ ISO8+4 smoothing is the best mean in 3 of 4 cases and the only arm with no degenerate
(vanishing-gradient) node; raw central differences collapse to axis-only normals (100 % error).

## ⑤ Convergence / sensitivity

**Smoothing iterations** (mean rel error [%], upstream value = 4):

| case | 0 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| sphere_n48_R10_offcenter | 12.6 | 7.54 | 5.36 | 3.43 | 2.02 | 1.14 |
| sphere_n64_R14_offcenter | 12.7 | 7.52 | 5.27 | 3.29 | 1.95 | 1.12 |
| plane_n40_alpha26.565 | 12.3 | 4.51 | 2.91 | 2.47 | 2.69 | 3.08 |
| plane_n40_alpha45 | 2.12 | 2.55 | 2.85 | 3.33 | 4.05 | 5.12 |

Sphere error is monotone in the iteration count (12.6 % → 1.1 %): upstream's 4 iterations is a cost/accuracy
compromise, **not** converged for spheres. The 26.565° plane has a minimum near 4; the 45° plane prefers 0
(2.1 %) and degrades with smoothing because smoothing rounds the sharp corners that carry the 45° orientation.
Reported as a trade-off; the anchor keeps 4 frozen rather than tuning it.

**Resolution**: sphere R=10 (n=48) → R=14 (n=64): mean 3.43 % → 3.29 %, p95
4.12° → 4.12° →
resolution-stable to 0.14 pp; the residual is the staircase (h/R) effect, not an implementation artefact.

**Operator isotropy (K3)**: ISO8 directional gain 0.880952 (anisotropy ratio 1.0,
band 1.000001) ✓ PASS. The failure mode this protects against was measured during development: using one
component's pair set for all three components tilts the normals by 22°.

**Curvature (K4, downstream)**: κ = −div(n) of the ISO8 level set vs 2/R:

| case | κ_ref | κ_ISO8 | rel err [%] | κ_central |
|---|---|---|---|---|
| sphere_n48_R10_offcenter | 0.2 | -0.1652 | 17.4 | -0.1637 |
| sphere_n64_R14_offcenter | 0.1429 | -0.119 | 16.7 | -0.1176 |
| plane_n40_alpha26.565 | 0 | 0.000556 | n/a | 0.0006871 |
| plane_n40_alpha45 | 0 | -0.02693 | n/a | -0.0273 |

✗ **FAIL (16–18 %, band 15 %)**, reported honestly: the estimator is biased low because div(n) is evaluated
on the ±1-cell staircase band, not on the continuous level set, and both stencils give the same biased value
(<1 % apart), so K4 measures the estimator rather than the table. Kept as a documented limitation
(level-set projection before differentiating would be the fix; out of scope).

## ⑥ Failure modes and artefacts

* **D7 upstream defect**: `mirror_bc` is read before assignment (undefined behaviour); the harness pins it with
  `-finit-integer=0`, so the OSS arm is deterministic but the defect would bite another toolchain.
* **My own bug, caught by the OSS arm**: writing the wall file in C order instead of Fortran i-fastest order
  transposes the geometry — invisible on an axis-symmetric sphere (the first four geometries matched exactly!)
  and exposed only by an off-centre centre. Voxel I/O here must always be tested with an asymmetric geometry.
* **Single pair set for all components** → 22° systematic tilt.
* **Padding with `ghost` instead of `ghost+overlap`** → up to 21.8° error near the outer listed nodes.
* Artefacts `artifacts/xval_iso8_wall_normal_table/`: `results.json`, `oss_iso8_stencil.json`,
  `accuracy_table.csv`, `exactness_table.csv`, `sensitivity_table.csv`, `oss_fixture_normal_table.csv`
  (regression fixture), `figures/fig1_sphere_normal_field.png`, `fig2_smoothing_sensitivity.png`,
  `fig3_stencil_comparison.png`, `fig4_operator_anisotropy.png`.
* Reproduce: `python3 scripts/xval_iso8_wall_normal_table.py run` (37 s, CPU only, no cluster);
  `python3 -m pytest tests/test_wall_normal_iso8.py` → 6 passed.

## Difference list — TensorLBM vs OSS, and transplant trade-offs

| # | item | OSS (lanl/MF-LBM) | TensorLBM after transplant | trade-off |
|---|---|---|---|---|
| D1 | gradient stencil | ISO8, 34 pairs, 5×5×5 | identical (parsed, test-enforced) | none, kept bit-compatible |
| D2 | pair sets | one set per component (x: xy+xz edges, y: xy+yz, …) | reproduced per component | mandatory: one shared set tilts normals 22° |
| D3 | pre-smoothing | 4 × D3Q27 conv, lattice weights, mass conserving | identical | upstream 4 kept; not converged for spheres |
| D4 | padding | `ghost_layers + overlap_phi` = 10 | identical (6+4) | padding is algorithmic, not cosmetic |
| D5 | node listing | Fortran 1-based, i fastest, includes 4-cell overlap band outside the box | identical order/band + `in_physical_domain()` filter | overlap only matters for MPI subdomains |
| D6 | data structure | preprocessed binary table (96 B / 36 B records) | in-memory NumPy dict, no serialiser | we save I/O, pay RAM; serialiser out of scope |
| D7 | `mirror_bc` | read but never assigned (undefined) | explicit keyword, default 0 | defect pinned, not inherited silently |
| D8 | neighbour set | 18-direction half set for `neighbor_list`/`la_weight` | identical, also exposed | — |
| D9 | sign convention | normal = +grad(solid fraction) → **into the solid** | identical, documented | callers must flip; wetting code needs the inward sign |
| D10 | language/perf | Fortran+OpenMP over the padded array | vectorised NumPy, ~2–7× slower (0.86 s vs 0.13 s @64³) | fine for once-per-geometry preprocessing; torch port is follow-up |
| D11 | downstream | `Phase_gradient.F90`, 3-D geometric wetting | **not wired**: TensorLBM has 2-D wetting + 3-D SDF IBB, no 3-D wetting BC | separate anchor; this table is its precondition |
| D12 | solver coupling | normals computed once, reused every step | same intended use, no coupling in this task | — |

## Verdict

* K1 **PASS** (8.9e-16): the transplant is exact, so any later coupled-simulation difference is not the table's.
* K2 **PASS on the mean** (2.5–3.4 % < 15 %), **honest FAIL per-node** (up to 43.6 % at 45° corners, untrimmed).
* K3 **PASS** (anisotropy 1.0).
* K4 **FAIL** (16–18 %), band-estimator bias, identical for both stencils.
* Decision: **adopt** the ISO8 table as TensorLBM's wall-orientation preprocessor; open items = 3-D wetting
  BC wiring (D11) and upstream's non-converged 4 smoothing iterations.
* Limitation: the OSS binary is a *preprocessing* code, so this is a geometry/algorithm XVAL (normals vs
  analytic); hydrodynamic acceptance must come from a 3-D wetting/droplet run (follow-up).
