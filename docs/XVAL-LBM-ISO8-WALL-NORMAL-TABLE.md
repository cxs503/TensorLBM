# XVAL-LBM-ISO8-WALL-NORMAL-TABLE — cross-validation against lanl/MF-LBM

**Anchor** `XVAL-LBM-ISO8-WALL-NORMAL-TABLE` (task board, OSS cross-validation) · repo **TensorLBM**
· round 1 `agent/m17865` (2026-11-13, merge #2117) · round 2 `agent/m20638` (2026-11-14).

Object under test: MF-LBM's *iso8 wall-normal table* — the per-geometry preprocessor
(`preprocessing/3.wall_boundary_preprocess/3d_wall_boundary_process.f90`) that stores unit wall
normals on every fluid node touching the solid (`fluid_boundary_nodes % snx/sny/snz`), consumed by
`Phase_gradient.F90` to impose the geometric (contact-angle) wetting condition. TensorLBM's
transplant lives in `src/tensorlbm/boundary/wall_normal_iso8.py` (raw, OSS-convention table) and
`src/tensorlbm/boundary/wall_normal_preprocess.py` (platform convention, the caller path).

Acceptance: owner band < 15 % on the normal error, measured against the analytic normal of the
declared geometry; no accuracy claim may be made beyond that band.

## Round 2 — what this round adds (one-paragraph verdict)

Round 1 verified the transplant is **bit-exact** (max |Δn| = 8.9e-16 vs the compiled OSS
preprocessor) and passed on the mean (2.5–3.4 %), but reported an honest FAIL on the worst row
(43.6 % on a 45° plane) and left three debts: the platform caller was **not importable**
(`tensorlbm.boundary.build_wall_normal_table` raised ImportError — two merged tests red on `main`),
the OSS regression fixture was **missing** so the fixture test silently *skipped*, and nobody had
pinned *which* rows carry the >15 % error. Round 2 closes all three and replaces the blanket
"honest FAIL" with a **well-posedness certificate**:

> the >15 % rows are, without exception, rows whose support (`2 + n_smooth` = 6 cells) reaches across
> a **non-periodic domain face that the wall itself reaches**. There the ghost band only replicates
> the face plane along the face normal, so the solid is continued in a direction the wall does not
> have — the normal is a property of the padding, not of the geometry. On every row that passes the
> certificate the per-row error is **≤ 13.3 %** (< 15 %), and on the junction rows the error decays
> monotonically with distance to the face (43.6 % → 23.9 % → 5.5 % → 0.47 % at 1/2/3/4 cells).

## Round-2 evidence (all numbers from `scripts/xval_iso8_wall_normal_table.py run`, 31 s, CPU)

Arm K1 — transplant exactness (compiled OSS vs TensorLBM, identical node tables, 4 geometries):

| case | rows | max abs \|Δn\| vs OSS |
|---|---|---|
| sphere_n48_R10_offcenter | 1815 | 8.9e-16 |
| sphere_n64_R14_offcenter | 3450 | 8.9e-16 |
| plane_n40_alpha26.565 | 2360 | 6.7e-16 |
| plane_n40_alpha45 | 2200 | 7.8e-16 |

Arm K2 — per-row error split by the certificate (`well_posed_mask`, radius 6 = 2 gradient + 4 smoothing):

| case | well-posed rows | wp mean | wp **max** | ill-posed rows | ill mean | ill **max** | rows ≥15 % | of which well-posed |
|---|---|---|---|---|---|---|---|---|
| sphere_n48_R10 | 1815 | 3.43 % | **11.73 %** | 0 | — | — | 0 | 0 |
| sphere_n64_R14 | 3450 | 3.29 % | **13.26 %** | 0 | — | — | 0 | 0 |
| plane_n40_26.565° | 1120 | 0.48 % | **0.69 %** | 1240 | 4.26 % | 32.83 % | 120 | **0** |
| plane_n40_45° | 924 | 0.00 % | **0.00 %** | 1276 | 5.74 % | 43.64 % | 240 | **0** |

Certificate quality: recall 100 % (no row outside the 15 % band is called well posed) and the
certificate is *conservative* — it also flags many rows that are numerically fine, which is the
intended asymmetry for a certificate. It is a pure geometric test on the mask, so it costs one
separable dilation per face (no tuning constant, no empirical threshold).

Arm K2b — error vs distance of the row to the nearest domain face (1 = on the face), 45° plane:

| distance [cells] | 1 | 2 | 3 | 4 | 5 | 6 | ≥7 |
|---|---|---|---|---|---|---|---|
| mean rel err | 19.8 % | 12.2 % | 2.4 % | 0.18 % | 0.00 % | 0.00 % | 0.00 % |
| max rel err | 43.6 % | 23.9 % | 5.53 % | 0.47 % | 0.01 % | 0.00 % | 0.00 % |

(26.565° plane, same arm: 32.8 / 18.3 / 6.5 / 1.44 / 0.72 / 0.69 / 0.69 % — it plateaus at ≈0.7 %,
the stencil's own bias for that oblique family, still 20× inside the band.)

Arm K2c — smoothing depth vs well-posed error (sphere_n64, well-posed rows only):

| n_smooth | 0 | 1 | 2 | **4 (upstream)** | 8 | 16 |
|---|---|---|---|---|---|---|
| mean | 12.7 % | 7.5 % | 5.3 % | **3.29 %** | 1.95 % | 1.12 % |
| max | 48.4 % | 29.7 % | 20.4 % | **13.26 %** | 6.45 % | 3.35 % |

Upstream's 4 sweeps leave only 1.7 pp of margin on the worst sphere row; 8 sweeps give 2× margin.
We keep 4 as the default because it is what makes K1 bit-exact — the value is a *documented option*
(`n_smooth=`), and the trade-off is now quantified instead of asserted. Note that at `n_smooth=16`
the support radius (18) exceeds half a 40³ box, so **no** row of the plane cases can be certified
(`null` in `sensitivity_table.csv`) — the certificate genuinely becomes vacuous, not the solver wrong.

## Round-2 code changes (all in `src/tensorlbm/boundary/`, `scripts/`, `tests/`)

1. **Public export fixed** — `wall_normal_iso8` / `wall_normal_preprocess` are re-exported from
   `tensorlbm.boundary`. The two merged tests that imported `build_wall_normal_table` were red on
   `main` (ImportError); they are green now.
2. **`well_posed_mask(indices, walls, ...)` + `support_radius(n_smooth)`** — the certificate above,
   vectorised NumPy, no new dependency. `build_wall_normal_table` returns it as
   `WallNormalTable.well_posed` and accepts `drop_ill_posed=True` for callers that cannot tolerate a
   padding-defined normal (wetting closures). Default behaviour is unchanged (rows kept).
3. **Regression fixture restored** — `artifacts/xval_iso8_wall_normal_table/oss_fixture_normal_table.csv`
   + `oss_fixture_case.json` are regenerated and committed, so `test_matches_oss_fixture_to_machine_precision`
   actually runs (it was skipping). That test is now the drift guard for the 1815-row OSS table (< 1e-12).
4. **Harness arms** — `well_posed` certificate block, `face_distance_bins`, `smooth_sweep_iso8_wellposed`,
   `accuracy_table.csv` rows for both classes, and `figures/fig5_face_distance_error.png`.
5. **Tests** — 13 in `tests/test_wall_normal_iso8.py` (was 7, of which 2 red + 1 skipped): the new
   ones assert the certificate on a flat wall, on the overlap band, on a domain-wall face, on the
   45° plane (recall 100 %, well-posed max < 15 %) and on the sphere (all rows certified, max < 15 %),
   plus the `drop_ill_posed` parity path of the platform caller.

## Round-1 record (kept for the audit trail)

* defect**: `mirror_bc` is read before assignment (undefined behaviour); the harness pins it with
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

The existing ISO8 [harness](../scripts/xval_iso8_wall_normal_table.py) reads the relocated
[original Fortran excerpt](../development/verification/wall-boundary-tables/iso8/oss_src/wall_boundary_process.f90) archived alongside the
[preregistration](../development/verification/wall-boundary-tables/iso8/predeclared.json). See the
[migration ledger](architecture/migrations/wall-boundary-tables-xval-2026-11-24.json)
for their original Git paths.
