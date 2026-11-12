# GAP-LBM-T5-ICE-COUPLING — 3-D coupling-interface design (round 1)

* Companion to `docs/GAP-LBM-T5-ICE-COUPLING_dependency_audit.md`.
* Reference implementation: `src/tensorlbm/three_way/interface3d.py`.
* Gates: `scripts/three_way_interface3d_gates.py` → `runs/three_way_interface3d/gates.json` (mirrored to the tracked `artifacts/three_way_interface3d/`);
  numbers are rendered to `docs/generated/GAP-LBM-T5-ICE-COUPLING_gate_tables.md` by the
  same script (no hand transcription anywhere).
* Status: **interface prototype, gated against analytic references.** This is *not* a
  verified T5 chain (see the audit's round-1 verdict).

## 1. Design goal and non-goals

**Goal.** Define the single exchange path through which all three T5 legs talk, so that
each leg can be developed and graded independently:

```
   LBM (fluid)  ──surface tractions──▶  INTERFACE  ──nodal forces──▶  FEM (structure)
                                        │  ▲
                            DEM (ice)  ──┘  └── particle positions / velocities
```

**Contract.** The interface must be

1. **conservative** — the transferred load reproduces the resultant and the moment of the
   applied load (`Σf = ∫t dA`, `Σ r×f = ∫ r×t dA`);
2. **work-conjugate** — the velocity transfer is the transpose of the load transfer, so no
   interface work is created (`Σ f_i·u_i = ∫ t·u dA` for affine velocity fields);
3. **declared in time** — sub-cycling ratio, coupling lag and integration rule are
   explicit objects, not implicit in a driver loop;
4. **guarded** — the partitioned scheme refuses configurations where the added-mass
   instability is unavoidable (`μ = m_a/m_s ≥ 1`);
5. **auditable** — every claim is a gate with an independent reference, written to JSON.

**Non-goals for round 1.** No 3-D LBM solve, no shell FEM, no bonded 3-D ice. Those are
backends behind the contract (audit gaps G5-05/06/07).

## 2. Data model

| type | role | fields / key methods |
|---|---|---|
| `Surface3D` | Lagrangian surface quadrature (hull patch, floe surface, sphere) | `x,y,z,nx,ny,nz,dA`; `traction_force`, `pressure_force`, `moment`, `work`, `centroid`, `area` |
| `sphere_surface(...)` | closed sphere / open hemisphere / capped hemisphere | UV triangulation, geometric area-weighted normals, centroid rule |
| `rectangle_surface(...)` | planar patch with *exact* geometry | `n1×n2` quads → 2 triangles each |
| `HexGrid3D` | Q1 (trilinear) structural discretisation | `shape8`, `local_coords`, `containing_element`, `distribute_point_consistent`, `distribute_point_lumped` (control), `distribute_surface`, `gather_point_velocity` |
| `project_to_beam(...)` | hull-girder (Hermite cubic) reduction | same shape functions as `beam_fem.EulerBernoulliBeamFEM` |
| `CouplingClock` | partitioned clock | `dt_fluid`, `dt_struct`, `n_sub`, `lag`; `check()` refuses inconsistent clocks |
| `InterfaceLedger` | interface work bookkeeping | trapezoid work, `closure(ΔE)` |
| `added_mass_ratio`, `explicit_staggered_stable`, `relaxation_bound`, `staggered_spectral_radius`, `march_one_dof` | stability guard and its verification model | see §5 |

## 3. Exchange operators and why they are exact

Let `N_i(x)` be the Q1 shape functions of the structural discretisation, `r_i` the node
positions. Q1 elements are isoparametric: the interpolation of the nodal coordinates *is*
the geometry map, so two algebraic identities hold exactly (not approximately):

* **I1 partition of unity**: `Σ_i N_i(x) = 1` for any `x` in the element (including the
  clipped case where `x` lies outside — hence no renormalisation is needed);
* **I2 linear completeness**: `Σ_i N_i(x) r_i = x(ξ(x)) = x` for the isoparametric map.

**Load transfer** (`f_i += N_i(x_p) F`, `f_i += Σ_m N_i(x_m) t_m dA_m`).
Then `Σ_i f_i = F Σ_i N_i = F` (I1) and `Σ_i r_i×f_i = (Σ_i N_i r_i)×F = x_p×F` (I2):
resultant **and** moment are conserved to machine precision — no separate
"moment correction" pass is needed, and none is implemented (a correction pass would in
fact break work conjugacy).

**Velocity transfer** (`u_p = Σ_i N_i(x_p) u_i`) is the transpose of the load operator, so
for any affine velocity field `u(x) = u₀ + ω×(x−x₀)`:
`Σ_i f_i·u(r_i) = ∫ t·u dA` exactly (I3). This is what forbids the interface from creating
or destroying energy.

**Why the naive alternative is rejected.** Concentrating a point load on the nearest node
(`distribute_point_lumped`, kept only as the gate's control) conserves the resultant but
produces a moment error of order `h` — measured in I4 as **1.04 → 0.515 → 0.114** for
`h = 0.5 → 0.25 → 0.125` (observed order ≈ 1.0), against **≤ 3e-17** for the consistent
operator. Two resolutions are reported for every gate so this is a *measured* design
decision, not an assertion.

**What is approximate (and is labelled as such).** The marker rule is one point per
triangle, so the *surface quadrature* of a non-constant traction carries `O(h²)` error:
I3 reports the discrete identity at machine precision (`≤ 1.7e-16`) but the integral of the
*quadratic* work density against a 6×6 Gauss reference at **4.04e-4 → 1.01e-4 → 2.53e-5**
with observed order **2.00** (marker refinement 4→8→16 per side). Clipping markers that
fall outside the structural block also degrades moment conservation locally; the current
implementation clamps to the nearest cell and this is documented as a boundary limitation.

**Hull-girder reduction.** `project_to_beam` uses exactly the Hermite cubic basis of
`beam_fem`, so the transverse resultant and its first moment are conserved exactly and the
load is delivered to the same generalized coordinates the FEM integrates. Verified in I6:
a 1000 Pa panel load on a 0.1 m wide hull strip gives a tip deflection of
**−7.1398e-3 m** vs the Euler–Bernoulli `qL⁴/8EI` **7.1429e-3 m** (0.043 % at 16 stations,
observed order 2.00; 0.17 % at 8 stations).

## 4. Time integration scheme

Partitioned, explicit, sub-cycled:

1. advance the fluid `n_sub` steps of `dt_fluid`;
2. sample the surface traction at the (possibly lagged) fluid state `lag` structural steps
   back, project to structural DOFs;
3. advance the structure one step `dt_struct = n_sub·dt_fluid`;
4. update the ledger, check the guard, re-derive the DEM contact forces.

`CouplingClock.check()` raises if `n_sub·dt_fluid ≠ dt_struct` — a silent clock mismatch is
the most common way a "converged" coupled run turns out to have been reading stale loads
(gate `clock`: mismatch 0.0 for the consistent clock, 0.6 rejected, `ValueError` raised).
The lag is an explicit field so the artefact records which lag was used; the 2-D chain
already showed a half-interval lag is detectable (16.1 % response change, documented in
`docs/THREE_WAY_LBM_DEM_FSI.md`), which is the falsifiability test round 2 should repeat in
3-D.

## 5. Stability guard (added mass)

For an incompressible fluid the explicit Dirichlet–Neumann staggering has an added-mass
instability: the fluid reacts to the structural acceleration before the structure can
respond, and the loop diverges once `μ = m_a/m_s ≥ 1` (Causin–Gerbeau–Nobile 2005;
Förster–Wall–Ramm 2007 for the relaxation bound). The design:

* `added_mass_ratio(m_s, ρ_f, V, C_a)` — `μ = C_a ρ_f V / m_s`;
* `explicit_staggered_stable(μ)` — refuse `μ ≥ 1` (switch to implicit/strong coupling or
  increase structural mass / scale the added mass);
* `relaxation_bound(μ) = 2/(1+μ)` — bound on the under-relaxation factor of the interface
  fixed-point iteration.

Verified against a reduced (`Causin`) 1-DOF model, whose discrete recursion is a 3×3
matrix whose spectral radius is available in closed form, **independently** of the time
march:

| μ | `|λ|` closed form | growth per step (time-marched, log-slope fit) | rel err |
|---|---|---|---|
| 0.5 | 0.999989 | 0.999456 | 5.3e-4 |
| 2.0 | 2.000044 | 2.000044 | 6.7e-16 |

Bisection on `|λ|(μ) = 1` gives `μ_crit = 0.999975` (reference 1.0, rel 2.5e-5). The
relaxation bound is exact: at `ω = 0.6 < 2/3` the iteration contracts with measured rate
0.800000 (= `|1−ω(1+μ)|`), at `ω = 0.7333 > 2/3` it diverges with rate 1.200000.

Physical consequence for T5: a 1 kg floe displacing a 0.05 m sphere bears
`μ = 0.524` (gate `clock` output), so a *strongly* coupled scheme is affordable at small
model scale; a full-scale floe (`m_s ≈ ρ_ice h A`) in water has `μ ~ ρ_w/ρ_ice · O(1) > 1`
and must use the relaxed/implicit branch. The guard exists so this is caught by a check
rather than by a diverging run.

## 6. Invariants, gates, and their status

| id | invariant | gate | how it is checked | verdict |
|---|---|---|---|---|
| I1 | `Σ_i N_i = 1`, `Σ_m n_m dA_m = 0` on a closed surface | `I1` | closed sphere `|Σ n dA| = 7.6e-18`; open-hemisphere pressure resultant vs `pπa²`: 2.55 % → 0.64 % → 0.16 % (order ≈ 2.00) | PASS |
| I2 | resultant + moment of a transferred surface load | `I2` | uniform traction on a plate: `F_z = 2.000000000000` (rel 0), `|M| ≤ 5.7e-17` at 6 and 12 cells | PASS |
| I3 | work conjugacy (affine fields) | `I3` | discrete identity `≤ 1.7e-16`; quadrature error vs 6×6 Gauss order 2.00 | PASS |
| I4 | DEM point load: moment conservation vs control | `I4` | consistent `≤ 1.1e-16`; nearest-node 1.04/0.515/0.114 (order 1.01) | PASS |
| I5 | added-mass guard | `I5` | closed-form vs marched growth; `μ_crit = 0.999975`; relaxation bound exact | PASS |
| I6 | structural leg fed *through* the interface | `I6` | tip load `5.1e-12` vs `FL³/3EI`; `f₁` `1.3e-7`; panel UDL tip deflection 0.17 % → 0.043 % (order 2.00) | PASS |
| I7 | interface work ledger | `I7` | `|W−ΔKE|/E₀ = 5.7e-5 → 1.4e-5`, observed order 2.00 | PASS |
| — | clock consistency | `clock` | inconsistent clock raises; `added_mass_ratio` unit case 0.5236 | PASS |

All eight gates pass in 4 s; see `runs/three_way_interface3d/gates.json` and the generated
tables. `tests/test_three_way_interface3d.py` (14 tests, 4.8 s) pins the same invariants at
unit level so a regression is caught before the study runs.

## 7. Failure modes and mitigations

| failure mode | symptom | mitigation in this design | residual risk |
|---|---|---|---|
| added-mass divergence | amplitude grows geometrically | `explicit_staggered_stable` / `relaxation_bound` guard, explicit `μ` in the artefact | reduced 1-DOF model only; real fluid needs the round-2 check |
| moment loss at the hand-off | spurious torque on the structure | consistent Q1 transfer (I2/I4), lumped route kept only as control | clamped markers near boundaries |
| spurious interface work | energy drift not attributable to physics | transpose (work-conjugate) transfer + ledger (I3/I7) | quadrature error `O(h²)` (labelled) |
| silent time offset | loads read from the wrong state | `CouplingClock.check()`, declared `lag` | lag must be repeated in the 3-D run |
| quadrature under-resolution | loads smooth but under-integrated | two/three resolutions per gate with observed order | marker cost for fine patches |
| duplicated package roots | `ImportError` / wrong physics reused | audit F2: single canonical path required before round 2 | not fixed now |

## 8. How round 2 plugs its backends in

The interface deliberately knows nothing about the LBM or the DEM solver; it consumes

```python
traction = lambda surface: ...          # (M,3) per unit area, from the fluid solver
particle_positions, particle_forces     # (P,3) from the DEM solver
nodal = grid.distribute_surface(surface, traction)      # + distribute_point_consistent
u_p   = grid.gather_point_velocity(x_p, u_nodes)        # transpose
```

so round 2 is a wiring exercise with existing verified pieces:

1. **Fluid backend**: `solver3d` D3Q19 + `ibm.py` 3-D operators, on the *verified*
   `benchmarks/verified/sphere_re100` case (C_D error 2.77 %, control-volume closure
   1.5e-4) → replace the analytic traction with a real one (gap G5-07).
2. **Structural backend**: keep `beam_fem` for the hull girder (verified here); add a
   plate/shell backend behind the same nodal-force contract (gap G5-05).
3. **DEM backend**: `dem/{particle,contact}` point forces into
   `distribute_point_consistent` (conservation already gated); 3-D bonded breakage is the
   remaining physics gap (G5-06).
4. **Gate upgrade**: re-run I1–I7 with the real fields and add the 2-D chain's gates (a)
   and (d) in 3-D, plus the T1 multiphase prerequisite (blocked, audit F3).

## 9. Limitations stated plainly

* No 3-D LBM→FEM→DEM coupled simulation was run in round 1; the fluid side enters the
  gates only as analytic tractions. "Interface verified" ≠ "T5 verified".
* The added-mass guard is verified on a reduced 1-DOF model, not on the 3-D fluid.
* The structural leg is a 1-D hull girder; 3-D shell behaviour (local panel breathing,
  buckling) is outside the current structural backend.
* Surface quadrature is first-order accurate in the load *density* per triangle
  (`O(h²)` in the work integral); refine markers or use a higher-order rule for sharp
  contact-pressure spikes.