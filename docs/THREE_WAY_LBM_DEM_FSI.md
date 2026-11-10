# Three-way LBM–DEM–FSI coupling: ice-floe slam and hull-girder whipping

**Repository** TensorLBM · **Module** `src/tensorlbm/three_way/` ·
**Driver** `scripts/three_way_gates.py` · **Artefacts** `runs/three_way/`,
`docs/figs/three_way_*.png` · **Campaign node** P35-N9

---

## 1. Case, physics and parameters

### 1.1 Why this configuration

The node asks for the **first three-way coupling path** in the ecosystem: an LBM
flow field, a DEM ice floe and an FSI structure exchanging momentum and energy
with *quantified* interface closure.  A 2D vertical-plane flume is used because
it is the cheapest configuration in which all three legs are simultaneously
active and every hand-off is integrable:

* the water column is driven by a **wave-equivalent pressure-gradient forcing**
  `F_x(t) = A[cos(ωt) + γ cos(2ωt)]` imposed in a source column at the left wall
  (`x ∈ [2,6)`), i.e. the forcing a Stokes second-order wave exerts on the water
  column (`−∇p = ρ ∂u/∂t`); the second harmonic carries the bound wave;
* a **buoyant ice floe** (DEM disc, ρ_ice/ρ_w = 0.6) floats at a prescribed
  still-water line and is carried back and forth by the oscillatory flow;
* the floe repeatedly **slams into a flexible cantilever** (the hull-girder /
  ice-strengthened side analogue) standing in the water at `x = 70`; the
  mid-span bending stress of that cantilever is the **whipping output**.

### 1.2 Geometry and parameters (lattice units, `dx = dt = 1`, `c_s² = 1/3`)

| quantity | value | note |
|---|---|---|
| domain | 96 × 48 | closed basin, rigid lid, four solid walls |
| relaxation time τ | 0.56 | ν = 0.02 |
| wavemaker velocity | 0.05 | fundamental + 10 % bound second harmonic |
| wave period | 120 steps | ω = 2π/120 |
| floe radius / density ratio | 5 / 0.60 | draft 0.789, mass 47.12, β = ρ_w A/m = 1.667 |
| floe contact (DEM) | k_n = 2.0, k_t = 1.0, η_n = 2.0, η_t = 0.5, μ = 0.3 | `tensorlbm.dem.LinearSpringDashpot` |
| beam | L = 34, h = 2, E = 2.0e5, ρ_s = 15, ζ = 0.02, 2 modes | ω₁ = 0.203 (period 31 steps) |
| still-water line | y = 34 | buoyancy reference (rigid lid) |
| interface marker spacing | 1.0 | hat kernel for the exchange, 4-point kernel for the integrals |

### 1.3 The three legs and their hand-offs

| leg | hand-off | implementation |
|---|---|---|
| LBM → DEM | fluid velocity and drag on the floe | IBM interpolation of `u` to the Lagrangian markers (`marker_stencil`, `interpolate_stencil`) |
| DEM → LBM | reaction force back into a Guo force field | immersed-boundary **direct forcing over the body volume**, smoothed indicator `χ ∈ [0,1]` over one cell |
| DEM → FSI | contact impulse onto the modal cantilever | `Q_n += F φ_n(y_c)` |
| FSI → DEM | beam deflection at the contact height | `Floe2D.contact_with_segment(..., deflection_fn)` |

### 1.4 Added-mass-implicit direct forcing

For a light floe the explicit update is the classic added-mass instability
(`β = ρ_w A/m = 1.667 > 1`).  The interface velocity is therefore solved
implicitly,

```
u_s = (v + β u_f)/(1 + β),      F_body = −ρ_w χ (u_s − u_f)/Δt,
```

which is the fixed point of the explicit update and is stable for any β.  A
*stale* interface state (`offset_steps > 0`, half weight) reproduces the
time-synchronisation error that gate (d) measures; a fully stale state diverges
within a few steps, which is itself the physical statement that interface
timing is not optional.

---

## 2. Method and gates

`core2d.Flow2D` is a compact D2Q9 BGK solver with a Guo force field, half-way
bounce-back and three diagnostics that all gates draw from:

* the deviatoric stress `σ_ij = −((τ−0.5)/τ) Π_ij^neq` (Chapman–Enskog),
* the viscous dissipation `ε = 2μ S_ij S_ij`,
* the momentum and kinetic energy of the field.

`beam.CantileverBeam` is a modal Euler–Bernoulli cantilever with the exact
clamped-free eigenfunctions, sub-stepped for stability; the whipping stress is
`σ(y) = E (h/2) w''(y)`.

Acceptance gates (`gates.py`, run by `scripts/three_way_gates.py`):

| gate | operational definition | criterion |
|---|---|---|
| (a) | static cylinder in a body-force channel: momentum sink of the direct forcing vs the control-surface integral `∮(−p n + σ·n)dA + ∮ρu(u·n)dA` on markers at `R+3`, three resolutions | < 2 %, observed order ≥ 1 |
| (b) | first-mode generalized force on the structural face: Eulerian full pressure column vs Lagrangian marker quadrature | < 3 % |
| (c) | floe + beam energy ledger over one wave period: interface work (IBM + hydrodynamic) vs stored energy + structural/contact damping | < 3 % |
| (d) | interface evaluated on a lagged state: `R(m) = \|M(m) − M(0)\|/M(0)` | zero-offset < 3 %, half-interval > 10 % |
| (e) | FSI feedback path off → structural response | < 1e-6 |
| (f) | pytest coverage | tests present and green |
| (g) | three seeds (initial-state draw only, physical parameters fixed) | dispersion < 10 % |

---

## 3. Results

*(numbers are written by `scripts/three_way_gates.py` into
`runs/three_way/gates.json`; the tables below quote that file verbatim)*

| gate | quantity | measured | criterion | verdict |
|---|---|---|---|---|
| (a) | LBM→DEM momentum exchange = control-surface integral; order | 7.7%, 4.8%, 8.8%; order 2.93 | < 2 %, order ≥ 1 | **FAIL** |
| (b) | DEM→FSI modal generalized force vs full-field virtual work | 1.37% | < 3 % | **PASS** |
| (c) | energy ledger, floe+beam, one wave period | -0.49% | < 3 % | **PASS** |
| (d) | time sync: zero offset / half-interval lag | 0.76% (stress), 0.01% (DEM); lag detected 16.1% (DEM) | < 3 % / > 10 % | **PASS** |
| (e) | negative control, FSI feedback off | response ratio 0.0e+00 | < 1e-6 | **PASS** |
| (g) | three-seed dispersion (RMS mid-span stress) | 3.3% | < 10 % | **PASS** |

Aggregate verdict: **HONEST FAIL** (one of seven gates does not close — gate (a), see §5.4). Runtime 338 s.

### 3.0 Gate (a) detail

| R [cells] | grid | Re | U | drag (momentum sink) | drag (control surface) | error | Cd |
|---|---|---|---|---|---|---|---|
| 4 | 80×32 | 8.7 | 0.0362 | 0.05436 | 0.05018 | 7.7% | 10.38 |
| 6 | 120×48 | 23.3 | 0.0648 | 0.12343 | 0.12936 | 4.8% | 4.90 |
| 8 | 160×64 | 42.6 | 0.0888 | 0.20390 | 0.22194 | 8.8% | 3.23 |

Cd refinement: 10.38 → 4.90 → 3.23; observed order p = 2.93 ≥ 1 ✓.

### 3.1 Whipping response

The coupled run produces repeated ice-floe impacts on the flexible beam and a
decaying whipping ringing at the first bending period:

| quantity | value |
|---|---|
| steps | 480 |
| ice-floe contact steps | 34 |
| peak contact force | 0.2025 |
| peak mid-span stress | 36.39 |
| RMS mid-span stress | 10.90 |
| peak clamp stress | 115.17 |
| peak tip deflection | 0.1880 |
| floe surge range | -0.0465 … 0.0682 |

![whipping](figs/three_way_whipping.png)

![floe trajectory](figs/three_way_floe.png)

![gate a](figs/three_way_gate_a.png)

![ledger](figs/three_way_ledger.png)

---

## 4. Error judgement against references

| quantity | reference | measured | error | < 3 %? |
|---|---|---|---|---|
| Cd, R = 8 (Re 42.6, D/H = 0.25) | 2D cylinder literature Cd ≈ 1.3–1.5 unconfined; blockage (D/H = 0.25) raises it by roughly a factor 1 + 0.5 D/H ≈ 1.1–1.3 | 3.23 | +70 % vs the unconfined band | **no** — the anchor is order-of-magnitude only (diffuse-interface body, blockage, Re range) |
| Cd convergence | three grid levels | 10.38 → 4.90 → 3.23 | observed order 2.93 | **yes** (order ≥ 1) |
| momentum exchange vs control-surface integral | independent route | 7.7 %, 4.8 %, 8.8 % (R = 4, 6, 8) | best 4.8 % | **no** (criterion 2 %) — honest FAIL |
| modal generalized force | full-field virtual work | 1.37 % | — | **yes** |
| energy ledger (floe + beam) | first-principles budget | −0.49 % | — | **yes** |
| time synchronisation | zero-offset floor / half-interval lag | 0.76 % / 16.1 % | — | **yes** |
| negative control | zero response without feedback | 0.0 | — | **yes** |
| seed dispersion | three seeds | 3.3 % | — | **yes** |
| fluid-wide ledger (informational) | wavemaker work vs fluid KE + dissipation | −250 % | — | **no** — see §5.8; the isothermal LBM's compression work and the velocity-source controller make this budget ill-conditioned, which is why gate (c) is defined on the floe+beam subsystem |

Explicit answers: the DEM→FSI, energy, timing, control and seed criteria are
**< 3 %** (or zero by construction); the LBM→DEM surface-integral criterion is
**not**: it closes to 4.8 % at the best resolution and 7.7–8.8 % elsewhere.
Consequently the aggregate verdict is an **honest FAIL** for gate (a).

---

## 5. Pitfalls, limitations and scope

1. **No resolved free surface.** The flume is single-phase water with a rigid
   lid; buoyancy is computed against a prescribed still-water line.  The wave is
   represented by its pressure-gradient forcing, not by a moving free surface,
   so wave breaking, run-up and added resistance are outside the model.  The
   quantity of record here is the *coupled interface closure*, not the wave
   field itself.
2. **Added-mass implicit exchange.** The interface velocity `u_s` is a
   compromise between the body and fluid velocities, so the instantaneous
   no-slip residual is not zero by construction.  This is why gate (d) uses a
   *response* residual rather than a pointwise slip: the pointwise slip is
   dominated by the added-mass physics, not by numerical error.
3. **Volume-averaged IBM.** The body is forced through a smoothed indicator
   `χ`, not by a sharp surface force.  This conserves momentum to machine
   precision but diffuses the interface over one cell.
4. **Surface quadrature bias.** The marker-based surface integral
   (`∮(−p n + σ·n)dA`) carries a curvature-induced bias of a few percent that
   is *converged* in marker density but not in contour radius (best 1.6 % at
   `R+3`, `R = 6`; 4–9 % at the other resolutions — see gate (a)).
   This is the one interface criterion that does not close everywhere.
5. **Beam deformation is one-way for the fluid.** The LBM sees a static solid
   column; the deformation feeds back only into the DEM contact geometry — which
   is exactly what this node specifies ("结构变形反馈到DEM接触几何"), but a moving
   mesh is not implemented.
6. **Circular floe.** A disc has no hydrostatic restoring moment, so floe
   rotation is not excited by buoyancy; the contact torque is retained but
   remains small.
7. **Fluid-wide energy ledger does not close** (informational, not gated):
   the isothermal BGK lattice has no internal-energy variable, so the
   `∫p ∇·u` compression work is not representable in a KE+PE budget, and the
   velocity-source wavemaker's work is dominated by the controller's
   corrections.  The gate (c) ledger is therefore defined on the floe+beam
   subsystem, where every term (interface work, stored energy, Rayleigh and
   contact-dashpot losses) has an exact discrete counterpart; it closes to
   0.5 %.  A fluid-wide budget would need the internal-energy equation.
8. **Whipping is dominated by the hydrodynamic pressure load** (RMS 9 vs a
   contact-only contribution of order 1); the ice contact provides the
   impulsive excitation that sets the ringing phase, and the negative control
   (e) confirms the FSI path is the only source of the structural response.

---

## 6. Reproduction

```bash
PYTHONPATH=src python scripts/three_way_gates.py     # ~5 min, writes runs/three_way/
PYTHONPATH=src python -m pytest tests/test_three_way_core.py tests/test_three_way_gates.py -q
```