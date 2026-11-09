# IN70-N01 — DG convection–diffusion verification ladder (TensorLBM)

**Node:** IN70-N01 · campaign *DG from zero to machine-falsifiable*
**Repo:** `TensorLBM` · **module:** `src/tensorlbm/dg_verification/` · **branch:** `agent/m7709`
**Evidence bundle:** `artifacts/IN70_N01/` (`*.json` per benchmark, `VERDICT.json`, `fields_*.npz`, `figures/`)
**Machine gate:** `tests/test_dg_verification.py` (38 tests)

---

## 1. Case introduction — what is being verified, and why DG

The repository already owned a *nodal DG advection kernel* used inside the hybrid
DG-LBM (`src/tensorlbm/dg_advection.py`, `dg_lbm.py`), but it had **no
verification face of its own**: no convergence study, no manufactured solution,
no shock test, no limiter, no mass ledger. A DG discretisation that is only ever
used inside a coupled solver cannot be graded — its errors are entangled with the
LBM collision, the coupling band and the obstacle geometry. This node builds the
missing verification face: a small, LBM-independent DG library for textbook
conservation laws plus the five benchmarks of the campaign table.

Physics covered (all periodic Cartesian grids, float64, numpy):

| # | equation | exact / reference | what it grades |
|---|----------|-------------------|----------------|
| B1 | `u_t + ∇·(a u) = 0`, `a = (0.5−y, x−0.5)` (rigid rotation), `[0,1]²` | Lagrange cone returns to itself at `t = 2π` | L2/L1/L∞ order |
| B2 | `u_t + ∇·(a u) = νΔu + S`, `[0,2]²`, ν = 0.05 | manufactured `u = sin(πx)sin(πy)e^{−t}` | order vs theory `p+1` |
| B3 | 1-D ideal-gas Euler, γ = 1.4, Sod IC `(1,0,1)|(0.125,0,0.1)`, `t=0.2` | exact Riemann (Toro) | shock position/strength |
| B4 | 1-D **isothermal** Euler (`p = ρc_s²`, `c_s²=1/3`) — the D2Q9 closure — `ρ 1→0.25`, `dx=1` | exact isothermal Riemann **+ verified D2Q9 run** (`benchmarks/verified/sod_shock_tube`) | DG ⇄ LBM ⇄ analytic shock position |
| B5 | 1-D advection of a square wave, minmod/TVB limiter | TVD: no new extrema | limiter correctness |
| B6 | periodic mass ledger (advection + LDG diffusion) | `|∫u−∫u₀|/|∫u₀|` | conservation |

Parameter table (all runs): Gauss–Lobatto nodal DG, degree `p = 1…3`,
`p+3`-point Gauss quadrature, upwind (B1/B2) or local Lax–Friedrichs (B3/B4)
interface flux, LDG for the diffusion in B2, SSP-RK3, `float64`, CPU.

Discretisation (1-D building block, `ξ ∈ [−1,1]`, cell width `h`):

```
du/dt = (2/h) M⁻¹ ( +∫φ_i' f  − [f*_r φ_i(x_r) − f*_l φ_i(x_l)] )      (conservation law)
q     = (2/h) M⁻¹ ( −∫φ_i' u  + [û_r φ_i(x_r) − û_l φ_i(x_l)] )         (LDG: q = u_x)
```

The **sign of the surface term is the whole game**: putting it on the other side
turns the upwind flux into a downwind one and the scheme grows exponentially.
Two bugs of exactly this class were found and fixed during this node (§5).

---

## 2. How it was done — tool chain and workflow

```
src/tensorlbm/dg_verification/
  element.py    Lobatto nodes, V/Vr, mass matrix & inverse, Gr = ∫l_k'l_j,
                face selection S, nodal differentiation D, Legendre modal map
  limiter.py    modal↔nodal transforms, minmod, Cockburn–Shu TVB limiter
  scalar.py     advection (variable coefficient, upwind/Rusanov), LDG diffusion,
                SSP-RK3, quadrature integrals, Gauss-point error norms
  euler1d.py    1-D Euler DG (isothermal & ideal gas), exact Riemann solvers,
                positivity fallback, shock locator
  stability.py  measured spectral radii of the assembled 1-D operators →
                the time step (c_adv = 6.0/11.84/19.16, κ = 36/148/439 for p=1/2/3)
  benchmarks.py the six benchmarks; every number below is produced by it
scripts/run_dg_verification.py   → artifacts/IN70_N01/*.json + fields_*.npz
scripts/plot_dg_verification.py  → artifacts/IN70_N01/figures/*.png
scripts/verdict_dg_verification.py → artifacts/IN70_N01/VERDICT.json
tests/test_dg_verification.py    → 38 machine gates
```

Workflow: kernels first, then `pytest` on operator consistency / conservation /
exact solvers, then the benchmarks at production resolution, then the verdict
aggregation (`VERDICT.json`) and the figures. Nothing in the report is copied by
hand: every table below names the JSON it came from, and the JSON is written
directly from the in-memory results of the run.

Six invariants are checked *before* any convergence claim (they are the tests
that caught the implementation bugs):

1. a constant field is steady (`|rhs| < 1e-12`) for both operators;
2. the periodic RHS annihilates global mass for random fields;
3. the operator is *consistent*: `rhs ≈ −a·∇u` (advection) and `rhs ≈ +νΔu`
   (diffusion), measured by the mean ratio `rhs/exact = 1.000 ± 0.03`;
4. the assembled 1-D advection operator has `max Re λ = 0` (no growth mode);
5. exact solvers reproduce the literature values (Toro table 4.1);
6. a uniform Euler flow translates without deformation to machine precision.

---

## 3. Results

### 3.1 Rotating cone / smooth rotation (B1) — `rotating_cone.json`, `rotating_gaussian.json`

L2 error of the field after one full revolution (`t = 2π`), LeVeque cone
(centre (0.5,0.75), r = 0.15, h = 1), upwind flux, `dt` from the measured
spectral radius:

| p | N = 16 | N = 32 | N = 64 | order (16→32 / 32→64) | LSQ fit |
|---|--------|--------|--------|----------------------|---------|
| 1 | 3.486e−2 | 1.121e−2 | 4.651e−3 | 1.64 / 1.27 | 1.45 |
| 2 | 6.791e−3 | 2.878e−3 | 1.150e−3 | 1.24 / 1.32 | 1.28 |

Smooth companion (Gaussian σ = 0.1 at the same centre) isolates the kink of the
cone initial data: P1 gives **2.15** (design order `p+1 = 2` ✓), while P2 stalls
at an **error floor ≈ 2e−3** (2.304e−3 → 1.981e−3, apparent order 0.22).

Time-integration error was excluded explicitly: halving `dt` at N = 64 changes L2
by 3.2e−5 (P1) and 1.3e−4 (P2) relative — three orders of magnitude below the
spatial error. Mass drift over the same run is ≤ 2.8e−13 (machine precision), so
the loss of order is **not** a conservation or a time-stepping defect.

### 3.2 MMS, convection–diffusion (B2) — `mms.json`

`u = sin(πx)sin(πy)e^{−t}` on `[0,2]²` with the analytically derived source
`S = u_t + a·∇u − νΔu` (`a = (1,1)`, ν = 0.05), `t_end = 0.2`, periodic:

| p | N = 8 | N = 16 | N = 32 | measured order | theory |
|---|-------|--------|--------|----------------|--------|
| 1 | 4.724e−2 | 1.202e−2 | 2.993e−3 | 2.01 | 2 |
| 2 | 1.431e−3 | 1.728e−4 | 2.143e−5 | 3.01 | 3 |
| 3 | 6.621e−5 | 4.129e−6 | (n/a) | 4.00 | 4 |

All three degrees land within **0.02** of theory. This also verifies the LDG
operator (the νΔu part) — the diffusion contribution is 2π²ν ≈ 0.99 per unit
time here, i.e. comparable to the advective term, so it is not a token term.

### 3.3 Sod shock tube (B3, ideal gas) — `sod.json` — **honest FAIL**

`γ = 1.4`, N = 100…400, TVB limiter (M = 0), LF flux, `t = 0.2`, exact reference
from `exact_sod` (p\* = 0.30313, u\* = 0.92745, ρ\*_L = 0.42632, ρ\*_R = 0.26557 —
all reproduced to 2e−4 by the solver).

| p | N | L1(ρ) | shock x (numerical) | exact | err / shock-contact distance |
|---|---|-------|--------------------|-------|------------------------------|
| 1 | 200 | 0.1622 | 0.8091 | 0.8504 | 25.0 % |
| 2 | 200 | 0.1688 | 0.8098 | 0.8504 | 24.6 % |

The solution is *not* usable: `|u|` reaches ≈ 0.9 far from the star region and
`L1(ρ) = 0.16`, i.e. a spurious large-scale wave. Diagnosis (measured, not
speculated): Rusanov/LF diffuses **mass across the contact without carrying
momentum or energy**; a pure contact test (`ρ 1→0.125` at equal p, exact u = 0)
already produces `|u|max = 0.886` with this flux, while the same code keeps
`u ≡ 0` to 1e−12 in a smooth acoustic run. An HLLC flux was prototyped to fix
this but did **not** reach a stable state inside this node, so it was removed
rather than shipped unvalidated. **This sub-item fails the `< 2 %` gate.**

### 3.4 Sod shock tube (B4, isothermal) — DG ⇄ LBM ⇄ analytic — `sod_isothermal.json`

Same closure and same initial data as the verified D2Q9 benchmark
(`benchmarks/verified/sod_shock_tube`: ρ 1→0.25, dx = 1, cs² = 1/3), so the two
solvers can be compared without a physics mismatch. Analytic isothermal Riemann:
ρ\* = 0.496623, u\* = 0.404101, W = 0.813735; the rarefaction invariant
`u* = c_s ln(ρ_L/ρ*)` reproduces u\* to 2e−16 (an independent cross-check of the
exact solver itself).

| p | t | shock x (DG) | exact | rel. err | LBM (verified run) | Δ vs LBM |
|---|---|--------------|-------|----------|--------------------|----------|
| 1 | 100 | 1081.3 | 1081.37 | 0.04 % | 1080 | −1.3 cells |
| 1 | 400 | 1325.6 | 1325.49 | 0.04 % | 1324 | −1.6 cells |
| 2 | 100 | 1082.1 | 1081.37 | 0.92 % | 1080 | −2.1 cells |
| 2 | 200 | 1163.5 | 1162.75 | 0.46 % | 1162 | −1.5 cells |
| 2 | 400 | 1673.9 | 1325.49 | **107 %** | 1324 | −350 cells |

Up to `t = 200` every point is well inside the 2 % gate, and the DG shock sits
1–2 lattice cells from the verified LBM shock for both degrees. The `P2, t = 400`
point degrades catastrophically — an unresolved long-time degradation of the
minmod-limited P2 run (it is reported, not hidden; see §5).

### 3.5 Square wave with the TVB limiter (B5) — `square_wave.json`

Advected one period (`a = 1`, `[0,1]`, jumps on cell faces), P1, minmod (M = 0):

| N | L1 vs IC (limited) | max u | min u | smeared cells | L1 vs IC (unlimited) | max u (unlimited) | min u (unlimited) |
|---|--------------------|-------|-------|---------------|----------------------|-------------------|-------------------|
| 100 | 0.0631 | 1.0000000 | 4.3e−6 | 100 | 0.0255 | 1.0633 | −6.33e−2 |
| 200 | 0.0402 | 1.0000000 | 4.8e−11 | 176 | 0.0155 | 1.0659 | −6.59e−2 |
| 400 | 0.0255 | 1.0000000 | 7.9e−21 | 226 | 0.0091 | 1.0641 | −6.41e−2 |

TVD holds to round-off: no overshoot above 1.0 and no undershoot below 0.0 for
any resolution, while the *same* grid without the limiter overshoots by 6.4 %
(Fig. 4). The limited solution is more diffusive than the oscillatory one, as
expected — the limiter trades formal accuracy for monotonicity.

### 3.6 Mass ledger (B6) — `mass_ledger.json`

| case | steps | relative drift (worst) |
|------|-------|------------------------|
| rotating divergence-free advection, P2, N = 32, 2 periods | 4 744 | 4.0e−13 |
| LDG heat decay, P2, N = 32, 200 steps | 200 | < 1e−15 |

Gate `< 1e−10` → **pass by 2.5 orders of magnitude**. This is structural, not
lucky: the limiter keeps the cell mean (its modal `m₀`) untouched, so limiting
and conservation are compatible (that property is itself a test).

### 3.7 Figures

* `figures/fig1_convergence.png` — log–log L2 vs h with theoretical slopes, MMS
  (left) and rotating cone + smooth companion (right).
* `figures/fig2_cone_fields.png` — cone fields at `t=0` and after one revolution
  (P1 and P2, N = 32), i.e. the 2-D cloud plot of the benchmark.
* `figures/fig3_sod.png` — ideal-gas density (DG vs exact) and the isothermal
  DG ⇄ LBM ⇄ analytic shock-position error with the 2 % gate.
* `figures/fig4_square_wave.png` — limiter on/off profiles at three resolutions.
* `figures/fig5_mass_ledger.png` — mass drift history vs the 1e−10 gate.

---

## 4. Error judgement against the campaign criteria

Criteria are aggregated programmatically in `artifacts/IN70_N01/VERDICT.json`.

| criterion (from the node table) | measured | verdict |
|---|---|---|
| rotating cone L2 order P1 ≥ 1.8 | 1.27 / 1.45 (cone, last pair / LSQ) | **FAIL** |
| rotating cone L2 order P2 ≥ 2.7 | 1.32 / 1.28 | **FAIL** |
| MMS order within 0.3 of theory | 2.01 / 3.01 / 4.00 (dev ≤ 0.02) | **PASS** |
| Sod shock position error < 2 % | isothermal P1/P2 ≤ 200: 0.04–0.92 % | **PASS** |
| — same, ideal gas γ = 1.4 | 25 % | **FAIL (honest)** |
| — same, isothermal P2 at t = 400 | 107 % | **FAIL (honest)** |
| mass conservation < 1e−10 | 4.0e−13 | **PASS** |
| square wave TVD (no new extrema) | overshoot 0, undershoot 0 (limited) vs ±6.4 % (unlimited) | **PASS** |

Question the discipline requires to be answered explicitly: **are the key
quantities within 3 % of their reference?** — For the *smooth manufactured*
convection–diffusion case and for the *isothermal* Riemann problem up to
`t = 200`, yes (orders within 0.02 of theory; shock position 0.04–0.92 %). For
the *rating* quantities of the rotating cone (which has a kink) and for the
ideal-gas Sod tube, **no** — those two sub-items are honest FAILs and are not
counted as verified.

---

## 5. Pitfalls, limitations, applicability

**Bugs found and fixed in this node** (each is now a regression test):

1. **Surface-term sign.** The weak form `M u̇ = +∫fφ′ − [f*φ]` had the surface
   term with the wrong sign, i.e. the "upwind" flux acted as a downwind one;
   the operator had a growth mode (`λ = +6` on a two-DOF periodic cell) and
   every time-dependent run diverged. Detected by the consistency test
   (`rhs/exact = −1`).
2. **Axis placement of the face term.** The 2-D y-direction inserted the
   face-selection matrix on the *last* axis instead of the y-node axis → the y
   flux landed in the wrong slots. Detected by the 2-D random-field mass test.
3. **Broken-cell initial data.** With Gauss–Lobatto nodes the interface node is
   shared by two elements, so sampling a discontinuous IC by *coordinate* heals
   the jump (both traces get the same value) and the Riemann fan never starts.
   Initial data for shocks must be assigned per *cell*. Symptom was a run that
   barely evolved.
4. **Limiter breaking conservation.** Limiting the two endpoints independently
   produces a profile whose mean is no longer `ū` (measured mean drift 1.8e−1).
   Fixed with the classical single symmetric slope
   `slope = minmod(2d_r/h, −2d_l/h, Δ₊/h, Δ₋/h)`.
5. **Limiter slope units.** The endpoint deviation of a linear profile is
   `slope·h/2`; mixing that with the cell-difference `slope·h` halves the slope
   and silently makes the scheme first-order (this is what produced the
   "everything is a cosine" Sod profiles before the fix).
6. **Time step for the diffusive part.** A guessed CFL constant for LDG is a
   trap: the P2 diffusion operator's spectral radius is ~4× the P1 one at the
   same `dx`. The step is now derived from the *measured* spectral radius of the
   assembled operator (`stability.py`), which is why the MMS converges at every
   degree instead of diverging at the finer grids.

**Open issues (not solved here, listed for the next node):**

* **P2 error floor ≈ 2e−3 in the rotating (variable-coefficient) advection.**
  P1 shows the design order (2.15) on the smooth Gaussian; P2 stops improving
  after N = 16 while the dt-refinement test and the mass ledger exclude time
  integration and conservation as causes. The prime suspect is the collocated
  evaluation of the variable-coefficient flux `a(x)u(x)` (aliasing) and/or the
  face-normal velocity extrapolation; over-integration of the volume term and a
  face-averaged velocity are the obvious next experiments.
* **Rotating-cone order.** The cone IC has a kink (discontinuous first
  derivative) at its base, which caps L2 convergence at `p+½` at best; the
  measured 1.45 / 1.28 is consistent with a kink-limited rate *plus* the P2
  floor above. The campaign gate (1.8 / 2.7) is therefore not met by this
  benchmark as specified.
* **Ideal-gas Sod.** Needs a contact-correct interface flux (HLLC/HLL-D) and a
  positivity-preserving limiter. The HLLC prototype was not stabilised in this
  node and was removed rather than shipped.
* **Isothermal P2, t = 400.** Long-time degradation of the minmod-limited P2 run
  (shock position off by 350 cells while P1 stays at 1.6 cells). Reproducible;
  root cause not isolated. Until then, treat the limited P2 isothermal path as
  validated only for `t ≲ 200` (≈ 100 shock crossings).
* **Applicability.** What is verified here is the *periodic* DG machinery
  (operators, limiters, LDG, conservation, orders) on unit-scale model problems.
  It does not by itself validate the coupled DG-LBM band
  (`dg_lbm.py`), wall boundaries, or anything 3-D.

---

## 6. Reproduce

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m pytest tests/test_dg_verification.py -q
PYTHONPATH=src python scripts/run_dg_verification.py --bench all      # ~6 min, 128-core box
PYTHONPATH=src python scripts/plot_dg_verification.py
PYTHONPATH=src python scripts/verdict_dg_verification.py
```

Artifacts: `artifacts/IN70_N01/{rotating_cone,rotating_gaussian,mms,sod,
sod_isothermal,square_wave,mass_ledger}.json`, `VERDICT.json`,
`fields_*.npz`, `figures/fig1..fig5.png`, run logs in
`artifacts/IN70_N01/logs/`.