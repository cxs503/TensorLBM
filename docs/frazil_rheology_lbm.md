# Frazil-ice slurry rheology surface for the D2Q9 lattice-Boltzmann solver

**Campaign node:** IN11-H2 (high-fidelity package H2, frazil rheology maturity) ·
**Repository:** TensorLBM · **Module:** `tensorlbm.frazil_rheology`, `tensorlbm.frazil_lbm` ·
**Artifact:** `artifacts/frazil_rheology.json` · **Figures:** `figures/frazil/` ·
**Generator:** `benchmarks/frazil_rheology_bench.py` · **Tests:** `tests/test_frazil_rheology.py`

---

## 1. Case introduction (physics, geometry, parameters)

### 1.1 Why frazil rheology

Frazil ice nucleates in turbulent supercooled water and grows as millimetre-scale
disc crystals. In the water column it forms a *slurry* whose viscosity is
several orders of magnitude above that of clear water. That viscosity is the
physical root of two downstream consumers:

* **C2 thrust-loss model (W9-1215):** the ice-induced thrust/torque loss scales
  with the local effective viscosity in the boundary layer and the momentum
  defect of the slurry wake.
* **A2 viscosity window:** the admissible operating window (ice fraction and
  supercooling) in which the propulsor keeps a usable thrust margin.

Until now the platform had no frazil closure at all: no concentration → viscosity
relation, no shear-thinning rheology for the ice slurry, and no
supercooling → crystallization window. The node therefore builds a *minimal but
verifiable* rheology surface plus the lattice-Boltzmann membrane that carries it.

### 1.2 Physics content

**(a) Effective-medium viscosity (Krieger–Dougherty, 1959).**
For a suspension of rigid, non-interacting spheres the dilute limit is
Einstein's `μ_r = 1 + [η]φ` with `[η] = 2.5`. Frazil flocs interact strongly, and
the standard closure in the sea-ice literature (Osterkamp & Gosink 1983;
Daly 1984) is

```
μ_r(φ) = (1 − φ/φ_max)^(−[η]·φ_max),    φ_max = 0.60,  [η] = 2.5
```

which is used **unfitted**, with the packed/floc limit `φ_max = 0.60` from the
frazil-slurry literature. Two consequences are built into the API:

* `μ_r → ∞` as `φ → φ_max`: the **viscosity jump window**. Across the last 0.05
  of volume fraction (0.55 → 0.60) the relative viscosity rises from 41.6 to
  1.3×10³, a factor **31.6** (evaluated to `φ_max − 0.005`; see
  `jump_window.mu_ratio`). This is the numerically hostile part of the closure
  and the reason the LBM membrane is verified against it explicitly.
* **No extrapolation past the packed limit.** The suspension is no longer
  flowing there, so the closure has no meaning. Requests at `φ ≥ 0.60` raise
  `FrazilRheologyError` (`truncate=False`) or are dropped and counted
  (`truncate=True`, the default of the curve builder). This is the
  "如实截断" requirement: the artifact never contains a viscosity above the
  packed limit.

**(b) Shear thinning (Ostwald–de Waele / power law).**
Frazil slurries are strongly shear-thinning because the flocs break down under
shear (Daly 1984; Barnes 1997):

```
μ_app(φ, γ̇) = K(φ)·γ̇^(n(φ)−1),     K(φ) = μ_w·μ_r(φ)
n(φ) = 1 − (1 − n_packed)·(φ/φ_max)^1.5,    n_packed = 0.45
```

so `K(φ)` is exactly the Krieger–Dougherty viscosity at `γ̇ = 1`, and `n` falls
from 1 (clear water) to 0.45 at the packed limit; the module clips `n` into
`[n_packed, 1]`. Only the thinning branch is physically admissible for frazil,
so `n ≤ 0` (mathematically singular: `ν ∝ γ̇^(n−1)` diverges) and `n > 1`
(shear thickening, which would bend the resistance model the wrong way) are
**rejected** (`validate_flow_index`, `FrazilRheologyError`).

**(c) Supercooling → crystallization growth-rate window.**
Growth is dormant below a nucleation threshold, then limited by two
resistances in series — interfacial attachment and heat diffusion:

```
v_kin  = A·(ΔT − ΔT_nuc)^b,     A = 2.5e−4 m s⁻¹ K^−1.5 , b = 1.5
v_diff = k_w·ΔT / (ρ_i L δ),    k_w = 0.561 W m⁻¹ K⁻¹, ρ_i = 917 kg m⁻³,
                                L = 3.3355e5 J kg⁻¹, δ = 1e−4 m
v      = (1/v_kin + 1/v_diff)⁻¹   for ΔT > ΔT_nuc = 0.02 K,  else 0
```

with the kinetic/diffusive crossover at `ΔT = 0.0408 K`: below it the
attachment kinetics is rate-limiting, above it the curve is linear
(diffusion-limited). The frazil (equiaxed) regime is flagged valid up to
`ΔT = 1 K`; beyond that columnar/dendritic ice forms and the closure is out of
range (flagged, not silently used).

### 1.3 Numerical setting

| Quantity | Value |
|---|---|
| Lattice | D2Q9, BGK, `float64` |
| Channel | height `H` cells, walls on rows 0 and `H+1`, half-way bounce-back at `y = 0.5` and `y = H+0.5`, periodic in `x` |
| Homogeneous direction | `nx = 8` (`4` for the `H = 64` grid-study tail) — the flow is x-invariant, so the profile is x-averaged |
| Peak velocity scale | `u_ref = 0.03` (≈ `Ma = 0.05`) |
| Reference wall viscosity | `ν_ref = 0.2` (pure power-law runs), `Ma` and `Re` held fixed across the grid study |
| Rheology closure | implicit relaxation time from the **discrete non-equilibrium stress** (see §2.2) |
| Steady state | drift of the centreline (peak-to-peak over the last 20 samples / mean) `< 1e-5`, then a 200-step time average |
| Forcing | Guo (2002) velocity-shift body force, momentum source `ρ·a` per step independent of `τ` |

Exact solutions used as references (`H`, half-height `Hh = H/2`, cell centres at
integer wall offsets, `y_c = (H+1)/2`):

```
Poiseuille:  u(y) = n/(n+1)·(a/K)^(1/n)·[ Hh^((n+1)/n) − |y−y_c|^((n+1)/n) ]
Couette:     u(y) = U_w·(y − 1/2)/H,        τ_xy = K·(U_w/H)^n
```

---

## 2. How it was done

### 2.1 Tool chain

```
tensorlbm/frazil_rheology.py     analytic closure (KD, n(φ), growth window, guards)
tensorlbm/frazil_lbm.py          D2Q9 channel membrane + exact-solution residuals
benchmarks/frazil_rheology_bench.py  case sharding, artifact + figure generation
tests/test_frazil_rheology.py    14 tests (5 negative)
```

The existing `tensorlbm.powerlaw` module (variable-viscosity BGK + Guo forcing)
is reused unchanged as the collision operator; `tensorlbm.frazil_lbm` adds the
moving-wall bounce-back, the steady-state driver and the closure. Nothing in the
existing modules was modified, so the pre-existing power-law test suite remains
valid.

### 2.2 Two closures, and why the consistent one was chosen

The pre-existing `tensorlbm.powerlaw` evaluates the shear rate by **central
differences** of the velocity field and feeds it into `ν = K γ̇^(n−1)`
("finite-difference closure"). That is cheap but not self-consistent: the
viscosity is a function of a stencil derivative while the lattice transports a
different (non-equilibrium) stress, and the mismatch acts as a spurious
`dν/dy · ∂u/∂y` transport term. Measured on the same case (`H = 32`, `n = 0.7`,
20 000 steps): **6.96 % L2 error**, and the solution keeps drifting instead of
reaching a fixed point.

The new module uses the **consistent closure**: for BGK the non-equilibrium
stress *is* the discrete strain rate,

```
Π_αβ^neq = −2 c_s² τ ρ S_αβ    ⇒   γ̇ = √(2 S:S) = 3‖Π^neq‖ / (√2 ρ τ)
```

so the power-law closure becomes one scalar equation per cell,

```
F(ν) = ν − K·( B/(1/2 + 3ν) )^(n−1) = 0,      B = 3‖Π^neq‖/(√2 ρ)
```

`F` is strictly increasing for `n ≤ 1`, hence the root is unique and Newton's
method converges in 6–12 iterations (implemented in
`powerlaw_relaxation_time_neq`, `newton_iters = 12`). Same case, same budget:
**0.24 % L2 error** — a factor **29** improvement, and a genuine fixed point.

### 2.3 Verification plan

1. **Grid study** — `n = 0.6 / 0.7 / 0.8` × `H = 24 / 32 / 48 / 64`.
2. **Production residuals** — finest converged run per index (`H = 48`).
3. **Couette** — `n = 0.6 / 0.7` × `H = 16 / 32 / 48`, plus the power-law wall
   stress `τ_xy = K(U_w/H)^n` measured from the near-wall gradient.
4. **Closure hand-off** — the same channel driven by `K(φ)`, `n(φ)` for
   `φ = 0.10 … 0.55`, i.e. across 2.5 decades of wall viscosity.
5. **Reproducibility** — the same case run twice under `torch.manual_seed(1234)`
   must give bit-identical profiles (SHA-256 of the profile buffer).
6. **Guards** — negative tests for `n ≤ 0`, `n > 1`, `φ ≥ φ_max`, `φ < 0`.

All runs are sharded per case (`--case` / `--ids`), each writing
`runs/frazil/<case>.json`, and `--assemble` merges them into the artifact, so no
number in this report is hand-copied.

---

## 3. Results

### 3.1 Concentration–viscosity curve and the viscosity jump window

| φ | μ_r | n(φ) |
|---|---|---|
| 0.00 | 1.00 | 1.000 |
| 0.10 | 1.31 | 0.963 |
| 0.20 | 1.84 | 0.894 |
| 0.30 | 2.83 | 0.806 |
| 0.40 | 5.20 | 0.701 |
| 0.45 | 8.00 | 0.643 |
| 0.50 | 14.70 | 0.582 |
| 0.55 | 41.57 | 0.517 |
| ≥ 0.60 | **refused** (packed limit) | — |

Jump window (`φ ∈ [0.55, 0.60]`): `μ_r` 41.6 → 1.31×10³ (**×31.6**),
`d ln μ_r/dφ = 60` at the window centre.
Truncation probe: request `[0.58, 0.60, 0.65, 0.75]` → 3 dropped, curve stops at
0.58; a strict call at `φ = 0.60` raises
`FrazilRheologyError("… >= packed limit 0.6: extrapolation refused …")`.

![concentration-viscosity](../figures/frazil/fig1_concentration_viscosity.png)

### 3.2 Growth-rate window

| ΔT (K) | 0.00 | 0.02 | 0.05 | 0.10 | 0.20 | 0.50 |
|---|---|---|---|---|---|---|
| v (µm/s) | 0 | 0 | 0.54 | 1.39 | 3.08 | 8.26 |

Kinetic/diffusive crossover at `ΔT = 0.0408 K`; `in_regime` is `False` above
`ΔT = 1 K`.

![growth window](../figures/frazil/fig4_growth_window.png)

### 3.3 LBM verification against the exact solutions

**Poiseuille, production resolution `H = 48`** (all converged, drift ≈ 1e-5):

| n | L2 relative error | max pointwise error (% of peak) | steps |
|---|---|---|---|
| 0.60 | **0.339 %** | 0.627 % | 18 900 |
| 0.70 | **0.257 %** | 0.269 % | 17 800 |
| 0.80 | **0.226 %** | 0.230 % | 16 800 |

**Couette** (production = finest converged run):

| n | H | L2 relative error | wall-stress error `τ_xy = K(U/H)^n` |
|---|---|---|---|
| 0.60 | 32 | **0.0076 %** | 0.0045 % |
| 0.70 | 32 | **0.0066 %** | 0.0045 % |

**Grid study** (L2, %; C = drift-converged):

| H | n=0.60 | n=0.70 | n=0.80 |
|---|---|---|---|
| 24 | 0.995 | 0.365 | 0.153 |
| 32 | 0.551 | 0.236 | 0.176 (C) |
| 48 | **0.339 (C)** | **0.257 (C)** | **0.226 (C)** |
| 64 | 0.614\* | 0.487\* | 0.263\* |

\* `H = 64` ran on a reduced step budget: its residual is still transient-limited
(drift 2e-4…3e-3), which is why it is excluded from any order estimate. At
`H ≥ 32` the error has already reached a ≈0.2 % floor (transient +
compressibility), so no defensible convergence order is quoted — the honest
statement is "every production resolution sits inside the 2 % band with a
≈0.2 % floor".

![profiles](../figures/frazil/fig2_poiseuille_profiles.png)

![grid](../figures/frazil/fig3_resolution_study.png)

### 3.4 Closure hand-off: slurry channel across the viscosity window

| φ | μ_r | n(φ) | wall ν (lattice) | lattice rescale λ | L2 error |
|---|---|---|---|---|---|
| 0.10 | 1.31 | 0.963 | 0.032 | 1.000 | 0.406 % |
| 0.20 | 1.84 | 0.894 | 0.066 | 1.000 | 0.263 % |
| 0.30 | 2.83 | 0.806 | 0.164 | 1.000 | 0.246 % |
| 0.40 | 5.20 | 0.701 | 0.200 | 0.383 | 0.235 % |
| 0.50 | 14.70 | 0.582 | 0.200 | 0.0747 | 0.660 % |
| 0.55 | 41.57 | 0.517 | 0.200 | 0.0195 | 1.230 % |

For `φ ≤ 0.30` the physical Krieger–Dougherty viscosity enters the lattice
*unscaled* (wall viscosity 0.032 → 0.164, a 5× sweep). For `φ ≥ 0.40` the
physical `μ_r` would push the relaxation time to `τ > 3`, where plain BGK is
over-diffusive; those runs use the exact invariance of the power-law channel
`(K, a) → (λK, λa)` (the velocity field is invariant, only the viscosity field
scales), with `λ` recorded per case. The residual is always measured against the
exact solution of the *same* lattice rheology, so what is verified is the
solver, and the closure enters through `n(φ)` and the reported `λ(φ)`.

![frazil channel](../figures/frazil/fig5_frazil_channel.png)

### 3.5 Reproducibility and guards

* Same-seed repeat (`φ = 0.30`, `H = 24`, 4 000 steps): profile SHA-256
  `2f65423c…d0bd8f2` identical for both runs → **bit-for-bit reproducible**.
* Guards exercised by tests: `n = 0, −0.5, 1.2, 1.4` rejected; `φ = 0.60, 0.65,
  0.9, −0.01` rejected; curve builder drops `φ ≥ 0.60` and counts it; strict
  mode raises.

---

## 4. Error judgement (explicit)

| Key quantity | Reference | Error | < 3 % ? |
|---|---|---|---|
| Poiseuille `u(y)`, n = 0.60, H = 48 | exact similarity solution | 0.339 % L2 | **yes** |
| Poiseuille `u(y)`, n = 0.70, H = 48 | exact | 0.257 % | **yes** |
| Poiseuille `u(y)`, n = 0.80, H = 48 | exact | 0.226 % | **yes** |
| Couette `u(y)`, n = 0.60/0.70, H = 32 | exact linear profile | 0.0076 % / 0.0066 % | **yes** |
| Couette wall stress `τ_xy` | `K(U/H)^n` | 0.0045 % | **yes** |
| Slurry channel, φ = 0.10…0.55 | exact power-law solution of the same lattice rheology | 0.235 – 1.230 % | **yes** (worst case 1.23 % < 3 %) |
| `μ_r(φ)` curve | literature *ranges* only (Krieger–Dougherty form + `φ_max ≈ 0.6`, `n ≈ 0.5–0.9`) | not pointwise comparable | **n/a — declared** (no digitised reference in the platform; see §5) |
| Growth window `v(ΔT)` | literature range anchors (`b ≈ 1.5`, `ΔT_nuc ≈ 0.01–0.05 K`) | not pointwise comparable | **n/a — declared** |

The 2 % node criterion is met on every channel case (worst: 1.23 %, and that is
the φ = 0.55 case that deliberately sits in the divergent part of the
Krieger–Dougherty curve). Where a closed-form reference exists the error is
quoted to four significant digits; where only literature ranges exist, this
report says so instead of inventing a percentage.

---

## 5. Notes, pitfalls, limitations

**Pitfalls found (worth keeping).**

1. *Wall location.* The pre-existing power-law channel test placed the first
   fluid cell centre on the wall plane (`y − 0.5`) instead of at one cell offset.
   With that convention even the Newtonian case shows a 4.9 % L2 error and the
   test tolerance had to be 5 %. Correcting the cell-centre convention (centres
   at `y = 1…H`, walls at `0.5` and `H+0.5`) drops the Newtonian error to
   **0.22 %** at `H = 32` — i.e. most of the "power-law error" originally
   attributed to the scheme was a bookkeeping offset.
2. *Closure consistency.* The finite-difference closure is not only inaccurate
   (6.96 % vs 0.24 %) but also fails to settle: the viscosity depends on a
   stencil derivative while the lattice transports a different stress, and the
   resulting feedback drifts. Under-relaxing the viscosity field (`θ = 0.3`) did
   **not** fix it. The non-equilibrium closure does.
3. *Moving wall sign.* The half-way bounce-back with a moving wall needs
   `+6 w_i (c_i·u_w)` once the opposite-direction gather is used; the wrong sign
   gives a mirror-image profile that still looks plausible (error 200 %), so it
   must be checked against the exact solution, not by eye.
4. *Convergence is not free.* The transient scales as `H²/ν` and the shear
   thinning feedback lengthens it; `H = 64` needs ~3× the steps of `H = 32`.
   Grid-study entries therefore carry their drift and a `converged` flag, and
   unconverged entries are excluded from order estimates.

**Limitations (honest).**

* The membrane is **2-D D2Q9**. There is no 3-D turbulence, no floc breakup
  kinetics and no population balance: the closure is local and
  generalised-Newtonian.
* The Krieger–Dougherty parameters (`φ_max = 0.60`, `[η] = 2.5`) and the index
  law `n(φ)` are literature closures anchored on *ranges and trends*, not fitted
  to a measured slurry. They are the accepted parameters in the sea-ice
  literature, but they are not validated pointwise here, and the artifact says
  so.
* The growth window gives the growth velocity that *drives* `φ`; it does not
  predict `φ` (no nucleation rate), and it is flagged out of range above
  `ΔT = 1 K`.
* For `φ ≥ 0.40` the lattice viscosity is rescaled (documented `λ` per case);
  the physical viscosity ratio is preserved but the lattice `τ` is not the
  physical one, so no statement is made about lattice `Re` at those
  concentrations.
* **Anchors are self-built.** `benchmarks/ice_nav_refs` (the IN10-L2 product) is
  **not present** in this worktree, so the citation anchors in
  `artifacts/frazil_rheology.json → anchors` were assembled from the primary
  literature (Krieger & Dougherty 1959; Osterkamp & Gosink 1983; Daly 1984;
  Michel 1978; Barnes 1997) and are declared as `source: self-built`. When
  IN10-L2 lands, its anchors should replace them.

---

## 6. Reproduction

```bash
# analytic curves + fast tests
python3 -m pytest tests/test_frazil_rheology.py -q

# regenerate the artifact (26 shards; each case is a single independent run)
python3 benchmarks/frazil_rheology_bench.py --plan
python3 benchmarks/frazil_rheology_bench.py --jobs 6
python3 benchmarks/frazil_rheology_bench.py --assemble
```

Artifact schema `frazil_rheology/v1`; every number in this document is read back
from `artifacts/frazil_rheology.json` (see `acceptance`, `lbm_validation`,
`rheology`, `growth_window`, `jump_window`, `anchors`).