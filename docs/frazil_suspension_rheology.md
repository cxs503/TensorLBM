# Frazil / grease-ice suspension rheology — hindered settling, drag transition and the concentration-coupled power law

**Campaign node:** IN54W1-N03 · **Repository:** TensorLBM · **Branch:** `agent/m7094`
**Modules:** `tensorlbm.frazil_suspension` (closure), `tensorlbm.frazil_settling_lbm` (lattice verification)
**Artifact:** `artifacts/frazil_suspension/summary.json` · **Figures:** `figures/frazil_suspension/`
**Generator:** `benchmarks/frazil_suspension_bench.py` · **Tests:** `tests/test_frazil_suspension.py` (28 tests, 6 negative)

---

## 1. Physical question and scope

Frazil / grease ice is a *suspension* of disc- and spicule-shaped ice crystals in
supercooled water. In a ship-resistance or ice-load model it is usually treated
as either clear water (dilute) or as a solid (packed), with nothing in between.
That gap is what this node closes: over `c_v = 0 → 0.4` we need three numbers

1. **how the effective viscosity grows** — and how much of that growth is
   cancelled by shear thinning at the strain rates the flow actually sees,
2. **how fast crystals settle out** — the hindered settling velocity `v_s(c)`,
   which controls how quickly a disturbed grease-ice layer re-stratifies,
3. **what drag coefficient a coupled solver must use** so that (2) is
   reproduced consistently.

The rheological pair `(K(φ), n(φ))` already existed in
`tensorlbm.frazil_rheology` (Krieger-Dougherty + power-law, validated in the
LBM channel by `tensorlbm.frazil_lbm`). What was missing — and is added here — is
the transport half (settling, diffusivity) and the drag closure that ties it
back to the rheology. **No previously verified case is reused**: the three new
verification targets are a *closed settling column* (Kynch front), a
*variable-viscosity power-law channel* (wall-concentrated frazil layer) and the
scalar advection-diffusion operator, all of which are new to the repository.

## 2. Model

### 2.1 Concentration → power-law rheology (existing, reused)

```
μ_r(c) = (1 - c/c_max)^(-[η]c_max),      c_max = 0.60, [η] = 2.5
n(c)   = 1 - (1 - n_packed)(c/c_max)^1.5,  n_packed = 0.45
K(c)   = μ_w μ_r(c)          (apparent viscosity at γ̇ = 1)
μ_eff  = K(c) γ̇^(n(c)-1)
```

### 2.2 Hindered settling (new)

```
v_s(c) = v_s0 (1 - c/c_max)^m,   m = m(Re_p)  (Richardson–Zaki 1954, piecewise)
v_s0   = sqrt( 4 Δρ g d / (3 ρ_f C_D(Re) ) ),   C_D = Schiller–Naumann
```

### 2.3 Drag closure consistent with (2.2) (new)

The drag a coupled solver must use follows from the requirement that the drag
balance reproduce the hindered velocity with the slip velocity the transport
solver uses:

```
Ω(c) ≡ C_D(c)/C_D(Re) = (1 - c/c_max)^(-2m),     C_D(c) = C_D(Re)·Ω(c)
```

This round trip (`C_D(c) → v_s(c)`) is verified numerically to machine precision
(relative error `1.6e-16`), so the settling law and the drag coefficient cannot
silently drift apart.

### 2.4 Transport equation solved by the LBM

```
∂c/∂t + ∇·(c u_eff) = ∇·(D_eff ∇c),      u_eff = u - v_s(c) ẑ
D_eff = D_m + α d² γ̇ c²(1 + ½ e^{8.8c})       (Leighton–Acrivos shear-induced)
```

The settling enters *inside* the flux divergence, so the D2Q5 solver transports
the conservative Kynch equation; the column walls are no-flux half-way
bounce-back, giving exact mass conservation.

### 2.5 Parameters

| parameter | value | note |
|---|---|---|
| floc diameter `d` | 1.0 mm | representative of the 0.1–3 mm Daly (1984) range |
| `ρ_ice`, `ρ_water` | 917 / 1000 kg m⁻³ | Δρ = 83 kg m⁻³ |
| `μ_water` (0 °C) | 1.8e-3 Pa s | |
| `c_max` | 0.60 | frazil floc packing limit |
| `n_packed` | 0.45 | power-law index at the packed limit |
| single-particle settling | `v_s0 = 15.23 mm/s`, `Re_p = 8.46`, `C_D = 4.68` | 1.65× the Stokes velocity → non-Stokes drag matters |
| RZ exponent | `m = 3.554` | from `Re_p = 8.46`, branch `1 ≤ Re < 500`: `4.4 Re^-0.1` |

## 3. Answer 1 — effective viscosity for `c_v = 0 → 0.4`

`μ_eff` in mPa·s at four shear rates (programmatic, `closure_sweep` of the artifact):

| c | μ_r (KD) | n(c) | γ̇=0.1 s⁻¹ | γ̇=1 | γ̇=10 | γ̇=100 | μ_eff/μ_w at γ̇=1 |
|---|---|---|---|---|---|---|---|
| 0.00 | 1.000 | 1.000 | 1.80 | 1.80 | 1.80 | 1.80 | 1.00 |
| 0.05 | 1.139 | 0.987 | 2.11 | 2.05 | 1.99 | 1.93 | 1.14 |
| 0.10 | 1.315 | 0.963 | 2.58 | 2.37 | 2.17 | 1.99 | 1.31 |
| 0.20 | 1.837 | 0.894 | 4.22 | 3.31 | 2.59 | 2.03 | 1.84 |
| 0.30 | 2.828 | 0.806 | 7.97 | 5.09 | 3.25 | 2.08 | 2.83 |
| 0.40 | **5.196** | **0.701** | **18.64** | **9.35** | **4.69** | **2.36** | **5.20** |

Three observations that matter for the ship-performance side:

* the KD thickening at `c = 0.4` is **5.2×**, inside the `n ≈ 0.5–0.8` target band
  (`n = 0.701`);
* that factor is only realised at `γ̇ = 1 s⁻¹`; at 100 s⁻¹ the *apparent* ratio
  collapses to **1.31×** — a grease-ice layer in a fast boundary layer looks
  almost Newtonian even at 40 % ice;
* conversely at 0.1 s⁻¹ (a slowly rotating propeller wake, or a stationary
  channel) the same slurry is **10.4×** thicker than water.

Flow capacity (uniform slurry channel at fixed driving force, exact power-law
result, LBM-confirmed to 0.05 %):

| c | Q(c)/Q(0) | equivalent flow loss |
|---|---|---|
| 0.10 | 0.609 | 1.6× |
| 0.20 | 0.266 | 3.8× |
| 0.30 | 0.0741 | 13.5× |
| 0.40 | 0.00936 | **107×** |

The 107× is dominated by the power law being evaluated at the channel's own low
shear rate (where `γ̇^(n-1)` is large), not by the 5.2× KD factor alone — this is
exactly the effect that a Newtonian effective-viscosity surrogate cannot capture.

## 4. Answer 2 — settling velocity decay

| c | v_s (mm/s) | v_s/v_s0 |
|---|---|---|
| 0.00 | 15.23 | 1.000 |
| 0.05 | 11.18 | 0.734 |
| 0.10 | 7.97 | 0.523 |
| 0.20 | 3.60 | 0.237 |
| 0.30 | 1.30 | 0.085 |
| 0.40 | 0.31 | **0.020** |

Over `c = 0 → 0.4` the settling velocity falls by a factor **49.6**; the last
0.1 of concentration (0.3 → 0.4) removes another 4.2×. Physically: a 40 %
grease-ice layer is effectively non-settling on the timescale of a ship
interaction (0.31 mm/s → 1 m of column in ~54 min), while a 5 % layer
re-stratifies in a minute.

## 5. Answer 3 — drag coefficient transition

Two transitions are reported.

**(a) Reynolds transition (dilute limit).** `C_D(Re)` moves from the Stokes line
`24/Re` (verified to `2.7e-4` relative at `Re = 1e-4`) to the Newton plate 0.44
(`Re > 800`). For 1 mm frazil at `Re_p = 8.46` the Schiller–Naumann drag is
`4.68`, i.e. 1.65× Stokes: **non-Stokes drag is already active for millimetre
flocs**, and the Stokes settling velocity (25.1 mm/s) over-predicts the true
15.2 mm/s by 65 %.

**(b) Concentration transition.** The RZ-consistent amplification `Ω(c)`:

| c | Ω(c) = C_D(c)/C_D(Re) | Di Felice (1994) `(1-c)^-χ` |
|---|---|---|
| 0.05 | 1.9 | 1.18 |
| 0.10 | 3.7 | 1.39 |
| 0.20 | 17.9 | 2.02 |
| 0.30 | 138 | 3.07 |
| 0.40 | **2463** | **4.99** |

**Honest statement of model spread:** the two closures disagree by two orders of
magnitude at `c = 0.4`. The reason is structural, not numerical: Di Felice uses
`ε = 1 − c` (packing at 1) and a fixed exponent `χ ≈ 3.7`, whereas Richardson–Zaki
with the frazil packing limit `c_max = 0.6` diverges at 0.6 and carries `m = 3.55`.
We use RZ (validated for batch settling of flocs) in the transport solver and
report Di Felice only as a comparison. **No `<3 %` claim is made for this
quantity** — there is no experimental anchor in hand, so the number is declared a
closure choice with a stated spread.

## 6. Numerical verification (lattice)

| case | scheme | reference | error | criterion |
|---|---|---|---|---|
| uniform power-law channel, `c = 0 / 0.2 / 0.4` | D2Q9 variable-τ BGK + Guo force, neq-stress closure | exact power-law Poiseuille | L2 **0.43 % / 0.39 % / 0.46 %** | < 3 % ✅ |
| stratified channel, `c_wall = 0.2 / 0.4` | same, per-cell `K(y)`, `n(y)` | quadrature of `d/dy[K(du/dy)ⁿ] = −a` | L2 **0.54 % / 0.38 %** | < 3 % ✅ |
| scalar advection-diffusion | D2Q5 BGK | exact damped travelling wave | L2 **0.030 %**, `D_num` err 1.39 %, advection-speed err 4e-6 | < 3 % / < 5 % ✅ |
| settling column, `c0 = 0.05…0.4` | D2Q5 + RZ drift, no-flux walls | Kynch shock speed `v_s(c0)` | **0.23 / 1.51 / 1.84 / 0.96 / 0.11 %** | < 3 % ✅ |
| column mass conservation | — | initial mass | drift ≤ **1.3e-15** | < 1e-10 ✅ |
| flow-capacity ratio | from the same channel runs | analytic `n/(2n+1)`-corrected power law | **0.05 %** | < 3 % ✅ |
| quadrature reference self-check | — | closed-form Poiseuille | **6e-8** | — |

The quadrature reference is itself checked against the closed-form power-law
solution (1e-8 for `n = 0.5/0.7/1.0`) before being used as the stratified
reference, and it returns `u(H+½) = 2e-15` (the symmetry condition) — so the
variable-viscosity reference is not an unverified second code.

Numerical notes:

* the viscosity range is handled by the wall-viscosity rescaling
  `(K, a) → (λK, λa)`, exactly invariant for a uniform `K` and hence
  verification-neutral; the stratified case uses one `λ` for the whole profile;
* `τ_max ≤ 1.04` in every production run, so BGK compressibility errors stay
  small; the shear-thinning singularity at the channel centre is the binding
  constraint on `τ`, not the wall;
* all runs are deterministic (no RNG); profile SHA-256 digests are stored in the
  artifact as a reproducibility anchor.

## 7. Pitfalls found while building this

1. **D2Q5 scalar LBM plus a sharp front is dispersive.** Behind the Kynch
   interface the linear D2Q5 equilibrium leaves `O(u²)` oscillations
   (`-0.7 %` of `c0` at `u = 0.05`, `D = 1e-3`). Adding the second-order
   equilibrium term changes nothing for an axis-aligned D2Q5 lattice (the cubic
   moment cancels identically) — the fix is to keep `D` moderate and measure the
   interface at the 50 % crossing, where the dispersion is symmetric.
2. **A periodic column cannot measure settling.** With `torch.roll` boundaries the
   sediment leaving the bottom re-enters at the top and the "interface" is a
   wraparound artefact. The solver now uses half-way bounce-back walls, which
   also makes mass conservation exact (1e-15) instead of approximate.
3. **Sign of the body-force momentum balance.** `dτ_xy/dy = −a ⇒ τ = −a(y−y_c)`;
   getting the sign wrong still produces a plausible-looking symmetric profile
   (it is exactly the negative solution, L2 error = 2.0). The uniform limit check
   against the closed form catches it in one line.
4. **The flow-capacity ratio is not `(K_w/K)^(1/n)`.** The prefactor
   `n/(n+1)·(n+1)/(2n+1)` and the height exponent `Hh^(1/n−1)` carry the `n`
   dependence; dropping them (the first version of `flow_capacity_ratio`) gave
   0.095 instead of 0.0094 at `c = 0.4`, i.e. a 10× error that the LBM
   immediately exposed as a 90 % "disagreement".
5. **Richardson–Zaki's piecewise exponent is discontinuous** (≈5 % jump at
   `Re = 1`, ≈40 % at `Re = 500`). It is kept as published and documented;
   smoothing it would hide a real modelling limitation.

## 8. Limitations / scope

* Monodisperse spheres: real flocs are porous aggregates with a smaller effective
  density deficit, so the absolute `v_s0` is an upper bound; the *shape* of the
  decay is less sensitive.
* One-way coupling only: the closure feeds the transport and the channel; a fully
  two-way run (concentration → local viscosity → velocity → concentration in a
  convecting flow) is represented only by the stratified-channel slice.
* `D_eff` includes molecular + shear-induced diffusion; the settling-induced
  dispersion is reported separately and deliberately *not* added (double
  counting with the resolved settling flux).
* No experimental anchor for the suspension drag in the frazil regime was
  available, so the drag transition is a closure choice (see §5), not a
  validated prediction.
* The scalar solver's `D_eff` is constant per run; the concentration dependence
  of `D_eff` is available from the closure but not yet used inside the LBM.

## 9. Reproduction

```bash
PYTHONPATH=src python3 benchmarks/frazil_suspension_bench.py      # ~57 s, writes artifact + 6 figures
PYTHONPATH=src python3 -m pytest tests/test_frazil_suspension.py   # 28 tests, ~11 s
```

## Figures

![effective viscosity](../figures/frazil_suspension/fig1_mu_eff_shear_thinning.png)

![hindered settling](../figures/frazil_suspension/fig2_hindered_settling.png)

![drag transition](../figures/frazil_suspension/fig3_drag_transition.png)

![settling column](../figures/frazil_suspension/fig4_settling_column.png)

![channel profiles](../figures/frazil_suspension/fig5_channel_profiles.png)

![flow capacity](../figures/frazil_suspension/fig6_flow_capacity.png)

## References

* Richardson & Zaki (1954) *Trans. Inst. Chem. Eng.* 32, 35.
* Schiller & Naumann (1935) *Z. Ver. Deutsch. Ing.* 77, 318.
* Krieger & Dougherty (1959) *Trans. Soc. Rheol.* 3, 137.
* Leighton & Acrivos (1987) *J. Fluid Mech.* 181, 415.
* Di Felice (1994) *Int. J. Multiphase Flow* 20, 153.
* Daly (1984) CRREL Report 84-11 (frazil floc sizes).
* Kynch (1952) *Trans. Faraday Soc.* 48, 166 (settling front kinematics).