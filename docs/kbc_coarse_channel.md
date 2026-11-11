# Three-segment KBC collision for coarse-grid channel turbulence

Campaign node `p495w4-n03-lbm-kbc-coarse-wall-turbulence` · repo `TensorLBM` ·
operator `src/tensorlbm/kbc_collision.py` (NumPy reference) +
`src/tensorlbm/kbc_channel_kernel.c` (OpenMP production kernel) ·
production driver `tools/kbc_channel_report.py` ·
gates `src/tensorlbm/kbc_channel.py` ·
report builder `tools/kbc_build_report.py` ·
ghost-rate calibration `tools/kbc_ghost_rate_calibration.py`.

## 1. Why this operator

The previous campaign node (`m9102`) implemented ELBM for the same problem and
returned an honest FAIL: at every affordable resolution (Δ+ = 5.6 … 12.3) the
H-theorem limiter engaged with field-mean α = 0.19 … 0.31, the extra
dissipation thickened the wall layer, the channel relaminarised and C_f landed
47–63 % off MKM-1999 DNS. The diagnosis was structural: **the entropy limiter
is not a sub-grid model** — it removes the non-equilibrium content it cannot
resolve, and the shear stress goes with it.

This node implements the fix the m9102 report pointed at: the
Karlin–Bösch–Chikatamarla (2014) decomposition, in which the non-equilibrium
part is split by Hermite projection into three orthogonal segments and **only**
the highest one is touched by the entropy condition:

```
f^neq = s + b + g                                    (exact identity)
f*    = f^eq + (1 - ω_ν) s + (1 - ω_B) b + (1 - α ω_g/τ) g
```

| segment | modes (D3Q27) | relaxation | consequence |
|---|---|---|---|
| K (shear) `s` | traceless 2nd order, 5 | `ω_ν = 1/τ` | `ν = cs²(τ-1/2)` exactly; shear stress can never be over-damped |
| B (bulk) `b` | trace of 2nd order, 1 | `ω_B = 1/τ_B` | bulk viscosity is a free knob with no shear leakage |
| C (ghost) `g` | 3rd order and above, 17 | `α ω_g/τ` | the only entropy-conditioned segment |

The mission text quotes 2 conserved + 4 shear / 1 bulk / 1 ghost: those are the
**D2Q9** counts. The same split on D3Q27 is 4 conserved + 5 shear / 1 bulk / 17
ghost (`f` carries 27 moments: 1 density, 3 momentum, 6 second order, 17
third-and-higher); the implementation is written for the general case and is
tested on D3Q27.

The entropy-condition solve for `α` is a monotone Newton/bisection on the
convex function `H(α)`: `f*(α)` is affine in `α`, `x ln x` is convex, so `H` is
convex and `H'(α)`, `H''(α)` are available in closed form (one logarithm per
direction). A per-cell relaxation scale `γ ∈ (0,1]` (linear in the populations)
guarantees positivity and `ΔH ≤ 0`; the number of firings is reported.

## 2. Two design decisions that came out of measurement (not assumption)

### 2.1 The ghost rate must be tied to `1/τ`

`α = 1, ω_g = 1` reproducing BGK exactly is *not* cosmetic: the `-1/2` in
`ν = cs²(τ - 1/2)` is carried by the O(Kn) ghost content of the
Chapman–Enskog distribution. With the ghost rate hard-wired to a `τ`-independent
value (`ω_g = 1` absolute instead of `1/τ`) the laminar channel came out **3.2×
too viscous** — measured effective viscosity `cs²τ` instead of `cs²(τ-1/2)`.
The regression is pinned by `test_alpha_one_equals_bgk` (1e-15) and
`test_laminar_poiseuille_regression` (<2 % on the centreline).

### 2.2 At the campaign's low `τ`, `α > 1` destabilises the ghost modes

`tools/kbc_ghost_rate_calibration.py` drives a Stokes shear layer
(`F_x = F0 cos(ky)`, steady `u = F0/(ν_eff k²)`) at the campaign viscosity:

| τ | α = 1 | α = 1.02 | α = 1.05 |
|---|---|---|---|
| 0.5026 | ν_eff/ν_nominal ≈ 1.0 | collapse | blow-up |
| 0.52 | ≈ 1.0 | collapse | blow-up |
| 0.7325 | ≈ 1.0 | ≈ 1.0 | drift |

Above `α ≈ 1.02` the ghost over-relaxation `1 - α ω_g/τ` changes sign with
magnitude > 1 at `τ ≈ 0.5`, and the entropy-monotone limiter then freezes the
mean flow instead of the ghost modes. Production therefore fixes
`α_min = α_cap = 1`, i.e. **the ghost segment relaxes at the BGK rate** — the
operator is then exactly BGK, and the entropy knob is recorded as a *measured,
rejected* option. This is a negative result: entropy-relaxed ghost modes are
not a usable implicit SGS at Δ+ ≈ 11–18.

## 3. Validation chain (`tests/test_kbc_channel.py`, 28 tests)

1. **Split exactness / orthogonality** — `s+b+g = f^neq`, `s` carries the whole
   traceless second moment, `b` the whole trace, `g` nothing of second order
   (1e-15).
2. **Exact shear viscosity** — `α=1, ω_g=1` equals BGK to 1e-15 for
   τ ∈ {0.5002 … 2.5}; transverse shear-wave decay rate equals `νk²` to <1 %
   for BGK and KBC.
3. **Ghost-rate regression** — a `τ`-decoupled ghost rate is *measurably* a
   different operator (pinned, so the 3.2× defect cannot come back).
4. **H-theorem and positivity** — per-cell `ΔH ≤ 0` and `f* > 0` on distorted
   states for τ ∈ {0.5002 … 2.0}; mass and momentum conserved to 1e-14.
5. **Kernel ≡ reference** — 27 populations and the entropy-solved `α` agree to
   1e-16 for τ ∈ {0.502, 0.6, 1.2, 2.0} at several cells (this test caught a
   wrong direction-product table in the C kernel).
6. **Laminar plane Poiseuille** — centreline velocity within 2 % of
   `(F/2ν)ξ(H-ξ)` on the kernel; the profile must be parabolic, not a wall
   layer (this test caught the ghost-rate/viscosity coupling).
7. **Gate layer** — a synthetic DNS-shaped profile returns `C_f` to <1 %, u+ to
   <10 % and a passing wall-consistency check, so the *comparison machinery* is
   verified independently of the solver.

## 4. Production configuration

* D3Q27, `u_τ = 0.01` (Ma_max < 0.2), body force `F = u_τ²/h`,
  `ν = u_τ h/Re_τ`, `τ = 0.5 + 3ν` (0.5024 at Re_τ = 590, 0.5026 at 180).
* Box `Lx = 1.5πh`, `Lz = 0.75πh` (≥ the MKM box in wall units for every case).
* Half-way bounce-back walls (no-slip planes at `y = 0.5` and `y = ny-1.5`),
  Guo forcing with the half-force velocity shift.
* `α_min = α_cap = 1`, `γ` safeguard on, entropy audit window on the
  `audit_L180_n33_entropy` case.
* Gate thresholds frozen *before* the runs in
  `tensorlbm.kbc_channel.GATE_SPEC` (C_f 3 % at Re_τ 395/590, 5 % at 180; u+
  pointwise 5 % for `y+ ≤ 100`; Reynolds stress 10 %; grid-convergence 5 %).

## 5. Results

See `artifacts/kbc/p495w4-n03/` (per-case `gates.json`, `meta.json`,
`profiles.csv`, `history.csv`) and `figures/kbc_p495w4_n03/report.md`. The
mission report is generated from those artefacts by
`tools/kbc_build_report.py` — no number in it is typed by hand.

## 6. Limits

* No wall model: the first fluid node sits at `y+ ≈ 5.6 … 9.2`, inside the
  buffer layer, and the near-wall cycle is unresolved. Mean-profile error is
  dominated by that, not by the collision operator.
* Statistics: 8–10 eddy turnover times; the Reynolds-stress peak carries a
  visible sampling/box error and is compared as a shape and magnitude.
* The `α` solver is not a globally optimal entropy minimiser when the
  positivity bracket binds (measured slack ≤ 4e-2 in `H` units); the hard
  guarantee (`ΔH ≤ 0`) comes from the `γ` safeguard.
* D3Q27 only in the kernel (the NumPy reference also supports D3Q19).
## 7. Production outcome (measured, honest)

Three grids — `Re_tau = 180` with NY = 33 (Δ+ = 11.6) and NY = 49 (Δ+ = 7.8),
and `Re_tau = 395` with NY = 49 (Δ+ = 16.8) — were run to 24 000 / 30 000 steps
with the production configuration (`α = 1`, i.e. the BGK-equivalent KBC
operator). All three **fail every accuracy gate**:

| case | Δ+ | C_f error | u+ worst (y+≤100) | Re-stress RMS | u_tau (stress) vs force |
|---|---|---|---|---|---|
| L180_n33 | 11.6 | 329 % | 47.2 % | 193 % | 450 % |
| L180_n49 | 7.8 | 315 % | 51.2 % | 43 % | 394 % |
| L395_n49 | 16.8 | 681 % | 77.0 % | 71 % | 352 % |

and the failure mode is the same on both grids, which is what makes it
diagnostic rather than accidental:

* the centreline `u+` saturates at 10.6 / 11.4 against the DNS 18.3 — the
  resolved turbulence is **not sustained** on these grids;
* a population reaches the `1e-30` floor within ~2 000 steps and the
  positivity/entropy safeguard then fires continuously (γ-limited fraction up
  to 1e-3 in steady state, zero sampled cells free);
* the total stress carried by the populations deviates from the momentum
  balance identity `-u_tau^2 (1 - y/h)` by a factor −30.8 / −9.1 at the first
  node, i.e. a spurious (parasitic) mode dominates the near-wall momentum flux.

Reading: a `Δ+ ≈ 8 … 12` channel needs a sub-grid model. The three-segment
operator guarantees that the *shear* segment is never over-damped (the m9102
defect), but it does not supply the missing sub-grid stress — and the one knob
in the operator that could act as an implicit SGS, the entropic ghost rate, is
measured to destroy the solution at these `τ` (§2.2). The honest conclusion is
therefore a **falsified hypothesis**: the KBC three-component operator with an
entropy-relaxed ghost segment is not, by itself, a route to `< 3 %` `C_f` at
`Δ+ ≈ 8 … 18`; the m9102 failure was not only the H-limiter.

Bounded resources: the `Re_tau = 590` case and the `NY = 65` grid were not
completed inside the mission budget (the cluster queue held ~260 jobs for the
whole session and the two nodes were saturated by the rest of the fleet); the
report omits them rather than estimating.

Grid convergence *of the wrong solution*: the two `Re_tau = 180` grids agree in
`C_f` to 3.4 %, i.e. the discretisation error is small — the channel converges
to a solution that is simply not the DNS one. A small grid spread is therefore
reported as evidence about discretisation, not as evidence of accuracy.
