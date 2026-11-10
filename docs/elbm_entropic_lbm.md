# ELBM: entropy-conditioned lattice Boltzmann with an exact H-theorem (D3Q27)

**Node**: `p485w1-n04-lbm-elbm-turbulence` (TensorLBM)
**Scope**: entropic (H-theorem based) collision, D3Q27 closed-form entropic
equilibrium, local Newton solve of the entropy condition, OpenMP channel
kernel, turbulent-channel validation against Moser-Kim-Mansour (1999) DNS.

---

## 1. What is implemented

| Component | Module | Evidence |
|---|---|---|
| Entropic equilibrium (Gauss) closed form, D3Q27 | `tensorlbm.elbm.entropic_equilibrium_gauss_d3q27` | REAL — equals the Lagrange-dual Newton minimiser to 1e-16 |
| Entropic equilibrium (Gauss / Bose-Einstein), any lattice | `tensorlbm.elbm.entropic_equilibrium_dual` | REAL — 4-unknown Newton on `∇H = λ·(1,c)` |
| Discrete entropy functionals `H_G`, `H_BE`, KL excess | `gauss_entropy`, `bose_einstein_entropy`, `relative_entropy` | REAL |
| Entropy condition `α` (local Newton + positivity bracket) | `solve_entropy_condition` | REAL |
| ELBM collision D3Q27 / D3Q19 | `collide_elbm_d3q27`, `collide_elbm_d3q19` | REAL |
| Channel driver (Guo forcing, half-way bounce-back, statistics) | `elbm_channel` | REAL |
| Fused OpenMP kernel (same algorithm) | `elbm_channel_kernel.c`, `elbm_channel_native` | REAL, cross-checked against the torch path |

### 1.1 The equilibrium

The discrete Gauss entropy `H(f) = Σ_i f_i ln(f_i/w_i)` is convex, so its
constrained minimum at fixed density and momentum is unique. Stationarity of
`H + λ(Σf − ρ) + λ⃗·(Σ c f − ρu)` gives the exponential form
`f_i ∝ w_i exp(−a·c_i)`; for D3Q27 — whose weights *are* the tensor product of
three D1Q3 weight sets — the 4×4 dual problem factorises and the equilibrium is
available in closed form. Per axis, with `S = sqrt(1+3u²)`, `r = (S−2u)/(1+u)`
and `P = 6/(4 + r + 1/r)`:

```
g(+1) = P/r,   g(0) = P,   g(−1) = P·r,     f_i^eq = w_i ρ Π_α g(c_iα)
```

This reproduces ρ and ρu **exactly** (machine precision) and is positive by
construction. The test-suite pins it against the independent dual Newton solve
(1e-11) and verifies that it minimises `H` along moment-preserving line
searches. The same dual solver provides the Bose-Einstein equilibrium
(`f_i ∝ w_i/(e^{θ+a·c_i} − 1)`) and the D3Q19 equilibrium, whose weights are not
a tensor product and therefore admit no product closed form.

### 1.2 The collision and the entropy condition

```
f* = f + (α/τ)(f^eq − f),      H(f*) ≤ H(f),      α ∈ (0, 1]
ν(α) = c_s²(τ/α − 1/2),        α = 1 ⇔ BGK exactly
```

`α` is solved **per cell and per step**, in two stages:

1. Evaluate `H` at the admissibility bound `α_hi = min(τ·t_pos, 1)`, where
   `t_pos` is the largest rate that keeps every population positive. If `H` did
   not increase there, `α = α_hi`: this is the ordinary resolved-cell case and
   gives plain BGK (`α = 1`). The stage is what makes the solve immune to
   floating-point noise in `H` — a degenerate state (`f ≡ f^eq`) takes the cap
   instead of a noise-driven Newton step.
2. Otherwise a root exists inside; a safeguarded Newton/bisection iteration on
   `[0, α_hi]` returns `lo`, the last iterate whose entropy was *verified* not
   to increase. The H-theorem therefore holds by construction (up to the
   tolerance `1e-6(|H|+1)` used for the logarithms), not by luck.

### 1.3 Why the cap is `α ≤ 1` and not the classical `α_lit ≤ 2`

Along the line `f → f^eq` the discrete entropy is convex and, to second order,
symmetric: `H(f(t)) − H(f^eq) = (1−t)²·H2`. The entropy root of a
near-equilibrium state therefore sits at the *mirror point* `t = 2`
(rate `2`, i.e. `α = 2τ`) regardless of `τ`. That root is **not** the BGK rate:
the entropy condition alone cannot fix the transport coefficient. The classical
papers write the collision as `f* = f + α_lit β(f^eq − f)` with `β = 1/(2τ)`
and cap `α_lit ≤ 2`, which in the convention used here (where the rate is
`α/τ`) is `α ≤ 1` — the BGK rate. Consequences, all machine-checked:

* resolved cells: `α = 1` ⇒ exact BGK viscosity, H-condition satisfied with slack;
* non-equilibrium cells (`α < 1`): under-relaxation ⇒ extra dissipation ⇒
  positivity and H-monotonicity at **any** `τ`, which is the ELBM stability
  mechanism. No eddy-viscosity model is used anywhere in this node.

Measured on the production runs: `α ∈ [0.94, 1.00]`, mean `≈ 0.999`, i.e. the
limiter is idle in almost the whole field and acts only in the few cells where
BGK would violate the entropy condition. (Had the cap been left at 2, the
laminar validation below would have run at `α ≈ 1.1 > 1`, i.e. at *negative*
effective viscosity — a real defect that this reasoning caught.)

---

## 2. Numerics of the channel driver

* **Forcing**: Guo (2002), with the half-force velocity shift (the equilibrium
  is built at `u + F/(2ρ)`, the source delivers the remaining `(1−1/(2τ))F`).
  Using the source *without* the shift under-drives the flow by `1−1/(2τ)`
  (measured: a factor 10 at `τ = 0.56`, found by the laminar validation).
* **Walls**: half-way bounce-back on the rows `y = 0` and `y = ny−1`, i.e. the
  no-slip planes are at `y = 0.5` and `y = ny − 1.5`, so the channel height is
  `H = ny − 2` (not `ny − 1`: that one-unit error shifted the whole profile by
  5 % before it was fixed and is now covered by the laminar test).
* **Statistics**: averages over the two homogeneous directions, then over time,
  starting after `warmup`; `u_tau` comes from the exact momentum balance
  `u_tau = sqrt(g h)` (independent of the viscosity), and the viscosity is
  cross-checked from the measured viscous-sublayer slope.
* **Kernel**: `elbm_channel_kernel.c`, OpenMP, `-O3 -march=native -ffast-math`,
  cell-major loops (memory order `z→y→x`). The torch reference
  (`elbm_channel.run_elbm_channel`) runs at ~0.1 Mcell/s (the entropy solve is
  `(1+n_iter)·27` logarithms per cell executed as separate bandwidth-bound
  passes); the fused kernel reaches O(10) Mcell/s per node, which is what makes
  a wall-resolved ELBM channel affordable. The two implementations are pinned
  to each other in `tests/test_elbm.py` (1e-4 relative per population).

---

## 3. Validation chain (all reproducible with pytest)

| Check | Result |
|---|---|
| Closed-form equilibrium = dual-Newton minimiser | agreement **5e-16** |
| Equilibrium moments (ρ, ρu) | exact to **1e-16** |
| `H(f^eq) ≤ H(f^eq + λd)`, moment-preserving `d` | holds for all λ tested |
| H-theorem, per cell, τ ∈ {0.5001 … 2.0}, distorted states | `ΔH ≤ 0` to 1e-9 |
| Positivity of the post-collision state | `min f > 0` in every test |
| `α` admissible, `α = 1` for a resolved state | `α ∈ (0,1]`, exact cap |
| Shear-wave decay rate vs `ν k²` | within **5 %** |
| D3Q19 path (dual-Newton equilibrium) | H-theorem + positivity hold |
| Native kernel vs torch reference, one step | **1.6e-5** relative (24 Newton iters) |
| Laminar channel vs exact Poiseuille parabola | **0.3 %** (H = 19, 60k steps) |
| DNS gate machinery self-test | passes the DNS profile, rejects a −20 % profile |

---

## 4. Limits (honest scope)

* The turbulent-channel runs reported in `docs/reports/` are **under-resolved
  by DNS standards** (Δy+ ≈ 5–12 versus Δy+ ≲ 1 for a spectral DNS): they can
  demonstrate stability, positivity and the shape of the wall law, but not
  5 %-accurate friction coefficients. Criterion verdicts are reported as
  computed, including FAIL.
* The entropic *equilibrium* is used together with a **single** relaxation
  factor. The KBC/ghost-mode decomposition (already in
  `tensorlbm.entropic_kbc`) and the two-population variants are a different
  route to the same H-theorem and are not exercised here.
* `α` is solved on the *approximate* entropy surface of the truncated lattice;
  the H-theorem statement is discrete (that is exactly what the tests check),
  not a continuum statement.
* The Bose-Einstein equilibrium is implemented and tested for moment
  consistency and entropy ordering, but the production collision uses Gauss.