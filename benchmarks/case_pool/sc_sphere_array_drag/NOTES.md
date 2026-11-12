# CASE-LBM-CUBIC-ARRAY-DRAG — simple-cubic sphere array, Stokes drag (Darcy resistance)

Case-pool entry (owner 2026-11-12 算例池令). Anchor: `CASE-LBM-CUBIC-ARRAY-DRAG`.

## 1. Target quantity and reference (locked BEFORE any run)

Simple-cubic (SC) periodic array of identical spheres, creeping flow driven by a
uniform body force.  Dimensionless drag / Darcy resistance

    K(phi) = F / (6*pi*mu*a*U_s),   U_s = (1-phi) * <u>_fluid   (superficial / Darcy)

with `F` = drag on one sphere, `a` = sphere radius, `phi` = true solid volume
fraction of the array, `mu = rho*nu`.

### 1.1 Hasimoto (1959) — analytic periodic fundamental solution, JFM 5:317

Hasimoto's semi-analytic SC sedimentation series (as reproduced verbatim in
`benchmarks/verified/permeability3d/NOTES.md` §1.6, itself two-source locked):

    U/U0 = 1 - 1.7601 c^(1/3) + c - 1.5593 c^2 + 3.9799 c^(8/3) - 3.0734 c^(10/3)
    K    = U0/U

Evaluated **in code** (`run.py:hasimoto_K`) from the coefficients above — never
typed as a result value.  Validity: dilute expansion (c << 1); used here as an
independent analytic cross-check at `phi = 0.125` and, flagged OUT-OF-VALIDITY,
as an extrapolation at `phi = 0.5`.

### 1.2 Sangani & Acrivos (1982) / Zick & Homsy (1982) tabulated SC values

The ecosystem's locked SC table (`benchmarks/verified/permeability3d/NOTES.md`
§1.2–1.4, two independent witnesses: Basilisk `src/test/spheres.c` verbatim
transcription + Holmes/Williams/Tilke 2011 literature column) contains

| phi | K |
|---|---|
| 0.027 | 2.0077 |
| 0.064 | 2.8102 |
| **0.125** | **4.292** |
| 0.216 | 7.4423 |
| 0.343 | 15.402 |
| **0.45** | **28.09** |
| 0.5236 | 41.99 (divergent, excluded) |

**Deviation from the instructed point set (registered, reason):** the instructed
second point `phi = 0.5` has **no locked table row** in any source available
offline (rows are at 0.45 and 0.5236; 0.5236 = pi/6 is the touching limit and is
divergent), and it is also the hardest point geometrically (surface gap
`h/a = 0.031` needs `Delta_x <~ h/3`, i.e. `N >~ 250` cells per lattice constant).
Protocol therefore carries **`phi = 0.125` and `phi = 0.45` as verdict-bearing**
(table-exact, anti-shopping: no interpolation of a reference) and reports
`phi = 0.5` as a **diagnostic third point** whose reference is WITHHELD
(no locked row); its expected bracket is the log-log interpolation of the locked
table, `K_interp(0.5) = 34.1`, quoted as a range only, never as a verdict.

## 2. Measurement channels (two independent force channels + one velocity channel)

| channel | definition | role |
|---|---|---|
| `K_mx` | drag from **momentum exchange** on the interface links: `F_x = -sum_{solid x_s} sum_{i: x_s - c_i fluid} 2 c_ix f_i(x_s, post-BB)`; `K_mx = F_x / (6 pi nu a_nom U_s)` | primary (verdict) |
| `K_mb` | momentum balance `F = a_body * (1 - phi_actual) * N^3` (steady state: injected momentum = drag), `K_mb = F / (6 pi nu a_nom U_s)` | consistency (must agree with `K_mx`) |
| `K_qv` | `K = a_body N^3 / (6 pi nu a_nom U_s)` from the mean fluid velocity only | legacy channel of the previous in-repo SC attempt; reported **only** as a comparison to show the factor disagreement |

`U_s = (1 - phi_nom) * <u_x>_fluid` with `phi_nom` the nominal (input) solid
fraction; a diagnostic channel uses `phi_actual` (voxel count) and
`a_eff = a_nom (phi_actual/phi_nom)^(1/3)`.

## 3. Numerical set-up (predeclared)

- D3Q19, BGK, `tau = 1` (`nu = 1/6` in lattice units), `float32`, `torch`.
- Library kernels only (`src/tensorlbm`): `d3q19.equilibrium3d`,
  `solver3d.collide_bgk3d`, `solver3d.stream3d`,
  `boundary.boundaries3d.bounce_back_cells_3d`, `d3q19.macroscopic3d`.
  Body force = disclosed first-order Guo term (`apply_body_force_3d`, a D3Q19
  transcription of `turbulent_channel._apply_body_force_2d`), applied to fluid
  nodes only.
- `a_body` chosen so the achieved particle Reynolds number
  `Re_p = U_s*2a/nu ~ 0.05` (creeping limit; `Ma <= 1e-3`).
- Ladder (three tiers per point, `N` = cells per lattice constant):
  `phi = 0.125`: N = 32, 48, 64, 96;  `phi = 0.45`: N = 48, 64, 96, 128;
  `phi = 0.5`: N = 64, 96, 128 (see caveat 1.2).
- Steady state: drift of `<u_x>_fluid` over a 2000-step window `< 1e-6`
  for 3 consecutive evaluations, plus 4000 extra steps; measurement window =
  last 4000 steps.
- `max_steps = min(4 N^2, 200000)`; a case records `steady_detected`; a case
  that does not reach steady state is reported, never silently accepted.

## 4. Predeclared thresholds and stopping rule

- **PASS** criterion (per point, verdict-bearing): `|K_mx/K_ref - 1| < 0.05`
  at the **finest tier**; `< 0.15` at the coarsest tier is not required.
- **Convergence**: successive-tier relative change `<= 0.02` at the two finest
  tiers, else the point is reported as **not grid-converged** and the verdict is
  `WITHHELD` regardless of the finest-tier error.
- **Channel consistency**: `|K_mx/K_mb - 1| < 0.01` required, else the case is
  flagged `channel_inconsistent` and excluded from the verdict.
- Stopping rule: no threshold is revised after data are inspected; if the
  finest tier is not converged the ladder is extended one tier *upward* (never
  downward) and the extension is recorded.

## 5. Known confounders

- Staircase (voxel) geometry: `phi_actual != phi_nom` and the effective radius
  differs; diagnostic channel quantifies it, nominal stays verdict-bearing.
- Bounce-back wall location vs `tau` (kept at `tau = 1`, the least
  tau-sensitive choice).
- Compressibility (`Ma` from `u_max`), reported per case; must stay `< 1e-2`.
- Dense points: surface gap resolution `h/Delta_x = 2 a (1 - (phi/pi*6)^(1/3)) /
  Delta_x` — reported per tier; `< 3` cells is flagged `under_resolved`.

## 6. Reproduce

```bash
python benchmarks/case_pool/sc_sphere_array_drag/run.py case --phi 0.125 --n 64 --out out
python benchmarks/case_pool/sc_sphere_array_drag/run.py ladder --phi 0.125 --out out   # all tiers
python benchmarks/case_pool/sc_sphere_array_drag/run.py report --out out --figures figures
```
## 7. Amendment 1 (registered after the N=16 diagnostic, before the verdict)

The link-pair momentum-exchange formula of section 2 assumes a pairwise reflection
at every fluid-solid link.  With the library's `bounce_back_cells_3d` the swap is
applied to **all 19 populations of every solid node**, so at staircase corners the
population leaving the interface link is taken from the *opposite* side of the
solid node and the pairwise identity fails.  Measured consequence (phi=0.125,
N=16, converged): `F_mx/F_inj = 1.326` while the total lattice momentum is
stationary (`dP_all/dt = 0` to float32 noise).  Therefore:

- **primary (verdict-bearing) channel := `K_mb`**, the exact momentum balance
  `F = a_body * n_fluid` (valid because total momentum is stationary and `dP/dt`
  was measured to be zero); `K_mb = a_body*N^3/(6 pi nu a_nom <u_x>_fluid)`
  (the `(1-phi)` of the superficial velocity cancels analytically).
- `K_mx` (link-pair) is demoted to a **documented artifact diagnostic**
  (`F_mx/F_inj` reported per case, must be ~1 for the formula to be usable).
- The legacy in-repo channel `K_qv = K_mb/(1-phi)` (used by
  `benchmarks/pending/permeability3d`) is reported for comparison only: it is
  inflated by `1/(1-phi)` (2.7% at phi=0.027, 14.3% at phi=0.125) and therefore
  cannot be used to claim agreement at dense points.
