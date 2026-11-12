## 1. Case introduction (physics / geometry / parameters)

Creeping flow through a **simple-cubic (SC) periodic array of solid spheres** driven by a
uniform body force. The target is the dimensionless drag (Darcy resistance)

  `K(phi) = F / (6*pi*mu*a*U_s)`, `U_s = (1-phi) <u_x>_fluid` (superficial velocity),

which is the classical benchmark of **Hasimoto (1959), J. Fluid Mech. 5:317**
(periodic fundamental solutions; dilute series
`U/U0 = 1 - 1.7601 c^(1/3) + c - 1.5593 c^2 + 3.9799 c^(8/3) - 3.0734 c^(10/3)`, `K = U0/U`)
and **Sangani & Acrivos (1982), Int. J. Multiphase Flow 8:343** (SC/BCC/FCC tables). The SC
values used here are the ecosystem's locked two-witness table
(`benchmarks/verified/permeability3d/NOTES.md` §1.2-1.4: Basilisk `src/test/spheres.c`
verbatim + Holmes/Williams/Tilke 2011 literature column):
`phi = 0.125 -> K = 4.292`, `phi = 0.45 -> K = 28.09`, `phi = 0.027 -> 2.0077`,
`phi = 0.5236 -> 41.99` (touching limit, divergent, excluded).
Hasimoto's series evaluated **in code** gives `K_H(0.125) = 4.2894`
(0.06% from the table row) - the two references agree at the protocol point.

**Registered deviation (declared, not hidden):** the instructed second point `phi = 0.5` has
**no locked table row** (rows sit at 0.45 and 0.5236) and is geometrically the hardest point
(surface gap `h/a = 0.031`, needing `N ~ 250` cells per lattice constant). Protocol therefore
carried `phi = 0.125` (diluted) and `phi = 0.45` (dense, table-exact) as verdict-bearing, with
`phi = 0.5` as a diagnostic third point (reference WITHHELD).

Geometry: sphere of `a = N*(3*phi/(4*pi))^(1/3)` at the cell centre, periodic in all three
directions; `N` = cells per lattice constant; `N^3` = one unit cell.

## 2. Method / implementation

* D3Q19 BGK, `tau = 1` (`nu = 1/6` lattice units), `float32`, PyTorch CPU
  (32 threads, torch 2.7.1, HEAD `ba6428034856`).
* Library kernels only, from this worktree's `src/tensorlbm`
  (`d3q19.equilibrium3d`, `solver3d.collide_bgk3d`, `solver3d.stream3d`,
  `boundary.boundaries3d.bounce_back_cells_3d`, `d3q19.macroscopic3d`); the single
  driver-level kernel is the disclosed first-order Guo body force (fluid-node masked).
* Step order: collide -> stream -> masked body force -> bounce-back. Steady state by velocity
  drift; measurement window = last 4000 steps.
* **Two force channels** were implemented and cross-checked: (i) exact momentum balance
  `F = a_body * n_fluid` (`K_mb`), (ii) link-pair momentum exchange (`K_mx`). Channel (ii)
  was **falsified** for this library's bounce-back (see §5).
* `a_body` sized for `Re_p = U_s*2a/nu ~ 0.05`; achieved `Re_p` and `Ma` are reported per tier.

## 3. Parameter table

| quantity | value |
|---|---|
| lattice | D3Q19, BGK, `tau = 1` |
| viscosity | `nu = (tau-0.5)/3 = 0.166667` |
| body force `a_body` (phi=0.125, N=32) | `1.9575e-06` (lattice units) |
| sphere radius `a_nom` (N=32) | `9.9256` cells |
| gap between neighbours (N=32) | `12.149` cells |
| voxel fraction `phi_actual` (N=32) | `0.12411` (nominal 0.125) |
| `Re_p` / `Ma` (N=32) | `0.0615` / `1.93e-03` |
| steps, N=32 | `6000`, mass drift `5.10e-05` |

## 4. Results: reference vs measured

| phi | N | gap [cells] | phi_actual | K_mb (primary) | K_qv (legacy) | K_mx (link-pair) | err(K_mb) vs table | err(K_mb) vs Hasimoto | Ma | Re_p | F_mx/F_inj | steady | steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.125 | 16 | 6.07 | 0.11841 | 3.1104 | 3.5282 | 4.1248 | -3.89% | -3.84% | 4.08e-03 | 0.0690 | 1.3261 | no | 5000 |
| 0.125 | 32 | 12.15 | 0.12411 | 3.4907 | 3.9854 | 4.7001 | +9.51% | +9.57% | 1.93e-03 | 0.0615 | 1.3465 | no | 6000 |

Verdicts (machine-written in `out/result.json`):
* phi=0.125: **INCONCLUSIVE_NOT_STEADY** (err_finest=-0.1867, grid change=0.1223, channel spread=0.3465)

Both computed tiers undershoot the reference: `K_mb(16) = 3.110` (-27.5%),
`K_mb(32) = 3.491` (-18.7%); the change **between tiers is 12.2%**, i.e.
the ladder is **not grid-converged** and the preregistered `< 2%` convergence gate fails.
The legacy channel `K_qv = K_mb/(1-phi)` (used by
`benchmarks/pending/permeability3d`) reads `3.985` (-7.2%) at N=32;
it differs from `K_mb` by exactly `1/(1-phi)` (14.3% at phi=0.125, 2.7% at phi=0.027).

## 5. Pitfalls, limitations, and what was falsified

1. **Link-pair momentum exchange is invalid with this bounce-back.** `bounce_back_cells_3d`
   swaps *all 19* populations of every solid node, so at staircase corners the population
   leaving an interface link comes from the opposite side of the solid node and the pairwise
   `2*c_i*f_i` identity fails. Measured `F_mx/F_inj = 1.346` while the total lattice momentum is
   stationary (`dP_all/dt = 0` to float32 noise) - so `K_mx` over-counts by 33% and cannot be
   used as a verdict channel. Registered as Amendment 1 in `NOTES.md` §7.
2. **Normalisation trap.** `K_qv = a_body*N^3/(6 pi nu a (1-phi) um)` is inflated by
   `1/(1-phi)` relative to the momentum-balance derivation; at phi=0.027 the two differ by
   2.7% (which "passes"), at phi=0.125 by 14.3% (which decides pass/fail). Any agreement claim
   at dense points must state which convention is used.
3. **Voxel-fraction deficit.** `phi_actual < phi_nom` at coarse N (0.1184 vs 0.125 at N=16,
   0.1241 at N=32); using `a_eff` moves K by -0.4% (N=32) but by -7% at N=16.
4. **Residual transverse flow.** `uz_rms/ux_rms = 0.074` (max 21%) persists in the steady field
   of a geometry that is symmetric under all axis reflections and permutations; it is localised
   on staircase corners (see `figures/field_slice.png`) and is the most likely co-source of the
   drag deficit. Not resolved within this mission - it needs a volume-averaged transverse
   metric and a corner-consistent (interpolated) bounce-back.
5. **Not completed:** the preregistered ladder `N = 48/64/96` and the dense points
   `phi = 0.45` and `phi = 0.5` did **not** run to completion: the shared control host was at
   load 150-190 on 128 cores (37-38 ms/step at N=64-equivalent sizes) and the cluster queue was
   420 jobs deep (`cj1248` still `queued` after 13 min). This is recorded as ecological friction.

## 6. Error verdict (explicit)

* Key quantity `K(phi=0.125)`: measured `3.491` at N=32 vs reference `4.292`
  -> **-18.7% (primary channel)**, vs `K_qv` `3.985` -> -7.2%.
  **Is it < 5%? NO.** The `<5%` criterion is **not met** at either computed tier (-27.5% at N=16),
  and the convergence gate (2% between tiers) also fails (12.2%).
* Key quantity `K(phi=0.45)`: **no result** (run did not complete) -> WITHHELD.
* `K(phi=0.5)`: **no locked reference row** -> reference WITHHELD by rule; not attempted.
* Honest verdict: **FAIL (stage = ladder completion / discretisation accuracy)**, not a
  refutation of the reference values. Two independent reference evaluations (Hasimoto series
  computed in code and the locked SC table) agree to 0.06% at phi=0.125, so the *reference* is
  solid; the *measurement* is what fails.
