# Pre-registration — drag closure from the Stokes limit to Re = 50

`node P54-W3-N11 · mission 10330 · repo TensorLBM · branch agent/m10330`

## 0. What this document freezes

The acceptance criteria, the numerical protocol and the **verdict rules** of
the LBM/DEM drag study, written before the production runs. Anything measured
afterwards that contradicts this text is reported as a result, not redefined.

This is the second attempt at this node. The first attempt is summarised in
§6 (what it did, and the three measurement traps it left behind); the protocol
below exists because of them.

## 1. The gap

`m7441` (node chain behind `docs/sphere_settling_gcs_20261107.md`) certified the
sphere force instrument **only in the Stokes limit**: the production matrix ran
`Re = 0.02 … 0.20`, and the report states explicitly that `Re >= 1` was not
attempted. The DEM solver of this repository therefore offers exactly two
options above the Stokes regime: pure `stokes` (documented here to be off by
186 % in terminal velocity at `Re ~ 40`) or the bare empirical
`schiller_naumann`, with no asymptotic behaviour at small `Re`, no documented
validity range, and no verification against the LBM solver of this repo.

Deliverable of this node: an analytic bridge `C_D(Re)` for `1 <= Re <= 50`
(Oseen asymptotic branch → Schiller-Naumann empirical branch), a resolution /
confinement convergence study of the LBM drag measurement, a deviation table
against the closure, and a DEM-coupling verification.

## 2. Frozen criteria

| # | criterion | verdict rule |
|---|---|---|
| a | Stokes limit: `C_D Re -> 24` as `Re -> 0` | closure: relative deviation `< 1 %` at `Re <= 1e-3` (analytic identity, tested); LBM: `C_D * Re` at the lowest simulated `Re` reported against 24 (no pass/fail bound — Stokes is *known* to be 10 % off at `Re = 1`) |
| b | Oseen vs LBM, `Re = 1 … 50`, `< 5 %` | as written this is evaluated honestly: pure Oseen everywhere in the band. Additionally the bridged closure is evaluated in the same band; the criterion is judged PASSED only if a *closure that is exactly Oseen as Re→0* stays within 5 % for `Re <= 20` |
| c | resolution convergence: `C_D` converges with resolution | least-squares fit of `C_D` vs `1/D` at `lambda = 1/6` (D = 8, 12, 16), `R^2 >= 0.99` |
| d | drag prediction error `< 10 %` for `Re = 1 … 50` (the m7441 extension) | bridged closure vs LBM (`lambda -> 0` extrapolated), and the DEM terminal velocity of the bridged closure vs the standard drag curve |
| e | negative control: zero drive ⇒ zero drag | exact `F = 0` (no applied force, no flow) and `C_D` undefined/zero |
| f | engineering delivery | `pytest >= 15` cases in `tests/test_drag_closure.py`, `results.json` carrying the `C_D(Re)` table, the convergence data and the deviation table |

## 3. Frozen numerical protocol

| item | value | reason |
|---|---|---|
| lattice / collision | D3Q19 / BGK | repository default (`solver3d`) |
| geometry | one voxel sphere, periodic cube, sphere at rest | reuse the validated instrumentation of `sphere_settling_gcs` |
| `u_char` | 0.05 lattice (`Ma = 0.087`, `Ma^2 = 0.8 %`) | compressibility below the 5 % band of every criterion |
| `nu` | `u_char * D / Re` | pins the Reynolds number |
| `tau` | `0.5 + 3 nu` (0.54 … 2.3 over the matrix) | BGK stability; the `tau >= 5` wall defect of m7441 is avoided by construction |
| driving | velocity-pinned body force: `g` from a drag feed-forward plus a gentle trim on the filtered mean velocity | a *prescribed* `g` needs `rho V/(6 pi mu R) ~ 1.6e4` steps of spin-up at `Re = 10` |
| force bookkeeping | `F = (1 - 1/(2 tau)) g V_fluid` | the Guo source of this library injects `(1 - 1/(2 tau)) x F` per step — **measured**, §5 |
| steady gate | tail-half drift of `u` and of `g` `< 1 %` (tail = last 50 samples, every 20 steps) | else the case is reported as unconverged, not as a verdict |
| confinement | D = 12, `L = 36, 48, 60` (`lambda = 1/6, 1/8, 1/10`), linear extrapolation in `lambda` | the ambient state of a periodic cell is not an unbounded stream |
| resolution | `lambda = 1/6` fixed, D = 8, 12, 16 (`L = 3D`) | isolates the body representation |
| step count | `n_conv * D/u_char`, `n_conv = 20 … 34` (more at larger Re) | wake development, floor of 3000 steps |

## 4. Verdict rules

* Every criterion is PASS/FAIL by the rule in §2, evaluated by
  `merge_results.py`, written to `results.json["criteria"]`.
* A criterion that fails stays failed in the report; the physical reason is
  documented, the criterion is not moved.
* Criterion (b) has a *literal* reading that is physically impossible (Oseen is
  a weak-inertia expansion; its own 5 % band closes at `Re = 1.24`, computed by
  `drag_closure.oseen_error_ceiling`). Both readings are reported.

## 5. Instrument calibrations run *before* the freeze

**Guo momentum injection.** A sphere-free periodic box (`L = 16`, uniform body
force `g = 1e-5`) accelerates at `dU/dt = 2.65e-6` for `tau = 0.68`, i.e. the
injected momentum is `0.2647 g V` and the analytic value
`1 - 1/(2 tau) = 0.2647` is reproduced to `< 1 %` at `tau = 0.536 / 0.68 / 2.3`.
Every force below therefore carries that factor; treating `g V` as the force
would bias the drag by 1.5x … 15x depending on `tau`.

**Three independent force estimators** are available and are all reported:

| estimator | status |
|---|---|
| momentum balance `(1-1/2tau) g V` | used for the verdict (needs steady state, which the steady gate checks) |
| kinetic-flux control surface (m7441 instrument) | agrees with the momentum balance to ~20 % at `Re = 0.04` (the regime it was certified in) but reads a factor ~4 too high at `Re = 10`; the ambient state around the surface is a forced, confined flow, not a uniform stream |
| wall momentum exchange, single time level | reported as a cross-check; single-level pairing (the G15 repair) |

Because two of the three disagree at finite `Re`, the disagreement itself is a
deliverable (report §5) — but only the momentum balance enters the verdict,
since it is an exact statement about the periodic cell.

## 6. What the first attempt got wrong (kept for the record)

1. **Prescribed body force** in a periodic cell: the momentum stored in the
   fluid bulk (`rho V u`) makes the spin-up time `rho V/(6 pi mu R)`; at
   `Re = 10` (D = 12) that is 1.6e4 steps, i.e. the first attempt measured a
   transient, not a steady state.
2. **`g V` as the force**: the `(1 - 1/(2 tau))` factor of the Guo source was
   not accounted for, which at `Re = 50` (`tau = 0.54`) is a factor 15.
3. **Surface-flux instrument above `Re ~ 1`**: the m7441 estimator is exact
   only for a uniform ambient state; in a forced confined cell it reports
   `C_D = 14.3` where the momentum balance gives `5.2` (`Re = 10`, `lambda =
   1/6`). Not a bug in the estimator — a domain of validity that was never
   stated.

## 7. Honest declaration

* No moving boundary, no empirical drag input inside the simulation: the
  closure is only used to *set* the driving force, never to compute the answer.
* `tau < 0.6` cases (`Re = 50` at D = 12/16) are kept but flagged: BGK accuracy
  degrades as `tau -> 0.5`, and `lambda -> 0` extrapolation at `tau = 0.54` is
  the weakest point of the study.