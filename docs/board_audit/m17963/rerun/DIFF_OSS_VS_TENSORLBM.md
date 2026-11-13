# Diff list: TensorLBM vs haimingz/vivsim (force-corrected NEBB + body-force driver)

Round 1 XVAL, anchor `XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S`.
Full narrative + numbers: `REPORT_XVAL_NEBB_BODYFORCE.md`.

## Approach differences

| aspect | vivsim (OSS) | TensorLBM | impact at Re≈2.25, Ma≈0.09 |
|---|---|---|---|
| NEBB kernel | shear-reconstruction `f_in = f_out ± ½(f_t0−f_t1)·s + (⅙)un·ρ ± ½ut·ρ` | feq-difference `f_in = feq_in − feq_out + f_out` (Zou/He) | agree 1st order; ~1 % (NY=20) to ~4 % (NY=10) apart |
| wall density | fixed `rho_wall = 1` | field-recovered (`wave_bc_common`) | TL more general; OSS simpler for ρ≈1 |
| Guo term velocity | corrected `u*` in force term | actual `u` in force term; `u*` in `feq` | O(Ma²), negligible |
| Guo scaling | EDM full term / Guo-BGK `(1−ω/2)` | `(1−1/(2τ))` | identical for BGK |
| velocity correction | `u_phys = u + g/(2ρ)` | same | identical |
| init | `f = feq(ρ, u − g/2)` | same | identical |
| collision | BGK / MRT / KBC | BGK (this XVAL) | MRT/KBC out of scope |

## Parameter / boundary-condition differences

* **Grid:** OSS example default `NX=NY=10` (H=9) is under-resolved; both
  arms show 5–9 % error there, 2–3 % at `NY=20`.
* **Walls:** both half-way no-slip at `y=0` and `y=NY−1`; both apply the
  force correction `u_wall ← u_wall − g/(2ρ)` to the *tangential* component.
* **Forcing:** both apply `Gx` to the whole domain (incl. wall rows) as a
  population increment; EDM ≡ Guo at this Ma.

## Porting tradeoffs (recommendation)

1. **Keep** TensorLBM's feq-difference NEBB (textbook, field-recovered ρ).
2. **Adopt** the OSS scaffold conventions (already in repo): pre-corrected
   wall velocity `−g/(2ρ)`, init `f = feq(ρ, u − g/2)`, post `u += g/(2ρ)`.
3. **No code change** needed for OSS-consistency; residual error is
   grid-driven, not a scheme defect.
4. **If 3 % is required:** refine the wall-normal grid (≥ 20 cells) or use a
   second-order wall (e.g. ghost-node / extrapolation) — out of scope here.
