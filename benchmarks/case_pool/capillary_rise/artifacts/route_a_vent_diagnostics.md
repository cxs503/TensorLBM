# Route A (vented free surface) - measured diagnostics (negative result)

SC D2Q9, G_eff=-5, tau=1, rho_l=1.957, rho_v=0.1596, buoyancy gravity
F_y = -(rho-rho_v) g, two-row baffles, pool at rows 20-22, domain 160x128.

## A1. Fixed-density vent + Young-relation wall (psi_w = 0.503, theta = 60 deg)

| step | total mass / initial | min rho | pool level (rows) |
|---|---|---|---|
| 400 | 0.866 | 0.048 | 19.5 |
| 2400 | - | 0.039 | 2.3 |
| 2600 | NaN (divergence) | - | - |

Mechanism: the wetting psi_w on a two-row baffle acts on the **vapour** on both
faces; the gas is 12x lighter, so the same force density gives ~12x the
acceleration and the gas jets along the wall (measured u_max,gas up to 0.44,
supersonic for the SC gas where dp/drho = 0.124). The gas leaves through the
constant-density reserve and drags the liquid out.

## A2. Phase-conditional wall + fixed-density vent

No leak (min rho stable 0.127, finite to 3000 steps), but the pool level drifts
monotonically 20.0 -> 30.7 rows in 3000 steps (+0.5 row / 250 steps): gas
condenses into the liquid because the reserve holds rho_v while the field's
vapour pressure is not p(rho_v). A vented weakly compressible vapour is not a
passive reservoir at Delta p_cap ~ 3.5e-3.

## A3. Existing CAC ladder route (tensorlbm.wetting.ladder::jurin)

Defaults (theta=60 deg, gap=20, g=2e-4, shape=(160,88)): h = -1.57, -7.67,
-14.2 at steps 100/200/300, NaN at step 400. Structural cause: the baffles are
drawn up to the top plate (mask[pool:ny-2, ...]) so the channel is a *closed
tube*; with an incompressible phase field the displaced vapour cannot leave.
ladder.sigma_flat works (sigma_meas = 0.01945 vs input 0.02, ratio 0.973) and is
reused as the surface-tension cross-check.

## Consequence

Route B (sc_closed_box.py) is the delivered driver; A and C are falsified with
the numbers above.
