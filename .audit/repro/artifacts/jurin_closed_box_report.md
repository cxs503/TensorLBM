# CASE-MCMP-CAPILLARY-RISE — Jurin law in a 2-D slit (TensorLBM)

**Verdict: FAIL** (gate: |error| < 10 %),
converged=False (final-window drift 0.01 %,
max liquid speed 9.89e-02).

## ① Problem and reference

Capillary rise between two vertical plates standing in a pool: the meniscus
climbs until the capillary pressure equals the hydrostatic head of the column,

    h = 2 sigma cos(theta) / (rho g r)                         (Jurin's law)

Jurin (1718) *Phil. Trans. R. Soc.* **30**, 739-747; de Gennes (1985) *Rev. Mod.
Phys.* **57**, 827 (both are the reference strings recorded in
`src/tensorlbm/contact_line/wetting_dynamics.py`).  For a 2-D slit of width `w`
the meniscus is a cylindrical arc whose mean curvature is `2 cos(theta)/w`, so
the same law holds with `r = w`.  The target number is **evaluated from the
closed form with the parameters of this run** — no table digit is transcribed.

## ② Method

* solver: Shan-Chen single-component multiphase D2Q9, Guo forcing, half-way
  bounce-back walls (`tensorlbm.sc_surface_tension`), library coexistence pair
  `rho_l=1.957`, `rho_v=0.1596`, `G_eff=-5.0`, `tau=1.0`.
* wetting: wall pseudopotential from Young's relation
  `psi_w = psi_v + cos(theta)(psi_l - psi_v)`; applied **only to solid cells in
  contact with liquid** (a dry, gas-facing wall is neutral) — without this the
  two-row baffle pumps vapour.
* gravity: buoyancy form `F_y = -(rho - rho_v) g` (identical force balance, but
  the vapour carries no hydrostatic gradient).
* closed box (bottom/top/side walls, baffles open at the top so the displaced
  vapour can pass over them).
* measurement: `tensorlbm.wetting.metrics.meniscus_height` (sub-grid `phi=0.5`
  crossing) in the central 60 % of the channel, referenced to the pool level
  measured far from the baffles.

## ③ Parameters (lattice units)

| parameter | value |
|---|---|
| domain | 150 x 128 |
| slit width w | 16 |
| baffle thickness | 2 |
| pool level (rows) | 22 |
| sigma | 0.05611 (tensorlbm.contact_line.engine.SIGMA_DEFAULT (measured, benchmarks/verified/laplace_droplet)) |
| rho_l / rho_v | 1.957 / 0.1596 (ratio 12.26) |
| contact angle theta | 60.0 deg |
| gravity g | 1.00e-04 |
| Bond number d_rho g w^2 / sigma | 0.788 |
| steps / sampling | 12000 / 250 |

## ④ Reference vs simulation

| quantity | reference | value | error | verdict |
|---|---|---|---|---|
| h (meniscus rise) | Jurin 2*sigma*cos(theta)/(d_rho*g*w) | 1.31 | 2.84 | +116.2% | FAIL |
| h (meniscus rise) | closed box eq.(1) (gas compression included) | 26.84 | 2.84 | -89.4% | FAIL |
| capillary pressure 2 sigma cos(theta)/w | measured hydrostatic + gas excess | 2.269e-04 | -3.919e-03 | +1827.3% | FAIL |
| theta (from wall_psi, Young relation) | input theta | 60.0 deg | 35.0 deg | 0.0 deg | PASS |
| theta (meniscus arc in situ) | input theta | 60.0 deg | 88.1 deg | +28.1 deg | FAIL |

## ⑤ Convergence / sensitivity

* final-window drift 0.01 % over the last fifth of the run;
* vapour compression measured from the field: `rho_gas = 0.1281`
  vs `rho_v = 0.1596` (-19.7 %), giving the
  closed-box pressure excess `-4.41e-03` (=
  -1943.6 % of the capillary
  pressure);
* liquid-phase speed floor `u_max,liquid = 9.89e-02`;
* mass drift (all phases) `9.04e-05`, liquid-only
  `8.79e-02`.

## ⑥ Failure modes / artifacts

* **Route A (vented free surface) is falsified** — see
  `artifacts/route_a_vent_diagnostics.md`: a fixed-density vent in a weakly
  compressible SC gas either drains mass (wetting wall pumps vapour) or injects
  it (p(rho_v) is not the field's gas pressure).  Both drifts were measured and
  are the reason this case uses a closed box.
* `right` 2-row baffle: the wetting potential must be phase-conditional,
  otherwise the gas adjacent to the wall is accelerated (measured u_gas up to
  0.4 lattice units).
* artifacts: `artifacts/jurin_closed_box.json` (all numbers),
  `artifacts/jurin_closed_box.png` (trajectory + compression), `jurin_closed_box.png`.
