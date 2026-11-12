# CASE-MCMP-CAPILLARY-RISE — capillary rise vs Jurin's law (TensorLBM)

**Verdict of this pass: FAIL** — measured rise 2.84 lattice units against a
Jurin reference of 33.25 built from the declared wall wetting
(error -91 %, gate 10 %).  The driver runs to a stable, converged state;
the failure is in the wetting boundary condition and in the scheme's
spurious-current floor, not in the geometry or the measurement chain.

## 1. Decision, alternatives, hypothesis

**Decision**: can TensorLBM reproduce `h = 2 sigma cos(theta)/(rho g r)`
(2-D slit: `r -> w`) to < 10 %?

| route | driver | outcome |
|---|---|---|
| B (delivered) | `sc_closed_box.py` — Shan-Chen D2Q9 + Guo forcing, closed box, buoyancy gravity `F_y=-(rho-rho_v)g`, phase-conditional wetting wall, volume-equivalent pool level | runs, converges, **FAIL** on the gate |
| A (falsified) | `run_case.py` — same kernels, vented free surface (constant-density vapour reserve) | mass is drained (wetting wall pumps vapour) or injected (`p(rho_v)` is not the field's gas pressure); see `artifacts/route_a_vent_diagnostics.md` |
| C (existing capability) | `tensorlbm.wetting.ladder::jurin` (CAC phase field) | NaN after 300-400 steps; the channel is a closed tube, so an incompressible vapour cannot be displaced |

**Hypothesis**: for `Bo = d_rho g w^2/sigma = 0.79 < 1` the meniscus is a
cylindrical arc of mean curvature `2 cos(theta)/w`; the wall strength is set by
Young's relation `psi_w = psi_v + cos(theta)(psi_l - psi_v)`.

## 2. Metrics, baseline, thresholds (predeclared)

* primary metric `h = y_meniscus - y_pool,flat` in lattice units, with
  `y_pool,flat = wall + sum(phi)/n_columns` (wetting side walls make the local
  pool surface non-flat, so the conserved liquid volume defines the reference);
* three references: Jurin with the *declared* wall wetting (`theta_Young`),
  Jurin with the contact angle measured on the meniscus arc (`theta_arc`), and
  the closed-box balance eq. (1) of `sc_closed_box.py` (gas compression added);
* gate `|error| < 10 %`, tight goal `< 5 %`;
* stopping rule: 12000 steps, convergence when the last-20 % h-spread
  < 2 % **and** `u_max,liquid < 0.02`;
* controls: no-wetting drive -> 0, closed-box gas term -> 0;
* confounders: spurious SC current, wall depletion layer, meniscus smearing
  (diffuse width ~4 over a 16-node slit), side-wall menisci (removed by the
  volume-equivalent pool level).

No repetition is warranted (deterministic run); the uncertainty is the
final-window spread (0.01 %) plus the scheme's velocity floor.

**Environment**: python 3.11, torch 2.7.1, 1 thread, cpu, `G_eff=-5`, `tau=1`,
`sigma=0.05611` (measured, `benchmarks/verified/laplace_droplet` via
`contact_line.engine.SIGMA_DEFAULT`); runtime 198 s.

## 3. Parameters (lattice units)

| parameter | value |
|---|---|
| domain | 150 x 128 (closed box) |
| slit width w / baffle / pool | 16 / 2 / 22 |
| rho_l / rho_v | 1.957 / 0.1596 (ratio 12.3) |
| wall fraction f_w = psi_w/psi_l | 0.85 |
| gravity g | 1.00e-04 |
| Bond number Bo | 0.79 |
| steps / sample | 12000 / 250 |

## 4. Reference vs simulation

| quantity | reference | got | error | gate |
|---|---|---|---|---|
| h vs Jurin, `theta_Young = 35.0 deg` | 33.25 | 2.84 | -91 % | FAIL |
| h vs Jurin, `theta_arc = 88.1 deg` (in situ arc) | 1.31 | 2.84 | +116 % | FAIL |
| h vs closed-box eq.(1) | 26.84 | 2.84 | -89 % | FAIL |
| contact angle: Young relation vs meniscus arc | 35.0 deg | 88.1 deg | +53.1 deg | FAIL |
| capillary `2 sigma cos(theta_Young)/w` | 2.27e-04 | hydrostatic `d_rho g h` = 4.91e-04 | residual +1827 % | FAIL |

**Root cause (measured, not hypothesised)**

1. The Shan-Chen wall at `f_w = 0.85` produces an *effective* contact angle of
   88.1 deg (meniscus arc) instead of the 35.0 deg implied by
   Young's relation — the wall is effectively neutral.  This is consistent with
   the library's own calibrated ladder (`tensorlbm.contact_line.engine`: the
   *measured* neutral wall sits at `f_w = 0.60`, not at `psi_v/psi_l = 0.17`), so
   the Young relation is not a usable wetting control for this scheme.
2. The closed-box vapour drifts off coexistence (`rho_g = 0.1281` vs
   `rho_v = 0.1596`), so the gas-pressure term -4.41e-03 is as large as the
   capillary pressure 2.27e-04: the vapour of a weakly compressible SC model
   cannot act as a passive reservoir at these capillary pressures.
3. The scheme's spurious-current floor in the liquid is
   `u_max,liquid = 0.10` lattice units, two orders of magnitude above the
   capillary-rise velocity of the case (~1e-3): a quasi-static rise sits at the
   edge of the noise floor (this is why the predeclared velocity-convergence
   flag reports False while `h` itself is converged to 0.01 %).

Liquid mass is conserved to 8.8e-02, so the measurement chain is trustworthy:
the gap is physical-model, not post-processing.

## 5. Reproduction

```
PYTHONPATH=src python3 benchmarks/case_pool/capillary_rise/sc_closed_box.py
```

writes `artifacts/jurin_closed_box.json` / `.png` / `_report.md`; every number in
this file was read back from that JSON (none typed by hand).
`tests/test_case_pool_capillary_rise.py` (4 tests, passing) pins the geometry,
the initial state, the sub-grid meniscus measurement and the Jurin scaling.

## 6. Interpretation limits

* Negative result for *this* SC formulation + wetting wall + closed box; it does
  not exclude another formulation (free-energy / phase-field with a calibrated
  wetting BC) from passing the case.
* The arc angle and the rise are not independent, so the run alone cannot
  separate "Jurin is wrong" from "the angle is wrong"; the Young-vs-arc column
  is what separates them.
* One gravity, one slit width: this is not a Bond-number sweep.
