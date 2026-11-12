# CASE-MCMP-CAPILLARY-RISE - case-pool pass 1 (TensorLBM)

**Anchor**: CASE-MCMP-CAPILLARY-RISE (capillary rise, Jurin's law).
**Ref**: h = 2 sigma cos(theta)/(rho g r) - Jurin (1718) Phil. Trans. R. Soc. 30,
739-747; de Gennes (1985) Rev. Mod. Phys. 57, 827 (reference strings as recorded
in src/tensorlbm/contact_line/wetting_dynamics.py). For a 2-D slit the same law
holds with r -> w because the meniscus mean curvature is 2 cos(theta)/w. The
target number is evaluated from the closed form with the run's own parameters
(no table digit transcribed).

**Gate**: |error| < 10 %. **Verdict: FAIL** (honest, no tuning).

## Deliverables

| path | content |
|---|---|
| benchmarks/case_pool/capillary_rise/sc_closed_box.py | delivered driver (route B): SC D2Q9 + Guo forcing + tensorlbm.sc_surface_tension kernels, closed box, buoyancy gravity, phase-conditional wetting wall, volume-equivalent pool level, meniscus arc fit; writes JSON + PNG + report table |
| benchmarks/case_pool/capillary_rise/run_case.py | route A (vented reserve) - kept as the falsified alternative |
| benchmarks/case_pool/capillary_rise/README.md | predeclared design: decision, alternatives, hypothesis, metrics, thresholds, stopping rule, confounders, environment, interpretation limits, results |
| artifacts/jurin_closed_box.json | every number of the run (trajectory, arc fit, gas compression, references, errors) |
| artifacts/jurin_closed_box.png | rise trajectory vs the two references + closed-box vapour compression |
| artifacts/route_a_vent_diagnostics.md | measured negative results of route A and of the existing CAC ladder route |
| tests/test_case_pool_capillary_rise.py | 4 tests pinning geometry, initial state, sub-grid meniscus measurement, Jurin scaling (passing) |

## Result

See benchmarks/case_pool/capillary_rise/README.md section 4 (numbers injected
from the JSON). Summary: the run converges (final-window spread 0.01 %, liquid
mass conserved to 1e-2) but the rise is far below the Jurin value built from the
declared wall wetting; the in-situ meniscus arc reads an effective contact angle
of 88 deg where Young's relation claims 31.5 deg.

## Ecosystem friction recorded

1. The Shan-Chen wall potential is not controlled by Young's relation
   psi_w = psi_v + cos(theta)(psi_l - psi_v): the library's own contact-line
   campaign measured the neutral wall at f_w = psi_w/psi_l = 0.60, i.e. 3.5x the
   Young value. Any wetting case built on wall_psi_from_contact_angle inherits
   this bias. Recommendation: expose the measured f_w -> theta ladder as the API
   and demote the Young formula to a nominal label.
2. tensorlbm.wetting.ladder::jurin (CAC route) is structurally unable to work:
   its baffles reach the top plate (closed tube) while the phase field is
   incompressible, and it is NaN within 400 steps at the repository defaults.
3. A vented (constant-density) boundary in a weakly compressible SC vapour is
   not usable at engineering capillary pressures (measured drift +0.5 row / 250
   steps, or outright mass drainage when the wall potential is not phase-aware).
4. The SC spurious-current floor in the liquid (~0.1 lattice units) is two orders
   of magnitude above the capillary-rise velocity of this case, so a quasi-static
   rise measurement needs a better-conditioned (low-Mach) scheme.

## Next round (hand-off)

* Calibrate f_w -> theta with the sessile-drop ladder in one run, then re-run the
  same closed box with the calibrated wetting; that alone decides whether the case
  can pass.
* Or replace route B with a free-energy/phase-field solver that has a
  mass-conserving, angle-calibrated wetting BC (the CSF CapillarySolver lacks a
  contact-angle BC and gravity, both of which would have to be added).
