# Flettner rotor sail (spinning cylinder) — LBM Magnus-force prediction, ship-wake coupling and voyage energy assessment

Mission IN23-N03 · repository TensorLBM · branch `agent/m5175` · 2026-11-07

## 1. What was built

New sub-package `src/tensorlbm/rotor_sail/`:

| module | content |
|---|---|
| `lbm2d.py` | D2Q9 solver: Bouzidi–Firdaouss–Lallemand (BFL) interpolated **moving-wall bounce-back** (Ladd momentum injection), optional **Smagorinsky LES**, prescribed inlet profiles, zero-gradient outlet, **two independent force estimators**: wall momentum exchange and control-volume momentum balance with the exact LBM momentum-flux tensor `Π_ij = Σ_q f_q c_qi c_qj` |
| `aero.py` | coefficient table/interpolation, deck boundary-layer + propeller-race inflow models, spanwise (strip) thrust integration vs the area-mean engineering formula, apparent-wind decomposition |
| `voyage.py` | Weibull wind statistics → apparent wind per heading → annual thrust assist → main-engine load reduction → fuel/CO₂ (HFO 3.114 t CO₂/t) |
| `scripts/run_flettner_validation.py` | thin CLI runner (`case`, `voyage`) |
| `tests/test_flettner_rotor_sail.py` | 13 fast unit tests + LBM smoke tests + artifact-level acceptance regression |

## 2. Physical setup (computed case)

| parameter | value |
|---|---|
| lattice | D2Q9, BGK, ρ₀ = 1 |
| Reynolds number | Re = 200 (laminar benchmark) |
| inlet velocity | u_in = 0.10 (Ma = 0.17) |
| cylinder diameter | D = 20 cells (R = 10) |
| domain | 18 D × 16 D (upstream 6 D, downstream 12 D) |
| spin ratio | α = ωR/u_in = 2.0 |
| ν = u D/Re | 0.01, τ = 0.5300 |
| steps / averaging window | 6000 / last 3000 (76 samples) |

## 3. Results (numbers produced by the runner, `artifacts/flettner_rotor/lam_D20_a2.json`)

| quantity | value |
|---|---|
| C_L (wall momentum exchange) | **+6.086** |
| C_L (control-volume momentum balance) | **+6.137** (0.8 % from the wall force) |
| C_D | +0.545 |
| C_Q (torque) | 0.222 |
| C_L, Mittal & Kumar (2003) Re = 200, α = 2 | 2.29 |
| relative error | **+166 %** |

**Verdict: FAIL** for the machine criterion “C_L(α=2) vs Mittal & Kumar Re = 200 DNS within 5 %”.
The control-volume cross-check passes (<10 %), so the two force estimators are mutually
consistent — the *flow field*, not the force extraction, is wrong.

Note on the sign convention: positive α is defined as clockwise rotation, giving positive
(upward) lift on a +x free stream; the raw CCW-Magnus force is negative, which the solver
reproduces correctly in sign.

## 4. Root-cause analysis of the FAIL (why the lift is ~2.7× too large)

1. **Confinement by the transverse boundary condition (dominant suspect).** The top/bottom
   rows impose a Dirichlet free stream (`u = u_in`, `v = 0`), i.e. they behave as solid
   wind-tunnel walls for the induced cross-flow. A cylinder at α = 2 drives a circulatory
   flow of the order u_in over the whole 16 D-high box; blocking its return flow inflates
   the lift. A moving-wind-tunnel correction or a much taller box (≥ 30 D) / periodic
   Fourier transverse direction is required. This was not tested within the budget.
2. **BFL far branch used the wrong node.** For wall fractions δ > 0.5 the interpolation
   used the same-node opposite population instead of `f_opp^+(x_b + e_opp)`. Patched
   during this mission (`lbm2d.py`, `fqb_nb`); the corrected case was launched but did not
   finish inside the wall-clock budget, so its number is **not** reported.
3. **No grid-refinement study completed** (D = 12/28 runs were queued but did not finish
   under a machine load average of 150–220 on 128 cores). A 2.7× lift bias is far beyond
   any plausible D = 20 discretisation error, which is consistent with (1).
4. **Transient not fully settled**: the sampled C_L still drifted from 6.40 to 6.10 over
   the averaging window, so the reported value is an upper bound.

## 5. What does work (evidence)

* The moving-wall bounce-back imposes the rigid-body rotation velocity (the torque
  coefficient is non-zero and converges: C_Q = 0.222) and the lift sign is physical.
* Wall force and control-volume momentum balance agree to **0.8 %** — an independent audit
  of the force extraction that does not rely on the wall model.
* The engineering chain (wake profile → strip integration → apparent wind → Weibull voyage
  → CO₂) is implemented and covered by 13 passing unit tests, including:
  * strip integration reduces exactly to the area-mean formula for uniform inflow;
  * a sheared deck boundary layer reduces thrust and stays within the 20 % band of the
    mean-velocity formula (model-level check, not yet an LBM result);
  * Weibull bins reproduce Γ(1+1/k) to 2 %;
  * voyage bookkeeping `CO₂ = net power × hours × SFOC × 3.114` closes exactly.
* Illustrative voyage numbers for a 180 m coaster (14 kn, 8 × 3000 nm round trips/yr,
  Weibull A = 8 m/s, k = 2, 4 rotors D = 3 m × H = 18 m, ω = 12 rad/s):
  mean thrust **9.75 kN**, rotor drive **4.1 kW**, propulsive saving **108 kW**,
  net **104 kW**, **64 t** fuel and **200 t CO₂** per year. These use the literature
  coefficient table, not yet the LBM-computed coefficients.

## 6. Limitations / not done

* No validated C_L(α) curve from LBM → the acceptance criterion for the laminar benchmark
  and the α-sweep are **not met**.
* Re = 1e5 LES case (D = 20, C_s = 0.16) was launched but did not complete; no high-Re
  number is claimed.
* Free-surface effect, finite span / tip vortices, rotor–hull aerodynamic interference and
  unsteady wind are outside the 2D section model; the wake coupling is a quasi-3D strip
  model fed by the 2D section coefficients.
* The reference coefficient table is transcribed from literature figures (digitisation
  uncertainty ~2–3 %); this uncertainty is small compared with the observed 166 % error.

## 7. Reproduce

```
python scripts/run_flettner_validation.py case --re 200 --u-in 0.10 --diameter-cells 20 \
  --alpha 2.0 --steps 6000 --warmup 3000 --upstream-d 6 --downstream-d 12 --height-d 8 \
  --save-fields --out artifacts/flettner_rotor/lam_D20_a2.json
python -m pytest tests/test_flettner_rotor_sail.py -q
```
