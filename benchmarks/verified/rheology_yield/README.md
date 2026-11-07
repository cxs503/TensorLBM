# rheology_yield — Herschel-Bulkley yield-stress verification (P55-N09)

**Status: HONEST PARTIAL / NOT VERIFIED (`all_gates_passed = false`).** The
integrals of the Bingham branch verify (flow rate ≤ 2.7 %), the plug-radius
probe, the plate (Couette) branch and the shear-thinning branch do **not** meet
the 3 % gate. Every number below is machine-produced by `run.py` into
`result.json` / `gates.json`; nothing is transcribed by hand.

## 1. Physical target

Yield-stress fluids (Herschel-Bulkley, `tau = tau_y + K gamma_dot**n`) stop
deforming below `tau_y`. The campaign node asks for three analytic anchors:

| # | Anchor | Analytic relation |
|---|--------|-------------------|
| 1 | pipe Poiseuille plug core | `r_c = R tau_y/tau_w` (yielded annulus `R(1-tau_y/tau_w)`) |
| 2 | plate Couette yield surface | pure Couette: uniform stress -> **no interior yield surface**; adverse-gradient Couette-Poiseuille: `y_yield = h tau_y/tau_w` |
| 3 | slush pipeline `DeltaP-Q` | closed-form H-B Poiseuille law `Q(tau_w,R,tau_y,K,n)` |
| 4 | negative controls | `tau_y = 0` -> no plug, pure shear never yields, sub-yield forcing -> no flow |

## 2. Method

* `tensorlbm.herschel_bulkley` — Papanastasiou-regularised H-B viscosity,
  closed-form pipe/slit flow rates and velocity profiles (Bingham and power-law
  limits verified to machine precision in `tests/test_herschel_bulkley.py`),
  Couette-Poiseuille adverse-gradient solution, slush concentration closure,
  and a 1-D exact solver for the *regularised* model
  (`hb_regularized_reference_profile`) used to separate discretisation error
  from regularisation bias.
* `tensorlbm.herschel_bulkley_lbm` — variable-viscosity BGK (D2Q9 + D3Q19),
  Guo forcing with per-cell relaxation time, half-way bounce-back (resting and
  moving walls), **wall-aware distance-weighted gradients** (a plain central
  difference next to a half-way wall underestimates the shear rate by up to a
  factor 2 and biases `nu ~ tau_y/gamma_dot`; fixing this moved the Bingham pipe
  flow rate from −9 % to −0.5 %), and threshold probes for the plug radius /
  yield surface.
* Geometry first: the staircase pipe's hydraulic radius is calibrated from an
  independent **Newtonian** run, `R_eff = (8 nu Q / (pi a))^(1/4)` =
  **20.27** at `R=20` (nominal), and the Newtonian profile then matches the
  Hagen-Poiseuille parabola to ≤ 2 % (5.9 % in the last fluid bin at the wall).

## 3. Results (this artifact set, `--fast`, R=14)

| quantity | LBM | analytic | error | gate |
|---|---|---|---|---|
| Bingham pipe flow rate | 1.35 % high | — | **+1.35 %** | of G3 (mixed with the n=0.6 branch) |
| Bingham plug radius (probe) | 3.52 | 4.87 | **−27.7 %** | FAIL (G1) |
| plug radius vs 1-D regularised ref. | 3.52 | 4.79 | −26.6 % | FAIL (G1b) |
| Bingham centreline velocity | — | — | +5.9 % | — |
| H-B n=0.6 pipe flow rate | — | — | **+89 %** | FAIL (G3) |
| Couette yield surface `y_yield` | 12.49 | 28.57 | **−56 %** | FAIL (G2) |
| pure-shear linearity (negative control) | — | — | 9.4 % | FAIL (N2, 3 % band) |
| sub-yield channel `u_max/u_ref` | 0.00084 | 0.02 ref | 4.2 % | FAIL (N3, 1 % band) |
| slush curve | diverged (`nan`) | — | — | FAIL (G3b) |
| Newtonian pipe, no plug (N1) | no plug found | no plug | — | **PASS** |

The full case set (`full_case_run.log`, R=20, six body forces) gives the same
picture on the Bingham branch with better hygiene: flow-rate errors
**+0.47 %, +2.74 %, −0.42 %** for `a = 0.70/1.00/1.45 × a_ref`, and
`−13 %/−19 %/−23 %` on the plug-radius probe. The shear-thinning branch is
`+30 %/+56 %/+77 %` (worse with increasing forcing), i.e. systematically
non-convergent in resolution.

## 4. Diagnosis and limits (why the gates fail)

1. **Regularised plug is soft and slow.** A regularised H-B model has no rigid
   core: it creeps at `sigma/nu_plug`. The plug probe therefore depends on
   `nu_max` and on run length; `converged=false` in every pipe case (the flow
   rate converges long before the core does). The exact 1-D regularised
   solution (`r_c` probe 4.79 vs ideal 4.87, i.e. 1.7 %) shows the LBM core is
   the deviating part — this is a *time-convergence* limitation, not an analytic
   error. Longer runs (30 000+ steps) and/or a stiffer plug with an implicit
   steady solver are required.
2. **Shear-thinning amplification.** For `n = 0.6` the viscosity scales like
   `gamma_dot^(-0.4)`; the staircase wall (no interpolated bounce-back) plus the
   half-way stencil makes the near-wall shear rate noisy/biased, and the power
   law amplifies it. A resolution study (R = 28/40) and BFL/interpolated
   bounce-back are the obvious fixes.
3. **Plate branch is transient-limited.** The adverse-gradient Couette plug is
   static, so its equilibration is diffusive through `nu_plug`; 6 000–20 000
   steps are not enough (pure-Couette linearity is off by 9 %, which is a
   convergence/wall-treatment signature rather than a physics error).
4. **Slush forcing was mis-scaled** in the reduced run (the slush consistency
   index is `O(1)`, so the body force must be `O(1e-3)`, not `O(3e-2)`); the
   run diverged to `nan` and the gate is reported as failed rather than dropped.
5. A **negative result is recorded**: smoothing the shear-rate input over a
   3×3 stencil (a common device in non-Newtonian LBM) made both the flow rate
   (+4 %) and the plug probe (5.53 vs 5.86) *worse* here, so the filter is off
   by default (`viscosity_smoothing=1`).
6. The campaign brief's `r_c = R(1 - tau_y/tau_w)` is inverted: the plug radius
   is `R tau_y/tau_w`, and `R(1 - tau_y/tau_w)` is the yielded annulus. Both are
   computed and reported.

## 5. Reproduce

```bash
python benchmarks/verified/rheology_yield/run.py --fast     # ~3 min, R=14
python benchmarks/verified/rheology_yield/run.py            # full case set, R=20/28
python -m pytest tests/test_herschel_bulkley.py -q
```
