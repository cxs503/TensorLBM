# XVAL-LBM-PHASE-FIELD-ALLEN-CAHN-D3Q15-D3Q19 - round 1 (honest FAIL, fail_stage = numeric application of the extracted operators)

## 1. Problem and reference provenance
Anchor: cross-validate TensorLBM's Allen-Cahn phase field against the waLBerla
`phase-field-allen-cahn-d3q15-d3q19` scheme
(`oss_solutions.id=363`, repo `lssfau/walberla`,
`apps/benchmarks/PhaseFieldAllenCahn/multiphase_codegen.py:10,29-37`).
waLBerla generates that app with **lbmpy**; the same library (lbmpy 2.0) is installed here,
so the OSS formulation is read at run time from `lbmpy.phasefield_allen_cahn`
(`AllenCahnParameters`, `chemical_potential_symbolic`, `interface_tracking_force`,
`laplacian_symbolic`, `isotropic_gradient_symbolic`). Nothing was hand transcribed and no
external copy of the waLBerla source was needed for the formulation itself
(no network: github is unreachable from this sandbox; pip and the local lbmpy install are).

OSS facts extracted (values printed by the run, `artifacts/probe.json`):
* beta = 12 sigma/W, kappa = 3/2 sigma W, omega_phi = 1/(0.5 + 3 M) -> tau_phi = 0.5 + 3M,
  i.e. M = (tau_phi - 1/2) cs^2 (from `AllenCahnParameters`, evaluated via `parameter_map()`).
* mu = 4 beta phi (phi-1/2)(phi-1) - kappa lap(phi) (`chemical_potential_symbolic`).
* interface-tracking force F_i = (1/3)(1-4(phi-1/2)^2)/W * n_hat_i (`interface_tracking_force`).
* the discrete laplacian/gradient are derived **on the LB stencil itself**: D3Q15 laplacian
  weights centre -32/9, faces 7/18, corners 11/72; D3Q19 uses its own set (extracted, see JSON).
* lattice split: **D3Q15 carries phi in [0,1]**, **D3Q19 carries hydrodynamics** with the split
  pressure / viscous / surface-tension force; rho = rho_L + phi (rho_H - rho_L) is explicit.

## 2. Method
`run_xval.py` extracts the two operators symbolically, proves that the coefficient tables
reproduce lbmpy's expressions **exactly** (`exact_reconstruction: [true,true,true,true]`,
`artifacts/probe.json`), rebuilds an OSS arm (BGK-D3Q15 phase field, lbmpy interface force,
`u_symp = u - F/2` consistent initialisation) and compares it with the shipped TensorLBM kernel
`tensorlbm.allen_cahn_lbm.step_allen_cahn_field`.
Metrics were predeclared (`predeclared.json`): M1 interface thickness, M2 curvature-driven
shrinkage `dR^2/dt = -4 M sigma`, M3 interface-force normal integral vs `sigma K`.

## 3. Parameters (dimensionless, lattice units)
sigma = 0.02, M = 0.2 (D_analytic = 4.0e-3), W in {4, 3}, box 64^3 (R0 = 12) and 48^3 (R0 = 9),
periodic, float64, 1200 / 900 steps.

## 4. Result table - NOT PRODUCED
The numeric application of the extracted operators is wrong: applying the extracted tables to a
quadratic field `q = 0.3x^2 + 0.7y^2 + 0.2z^2` gives
`lap_err = 691.2` (expected 2.4), `gradx_err = 79.2`, `grady_err = 184.8`
(`artifacts/probe.json`). Because the operators are the backbone of both arms, M1/M2/M3 numbers
are **withheld** - publishing them would be fabricated evidence. Both arm drivers raised
(caught in the probe: `complex` radius from a diverged field; a torch dtype/complex promotion in
`apply_op`).

## 5. Diagnosis of the FAIL (actionable)
1. `exact_reconstruction = true` proves the *coefficient tables* are faithful to lbmpy; the
   defect is in `apply_op`: the offsets returned by `Field.Access.offsets` are ordered
   `(x,y,z)` while the torch tensors are `(z,y,x)`; the mapping in this round used
   `shifts=(off[2], off[1], off[0])` for a D3Q15 laplacian (axis-symmetric, so harmless) and the
   same mapping for the **gradient**, where it is not. The factor-consistent check that would
   have caught it (linear field `q = x`: `grad_x` must be exactly 1) is missing from `selftest()`.
2. The gradient table printed in `artifacts/probe.json` shows coefficients 1/6 on diagonal
   neighbours, i.e. the gradient table is that of the *rotated* component, further evidence that
   component-to-offset bookkeeping (not the physics) is broken.
3. `shrinkage()` must guard `V > 0` and finite fields before taking the cube root.

## 6. What is nevertheless established (and can be built on next round)
* The OSS formulation is fully readable locally from lbmpy; no network is needed for future
  rounds of this anchor.
* Two convention-level differences that matter for any port, established **analytically** here
  and machine-checked against lbmpy's own expressions:
  - **Interface thickness:** OSS `phi = 1/2 (1 + tanh(2 s/W))` vs TensorLBM `psi = tanh(s/W)`.
    For the same label `W` the OSS interface is **half as thick**; the equivalence is
    `W_OSS = 2 W_TensorLBM`. Both are self-consistent (`sigma = kappa * int (dphi/ds)^2 ds`
    returns sigma in both conventions).
  - **Free-energy normalisation:** OSS `beta = 12 sigma/W, kappa = 1.5 sigma W` (phi in [0,1])
    vs TensorLBM `beta = 3 sigma/(2W), kappa = 3 sigma W/4` (psi in [-1,1]). Both give
    `sigma = (2 sqrt2/3) sqrt(beta kappa)` up to the affine map psi = 2 phi - 1.
* Lattice assignment differs: OSS D3Q15 (phase field) + D3Q19 (hydro) vs TensorLBM D3Q19 for
  both. The OSS split force (pressure + viscous + surface tension, divided by the local density)
  is the structural answer to the divergence that TensorLBM documents for its own Korteweg+Guo
  coupled step at 1000:1 - it is worth porting, but it was **not** validated this round.

## 7. Honest status
fail_stage = 3 (numeric operators); failure_reason = offset/axis bookkeeping in `apply_op`
plus a missing linear-field unit check; no metric gate was evaluated. Next round (round 2):
(1) add the `q = x` / `q = y` / `q = z` linear-field unit tests, (2) fix the axis mapping,
(3) re-run M1/M2/M3 with the thickness matching `W_OSS = 2 W_TensorLBM`, (4) then attempt the
D3Q19 split-force port for the constant-density Laplace case and only afterwards the 1000:1 case.
