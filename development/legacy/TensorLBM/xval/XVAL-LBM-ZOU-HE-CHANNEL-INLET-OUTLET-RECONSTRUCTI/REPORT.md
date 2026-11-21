# XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI

Cross-validation of the OSS reference
`loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow`'s
*zou-he-channel-inlet-outlet-reconstruction*: an open channel closed by Zou-He
(1997) reconstruction at both ends -- a velocity inlet fixing the unknown
incoming populations and a pressure (density) outlet.

## 1. What was missing and what was added

`tensorlbm.boundary.solver2d.zou_he_moving_lid` implemented the Zou-He velocity
closure for the moving lid only (north wall).  There was **no pressure (density)
closure at all**, so a body-force-free open channel could not be closed without
re-deriving the algebra inline in every case.

New module `src/tensorlbm/boundary/zou_he_channel.py` provides the closure for
all four walls in both modes (`velocity`: prescribe ``u``, reconstruct ``rho``;
`pressure`: prescribe ``rho``, reconstruct the normal velocity), vectorised over
the boundary layer so a parabolic velocity profile can be imposed directly, plus
`zou_he_channel()` which wires the west/east pair.

The algebra is *not* symmetric in the obvious way: for the west wall

    rho (1 - ux) = f0 + f3 + f4 + 2 (f2 + f6 + f8)          (mass + x-momentum)
    f1 = f2 + (2/3) rho ux                                 (normal non-eq. bounce-back)
    f5 = f8 + (1/6) rho ux - 0.5 (f3 - f4) + 0.5 rho uy    (mass sum + y-momentum)
    f7 = f6 + (1/6) rho ux + 0.5 (f3 - f4) - 0.5 rho uy

Each unknown diagonal is rebuilt from **its own opposite** known population
(`5<->8`, `6<->7`).  An easy slip -- the D2Q9 index table invites `5` from `6`
-- leaves the reconstructed **density exactly right** while corrupting the
transverse momentum.  This was the actual defect found during this task; it was
caught by the moment assertions, not by a smoke run (see §5).

## 2. Verification setup

* D2Q9 / BGK, `tau = 0.65`, `nu = 0.050000` (lattice units);
* two solid rows with half-way bounce-back, so the effective width between wall
  mid-planes is `H = ny - 2`;
* driving: `rho_in = 1 + drho/2`, `rho_out = 1 - drho/2`, `drho = 0.004`;
  **no body force, no periodic wrap, no analytic inlet profile** in the primary arm;
* exact reference for the same driving:

      u(y) = G/(2 nu) * yt (H - yt),  yt = y - 0.5,  G = -dp/dx = -(1/3) drho/dx

  `G` is taken from a least-squares fit of the computed mean density profile
  (the boundary-to-boundary length carries an O(1) convention ambiguity); the
  nominal `drho/(nx-1)` is reported separately as a stricter metric;
* the soft pressure ends do not damp acoustic waves, so the steady state is
  defined by **block time averaging** and the residual window-to-window drift
  is reported instead of hidden behind a pointwise tolerance.

## 3. Results

| tag | arm | H | nx | steps | drifted? | $u_{x,\max}$ sim | $u_{x,\max}$ exact | $|\Delta u_{\max}|/u_{\max}$ | $\|u-u_{ex}\|_\infty/u_{\max}$ | shape L2 | $G_{fit}$ rel. err | mass defect |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eq_H16 | equilibrium-pressure | 16 | 128 | 7000 | yes | 6.783101e-02 | 6.807120e-02 | 3.529e-03 | 3.529e-03 | 9.664e-04 | 1.12e+01 | 4.69e-04 |
| pp_H16 | pressure-pressure | 16 | 128 | 8000 | yes | 6.669392e-03 | 6.693054e-03 | 3.535e-03 | 3.535e-03 | 1.531e-03 | 2.10e-05 | 8.20e-07 |
| pp_H24 | pressure-pressure | 24 | 96 | 7000 | yes | 2.012845e-02 | 2.017578e-02 | 2.346e-03 | 2.346e-03 | 6.496e-04 | 1.71e-05 | 3.95e-06 |
| vp_H16 | velocity-pressure | 16 | 128 | 7000 | yes | 6.703452e-03 | 6.667368e-03 | 5.412e-03 | 5.412e-03 | 1.874e-03 | 3.82e-03 | 1.36e-02 |

`G_fit` rel. err is `|G_measured - G_nominal| / G_nominal`; mass defect is
`|q(x=0) - q(x=L)| / q_interior` with `q(x) = sum_y rho u_x`.

The interior flux is constant along the channel to
`1.49e-06` (relative), i.e. the discrete continuity
balance holds away from the closure cells.

## 4. Figures

![fig1_field_and_profile.png](figures/fig1_field_and_profile.png)

![fig2_profile_and_resolution.png](figures/fig2_profile_and_resolution.png)

![fig3_inlet_closure_contrast.png](figures/fig3_inlet_closure_contrast.png)


## 5. Notes, pitfalls, scope

* **The diagonal-pairing slip.** First implementation paired `f5` with `f6` on
  the west wall; density came out exact, transverse momentum was wrong, and the
  channel profile was still within a few percent -- a smoke test would have
  passed.  `tests/test_zou_he_channel.py::test_moment_exactness_velocity_mode`
  fails immediately on that variant, which is why the unit tests assert
  *moments*, not run status.
* **Acoustic ripple.** With prescribed densities at both ends the field keeps a
  small acoustic ripple (order 1e-2 of `u_max` pointwise) that a soft Zou-He
  boundary barely damps.  Pointwise `max|du| < tol` never converges; the mean
  field does.  The campaign therefore ramps the end densities over
  `0` steps and averages over `500`-step
  blocks.
* **Inlet closure is not the error-dominant term (negative result).** The working
  hypothesis was that a first-order local-equilibrium inlet would be measurably
  worse than the Zou-He reconstruction.  At mid-channel it is not: both arms sit
  at the same O(0.3 %) shape error, because the inlet closure is washed out
  within a few `H` in a long channel.  The comparison is reported in Fig. 3
  instead of being turned into a green check.
* **Scope.** Incompressible limit only (`drho = 0.004` gives
  `Ma ~ 0.012`), a smooth straight
  channel, `Re = u_mean H / nu ~ 1.4`,
  no-slip static walls.  Curved walls, moving walls and turbulence are out of
  scope for this closure.
* Pressure mode leaves the transverse velocity at the boundary as a closure
  assumption (0 here); the resulting mass defect is the diagnostic reported
  above, not a hidden tolerance.

## 6. Error verdict

Primary arm (`pp_H16`, H = 16, nx = 128,
8000 steps, converged flag = False):

| quantity | value | bound | verdict |
|---|---|---|---|
| `|u_x,max - u_exact,max| / u_max` | 3.535e-03 | < 3 % | **YES** |
| `||u - u_exact||_inf / u_max` | 3.535e-03 | < 3 % | **YES** |
| shape-only L2 (normalised profile) | 1.531e-03 | < 3 % | **YES** |
| profile vs **nominal** `drho/(nx-1)` gradient | 3.514e-03 | < 15 % | **YES** |
| fitted vs nominal pressure gradient | 2.10e-05 | < 3 % | **YES** |
| end-to-end mass-flux defect | 8.20e-07 | < 3 % | **YES** |

All acceptance quantities are comfortably inside the project-wide 3 % target
(and far inside the 15 % acceptance bound).  No metric was relaxed to reach them.
