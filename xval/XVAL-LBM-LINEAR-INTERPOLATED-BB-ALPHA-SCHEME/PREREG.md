# PREREG — XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME (written before any run)

## Decision this experiment must support
Does TensorLBM's 2-point interpolated bounce-back, once generalised to the OSS
`alpha`-scheme family (pylbm `LinearBounceBack` + post-collision correction),
reproduce the OSS accuracy claim on a force-driven plane-Poiseuille channel —
and under which carrier (BGK vs TRT) does it clear the owner acceptance line?

Alternatives (arms): `halfway` production operator; `bfl` = alpha-scheme with the
BFL rule, no correction (bit-equivalent to the operator shipped before this
mission); `bfl_par` = same + post-collision correction `K_hat = 1 - alpha_hat`;
`cli` = central linear interpolation (alpha_hat = 1, correction identically 0).

## Hypotheses
H1 (transplant fidelity, OSS claim): with the OSS exactness recipe
(`Lambda+ Lambda- = delta^2/2`, force added post-collision with isotropic
weights, reported velocity = stored momentum − F/2, pylbm
`tests/test_linear_bounce_back.py::poiseuille`) the parametrized members
(`bfl_par`, `cli`) are exact: `l2_rel < 1e-10` at every delta.
H2 (carrier is the limiting factor): on BGK the same closure stays above the
acceptance line for at least one delta, i.e. the closure alone does not fix the
wall position — it is the TRT magic product that does.
H3 (necessity of the correction): without the `K_hat` term at fixed
`Lambda+ Lambda- = delta^2/2` the scheme is no longer exact (`l2_rel > 1e-3`).

## Metrics (predeclared, no post-hoc choice)
* primary `l2_rel = ||u_num - u_ref||_2 / ||u_ref||_2` on the fluid nodes, with
  `u = momentum - F/2` (OSS convention) — acceptance `l2_rel < 0.15` (owner).
* `max_rel = max|u_num - u_ref| / max|u_ref|`.
* `w_bot, w_top`: effective no-slip distance from the first/last fluid node,
  from the roots of a least-squares parabola fitted to the numerical profile
  (`w = 0` means the wall is placed exactly at `delta`); `fit_res` = relative
  residual of that fit (flags non-parabolic contamination).
* `q_err` = (sum u_num - sum u_ref) / sum u_ref over the same nodes.

## Controls / confounders
* `delta = 0.5` control: all closures collapse onto half-way bounce-back
  (BFL `alpha = 2`, CLI `alpha = 2`), so `max|u(bfl) - u(halfway)| < 1e-10`
  must hold; a larger difference means the operator, not the closure, differs.
* wrap control: streaming is periodic in x and *open* in y (outermost solid rows
  reset to rest) so the two walls cannot feed each other.
* one lattice spacing, one rho, one F (`u_max ~ 2e-2`, Ma ~ 3.5e-2, Re_H < 4):
  the comparison is a wall-model comparison, not a resolution study.
* convergence: steady when `max|du| < 1e-12 * u_max` over 200 steps after 4000
  steps; otherwise `converged = false` and the point is reported as such.

## Stopping rule
Sweep is complete at 3 carriers x 4 closures x 5 deltas x 2 resolutions; no
threshold is changed after looking at results. If H1 fails, the report states
`honest FAIL` for the exactness arm and keeps the measured numbers.
