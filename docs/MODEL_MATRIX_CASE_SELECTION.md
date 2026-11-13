# Case selection for the model matrix: one falsified hypothesis

## Hypothesis (H1, withdrawn)

*"A shear wave at `U0=0.2` (Ma≈0.35) exposes the O(Ma²) compressibility error of
each collision operator, so the same analytic reference at two Mach numbers
separates BGK from regularized/cascaded/cumulant models."*

## Measurement (falsified)

The finite-Mach shear case was implemented exactly like the low-Mach one, only
with `U0=0.2`, and run over the gate models (res=16, tau=0.6/0.8):

| model | nu err % at U0=0.01 | nu err % at U0=0.2 |
|---|---|---|
| d3q19/bgk | +1.608229 | +1.608229 |
| d3q27/kbc | +1.201 | +1.201 |

The two coefficients agreed to **6 significant digits** — i.e. the measured
viscosity carried *no* Mach-number information at all.

## Root cause

For `u = (u_x(y), 0, 0)`, `(u·∇)u = 0` identically, and the linear-response
viscosity of a collision operator is Mach-independent by construction. The
planned discriminator cannot exist for a shear wave, at any amplitude.

## Replacement (adopted)

`tg_decay` (Taylor-Green vortex), whose velocity field *does* have a non-zero
convective term and therefore a genuine O(Ma²) compressibility error. It is
registered as an **advisory** case: deviations are recorded but do not gate the
build, because the incompressible analytic solution is not exact at finite Ma.

Measured (res=24, tau=0.6): nu error +0.54/+0.49/+0.32 % for Ma
0.035/0.087/0.173 (BGK); cascaded/KBC +0.44/+0.39/+0.24 %. The residual is not
compressibility-dominated at these Mach numbers — an honest, if less dramatic,
conclusion than the withdrawn hypothesis.

## Lesson for the matrix

A case is only worth a column if a *wrong* model can fail it. Both cases kept in
the matrix are checked for that property: `test_negative_control_wrong_viscosity_is_rejected`
and the `fit_r2` criterion (a non-exponential decay is rejected even if the
fitted slope is right).
