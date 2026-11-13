# XVAL-LBM-DIFFUSE-INTERFACE-WETTING-VIA-WALL-ORDER-PARAMETER

Cross-validation of TensorLBM's diffuse-interface wetting against the open-source
implementation `ebinan92/Fingering_dynamics` (`lattice_boltzmann/validation.py`,
commit `150a5739`, 2019-02-28; the repository ships no LICENSE file, the file is
vendored at `xval/LBM_WALL_ORDER_PARAMETER_WETTING/oss_src/` **as the reference
source only**).

## (1) Problem and reference provenance

The reference imposes wettability with a **wall order parameter**: a ghost layer of
the phase field `psi_wall_list` is stacked above and below the fluid domain and every
finite-difference stencil of the order parameter (the isotropic 9-point gradient and
Laplacian that build the chemical potential)

    mu = a psi (psi^2 - 1) - kappa lap(psi),   a = 2 kappa / xi^2,  kappa = (3/4) sigma xi

reads that ghost value. There is no wetting potential, no contact-angle input: the
**static contact angle is an emergent function of one scalar `psi_w`**. The reference
ships a picture (`wettability_validation.png`) but no measured angle and no law.

Question of this XVAL: *can TensorLBM reproduce that mechanism, and what is its
quantitative contact-angle law `theta(psi_w)`?*

## (2) Method and implementation

New capability `src/tensorlbm/wetting/wall_order_parameter_2d.py`:

* `grad_laplacian` — the reference's isotropic 9-point stencils (axis weight 4,
  diagonal 1, `1/(12 dx)` / `1/(6 dx^2)`), with the two ghost rows carrying `psi_wall`
  (lower) and `psi_wall_top` (upper). Unit-tested to be exact for quadratics and linear
  fields, and to show the ghost entering only the wall-adjacent rows.
* `step` — a faithful port of the reference's **two-distribution scheme**: momentum `f`
  with the Korteweg force `F = mu grad psi` through the Guo term, order parameter `g`
  with the reference equilibrium (mobility term `Gamma mu/cs^2`, rest-direction
  `-(1-w_0)` correction) and half-way bounce-back for *both* distributions.
* `relax_to_equilibrium` — mass-conserving Allen-Cahn descent `psi -= omega (mu - <mu>)`
  on the *same* free energy, stencils and wall ghost. Its fixed point `mu = const` is
  the *same* equilibrium condition as Cahn-Hilliard (`div(M grad mu) = 0`, no-flux
  walls), but it is robust at the interface widths of interest; it carries the
  quantitative law (see (5) for why the LBM closure cannot).
* `contact_angle_from_field` with two independent estimators: inversion of the
  Laplace-cap relation from the **conserved excess area** and the apex height
  (`tan(theta/2) = h/a`, area form), and a circle fit of the `psi = 0` level set as a
  geometric cross-check. Wall plane = halfway bounce-back plane `y = -1/2`.
* `analytic_contact_angle` — the Young law implied by the Dirichlet wall value,
  `cos(theta) = (3 psi_w - psi_w^3)/2` (`psi_w = 0 -> 90 deg`, `+1 -> 0 deg`,
  `-1 -> 180 deg`), which is the predeclared reference.

Arms: `port` (TensorLBM, above), `lbm` (faithful LBM port, same geometry),
`oss` (vendored reference code). Harness `run_xval.py`, predeclaration
`predeclared.json`, raw data `runs/<arm>/result.json` + `psi.npz`.

## (3) Parameters (all lattice units, dimensionless groups complete)

| quantity | value |
|---|---|
| grid | ny = 56, nx = 80 (x periodic) |
| initial cap | circular, radius 16, centre on the lower wall |
| interface width `xi` | 3 (`xi/dx` = 3; swept 2, 3, 4) |
| surface tension `sigma` | 0.05 (model value `4 kappa/(3 xi)` = 0.05, verified) |
| `kappa`, `a` | 0.1125, 0.025 |
| mobility / descent step | Allen-Cahn `omega` = 0.4 (`Gamma` = 0.1, `tau_g` = 0.7 in the LBM arm) |
| `tau_f`, `nu` | 0.8, 0.1 (LBM arm only) |
| wall order parameter `psi_w` | -0.50, -0.35, 0, +0.35, +0.50, +0.70 |
| upper wall | `psi_w_top` = -1 (far field of the wetting phase) |
| dimensionless numbers | Re, Ca, Pe are **not** control parameters of a static contact angle; the only groups are `xi/dx` and `psi_w`. Capillary number Ca = 0 (static), Bond = 0 (gravity absent) |

## (4) Comparison table (reference = analytic Young law; gate 15 %)

| `psi_w` | `xi` | theta port [deg] | theta ref [deg] | err | pass |
|---|---|---|---|---|---|
| -0.50 | 3 | 133.18 | 133.43 | -0.19 % | yes |
| -0.35 | 3 | 118.60 | 120.24 | -1.36 % | yes |
|  0.00 | 3 | 88.11 | 90.00 | -2.10 % | yes (exact control) |
| +0.35 | 3 | 59.86 | 59.76 | +0.15 % | yes |
| +0.50 | 3 | 47.17 | 46.57 | +1.29 % | yes |
| +0.70 | 3 | 20.11 | 28.54 | -29.52 % | **no** |
|  0.00 | 2 | 89.04 | 90.00 | -1.07 % | yes |
|  0.00 | 4 | 86.70 | 90.00 | -3.67 % | yes |
| +0.35 | 2 | 60.87 | 59.76 | +1.85 % | yes |
| +0.35 | 4 | 58.55 | 59.76 | -2.04 % | yes |

median |err| = 1.60 %, 9/10 rows inside the 15 % gate; mass error < 1e-12 in every row.

**Reference arm (`oss`)** — the vendored code cannot supply a numeric angle:
* as committed it **does not run**: `getNabla_psix/psiy/psi2`, `getMix_tau`,
  `updatePsi`, `updateF`, `updateG` are defined at module level but called as methods
  (`AttributeError: 'Compute' object has no attribute 'getNabla_psi2'`). Documented
  patch P1 binds them (no formula change).
* its `main()` puts the drop at the **box centre** (`create_circle`), so no contact
  line exists; patch P2 rests the drop on the lower wall (geometry only).
* with P1+P2 at the xval resolution (32x48) the scheme is finite for 100 steps
  (`psi in [-1.19, 1.08]`, overshoot from the sharp step initial condition) and goes
  **NaN by step 200**; a run at its own published box (200x250, 1200 steps,
  `psi_w in {0, +0.35, -0.50}`) was submitted to the cluster queue (job `cj1656`,
  still queued when this report was written) — the gate therefore rests on the
  analytic reference, not on the OSS numbers.

## (5) Convergence and sensitivity (>= 2 points)

* **Resolution in `xi`** (table, rows 3/7/8 and 4/9/10): theta changes by 2.7 % over
  `xi` = 2 -> 3 -> 4 at `psi_w = 0` and by 3.9 % at `psi_w = +0.35` (gate 10 %), i.e.
  the law is resolution independent to a few per cent.
* **`psi_w` sweep** (6 points, -0.50 ... +0.70): monotone, and the deviation from the
  Young law grows only for strong wetting (+0.70), where the cap degenerates into a
  pancake (`h/a -> small`, the area inversion and the circle fit both become
  ill-conditioned: the two estimators differ by more than 5 deg there).
* **Iteration count**: all rows ran to the 30000-iteration cap with the residual drift
  ~1e-5 per 200 iterations (declared tolerance 1e-8 was **not** reached; `converged =
  false` is flagged in every `result.json`). Re-running `psi_w = +0.35` from a
  converged field changes theta by < 0.05 deg, so the angle is stationary well before
  the tolerance.
* **Spurious diffusion of the reference closure (defect found by this XVAL)**: in the
  `lbm` arm (faithful port, `Gamma` = 0.1, 1200 steps) the cap peak falls from 1.0 to
  0.054 (`dissociation_ratio` = 0.054, `fig3`) while the order-parameter mass is
  conserved to 1e-12. Cause: the second moment of the order-parameter equilibrium
  carries `cs^2 psi`, so the `g`-equation contains an unphysical diffusion
  `-(tau_g - 1/2) cs^2 lap(psi)` with the *same* magnitude scale as the physical
  mobility term; the resulting steady state satisfies `mu - psi/Gamma = const`, i.e. a
  linear term of relative size `1/Gamma` deforms (and for `Gamma` ~ 1 destroys) the
  double well. `Gamma >= 10` removes the effect but is numerically unstable at
  `dt = 1` here. This is why the port carries the law with the Allen-Cahn equilibration
  (alternative A3), which shares the wall condition but not the closure.

## (6) Failure modes, limitations, artefacts

* `|psi_w| -> 1`: theta -> 0/180 deg, where a *relative* error is ill-conditioned and
  the cap degenerates into a film; the sweep is deliberately bounded at 0.7.
* The area-based inversion assumes a circular arc; the diffuse tanh tail adds
  ~perimeter x `xi` of area, biasing the angle by a few degrees at `R/xi` = 15
  (characterised by the xfailed synthetic-cap test: 60/90/120 deg -> 57.8/88.5/116.9).
* The upper wall is a truncation; it sits 40 lattice units above the wall plane.
* `psi_w` is a *model* parameter, not a physical contact angle: the mapping to a
  physical wetting angle must go through the law in (4) or an equivalent calibration.
* Static equilibrium only; dynamic wetting (Cox-Voinov) and hysteresis are out of scope.

Artefacts: `xval/LBM_WALL_ORDER_PARAMETER_WETTING/{predeclared.json, run_xval.py,
oss_runner.py, make_figures.py, summary.json, runs/**, figures/fig1..fig3}`;
tests `tests/test_wall_order_parameter_wetting.py` (15 passed, 1 xfailed);
`docs/XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING.md` remains the *other* (colour-gradient)
wetting route — the two are complementary, not duplicates.

## (7) Diff list: this repo vs the OSS reference, and the porting trade-offs

| aspect | OSS `ebinan92/Fingering_dynamics` | TensorLBM port | trade-off |
|---|---|---|---|
| wall condition | ghost layer `psi_wall_list` stacked above/below, read by all `psi` stencils | identical (two ghost rows, isotropic 9-point stencils), plus a separate `psi_wall_top` | kept verbatim; the top value is separated so the far field can be neutral (`-1`) instead of being forced to the same wetting value (the OSS applies one value to both walls) |
| momentum equilibrium | He/Liu incompressible pressure split (`A0`, `A1_8`) with a power-law viscosity that its own source disables | standard weakly compressible D2Q9 BGK, Newtonian | static angle unaffected (momentum is a diagnostic; spurious currents reported). The OSS non-Newtonian branch is dead code (`''''  ''` comment in the source) |
| force | `getLarge_F`: Guo term with `F = mu grad psi`, recomputed per direction | Guo term, `F = mu grad psi`, single evaluation | same physics, ~9x fewer stencil evaluations |
| order-parameter closure | mobility term `w_i Gamma mu/cs^2` + rest correction; second moment `cs^2 psi` | selectable: `reference` (verbatim) and `isotropic`/`zero_second_moment` variants | **the reference closure is not stationary** for a flat interface nor for a cap (see (5)); the quantitative law therefore uses the Allen-Cahn equilibration |
| initial condition | sharp 0/1 step (`create_circle`) | `tanh((R - r)/xi)` equilibrium profile | the sharp step produces a first-step overshoot `psi in [-1.19, 1.14]` in the reference; the smooth start is a stabilisation, not a physics change |
| boundaries | periodic in x, half-way bounce-back in y for `f` and `g` | identical | — |
| drop geometry | drop at the box centre (no contact line) | cap resting on the lower wall | required to have a contact angle at all (patch P2) |
| validation | a `pcolor` picture only | measured `theta(psi_w)` law vs analytic Young law, 15 % gate, resolution sweep | the XVAL supplies the measurement the reference lacks |

## (8) Decision

Adopt alternative **A3**: the wall-order-parameter wall condition (the OSS mechanism)
is ported into `tensorlbm.wetting.wall_order_parameter_2d`, `theta(psi_w)` is fixed by
`cos(theta) = (3 psi_w - psi_w^3)/2` with a measured agreement of 1.6 % (median) inside
a 15 % gate for `|psi_w| <= 0.5`; the reference's order-parameter closure is **not**
adopted (spurious `nabla^2 psi` diffusion, quantified), and the strong-wetting branch
`psi_w >= 0.7` is reported as out of the validated range.
