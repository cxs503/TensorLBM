# TensorLBM AMR interface literature audit

This note records which published multi-resolution LBM requirements are
implemented, which are only approximated, and which remain validation work. It
is a design-control document, not evidence that TensorLBM matches a commercial
solver.

## Primary references

- Lagrava, Malaspinas, Latt and Chopard, *Advances in multi-domain lattice
  Boltzmann grid refinement*, Journal of Computational Physics 231 (2012),
  4808–4822, [doi:10.1016/j.jcp.2012.03.015](https://doi.org/10.1016/j.jcp.2012.03.015).
- Guzik, Weisgraber, Colella and Alder, *Interpolation methods and the accuracy
  of lattice-Boltzmann mesh refinement*, Journal of Computational Physics 259
  (2014), 461–487,
  [doi:10.1016/j.jcp.2013.11.037](https://doi.org/10.1016/j.jcp.2013.11.037).
- Dorschner, Frapolli, Chikatamarla and Karlin, *Grid refinement for entropic
  lattice Boltzmann models*, Physical Review E 94 (2016), 053311,
  [doi:10.1103/PhysRevE.94.053311](https://doi.org/10.1103/PhysRevE.94.053311).

## Requirement-to-code map

| Published requirement | TensorLBM implementation | Current evidence/status |
|---|---|---|
| Convective 2:1 time/space scaling and non-equilibrium rescaling | `convective_refined_tau` and `rescale_nonequilibrium` | Unit-tested; pulse benchmark conserves mass |
| Remove fine scales before fine-to-coarse transfer | 2×2×2 volume averaging plus optional second-order Hermite regularization | Implemented and enabled in current SUBOFF candidates |
| Time interpolation at coarse-to-fine ghost fills | Linear interpolation between the two parent time states at every fine substep | Implemented and schedule-tested |
| Spatial reconstruction at cell-centred ghost sites | Cell-centred trilinear interpolation; optional Hermite regularization of prolongated non-equilibrium | Implemented; regularized variant pulse-admitted |
| Population-wise conservative interface transport | Link-local kinetic transfer registers and positivity-limited reflux | Implemented; per-interface residual is recorded |
| Independent force must not sample numerical interface treatment | Geometric CV/filter clearance gate including the radius-one streaming source stencil | Implemented; fails closed in preflight |
| High-Re interface stabilization | Cumulant collision, filtered restriction, optional regularized prolongation and kinetic-only transition filter | L90/L120 A/B campaign still active; not yet promoted |
| Geometrically similar refinement and observers across a grid sequence | v10 exact 3:4:5 inner-patch, CV and wall-sampling scaling | Implemented in launcher and convergence gate |

## Important distinctions

Lagrava et al. use a node-centred interface and show that higher-order spatial
reconstruction and filtering of fine-grid non-equilibrium content matter at
high Reynolds number. TensorLBM is cell-centred, so their four-point midpoint
formula must not be copied blindly. The current 2×2×2 restriction is already a
coarse-width spatial low-pass, and Hermite regularization additionally removes
unresolved kinetic modes.

Guzik et al. treat the cell-centred case and emphasize space-time ghost
interpolation together with population-direction conservation and stream
register correction. TensorLBM has these structural pieces, but the present
trilinear coarse-to-fine reconstruction still needs an explicit manufactured
population-flux conservation benchmark; a small reflux residual alone does not
prove local interpolation accuracy.

Dorschner et al. demonstrate that a correctly formulated entropic stabilizer
can adapt at resolution transitions. TensorLBM contains an experimental KBC
collision path, but it is not a SUBOFF production default: its recovered shear
viscosity and interface behaviour require an independent decay/Poiseuille audit
before it can replace the cumulant path.

## Promotion gates

No interface treatment becomes a production default until all of the following
hold:

1. the uniform-fine pulse comparison remains mass conservative and within its
   density/velocity error targets;
2. an L90 continuation survives beyond the historical 1400–2400-step failure
   interval with positive populations and zero material limiter use;
3. primary and auxiliary CV flux stencils remain outside every filter shell;
4. the complete L90/L120/L150 configuration is geometrically and temporally
   similar;
5. force stationarity, independent observers and experimental comparison pass
   without using empirical correction factors.
