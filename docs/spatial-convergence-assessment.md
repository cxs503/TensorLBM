# Spatial convergence assessment

`tensorlbm.spatial_convergence.assess_spatial_convergence` fits a declared
resolution sequence to

`phi(N) = phi_infinity + a N^(-p)`.

It requires at least three strictly increasing characteristic resolutions and
reports:

- whether the sequence is monotonic;
- fitted observed order `p`;
- extrapolated infinite-resolution value;
- relative distance of the finest result from that limit; and
- relative RMS residual of the fit.

The order is found by bounded nonlinear search while the limit and amplitude
are solved by linear least squares at each candidate order.  Admission is
fail-closed for non-finite or non-monotonic sequences and has explicit limits
for minimum order, finest-grid uncertainty and fit residual.  A manufactured
`phi=1.25+3/N^2` sequence recovers second order and the exact limit in tests.

This is a discretisation-evidence tool, not an experiment-fitting tool.  The
same solver, domain proportions, boundary treatment, physical parameters,
averaging rule and wall exchange height in physical units must be held fixed
across the input sequence.  TensorLBM will use it for L256/L384/L512 flat
plates, sphere resolution studies and ultimately AFF-1/AFF-8; two-grid
agreement alone is never promoted as a convergence claim.
