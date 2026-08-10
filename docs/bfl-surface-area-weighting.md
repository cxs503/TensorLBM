# BFL local surface-area weighting

Wall stress is an area integral.  Assigning the same share of an analytical
total area to every boundary node gives the correct total only for uniform
stress and distorts curved-surface shear distributions.  It also cannot add
sail and control-surface area for AFF-8.

`bfl_surface_area_weights` uses the axial BFL links as projected surface faces.
For a patch with normal `n`,

`dA = N_axial / (|nx| + |ny| + |nz|)`.

This is exact for an axis-aligned plane and recovers `sqrt(2)` for a patch at
45 degrees with one x and one y projection.  A supplied analytical bare-hull
area calibrates only the global scale; it does not flatten the local
orientation distribution.  The active six-neighbour wall mask is explicit,
so diagonal-only BFL nodes that do not receive wall traction are excluded from
both area and coverage diagnostics.

For AFF-1, analytical SUBOFF normals and analytical bare wetted area are used.
For AFF-8, a bare-hull voxel-gradient proxy first determines the scale, then
the same scale is applied to the full hull+sail+stern-plane proxy.  A coarse
L24 composition probe gives 179.79 square lattice units for the calibrated
bare hull and 195.41 for the full geometry; all 186 active full-geometry nodes
receive nonzero area.  These coarse values verify composition, not physical
appendage area convergence.

Direct SUBOFF checkpoint/result schema v4 records the weighting method and
diagnostics.  Earlier force histories with uniform per-node area are not
restart-compatible with v4 and cannot be mixed into the new wall-shear
sequence.
