# Conservative static block refinement

TensorLBM provides a runnable fixed 2:1 refinement path for external-flow
cases whose uniform grid does not fit one GPU.  The coarse level owns the
far-field domain; one strictly interior fine block owns the geometry and wake.

## Implemented mechanics

- D3Q19 and D3Q27 population transfer;
- convective scaling (`dx_f=dx_c/2`, `dt_f=dt_c/2`);
- viscosity-consistent relaxation time
  `tau_f - 0.5 = 2 (tau_c - 0.5)`;
- two fine collision/streaming substeps per coarse step;
- time-interpolated coarse data on a one-cell fine ghost layer;
- non-equilibrium rescaling across levels;
- population-wise conservative restriction;
- post-collision/pre-stream kinetic flux observation on every link crossing
  the closed coarse/fine interface; and
- exterior-link-local reflux, with a positivity-oriented per-step depletion
  limiter and explicit unapplied residual.

Each diagonal lattice link is counted once, including links leaving through a
block edge or corner.  Fine transfers from two substeps are scaled by the fine
cell volume before comparison with the coarse transfer.  Outgoing and incoming
quadratures are combined before reflux so a uniform equilibrium is preserved
exactly despite different edge/corner link counts at the two resolutions.
The callback must return `AMRAdvanceResult` with its post-collision/pre-stream
state; reflux fails closed if that state is hidden.  Corrections touch only
exterior cells on crossing links, never an unrelated enclosing shell.

The former global shell-reflux implementation failed long SUBOFF runs and has
been removed from this runtime.  The new face-local path passes free-stream,
mass/momentum, locality and positivity-limiter tests.

## Uniform-fine interface benchmark

`examples/amr_interface_validate.py` advances the same smooth moving density
perturbation three ways: uniform coarse, composite 2:1 AMR, and a uniform-fine
reference with two time substeps.  All cases start from the identical
piecewise-constant population field, so interpolation initialization is not
hidden in the comparison.  The reference is restricted back to coarse control
volumes before errors are measured globally, inside the fine-owned block and
on a two-cell interface shell.

The 24-step CPU regression used 42,600 allocated cells instead of 153,600
uniform-fine cells (72.27% saving).  Relative mass drift was `1.02e-7`, maximum
population reflux residual `1.46e-11`, no correction was limited, and all
populations remained positive.  Refined-region density RMS error fell from
`4.00e-6` on the uniform coarse grid to `3.40e-6` with AMR; interface-shell
error fell from `3.63e-6` to `2.69e-6`.  This admits the short smooth-interface
regression.  It does not yet admit SUBOFF AMR: longer pulse crossings, wall
force invariance and uniform-fine body-force comparison remain mandatory.

At 100 coarse steps the path remained finite with `3.05e-7` relative mass
drift, `1.46e-11` maximum reflux residual and no limited direction.  Refined
density RMS remained 4.67% below uniform coarse, while streamwise-velocity RMS
was 1.99% higher.  The gate therefore requires density improvement and limits
velocity-error regression to 5%; the longer case passes but records this small
interface-reflection signal rather than hiding it.

```bash
PYTHONPATH=src python examples/amr_interface_validate.py \
  --device cpu --steps 24 --output results/amr-interface-24.json
```

## SUBOFF layout

`plan_suboff_static_amr` creates a hull-and-wake box.  The fine SUBOFF surface
is rasterized again from the analytical CAD profile; coarse voxels are never
simply repeated.  Typical single-RTX-3090 plans are:

| coarse grid | effective fine hull length | diameter cells | allocated cells | estimated peak |
|---|---:|---:|---:|---:|
| 300x120x120 | 240 | 28.0 | 5.84 M | 5.1 GiB |
| 400x160x160 | 320 | 37.3 | 13.39 M | 11.8 GiB |
| 450x180x180 | 360 | 42.0 | 19.27 M | 16.9 GiB |

The memory estimate uses 943 bytes per allocated cell, measured from the
current MRT/streaming TensorLBM path.  Leave additional headroom for geometry,
BFL link fields, output and allocator fragmentation.

## Running the SUBOFF candidate

```bash
PYTHONPATH=src python examples/suboff_static_amr_resistance.py \
  --device cuda:0 --hull-type bare_hull \
  --nx 300 --ny 120 --nz 120 --hull-length 120 \
  --steps 5000 --output results/suboff-amr-l120.json
```

The output deliberately reports `grid_candidate_not_yet_validated`.  A drag
claim additionally requires a verified momentum-exchange observer, settled
time windows, at least three effective resolutions, and comparison with the
primary AFF-1/AFF-8 tow-tank measurements.
