# Compressed fluid-node table as a production backend of the porous solver

Anchor: `XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE` (follow-up of mission m22754).
Scope: make the verified compressed layout a *callable part of the main solver*,
not a stand-alone cross-validation script, and register the resulting numbers.

## 1. What was missing after m22754

m22754 shipped the transplanted kernel (`src/tensorlbm/porous/fluid_node_table.py`)
and a stand-alone XVAL harness, verified as a level-2 artefact. It had

* no caller in the solver: nothing outside `xval/` and its own test imported the
  module, so the layout could not be used by any production path,
* no reference-vs-got table with a reproduction command,
* one declared run artefact (`figures/xval_compressed_node_table.png`) was never
  generated.

## 2. What this mission adds

`tensorlbm.porous_media.run_single_phase_darcy(config)` — a single-phase
periodic-slab Darcy channel with a selectable lattice layout, inside the porous
media solver module (`src/tensorlbm/porous_media.py`), exported as
`tensorlbm.run_single_phase_darcy` / `tensorlbm.SinglePhaseDarcyConfig`:

| `layout` | populations | wall rule | role |
|---|---|---|---|
| `compressed` (default) | fluid nodes only, `f.shape == (9, K)` | half-way, fluid-node push scatter through the neighbour table | **the new backend** |
| `dense_halfway` | full lattice | half-way, per-direction reflection on top of `tensorlbm.solver.stream` | independent control of the same physics |
| `dense` | full lattice | the shipped update `stream` + `bounce_back_cells` | documents the existing convention |

Every arm reports the superficial Darcy permeability `k = nu <u_x> / g`
(`<.>` over all lattice sites) against the analytic periodic-slab value
`k_ref = (H / ny) H^2 / 12`.

Reproduction:

```
PYTHONPATH=src python3 benchmarks/xval_compressed_node_table_prod/run_bench.py --cases 12x8,24x16
```

## 3. Results (g = 1e-4, tau = 0.8, nu = 0.1, porosity H/ny = 2/3)

| case | steps | k_ref (analytic) | k_got | error | converged |
|---|---|---|---|---|---|
| `compressed_ny12_h8` | 1500 | 3.555556e+00 | 3.540000e+00 | -0.4375 % | yes |
| `dense_halfway_ny12_h8` | 1500 | 3.555556e+00 | 3.540000e+00 | -0.4375 % | yes |
| `dense_ny12_h8` | 1800 | 3.555556e+00 | 4.662222e+00 | +31.1250 % | yes |
| `compressed_ny24_h16` | 5600 | 1.422222e+01 | 1.420667e+01 | -0.1094 % | yes |
| `dense_halfway_ny24_h16` | 5600 | 1.422222e+01 | 1.420667e+01 | -0.1094 % | yes |
| `dense_ny24_h16` | 6150 | 1.422222e+01 | 1.641732e+01 | +15.4343 % | yes |

* **G1 representation identity** (compressed vs `dense_halfway`):
  `k` relative difference 6.90e-15 (H=8) and
  2.63e-15 (H=16); profile max relative difference
  9.11e-15 / 5.45e-15.
  The compaction plus neighbour-table streaming is therefore a representation
  change, not a physics change.
* **G2 physics**: compressed arm `-0.4375 %`
  (H=8) and `-0.1094 %` (H=16) — inside the
  3 % acceptance band, and the error decreases with resolution as a
  second-order half-way bounce-back should.
* **G3 wall convention (finding)**: the shipped dense update is **not** the same
  wall rule. `bounce_back_cells` reflects *at the solid node*, so the no-slip
  plane sits on the solid-node centre and the effective channel width is
  `H + 1` instead of `H`. Measured bias: `+31.125 %`
  (H=8) and `+15.434 %` (H=16); the ratio
  2.0166 matches the `H_big/H_small = 2` predicted
  for a fixed half-cell offset. The permeability-implied effective widths are
  8.756 (H=8) and
  16.784 (H=16), both between `H + 0.5` and
  `H + 1`.
  This is *not* a bug in the two-phase entry points: their analytic companion
  `_two_phase_poiseuille_analytical` also puts its walls on node centres, so the
  pair is self-consistent. It does mean that any single-phase use of the shipped
  dense update must either widen the analytic reference by one lattice unit or
  switch to the half-way rule as done here.

## 4. Tests

`tests/test_porous_darcy_layouts.py` — 16 tests: identity to round-off, analytic
permeability at H = 8/10/12, the shipped-convention bias (magnitude *and*
effective width, so a silent change of the wall rule fails), reproduction of the
m22754 XVAL record (-0.1094 % at H=16), determinism, mass conservation
(`rho = 1 +/- 1e-12`), artefact writing, configuration validation and the
library-level export.

## 5. Limits / not run

* Single-phase only. The two-phase Shan-Chen porous entry points
  (`run_two_phase_poiseuille`, `run_capillary_invasion`, `run_porous_drainage`)
  still run on the dense layout; wiring the compressed layout into the
  two-component kernels is not attempted here (the interaction force needs the
  neighbour stencil per component, a larger change than this mission allowed).
* Steady state is reached by a `max|u(n) - u(n-check_every)| / max|u(n)| < tol`
  criterion, not by a fixed step count; every arm reports its own step count.
* Fitting the full `1/H` ladder at H=32 was queued on the cluster but not
  dispatched inside this mission's window; the two-point ladder H=8/16 already
  determines the offset hypothesis (ratio 2.017 vs 2.0).
* No GPU run: the transplanted OSS kernel is CUDA, this port is a CPU torch
  implementation of the same semantics.
