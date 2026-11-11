# `rotating_w3` — certification gate for the rotating-frame interior-flow ladder

Node `P158W3-N01` (TensorLBM).  Waves 1-2 verified the *physics* of the rotating
frame (`rotating_interior_frame`: rotating annulus Couette + rotating plane
channel Ekman; `rotating_duct_frame_w2`: Guo/EDM force schemes, Ekman scaling
law, reverse Couette, rotating square duct).  This benchmark adds the
**certification layer** and one **negative result**, and does not re-run any
verified case.

| criterion | what is measured | how |
|---|---|---|
| A1 | Taylor-Couette azimuthal profile | 10 radial samples vs the exact `Ar+B/r` on the **effective** (staircase-corrected) geometry |
| A2 | dimensionless torque `C_M = T/(2 pi rho nu^2 L)` | profile identity `T(r)=2 pi rho nu L r^2 (du/dr - u/r)`, three grids + reverse rotation |
| A3 | rotating square-duct secondary flow | **cited** from the verified W2 record (provenance re-read programmatically; re-running a verified case is forbidden) |
| A4 | Ekman thickness `delta_E` | blind one-parameter fit; the *resolution limit* below the verified `delta >= 4` is measured here (`delta = 2, 3`) |
| A5 | Ekman spiral angle | wall-shear/geostrophic angle at `delta = 6, 8, 12, 16`, extrapolated to `1/delta -> 0` vs the 45 deg theory (the angle is a wall property, the first interior node is one link inside the layer) |
| A6 | grid-convergence order | Roache GCI on the **bulk shape residual** (deviation from the exact solution family, wall shells excluded) -- the nominal-radius error is non-monotone because the staircase puts the wall at the nearest half link |
| A7 | co-rotating null test | both cylinders at the same `omega`: exact solution is rigid-body, `B = 0`, every torque must vanish |
| (extra) | rotation-rate stability boundary | frame-rate sweep; the pivot is `Omega_f^2 r / c_s^2` (centrifugal acceleration vs the lattice scale), not `Ro` |

## Run

```bash
PYTHONPATH=src python benchmarks/verified/rotating_w3/run.py           # ~20 min
PYTHONPATH=src python benchmarks/verified/rotating_w3/run.py --quick   # smoke test
python benchmarks/verified/rotating_w3/make_figures.py
python benchmarks/verified/rotating_w3/make_doc.py
```

`run.py` writes `records/rotating_w3.json` (criteria + per-case numbers +
curve data) and `records/rotating_w3_fields.npz`; `make_doc.py` regenerates
`docs/ROTATING_W3_P158W3_N01.md` from the record, so no number in the report is
typed by hand.

## Negative result (why the wall treatment did not change)

Three sub-grid wall schemes were implemented on a *sub-grid plane Poiseuille
channel* (analytic solution known, wall 0.7 links away) and all three fail:
whole-population linear extrapolation diverges (boundary gain `1+q > 1`);
Bouzidi's split scheme degenerates to the half-way scheme in a steady state
(its `q >= 1/2` branch is a time extrapolation); non-equilibrium extrapolation
has no measurable effect (the non-equilibrium part is flat across the first two
nodes, so the stencil carries no wall-position information).  Consequences and
the measured numbers are in the module docstring of
`src/tensorlbm/rotating_certification.py`; the regression guard is
`tests/test_rotating_certification.py::test_subgrid_wall_error_scales_like_h`.