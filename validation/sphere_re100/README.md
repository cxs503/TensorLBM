# VAL-SUITE: 3-D Sphere Flow at Re=100

Validation case for 3-D uniform flow past a stationary sphere at Reynolds number Re=100,
using the TensorLBM D3Q19-BGK lattice Boltzmann method with far-field boundary conditions.

## Physical Setup

| Parameter | Value |
|-----------|-------|
| Reynolds number | Re = 100 |
| Geometry | Sphere (3-D) |
| Flow | Uniform stream at u_in = 0.1 (lattice units) |
| Reference C_D | Schiller-Naumann: C_D = 24/Re·(1 + 0.15·Re^0.687) = **1.0917** |
| Acceptance band | 1.03 ≤ C_D ≤ 1.10 |

## Method

- **Lattice**: D3Q19 (19 velocity directions)
- **Collision**: BGK (single relaxation time)
- **Boundary conditions**: Far-field (free-stream equilibrium on inlet + lateral faces, zero-gradient outlet, bounce-back on sphere)
- **Force measurement**: Momentum exchange method (Ladd 1994)
- **Blockage correction**: First-order, C_D(corr) = C_D(raw) / (1 + 2.5·β), where β = πR²/ny²

## Grid Convergence Study

Three resolutions with sphere diameter D = 8, 10, 12 lattice units:

| D | Grid (nx×ny×nz) | τ | β | C_D(raw) | C_D(corr) | Error vs S-N |
|---|-----------------|------|--------|----------|-----------|-------------|
| 8  | 80×20×20  | 0.524 | 0.126 | 1.5375 | 1.1700 | 7.17% |
| 10 | 100×25×25 | 0.530 | 0.126 | 1.4988 | 1.1405 | 4.47% |
| 12 | 120×30×30 | 0.536 | 0.126 | 1.4860 | 1.1307 | 3.57% |

## Verdict

| Criterion | Result |
|-----------|--------|
| C_D(corr) in [1.03, 1.10] | **FAIL** (C_D = 1.1307) |
| Monotonic convergence | **PASS** (7.17% → 4.47% → 3.57%) |
| **Overall** | **FAIL** |

The C_D converges monotonically toward the Schiller-Naumann value but the finest
resolution (D=12) still yields 3.57% error, placing C_D = 1.1307 just outside the
acceptance band upper limit of 1.10.  Higher resolution (D ≥ 16) or GPU acceleration
is needed to reach the acceptance band.

## Files

- `run_sphere_re100.py` — Main validation driver
- `runs/sphere_re100/convergence_summary.json` — Per-resolution C_D data
- `runs/sphere_re100/verdict.json` — Machine-readable PASS/FAIL verdict
- `runs/sphere_re100/symmetry_velocity_contour.png` — Symmetry-plane |u| field
- `runs/sphere_re100/grid_convergence.png` — C_D and error vs resolution

## Usage

```bash
PYTHONPATH=src python validation/sphere_re100/run_sphere_re100.py
PYTHONPATH=src python validation/sphere_re100/run_sphere_re100.py --device cuda
```
