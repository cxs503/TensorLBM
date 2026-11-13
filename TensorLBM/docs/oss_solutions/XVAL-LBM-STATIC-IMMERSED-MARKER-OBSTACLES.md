# OSS Solution: XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES

**feature_slug**: `XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES`
**date**: 2026-11-14
**author**: m18749

## Problem
Represent stationary solid obstacles (cylinders, arbitrary shapes) in LBM flow
using Lagrangian immersed boundary markers, with validation against canonical
benchmarks.

## Method Selected
**Peskin's Immersed Boundary Method (IBM) with iterative direct forcing**

- Delta function: Peskin 4-point (support |r| ≤ 2, C¹ continuous)
- Force spreading: tensor-product stencil, vectorized scatter
- Velocity interpolation: tensor-product stencil, vectorized gather
- Forcing scheme: Uhlmann (2005) direct forcing with Guo half-force correction
- Iterative correction: re-interpolate spread force to achieve round-trip consistency

## OSS References
| repo | file | relevance |
|---|---|---|
| haimingz/vivsim | vivsim/ib/{kernels,stencil,mdf}.py | Peskin stencil + MDF (validated in XVAL-LBM-PESKIN) |
| pylbm | demo/2D/Karman_vortex_street.py | Cylinder flow benchmark (analytical geometry, not IBM) |

## Implementation
- `src/tensorlbm/ibm.py` — IBM core operators
- `src/tensorlbm/verification/cylinder_flow/__init__.py` — cylinder flow solver

## Validation Target
- Re = 20 cylinder flow: Cd = 2.05 (Dennis & Chang 1970)
- Acceptance: |Cd_computed − 2.05| / 2.05 < 15%

## Cluster Job
- Job ID: cj2024
- Command: full Re=20 simulation (320×160, 8000 steps)
- Output: `xval/XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES/artifacts/cylinder_flow_re20.json`
