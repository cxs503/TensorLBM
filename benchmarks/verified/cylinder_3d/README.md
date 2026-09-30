# cylinder_3d — 3D extruded circular cylinder, Re = 40  ✅ VERIFIED

Status: **verified** (2026-09-30). Two grids within 3% of the 2D infinite-span numerical
cluster and mesh-converged (span 2.64%).

## Physics

Viscous incompressible flow past a circular cylinder extruded along z with **periodic z**
(= infinite span), Re = U·D/ν = 40. Below the shedding threshold (Re_c ≈ 47) the wake is
steady; the quantity of interest is Cd = Fx / (½ρU²D).

## Reference convention

3D extruded + z-periodic is **physically equivalent to 2D infinite span**, so the correct
comparison is the **2D free-stream numerical cluster**, not a finite-span experiment:

| source | Cd |
|---|---|
| Dennis & Chang 1970 (2D numerical) | 1.522 |
| Fornberg 1985 (2D spectral) | 1.498 |
| Takami & Keller 1969 | ≈1.48 |
| **adopted** | **1.50 ([1.48, 1.52])** |

Tritton 1959 (≈1.54) is a **finite-span wind-tunnel experiment** and must not be used —
it inflated the historical "−11% error" by itself.

## Force caliber — surface-only Ladd MEM (the key point)

The Ladd momentum-exchange sum `Σ_solid 2 c_ix f_i` splits as
`Σ_surface + Σ_interior`. On a curved voxel body the **interior solid cells carry a
non-degenerate pseudo-force** (measured 0.043 at D=20, 0.063 at D=40 — growing with D),
which is why summing over *all* solid cells over-predicts Cd. Restricting the sum to the
**surface solid cells (6-neighbourhood touching fluid)** recovers the correct value. This
is also the true cause of the historical sphere MEM "+264%" failure (all-solid sum).
See `docs/mem_surface_caliber_finding.md`.

| grid | Cd (surface MEM) | err | Cd (all-solid, diagnostic) | Cd (interior, pseudo) |
|---|---|---|---|---|
| D=20 (640²) | **1.5368** | **+2.45%** | 1.5801 (+5.34%) | 0.0433 |
| D=40 (1280²) | **1.4972** | **−0.19%** | 1.5596 (+3.97%) | 0.0625 |

Both grids ≤3%, grid span 2.64% ≤3% → **verified**.

Note: the pressure+friction decomposition on the *same* fields reads 1.305 / 1.323
(−13.0% / −11.8%) — the augmented pressure/friction split on staircase wall geometry
is systematically low (the same caliber effect diagnosed on sphere and square_cylinder).

## Configuration

D3Q19, BGK, u_in = 0.08, lateral domain 32D (blockage 3.125%), nz = 1 (z-periodicity is
bit-exact identical to nz = 4), 30 000 steps to plateau (all plateau windows equal).
A 40D-domain run (blockage 2.50%) with D=20 reads 1.5244 (+1.62%), consistent.

## Reproduce

```bash
PYTHONPATH=src python benchmarks/verified/cylinder_3d/run.py single 20 --steps 30000 \
    --lateral 32 --nz 1 --device sdaa:0 --compile-mode eager
# repeat with 40 for the second grid (~4 h each on SDAA)
```

## Compile routing (SDAA caveat)

Runs through `benchmarks/compile_route.py`; on this SDAA host the teco-inductor backend
fails on the LBM step chain (5th backend defect), so `compile_route` automatically
**falls back to eager** — bit-identical results, recorded as
`compile_mode_effective="eager (fallback)"` in `result.json`. On CUDA hosts
`--compile-mode default` really compiles.