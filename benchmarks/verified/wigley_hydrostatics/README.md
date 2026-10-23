# wigley_hydrostatics — Wigley parabolic hull form coefficients  ✅ VERIFIED

Status: **verified** (2026-10-05).

**Scope: hydrostatic form coefficients (geometry + static closure only) — NOT
seakeeping / wave-body motion.** The free-surface architecture still carries the
documented "no free pressure field" block (see
`docs/free_surface_architecture_gaps.md`), so a波浪中运动 benchmark is not claimed.

## Result (two ladders, L = 640 / 960)

| coefficient | analytic | L=640 err | L=960 err | span |
|---|---|---|---|---|
| C_B   | 4/9   | +1.97% | +1.29% | 0.68% |
| C_wp  | 2/3   | -0.17% | -0.11% | 0.06% |
| C_M   | 2/3   | +2.25% | +1.22% | 1.01% |
| C_P   | 2/3   | -0.27% | +0.06% | 0.33% |
| KB/T  | 5/8   | +1.00% | +0.70% | 0.30% |
| I_T/(B^3 L) | 4/105 | -0.72% | -0.41% | 0.31% |
| I_L/(L^3 B) | 1/30  | -0.01% | -0.05% | 0.05% |

**max |err| = 2.25%, max span = 1.02%** — every coefficient within 3%, every span within 3%.

## Reference (multi-source)

Wigley (1934) parabolic hull closed-form integrals (Faltinsen *Sea Loads on Ships
and Offshore Structures*; Lewis *Principles of Naval Architecture*), **independently
re-derived with sympy** and cross-checked against the library's `wigley_hull_mask`
(bitwise-identical).

## Recorded boundary condition (important)

Voxel staircasing at low L is large: C_B error is **+89% at L=20, +21% at L=160,
+4% at L=320**, and only enters 3% at **L >= 640**. The two-ladder gate therefore
uses L = 640/960.

## Reproduce

```bash
PYTHONPATH=src python benchmarks/verified/wigley_hydrostatics/wigley_hydro_full.py
```
