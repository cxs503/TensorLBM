# SUBOFF geometry-axis campaign: exact C_D over (Re, sail_scale, fin_scale)

2026-08-22 · dataset `scan_suboff_geo_lhs_20260822` · code `main` @ 79b17f3

## Motivation

B1-5 (#218) opened a real geometry axis (`sail_scale` / `fin_scale`: an
inverse similarity transform anchored on the exact DARPA predicates). Its A/B
runs showed strict monotonicity of the raw force, but the self-projected C_D
proxy *saturated* between scale 2 and 3 — leaving open whether the geometry
axis carries separable information for surrogates. This campaign answers that
with the exact drag observer (#204) over a 3-factor design.

## Design

78 points = 64 LHS (log-uniform `re` ∈ [50, 800], uniform `sail_scale` /
`fin_scale` ∈ [1, 3], independently permuted pairings, seed 20260822)
+ 14 anchors (clean 1-D slices at re = 200 / 600: sail ∈ {1, 1.5, 2, 3} at
fin = 1, fin ∈ {1.5, 2, 3} at sail = 1).

Production chain throughout: case `suboff_n128` (n128, cumulant, u_in = 0.1,
full hull), 4000 steps, snapshots every 500, `DragSurveySpec(margin=4,
interval=25)`; worst-corner tau = 0.5288 (re = 800). 8× RTX 5090, ~8 min
wall, 72 MLUPS/card; max window drift across all 78 points 1e-4.
Launcher and analyzer are archived inside the dataset directory.

## C_D conventions

Two normalisations are reported side by side:

- `C_D_ref = 2·F_tail / (rho0·u_in²·69)` — the fixed reference area used by
  the v1.1 benchmark table (comparable across all prior datasets); measures
  *total resistance relative to the scale-1 geometry*.
- `C_D_proj` — same force over the point's true projected frontal area
  (voxel footprint from the offline builder), i.e. a shape-quality
  coefficient.

Note: the true scale-1 full-hull footprint is **73** cells, not 69 — the
v1.1 table's level carries a +5.8% reference-area bias (slopes unaffected).

## Anchor slices (exact, tail-window)

re = 200:

| scale | sail axis (fin=1) A / C_D_ref | fin axis (sail=1) A / C_D_ref |
|---|---|---|
| 1.0 | 73 / 7.299 | 73 / 7.299 |
| 1.5 | 75 / 7.533 | 79 / 7.618 |
| 2.0 | 93 / 8.036 | 117 / 8.309 |
| 3.0 | 117 / 9.681 | 165 / 10.601 |

re = 600:

| scale | sail axis (fin=1) A / C_D_ref | fin axis (sail=1) A / C_D_ref |
|---|---|---|
| 1.0 | 73 / 3.545 | 73 / 3.545 |
| 1.5 | 75 / 3.700 | 79 / 3.737 |
| 2.0 | 93 / 4.007 | 117 / 4.216 |
| 3.0 | 117 / 4.938 | 165 / 5.741 |

Findings:

1. **Monotonic on both axes at both Re, with no saturation in the exact
   total-resistance coefficient** (sail 1→3: +33%/+39%; fin 1→3: +45%/+62%
   at Re 200/600). B1-5's proxy saturation 2→3 was an artefact of
   normalising by the growing projected area: `C_D_proj` *falls* along the
   same slices (7.30 → 5.71 at re 200, sail axis) — the appendages grow
   frontal area faster than force, but the force itself keeps growing.
2. **The fin axis beats the sail axis** (+45% vs +33% at re 200; +62% vs
   +39% at re 600) — consistent with the fin's larger area growth (73→165
   vs 73→117 cells).
3. Cross-campaign reproduction: the (1,1) anchors reproduce the
   independent B1-5 A/B runs (2500 steps, different tail window) to ratios
   0.9998 / 0.9999.

## Regression baselines (log10 C_D_ref, all 78 points)

| model | R² | MAPE | coefficients |
|---|---|---|---|
| M1: re only | 0.9225 | 12.66% | intercept 2.454, re −0.645 |
| M2: + log sail + log fin | 0.9893 | 4.18% | re −0.632, sail +0.223, fin +0.301 |
| M3: + quadratics + sail·fin | 0.9961 | 2.61% | sail² 0.493, fin² 0.735, sail·fin −0.266 |

The geometry axis explains the bulk of the Re-only residual (12.7% → 4.2%
with two linear terms). The strong convex quadratics echo finding 1: force
growth *accelerates* toward scale 3 rather than saturating. These are the
baselines a geometry-aware surrogate (B1-7: FNO with scale-invariant inputs
à la B1-3, plus sail/fin as normalized inputs) must beat.

## Artefacts

- dataset: `/nfs/wangxi/datasets/scan_suboff_geo_lhs_20260822/`
  (launcher `b16_launch.py`, analyzer `b16_analyze.py`, per-point
  `drag_history.json` sidecars, `analysis.json` in
  `/nfs/wangxi/runs/b16_analysis_20260822/`)
- fields: 8 plane snapshots per point (z = nz/2), steps 500..4000, fp32
  HDF5 — FNO-ready on the same sample spec as B1-3.
