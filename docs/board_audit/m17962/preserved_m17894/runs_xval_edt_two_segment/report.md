# XVAL — EDT two-segment geometry tagging vs je-santos/MPLBM-UT

* anchor: `XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION`
* ref: `je-santos/MPLBM-UT` @ `a9fc496688486c8bbc882ec7d780619f6712fa54` (2023-12-01T16:44:09Z) — `mplbm_utils/pore_utils.py::create_geom_edist`
* got: `src/tensorlbm/edt_geometry_tags.py::build_geometry_tags`
* generated: 2026-11-13T10:11:50Z  |  python 3.11.6
* repro: `python tools/xval_edt_two_segment.py --out runs/xval_edt_two_segment`

## 1  Problem and reference source

MPLBM-UT prepares a Palabos geometry by (i) computing the Euclidean distance
transform inside the grain phase, (ii) forcing the six outer faces to depth 1 so
every boundary plane has bounce-back nodes, (iii) re-opening the pore cells,
(iv) segmenting the field into **two depth classes** (`0 < d < 2` boundary,
`d >= 2` interior) and (v) quantising to `int16` Palabos codes
2608 pore / 2609 boundary / 2610 grain (+2611 non-wetting seeds).  Step (iv)+(v)
is the *tag compression*: a float64 distance field (8 B/voxel) collapses to a
4-symbol int16 map (2 B/voxel).

## 2  Method

`got` re-implements the recipe independently: exact EDT through a k-d tree over
fluid-voxel centres (`exact_edt`), vectorised class assignment, lossless 2-bit
packing of the class map (`compress_label_map`), optional per-grain id side
channel, and the OSS on-disk format (`<name>_<nz>_<ny>_<nx>.dat`, int16 raw).
`ref` is a line-by-line transcription of the OSS functions executed in-process
(`oss_create_geom_edist` in `tools/xval_edt_two_segment.py`).

## 3  Parameters

| item | value |
| --- | --- |
| lattice spacing | 1 (lattice units), D3Q19 geometry convention |
| depth split | 2.0 cells (OSS) |
| BC faces | forced to depth 1 on all six faces (OSS) |
| Palabos codes | 2608 / 2609 / 2610 / 2611 |
| two-segment radius | L-bend 5 cells, V-bend 4.5, capsule 12, spheres 8 |
| threshold | 15% relative error (owner standard) |

## 4  ref vs got

| quantity | ref | got | error | pass |
| --- | --- | --- | --- | --- |
| two_segment_L: tag agreement vs OSS pipeline (fraction, higher better) | 1 | 1 | 0.000e+00 % (limit 15%) | PASS |
| two_segment_L: EDT algorithm: max |k-d tree - scipy| (cells, vs 1-cell band) | 0 | 0 | 0.000e+00 % (limit 15%) | PASS |
| two_segment_L: two-class agreement vs analytic geometry (fraction) | 1 | 0.99526 | 4.740e-01 % (limit 15%) | PASS |
| two_segment_L: EDT depth rel. error, cells deeper than split (mean) | 0 | 0.0196204 | 1.962e+00 % (limit 15%) | PASS |
| two_segment_L: EDT depth rel. error, cells deeper than 4 cells (mean) | 0 | 0.00367493 | 3.675e-01 % (limit 15%) | PASS |
| two_segment_L: EDT depth absolute bias, cells deeper than split (max, cells) | 0 | 1.12311 | 1.123 cells (limit 1.50) | PASS |
| two_segment_V: tag agreement vs OSS pipeline (fraction, higher better) | 1 | 1 | 0.000e+00 % (limit 15%) | PASS |
| two_segment_V: EDT algorithm: max |k-d tree - scipy| (cells, vs 1-cell band) | 0 | 0 | 0.000e+00 % (limit 15%) | PASS |
| two_segment_V: two-class agreement vs analytic geometry (fraction) | 1 | 0.992904 | 7.096e-01 % (limit 15%) | PASS |
| two_segment_V: EDT depth rel. error, cells deeper than split (mean) | 0 | 0.0952221 | 9.522e+00 % (limit 15%) | PASS |
| two_segment_V: EDT depth rel. error, cells deeper than 4 cells (mean) | 0 | 0.0264007 | 2.640e+00 % (limit 15%) | PASS |
| two_segment_V: EDT depth absolute bias, cells deeper than split (max, cells) | 0 | 1.26201 | 1.262 cells (limit 1.50) | PASS |
| capsule_r12_L20: tag agreement vs OSS pipeline (fraction, higher better) | 1 | 1 | 0.000e+00 % (limit 15%) | PASS |
| capsule_r12_L20: EDT algorithm: max |k-d tree - scipy| (cells, vs 1-cell band) | 0 | 0 | 0.000e+00 % (limit 15%) | PASS |
| capsule_r12_L20: two-class agreement vs analytic geometry (fraction) | 1 | 0.972444 | 2.756e+00 % (limit 15%) | PASS |
| capsule_r12_L20: EDT depth rel. error, cells deeper than split (mean) | 0 | 0.0359673 | 3.597e+00 % (limit 15%) | PASS |
| capsule_r12_L20: EDT depth rel. error, cells deeper than 4 cells (mean) | 0 | 0.017133 | 1.713e+00 % (limit 15%) | PASS |
| capsule_r12_L20: EDT depth absolute bias, cells deeper than split (max, cells) | 0 | 0.433981 | 0.434 cells (limit 1.50) | PASS |
| capsule_r12_L20: displaced volume (voxel count) vs exact | 16286 | 15863 | 2.597e+00 % (limit 15%) | PASS |
| slab_t4_gap24: tag agreement vs OSS pipeline (fraction, higher better) | 1 | 1 | 0.000e+00 % (limit 15%) | PASS |
| slab_t4_gap24: EDT algorithm: max |k-d tree - scipy| (cells, vs 1-cell band) | 0 | 0 | 0.000e+00 % (limit 15%) | PASS |
| slab_t4_gap24: solid fraction vs exact | 0.25 | 0.25 | 0.000e+00 % (limit 15%) | PASS |
| sphere_pack_r8: tag agreement vs OSS pipeline (fraction, higher better) | 1 | 1 | 0.000e+00 % (limit 15%) | PASS |
| sphere_pack_r8: EDT algorithm: max |k-d tree - scipy| (cells, vs 1-cell band) | 0 | 0 | 0.000e+00 % (limit 15%) | PASS |
| sphere_pack_r8: displaced volume (voxel count) vs exact | 6433.98 | 6309 | 1.943e+00 % (limit 15%) | PASS |
| two_segment_L_nw_padded: tag agreement vs OSS pipeline (fraction, higher better) | 1 | 1 | 0.000e+00 % (limit 15%) | PASS |
| two_segment_L_nw_padded: EDT algorithm: max |k-d tree - scipy| (cells, vs 1-cell band) | 0 | 0 | 0.000e+00 % (limit 15%) | PASS |

### 4b  EDT algorithm cross-check and compression

| case | max abs diff (k-d tree vs scipy) | RMS | float64 B | int16 B | 2-bit B | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| two_segment_L | 0.000e+00 | 0.000e+00 | 614400 | 153600 | 19200 | 32.00x |
| two_segment_V | 0.000e+00 | 0.000e+00 | 614400 | 153600 | 19200 | 32.00x |
| capsule_r12_L20 | 0.000e+00 | 0.000e+00 | 360000 | 90000 | 11250 | 32.00x |
| slab_t4_gap24 | 0.000e+00 | 0.000e+00 | 204800 | 51200 | 6400 | 32.00x |
| sphere_pack_r8 | 0.000e+00 | 0.000e+00 | 258048 | 64512 | 8064 | 32.00x |
| two_segment_L_nw_padded | 0.000e+00 | 0.000e+00 | 737280 | 184320 | 23040 | 32.00x |

## 5  Convergence and sensitivity

Capsule displaced-volume error vs exact `pi r^2 L + 4/3 pi r^3`:

| radius (cells) | voxels | ref volume | got voxels (phase-averaged) | rel err |
| --- | --- | --- | --- | --- |
| 6.0 | 9720 | 2261.947 | 2201.4 +/- 28.1 (n=8) | 2.678e-02 |
| 9.0 | 24192 | 7634.070 | 7560.4 +/- 86.2 (n=8) | 9.653e-03 |
| 12.0 | 48600 | 18095.574 | 17952.4 +/- 187.7 (n=8) | 7.913e-03 |

log-log slope of |volume error| vs h = **1.81** (second order expected for a voxelised curved surface).

Depth-split sensitivity (L-bend case):

| depth_split | boundary frac | interior frac | tag agreement vs OSS |
| --- | --- | --- | --- |
| 1.5 | 0.0205 | 0.0215 | 0.999818 |
| 2.0 | 0.0207 | 0.0214 | 1.000000 |
| 2.5 | 0.0301 | 0.0120 | 0.990612 |
| 3.0 | 0.0347 | 0.0073 | 0.985964 |

Resolution sensitivity (L-bend case):

| grid scale | voxels | boundary frac | boundary/solid |
| --- | --- | --- | --- |
| 0.75 | 32400 | 0.0296 | 0.6299 |
| 1.00 | 76800 | 0.0207 | 0.4919 |
| 1.50 | 259200 | 0.0163 | 0.3437 |

## 6  Failure modes, differences and artefacts

Differences vs OSS (see `docs/xval/EDT_TWO_SEGMENT_GEOMETRY_TAG_COMPRESSION.md`
for the full table): per-grain identity is not preserved by the 4-class map, so
mixed wettability needs the id side channel added here; `erase_regions` is a
no-op for binary images; the OSS code raises `NotImplementedError` for
`scale_2`/`add_mesh` (we keep the same semantics by not implementing them).

Failure modes: all-solid domain yields `inf` depths (same as scipy) and is
rejected upstream; anisotropy is honoured through `spacing`; `swap_xz` is
applied before padding exactly like OSS.

Artifacts:

* `runs/xval_edt_two_segment/summary.json`
* `runs/xval_edt_two_segment/table.csv`
* `runs/xval_edt_two_segment/report.md`
* `runs/xval_edt_two_segment/geometry`
* `runs/xval_edt_two_segment/table.csv`
* `runs/xval_edt_two_segment/report.md`

Figures:

* `runs/xval_edt_two_segment/figures/fig1_class_map.png` — Palabos geometry tags and two-class segmentation of the two-segment (L-bend) micro-model.
* `runs/xval_edt_two_segment/figures/fig2_distance_field.png` — Mid-plane depth: analytic capsule distance vs lattice EDT vs difference.
* `runs/xval_edt_two_segment/figures/fig3_convergence.png` — Displaced-volume error of the voxelised capsule vs 1/h (log-log).
* `runs/xval_edt_two_segment/figures/fig4_sensitivity.png` — Boundary/interior split vs threshold and vs grid resolution.
