# Ghia (1982) Re=400 external anchor — TensorLBM

generated: 2026-11-15T00:32:18+00:00

## 1. Problem setting and anchor source

* Geometry: square cavity 1x1, lid u=U_lid, no-slip on the other three walls.
* Re = U_lid*H/nu = 400; D2Q9 MRT, V3 pre-streaming half-way bounce-back on the three stationary walls, Zou/He moving-lid BC on the top row.
* Reference: Ghia, Ghia & Shin (1982), JCP 48(3):387-411, Table I (u vs y at x/L=0.5) and Table II (v vs x at y/L=0.5), 129x129 grid results.
* Provenance: BOUND. OCR text-layer witness sha256 `3637ef9ef72a76bb...`; in-repo CSVs are sha256-pinned and re-verified by `benchmarks/anchors/run_anchors.py` before scoring.

| table | file | sha256 (16) |
|---|---|---|
| I | `benchmarks/anchors/data/ghia1982_re400_tableI.csv` | `d1fd64cb8908dd8f` |
| II | `benchmarks/anchors/data/ghia1982_re400_tableII.csv` | `9c80212769831607` |

## 2. Method

* Solver: `tensorlbm.solver.collide_mrt` + `stream`, `tensorlbm.d2q9.equilibrium/macroscopic`, `tensorlbm.lid_driven_cavity.zou_he_moving_lid` (platform solver path, no benchmark-local numerics).
* Entry: `benchmarks/verified/cavity/re400/run.py`; scoring: `benchmarks/verified/cavity/re400/anchor_check.py`.
* Pre-declared metrics: M1 max-point deviation (|dU|/U_lid), M2 strict point-wise relative with ill-conditioning floor 0.05x peak, M3 extremum relative.
* Tolerances: M1 < 5%, M2 < 5%, M3 < 5%.

## 3. Results

| n | steps | last resid | M1 max-point dev | M2 strict rel | M3 u_min | M3 v_min | verdict |
|---|---|---|---|---|---|---|---|
| 65 | 60000 | n/a | 19.89% FAIL | 205.35% FAIL | 4.96% | 44.21% | FAIL |

### Point-by-point table (finest grid)

| table | coord | ref | got | delta | rel |
|---|---|---|---|---|---|
| I u | 1.0000 | +1.00000 | +1.00000 | -0.00000 | 0.0% |
| I u | 0.9766 | +0.75837 | +0.83869 | +0.08032 | 10.6% |
| I u | 0.9688 | +0.68439 | +0.75886 | +0.07447 | 10.9% |
| I u | 0.9609 | +0.61756 | +0.68688 | +0.06932 | 11.2% |
| I u | 0.9531 | +0.55892 | +0.61590 | +0.05698 | 10.2% |
| I u | 0.8516 | +0.29093 | +0.28571 | -0.00522 | 1.8% |
| I u | 0.7344 | +0.16256 | +0.16134 | -0.00122 | 0.8% |
| I u | 0.6172 | +0.02135 | +0.02158 | +0.00023 | 1.1% |
| I u | 0.5000 | -0.11477 | -0.11561 | -0.00084 | 0.7% |
| I u | 0.4531 | -0.17119 | -0.17255 | -0.00136 | 0.8% |
| I u | 0.2813 | -0.32726 | -0.31104 | +0.01622 | 5.0% |
| I u | 0.1719 | -0.24299 | -0.21676 | +0.02623 | 10.8% |
| I u | 0.1016 | -0.14612 | -0.12506 | +0.02106 | 14.4% |
| I u | 0.0703 | -0.10338 | -0.08515 | +0.01823 | 17.6% |
| I u | 0.0625 | -0.09266 | -0.07518 | +0.01748 | 18.9% |
| I u | 0.0547 | -0.08186 | -0.06501 | +0.01685 | 20.6% |
| I u | 0.0000 | +0.00000 | +0.00000 | +0.00000 | nan% |
| II v | 1.0000 | +0.00000 | +0.00000 | +0.00000 | nan% |
| II v | 0.9688 | -0.12146 | -0.08026 | +0.04120 | 33.9% |
| II v | 0.9609 | -0.15663 | -0.10183 | +0.05480 | 35.0% |
| II v | 0.9531 | -0.19254 | -0.12308 | +0.06946 | 36.1% |
| II v | 0.9453 | -0.22847 | -0.14024 | +0.08823 | 38.6% |
| II v | 0.9063 | -0.23827 | -0.20673 | +0.03154 | 13.2% |
| II v | 0.8594 | -0.44993 | -0.25102 | +0.19891 | 44.2% |
| II v | 0.8047 | -0.38598 | -0.27895 | +0.10703 | 27.7% |
| II v | 0.5000 | +0.05186 | -0.05463 | -0.10649 | 205.4% |
| II v | 0.2344 | +0.30174 | +0.31011 | +0.00837 | 2.8% |
| II v | 0.2266 | +0.30203 | +0.32489 | +0.02286 | 7.6% |
| II v | 0.1563 | +0.28124 | +0.42854 | +0.14730 | 52.4% |
| II v | 0.0938 | +0.22965 | +0.34289 | +0.11324 | 49.3% |
| II v | 0.0781 | +0.20920 | +0.28580 | +0.07660 | 36.6% |
| II v | 0.0703 | +0.19713 | +0.25325 | +0.05612 | 28.5% |
| II v | 0.0625 | +0.18360 | +0.22069 | +0.03709 | 20.2% |
| II v | 0.0000 | +0.00000 | +0.00000 | +0.00000 | nan% |

### Ill-conditioned points for the strict metric (M2)

|ref| < 0.05 * peak = 0.0500; a 1e-3*U_lid absolute error already means >=20% here, so these are reported, not silently dropped:

| table | coord | ref | rel |
|---|---|---|---|
| I | 0.6172 | +0.02135 | 1.1% |
| I | 0.0000 | +0.00000 | 0.0% |
| II | 1.0000 | +0.00000 | 0.0% |
| II | 0.0000 | +0.00000 | 0.0% |

## 4. Excluded entries (suspect list) and why

Every reference entry registered in `benchmarks/anchors/ghia1982_provenance.json -> suspect_entries` is excluded from this comparison; the BOUND tables above are the only reference used.

| Re | table | reason | action |
|---|---|---|---|
| 400 | GHIA_RE400 (reference_data.py inline constant) | Fabricated data: values do not match OCR source at any coordinate. Max relative error >400% at x=0.5 (existing=-0.15612 vs OCR=0.05186). Coordinate grid also wrong (14 uniform-spaced points vs 17 non-uniform Ghia grid points). See docs/audit_m15543.md finding F-A for similar Re=100 issue. | REPLACE with BOUND transcription from OCR witness |
| 100 | GHIA_RE100 (reference_data.py inline constant) | UNVERIFIED_OFFLINE per reference_data.py docstring: hand-entered without page/table/sha256 record. Centre value u(0.5)=-0.15612 vs literature -0.20581 (24% error). Coordinate grid wrong (14 points vs 17 Ghia points). | REPLACE with BOUND transcription from OCR witness |
| 1000 | GHIA_RE1000 (reference_data.py inline constant) | UNVERIFIED_OFFLINE per reference_data.py docstring: hand-entered without page/table/sha256 record. Coordinate grid wrong (14 points vs 17 Ghia points). | REPLACE with BOUND transcription from OCR witness |
| 400 | GHIA_RE400 (inline constants in src/tensorlbm/lid_driven_cavity.py and src/tensorlbm/lid_driven_cavity_3d.py) | Four v-values drifted from the witness: v(x=0.9063) carried 0.8047's value -0.38598 instead of -0.23827 (/dv/ = 0.14771 = 33% of the v-extremum), v(x=0.1563)=+0.28003 vs +0.28124, v(x=0.0938)=+0.22980 vs +0.22965, and the 0.2266/0.2344 pair swapped (+0.30174/+0.30203). Present in main since 3d5ef2fa6 (2026-05-23); invisible until a code-vs-witness drift guard existed. | CORRECTED 2026-11-15 to BOUND values; guarded by run_anchors.py step [5] and tests/test_ghia_re400_anchor_lit.py |

## 5. Verdict

* **M1: FAIL** (see table).
* **M3 (extremum relative error) < 5%: FAIL** (u_min 4.96%, v_min 44.21% at n=65).
* **M2 (strict point-wise relative) < 5%: FAIL** (max 205.35% over the 30 conditioned points).

## 6. Reproduce

```
PYTHONPATH=$PWD/src python benchmarks/verified/cavity/re400/run.py --nx 65 --steps 60000 --u-lid 0.1 --out benchmarks/verified/cavity/re400/artifacts/run_raw.json
python benchmarks/verified/cavity/re400/anchor_check.py
python benchmarks/anchors/run_anchors.py
```

