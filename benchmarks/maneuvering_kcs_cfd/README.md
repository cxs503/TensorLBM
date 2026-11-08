# KCS drift / rudder / circular-motion LBM sweep → MMG derivative identification

Node **IN59-N02** (TensorLBM). Code: `src/tensorlbm/maneuvering_cfd.py`.
Report: `docs/IN59_N02_KCS_MANEUVERING_CFD.md`.

## What this benchmark does

1. Voxelises the KCS hull (Lpp = 120 lattice cells, `Cb` measured from the mask
   and compared with the locked 0.6505 — fail-closed) at a prescribed drift
   angle `beta`, rudder angle `delta`, or yaw rate `r'`.
2. Runs steady D3Q19 lattice-Boltzmann flow (solid-node bounce-back, Ladd
   momentum exchange for force and yaw moment).
3. Non-dimensionalises with the SNAME prime system and identifies the MMG
   polynomial coefficients by least squares with parameter covariance.

## Run

```bash
# 13-case sweep, one process per case (no multiprocessing pool: torch/OMP fork
# deadlocks were observed on this host)
python benchmarks/maneuvering_kcs_cfd/run.py --mode specs | \
  while read -r spec; do
    CASE_THREADS=2 PYTHONPATH=src python benchmarks/maneuvering_kcs_cfd/run.py \
        --mode one --spec "$spec" &
  done; wait

PYTHONPATH=src python benchmarks/maneuvering_kcs_cfd/run.py --mode analyze
PYTHONPATH=src python benchmarks/maneuvering_kcs_cfd/report.py
```

`--mode analyze` writes `artifacts/maneuvering_kcs_cfd/result.json` (all criteria
numbers + provenance) and figures 1-6; `report.py` renders the markdown report
from that JSON so no number is typed by hand.

## Case matrix

| family | sweep | derivatives identified |
|---|---|---|
| static drift | `beta = -20, -10, 0, 10, 20 deg` | `X'_0, X'_bb, X'_bbbb, Y'_b, Y'_bbb, N'_b, N'_bbb` |
| rudder | `delta = -35, -20, 20, 35 deg` | `Y'_d, N'_d, X'_dd` |
| circular motion (hull-fixed frame) | `r' = -0.4, -0.2, 0.2, 0.4` | `Y'_r, Y'_rrr, N'_r, N'_rrr, X'_rr` |

## Evidence grading (what is and is not claimed)

* **REAL** — voxel geometry with measured `Cb`; D3Q19 flow; momentum-exchange
  force (asserted identical to `obstacles.compute_obstacle_forces_3d`); least
  squares identification with covariance; synthetic round-trip recovery.
* **REAL but Re-limited** — the derivative *values*: `Re_lattice ~ 3e2` vs the
  KCS towing-tank `Re = 1.4e7`. They are not a model-scale prediction.
* **WITHHELD** — accuracy versus SIMMAN 2015 KCS experiments: no digit-exact
  reference table was available offline, and the Reynolds gap forbids an
  experimental error bar. See report §6 (honest FAIL of that gate).

## Tests

```bash
pytest tests/test_maneuvering_cfd.py -q      # ~20 s, 14 tests
```