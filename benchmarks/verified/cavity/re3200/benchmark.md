# Cavity Re = 3200 -- ref vs got (anchor REPRO-LBM-HOU1995)

reference: U. Ghia, K. N. Ghia, C. T. Shin, J. Comput. Phys. 48 (1982) 387-411 -- Table I (u along the vertical centreline x/L=0.5) / Table II (v along the horizontal centreline y/L=0.5), Re = 3200 column
content sha256: `42470d4e2526494754fb5b4a52777412a1c4d75f906895a6074ce4a41e39e2b9` (verified: True)
tolerance: 5.0 % of the lid speed, pointwise

**verdict: FAIL** on the finest scored grid n=128 -> max |err| = 96.44 %

| n | status | steps | converged | max abs err [%] | rmse [%] | pass |
|---|--------|-------|-----------|-----------------|----------|------|
| 128 | ok | 6000 | False | 96.44 | 42.98 | FAIL |
| 192 | WITHHELD | - | - | - | - | - |
| 256 | WITHHELD | - | - | - | - | - |

## worst rows (finest grid, wall convention)

* u: coord 0.9766 ref +0.53236 got -0.43206 err -96.44 %
* v: coord 0.9453 ref -0.54053 got +0.02355 err +56.41 %

## how to reproduce

* run: `python benchmarks/verified/cavity/re3200/run.py --n 128 192 256 --steps 150000 --monitor-every 5000 --resid-tol 2e-06`
* score: `python benchmarks/verified/cavity/re3200/benchmark.py`
* figures: `python benchmarks/verified/cavity/re3200/analyze.py`
