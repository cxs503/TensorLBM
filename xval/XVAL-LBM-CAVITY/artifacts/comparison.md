# XVAL-LBM-CAVITY · three-party metric table

| Re | W | τ | 方 | u_min | ref u_min | 误差 | v_max | ref v_max | 误差 | u(0.5,0.5) | ref | 误差 | RMSE(u) | max_dev | ≤3%? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 128 | 0.7304 | A TensorLBM | not_run | -0.21090 | — | 0.17527 | — | — | — | -0.20581 | — | — | — | not_run |
| 100 | 128 | 0.7304 | B pylbm | not_run | -0.21090 | — | 0.17527 | — | — | — | -0.20581 | — | — | — | not_run |
| 100 | 64 | 0.6152 | A TensorLBM | -0.21585 | -0.21090 | -2.35% | +0.17978 | +0.17527 | +2.57% | -0.21249 | -0.20581 | -3.25% | 0.0048 | 0.83% | ✅ |
| 100 | 64 | 0.6152 | B pylbm | -0.20672 | -0.21090 | +1.98% | +0.17389 | +0.17527 | -0.79% | -0.20426 | -0.20581 | +0.75% | 0.0117 | 2.49% | ✅ |
| 400 | 128 | 0.5576 | A TensorLBM | not_run | -0.32726 | — | 0.30203 | — | — | — | -0.11477 | — | — | — | not_run |
| 400 | 128 | 0.5576 | B pylbm | not_run | -0.32726 | — | 0.30203 | — | — | — | -0.11477 | — | — | — | not_run |
| 400 | 64 | 0.5288 | A TensorLBM | not_run | -0.32726 | — | 0.30203 | — | — | — | -0.11477 | — | — | — | not_run |
| 400 | 64 | 0.5288 | B pylbm | not_run | -0.32726 | — | 0.30203 | — | — | — | -0.11477 | — | — | — | not_run |

| Re | W | 工具间差 max|Δu| | max|Δv| | Δu_min | Δv_max | 涡心距 | ≤3%? |
|---|---|---|---|---|---|---|---|
| 100 | 128 | not_run | — | — | — | — | not_run |
| 100 | 64 | 3.12% | 1.49% | -4.42% | +3.39% | 0.0080 | ❌ |
| 400 | 128 | not_run | — | — | — | — | not_run |
| 400 | 64 | not_run | — | — | — | — | not_run |

Geometry conventions per party:
* Re100 W128 A: not_run (aspect None) | B: not_run (aspect None)
* Re100 W64 A: 3 walls half-way (pre-streaming reflection), lid at node row n-1 (Zou-He) (aspect 1.0078125) | B: BFL bounce-back (distance 0.5) on cell faces; lid = prescribed moments (aspect 1.0)
* Re400 W128 A: not_run (aspect None) | B: not_run (aspect None)
* Re400 W64 A: not_run (aspect None) | B: not_run (aspect None)