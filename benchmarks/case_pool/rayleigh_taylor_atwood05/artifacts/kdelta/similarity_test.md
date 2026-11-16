# kdelta ladder: is k*xi the knob, or the Allen-Cahn mobility?

Base ladder (`M = 20` fixed), similarity-preserving in zeta, Ma and
`a0/lam`; only `k*xi = 2 pi ac_width / lam` and the AC relaxation-rate
ratio `Pi` change.

| arm | lam | k*xi | zeta | Ma_peak | M | err_W vs inv | err_P vs inv |
|---|---|---|---|---|---|---|---|
| KD_l064 | 64 | 0.1963 | 1.5380e-02 | 0.247 | 20.0 | -11.52% | -7.76% |
| KD_l096 | 96 | 0.1309 | 1.5380e-02 | 0.250 | 20.0 | -9.45% | -6.24% |
| KD_l128 | 128 | 0.0982 | 1.5380e-02 | 0.249 | 20.0 | -8.55% | -5.53% |

| estimator | form | n | e0 (k*xi -> 0) | slope | R^2 | rms | AIC |
|---|---|---|---|---|---|---|---|
| W | linear | 3 | -5.53% | -30.37 | 0.9988 | 0.044 pp | -14.8 |
| W | quadratic | 3 | -7.64% | -101.1 | 0.9979 | 0.057 pp | -13.2 |
| W | proportional | 3 | +0.00% | -66.4 | -0.5223 | 1.531 pp | 4.6 |
| W | **LOO linear** | 3 | [-5.87, -5.31]% (full -5.53%) | — | — | — | — |
| P | linear | 3 | -3.28% | -22.78 | 0.9998 | 0.014 pp | -21.7 |
| P | quadratic | 3 | -4.86% | -75.71 | 0.9956 | 0.061 pp | -12.7 |
| P | proportional | 3 | +0.00% | -44.12 | 0.0493 | 0.907 pp | 1.4 |
| P | **LOO linear** | 3 | [-3.39, -3.21]% (full -3.28%) | — | — | — | — |
