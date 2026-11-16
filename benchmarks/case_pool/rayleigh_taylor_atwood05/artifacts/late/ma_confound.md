# Fr vs Ma from the *existing* late suite (no new run)

| case | lam | Ma_peak | Fr | err vs Fr_LZ | C_eff | gate | stop |
|---|---|---|---|---|---|---|---|
| late_l064_g2e-4_L10 | 64 | 0.539 | 0.6100 | +8.12 % | 0.855 | PASS | None |
| late_l064_g2e-4 | 64 | 0.548 | 0.5956 | +5.56 % | 0.897 | PASS | wall |
| late_l096_g3e-4 | 96 | 0.879 | 0.7833 | +38.84 % | 0.519 | FAIL | mach |

Power-law trend through the two usable arms: Fr ~ Ma^0.51 (Ma 0.539 -> 0.879, Fr 0.6100 -> 0.7833). The higher-Ma point is the `g`-propto-`lam` arm of the older ladder, so `lam`, `k*xi` and `Ma` move together: the trend is a *warning flag*, not a converged Ma dependence. That is exactly why the similarity ladder subordinates `lam`, `g`, `nu` to fixed `Ma` and `zeta`.
