# CASE-RT-INSTABILITY-ATWOOD -- late-time (nonlinear) front velocity

Gate: |Fr_meas / Fr_LZ(C=1) - 1| <= 15 %, window tau in [4.0, 6.0]; Fr = u / sqrt(A g lam), Fr_LZ = 1/sqrt(pi C) (Layzer-Goncharov, Boussinesq form).
Measured C_eff = 1 / (pi Fr^2) (compare C = 0.90 from the sibling case at A = 1/2).

| case | steps | tau_end | stop | Ma_peak | Fr_meas | Fr_LZ | err | gate | C_eff | robust Fr | subwindow drift |
|---|---|---|---|---|---|---|---|---|---|---|---|
| late_l064_g2e-4 | 5166 | 6.46 | wall | 0.548 | 0.5956 | 0.5642 | +5.56% | PASS | 0.897 | 0.5966 | -14.68% |
| late_l064_g2e-4_L10 | 5196 | 6.50 | completed | 0.539 | 0.6100 | 0.5642 | +8.12% | PASS | 0.855 | 0.6108 | -3.29% |
| late_l096_g3e-4 | 3594 | 4.49 | mach | 0.879 | 0.7833 | 0.5642 | +38.84% | INCOMPLETE | 0.519 | 0.7856 | +nan% |

Front-velocity estimator comparison (same runs, same window):

| case | u_bubble (tip) | u_spike (tip) | u_legacy (mean(zc)) | legacy vs tip | mean(zc) drift [lam] | multi-valued columns | front symmetry dev |
|---|---|---|---|---|---|---|---|
| late_l064_g2e-4 | 4.7657e-02 | 4.7636e-02 | 8.5926e-02 | +80.34% | 1.762 | 51 | 2.35e-02 |
| late_l064_g2e-4_L10 | 4.8731e-02 | 4.8872e-02 | 8.5177e-02 | +74.54% | 1.692 | 50 | 1.80e-02 |
| late_l096_g3e-4 | 9.4134e-02 | 9.3868e-02 | 1.1173e-01 | +18.86% | 0.650 | 78 | 1.28e-02 |

Two-fluid (finite-density-ratio) asymmetry -- measured, not assumed:

| case | u_spike/u_bubble (sim) | two-fluid prediction | err vs two-fluid spike |
|---|---|---|---|
| late_l064_g2e-4 | 1.0004 | 1.7321 | -25.37% |
| late_l064_g2e-4_L10 | 0.9971 | 1.7321 | -23.43% |
| late_l096_g3e-4 | 1.0028 | 1.7321 | -1.96% |
