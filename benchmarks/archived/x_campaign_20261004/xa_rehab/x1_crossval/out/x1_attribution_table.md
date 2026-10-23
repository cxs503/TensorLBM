# X1 cross-model attribution table (machine-rendered, zero hand-copy)

generated 2026-10-05T09:43:26Z; instrument = attribution, NOT verdict;
cross-family sigma VALUE comparison is banned (prereg §0).

| class | family | closure metric | value | u_max | density ratio | mode |
|---|---|---|---|---|---|---|
| Laplace | MCMP | err(N) range pp | 0.14 | 1.15e-02 | 6.02 | consumed |
| Laplace | SC | sigma(L) spread % | 2.83 | 2.06e-01 | 400-557 (per-run, file field) | consumed |
| Laplace | CAC | sigma_eff(R) spread % | 0.92 | 7.77e-06 | 1.00 | XA-3 dedup |
| Laplace | CG | sigma_eff(R@L200) spread % | 17.97 | 2.12e-03@R20/5.05e-03@R30/4.69e-03@R40 | 13.0 | XA-5 dedup, health-fail disclosed |
| Poiseuille mu2 | CG | relL2 | 0.32% / 0.21% | - | 1.0/13.0/6.03 | own run |
| Poiseuille mu2 | CAC | Eu | 7.70% / 35.55% | - | 1.0/13.0/6.03 | own run |
| Poiseuille mu2 | MCMP | max rel err % | 27.04%@H64 / 33.64%@H128 | - | 1.0/13.0/6.03 | consumed |
| Taylor lam1 Ca~0.1 | CAC | ratio_sh in-band | YES (0.9774) | - | 1.00 | consumed |
| Taylor lam1 Ca~0.1 | CG | - | shear not supported by frozen kernel (P0 probe) | - | 13.0 | probe |
