Maxwell anchor (analytic): rho_l=1.92887 rho_v=0.15289 ratio=12.62 p*=0.03421

| case | rho_in | err_l vs Maxwell | rho_out | err_v vs Maxwell |
|---|---|---|---|---|
| ref_edm_R14_tau0.8 | 1.9689 | 2.08% | 0.1607 | 5.13% |
| ref_edm_R14_tau1.2 | 1.9710 | 2.19% | 0.1622 | 6.08% |
| ref_guo_R10_tau1.0 | 1.9188 | 0.52% | 0.1265 | 17.25% |
| ref_guo_R14_tau0.8 | 1.9116 | 0.89% | 0.1241 | 18.84% |
| ref_guo_R14_tau1.0 | 1.9132 | 0.81% | 0.1252 | 18.14% |
| ref_guo_R14_tau1.2 | 1.9146 | 0.74% | 0.1262 | 17.46% |
| ref_guo_R18_tau1.0 | 1.9091 | 1.02% | 0.1243 | 18.73% |
| ref_guo_R7_L64_tau1.0 | 1.9358 | 0.36% | 0.1309 | 14.41% |
| ref_velocity_shift_R14_tau0.8 | 1.9336 | 0.24% | 0.1375 | 10.04% |
| ref_velocity_shift_R14_tau1.2 | 2.0266 | 5.07% | 0.1948 | 27.39% |
| repo_edm_R10_tau1.0 | 1.9908 | 3.21% | 0.1646 | 7.67% |
| repo_edm_R14_tau1.0 | 1.9701 | 2.14% | 0.1616 | 5.67% |
| repo_edm_R18_tau1.0 | 1.9611 | 1.67% | 0.1603 | 4.84% |
| repo_guo_R10_tau1.0 | 1.9188 | 0.52% | 0.1265 | 17.26% |
| repo_guo_R14_tau1.0 | 1.9132 | 0.81% | 0.1252 | 18.14% |
| repo_velocity_shift_R10_tau1.0 | 1.9908 | 3.21% | 0.1646 | 7.67% |
| repo_velocity_shift_R14_tau1.0 | 1.9701 | 2.14% | 0.1616 | 5.67% |
| repo_velocity_shift_R18_tau1.0 | 1.9611 | 1.67% | 0.1603 | 4.84% |

| arm:scheme:tau | sigma (slope) | R^2 | intercept/dp_max | sigma spread | <15% |
|---|---|---|---|---|---|
| ref:guo:tau=1.0 | 0.02660 | 0.99468 | 12.45% | 13.55% | YES |
| repo:edm:tau=1.0 | 0.04852 | 0.99811 | 7.32% | 6.52% | YES |
| repo:guo:tau=1.0 | 0.03114 | 1.00000 | 7.92% | 2.02% | YES |
| repo:velocity_shift:tau=1.0 | 0.04852 | 0.99811 | 7.32% | 6.52% | YES |

| R | dp repo | dp ref | rel diff | R_eq rel diff | <15% |
|---|---|---|---|---|---|
| R10 | 0.00254 | 0.00254 | 0.03% | 0.00% | YES |
| R14 | 0.00206 | 0.00206 | 0.02% | 0.00% | YES |