### W1 - static droplet (n=64, tau=1, 2500-step equal-time window)

| block | arm | G_12 | R_init | max\|u_mix\| | converged | rho1 centre | rho1 corner | dp | R meas | sigma = dp*R |
|---|---|---|---|---|---|---|---|---|---|---|
| W1_droplet_coupling | GOT-MIX | -1.50 | 12 | 0.02158 | **no** | 1.015 | 0.136 | 0.06095 | 12.67 | 0.7724 |
| W1_droplet_coupling | GOT-SELF | -1.50 | 12 | 0.0003249 | **no** | 0.228 | 0.228 | -0.0009818 | 20.80 | -0.02042 |
| W1_droplet_coupling | LBM4 | -1.50 | 12 | 0.03172 | **no** | 1.108 | 0.133 | 0.09566 | 11.99 | 1.147 |
| W1_droplet_coupling | LBM3 | -1.50 | 12 | 0.08319 | **no** | 1.127 | 0.527 | 0.1761 | 10.69 | 1.882 |
| W1_droplet_coupling | CTRL-NORM | -1.50 | 12 | 0.0003969 | **no** | 0.282 | 0.196 | -0.03494 | 18.69 | -0.6532 |
| W1_droplet_coupling | GOT-MIX | -2.50 | 12 | 0.0856 | **no** | 1.073 | 0.018 | 0.02757 | 12.42 | 0.3424 |
| W1_droplet_coupling | GOT-SELF | -2.50 | 12 | 0.009712 | yes | 0.915 | 0.141 | 0.01527 | 11.36 | 0.1735 |
| W1_droplet_coupling | LBM4 | -2.50 | 12 | 0.1817 | **no** | 0.915 | 0.015 | -0.05313 | 1.36 | -0.07206 |
| W1_droplet_coupling | LBM3 | -2.50 | 12 | nan | **no** | nan | nan | nan | nan | nan |
| W1_droplet_coupling | CTRL-NORM | -2.50 | 12 | 0.006433 | **no** | 0.751 | 0.158 | -0.04476 | 10.10 | -0.4519 |
| W1_droplet_radius | GOT-MIX | -2.00 | 8 | 0.05288 | **no** | 1.102 | 0.038 | 0.04553 | 8.43 | 0.3838 |
| W1_droplet_radius | GOT-MIX | -2.00 | 12 | 0.05177 | **no** | 1.094 | 0.043 | 0.04949 | 11.88 | 0.5878 |
| W1_droplet_radius | GOT-MIX | -2.00 | 16 | 0.05045 | **no** | 1.073 | 0.036 | 0.03484 | 15.63 | 0.5446 |
| W1_droplet_radius | LBM4 | -2.00 | 8 | 0.082 | **no** | 1.153 | 0.438 | 0.2177 | 7.14 | 1.555 |
| W1_droplet_radius | LBM4 | -2.00 | 12 | 0.07598 | **no** | 1.137 | 0.019 | 0.02401 | 11.12 | 0.2669 |
| W1_droplet_radius | LBM4 | -2.00 | 16 | 0.07415 | **no** | 1.116 | 0.560 | 0.19 | 14.30 | 2.717 |
| W1_droplet_radius | GOT-SELF | -2.00 | 8 | 8.895e-05 | **no** | 0.181 | 0.179 | -0.001227 | nan | nan |
| W1_droplet_radius | GOT-SELF | -2.00 | 12 | 0.0003046 | **no** | 0.230 | 0.226 | -0.00243 | 26.02 | -0.06322 |
| W1_droplet_radius | GOT-SELF | -2.00 | 16 | 0.00037 | **no** | 0.302 | 0.289 | -0.004415 | 27.64 | -0.122 |

### W1 - Young-Laplace consistency across R in {8,12,16} (G_12=-2.0)

| arm | sigma(R=8) | sigma(R=12) | sigma(R=16) | mean | radius residual |
|---|---|---|---|---|---|
| GOT-MIX | 0.3838 | 0.5878 | 0.5446 | 0.5054 | 24.1% |
| LBM4 | 1.555 | 0.2669 | 2.717 | 1.513 | 82.4% |
| GOT-SELF | -0.06322 | -0.122 | -0.09263 | 31.7% |

### W1 - deviation of each convention from GOT-MIX (u_max / sigma)

| G_12 | arm vs GOT-MIX | dev u_max | dev sigma | dev rho1_centre | u_max verdict |
|---|---|---|---|---|---|
| -1.50 | GOT-SELF | 98.5% | 102.6% | 77.5% | FAIL |
| -1.50 | LBM4 | 47.0% | 48.5% | 9.2% | FAIL |
| -1.50 | LBM3 | 285.6% | 143.7% | 11.1% | FAIL |
| -1.50 | CTRL-NORM | 98.2% | 184.6% | 72.2% | FAIL |
| -2.50 | GOT-SELF | 88.7% | 49.3% | 14.7% | FAIL |
| -2.50 | LBM4 | 112.3% | 121.0% | 14.7% | FAIL |
| -2.50 | LBM3 | nan% | nan% | nan% | NO-DATA |
| -2.50 | CTRL-NORM | 92.5% | 232.0% | 30.0% | FAIL |

### W2 - stratified Poiseuille vs closed-form two-layer solution

| block | arm | ny | tau_upper | G_12 | u_max sim | u_max analytic | err u_max | Q sim | Q analytic | err Q | err profile L2 | steady |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W2_poiseuille_primary | GOT-MIX | 25 | 1.00 | -2.50 | 0.02713 | 0.0198 | ** 37.01% | 0.3093 | 0.3036 |  1.86% | 18.26% | yes |
| W2_poiseuille_primary | GOT-SELF | 25 | 1.00 | -2.50 | 0.01913 | 0.0198 | ** 3.40% | 0.2896 | 0.3036 | ** 4.61% | 8.30% | yes |
| W2_poiseuille_primary | LBM4 | 25 | 1.00 | -2.50 | 0.0642 | 0.0198 | ** 224.23% | 0.7296 | 0.3036 | ** 140.31% | 150.03% | no |
| W2_poiseuille_primary | LBM3 | 25 | 1.00 | -2.50 | nan | 0.0198 |  nan% | nan | 0.3036 |  nan% | nan% | no |
| W2_poiseuille_primary | CTRL-NORM | 25 | 1.00 | -2.50 | 0.01204 | 0.0198 | ** 39.17% | 0.187 | 0.3036 | ** 38.41% | 39.16% | yes |
| W2_poiseuille_primary | LIB-ENTRY | 25 | 1.00 | -2.50 | 0.02713 | 0.0198 | ** 37.01% | 0.3093 | 0.3036 |  1.86% | 18.26% | yes |
| W2_poiseuille_shift_factor | GOT-MIX | 25 | 0.75 | -2.50 | nan | 0.02655 |  nan% | nan | 0.4066 |  nan% | nan% | no |
| W2_poiseuille_shift_factor | LBM4 | 25 | 0.75 | -2.50 | nan | 0.02655 |  nan% | nan | 0.4066 |  nan% | nan% | no |
| W2_poiseuille_shift_factor | LBM4-TAU | 25 | 0.75 | -2.50 | nan | 0.02655 |  nan% | nan | 0.4066 |  nan% | nan% | no |
| W2_poiseuille_resolution | GOT-MIX | 41 | 1.00 | -2.50 | 0.08316 | 0.057 | ** 45.89% | 1.64 | 1.482 | ** 10.68% | 18.07% | yes |
| W2_poiseuille_resolution | LBM4 | 41 | 1.00 | -2.50 | 0.1717 | 0.057 | ** 201.16% | 3.459 | 1.482 | ** 133.40% | 136.63% | no |
| W2_poiseuille_coupling | GOT-MIX | 25 | 1.00 | -2.00 | 0.02376 | 0.0198 | ** 20.00% | 0.3036 | 0.3036 |  0.00% | 9.79% | no |
| W2_poiseuille_coupling | LBM4 | 25 | 1.00 | -2.00 | 0.04989 | 0.0198 | ** 151.98% | 0.6492 | 0.3036 | ** 113.83% | 118.38% | no |

### W2 harness control (shipped entry point vs this harness at the same configuration)

| Q LIB-ENTRY | Q GOT-MIX | rel diff | u_max LIB-ENTRY | u_max GOT-MIX | rel diff | max abs profile diff | verdict |
|---|---|---|---|---|---|---|---|
| 0.30871 | 0.30871 | 0.00e+00 | 0.0271286 | 0.0271286 | 0.00e+00 | 0.00e+00 | HARNESS-OK |

### W2 - deviation from GOT-MIX (Q = band-integrated flux)

| block | arm | tau_upper | dev Q | dev u_max | err Q vs analytic (GOT) | err Q vs analytic (arm) | verdict |
|---|---|---|---|---|---|---|---|
| W2_poiseuille_primary | GOT-SELF | 1.00 | 6.4% | 29.5% | 1.86% | 4.61% | PASS |
| W2_poiseuille_primary | LBM4 | 1.00 | 135.9% | 136.6% | 1.86% | 140.31% | FAIL |
| W2_poiseuille_primary | LBM3 | 1.00 | nan% | nan% | 1.86% | nan% | NO-DATA |
| W2_poiseuille_primary | CTRL-NORM | 1.00 | 39.5% | 55.6% | 1.86% | 38.41% | FAIL |
| W2_poiseuille_shift_factor | LBM4 | 0.75 | nan% | nan% | nan% | nan% | NO-DATA |
| W2_poiseuille_shift_factor | LBM4-TAU | 0.75 | nan% | nan% | nan% | nan% | NO-DATA |

### W3 - momentum ledger (n=48, periodic, no body force, 400 steps)

| arm | drift P_x | drift P_y | drift/step P_y | box sum (F1+F2)_x | box sum (rho1 F1 + rho2 F2)/rho_tot | finite |
|---|---|---|---|---|---|---|
| GOT-MIX | -5.198e-05 | 5.341e-05 | 1.335e-07 | 0.000e+00 | 0.000e+00 | yes |
| GOT-SELF | -2.289e-05 | -4.959e-05 | -1.240e-07 | 0.000e+00 | 0.000e+00 | yes |
| LBM4 | -3.672e-05 | -1.450e-04 | -3.624e-07 | 0.000e+00 | 0.000e+00 | yes |
| LBM3 | nan | nan | nan | nan | nan | **no** |
| CTRL-NORM | -3.338e-06 | -1.311e+00 | -3.277e-03 | 0.000e+00 | 0.000e+00 | yes |
| LBM4-TAU | -3.672e-05 | -1.450e-04 | -3.624e-07 | 0.000e+00 | 0.000e+00 | yes |

### W4 - depleted-cell guard (one 2x2 cell at rho_1=1e-8, 1e-3 momentum)

| arm | finite | max\|u_mix\| in depleted patch | max\|u_mix\| global | rho_tot in patch |
|---|---|---|---|---|
| GOT-MIX | yes | 0.03245 | 0.06208 | 1.000e-08 |
| LBM4 | yes | 0 | 0.1242 | 1.000e-08 |
| LBM3 | yes | 0 | 0.1099 | 1.000e-08 |
