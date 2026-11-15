# XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY -- re-run summary

## A. m21062 ladder reproduction (D-ladder, beta=50, box=4D, KC=0.2)

| D | nx | tau | C_I_E1_area | C_I_E1_halfway | C_Q_E1_area | C_I_E2_area | resid_E1 | c_a_pot_area | err_vs_pot_pct | m21062_C_I | ratio_current_over_m21062 | a0_meas_over_nominal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 48 | 0.644 | 0.2222 | 0.1914 | 1.4426 | 0.051 | 0.3382 | 1.0057 | -77.91 | 2.1002 | 0.106 | 1.0259 |
| 16 | 64 | 0.692 | 0.1755 | 0.1573 | 1.4116 | 0.051 | 0.3366 | 0.9681 | -81.87 | 1.9897 | 0.088 | 1.0252 |
| 8 | 32 | 0.596 | 0.2649 | 0.2152 | 1.428 | 0.0295 | 0.3482 | 1.0214 | -74.06 | 2.1588 | 0.123 | 1.0274 |

## B. protocol / window convergence (D=16, beta=50)

| transient_periods | measure_periods | C_I_E1 | C_Q_E1 | C_I_E2 | C_Q_E2 | resid_E1 | resid_E2 | c_a_pot_area |
|---|---|---|---|---|---|---|---|---|
| 1.5 | 3.0 | 0.1755 | 1.4116 | 0.051 | 1.6001 | 0.3366 | 0.3436 | 0.9681 |
| 3.0 | 3.0 | -0.0662 | 1.8766 | -0.2154 | 2.1375 | 0.1243 | 0.125 | 0.9681 |
| 8.0 | 5.0 | -0.9268 | 2.3645 | -1.1832 | 2.7088 | 0.0812 | 0.0795 | 0.9681 |

## C. Stokes-number ladder (D=16, box=4D, t=8/5 periods)

_(no runs)_

## D. independent estimator E3 (global momentum balance)

| beta_nominal | C_I_E1 | C_I_E2 | C_I_E3 | C_Q_E1 | C_Q_E2 | C_Q_E3 | resid_P | c_a_pot_area | E1_vs_E3_delta |
|---|---|---|---|---|---|---|---|---|---|
| 50.0 | -0.0662 | -0.2154 | 0.0103 | 1.8766 | 2.1375 | 3.2789 | 0.0168 | 0.9681 | -0.0765 |

## E. diagnostics (compressibility / mass budget / pressure dipole)

| arm | U_amp_box_mean | drho_max_last_period | mass_rel_drift | p_dipole_amp | p_dipole_over_potential | pbar_amp | pbar_over_dipole | pbar_resid |
|---|---|---|---|---|---|---|---|---|
| e3_beta_50 | 0.040347 | 0.20866 | -0.064578 | 0.088885 | 1.768317 | 0.001129 | 0.012705 | 0.956407 |
