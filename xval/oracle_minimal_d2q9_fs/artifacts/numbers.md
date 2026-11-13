# XVAL numbers (programmatic dump — do not hand-edit)

## Run table

| label | arm | a | τ | g | ν | steps | u_max | Z max err % | mean err % | pass 15% | inventory drift % | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tensorlbm_koerner_a16_tau1 | tensorlbm_koerner | 16 | 1 | 5e-05 | 0.1667 | 1200 | 0.2377 | 22.297 | 20.213 | False | 0.0261 | 27.38 |
| oss_port_a16_tau1 | oss_port | 16 | 1 | 5e-05 | 0.1667 | 1200 | 0.8165 | 118.75 | 43.953 | False | 207.8417 | 35.88 |
| tensorlbm_koerner_a32_tau1 | tensorlbm_koerner | 32 | 1 | 5e-05 | 0.1667 | 1698 | 0.2442 | 13.806 | 6.837 | True | 0.043 | 102.82 |
| oss_port_a32_tau1 | oss_port | 32 | 1 | 5e-05 | 0.1667 | 1698 | 0.8165 | 124.377 | 59.606 | False | 213.4972 | 98.33 |
| oss_port_a32_tau1_rho_prev_fixed | oss_port | 32 | 1 | 5e-05 | 0.1667 | 1698 | 0.8165 | 15.961 | 7.914 | False | 19.0807 | 95.0 |
| oss_port_a32_tau1_if_rho_liquid | oss_port | 32 | 1 | 5e-05 | 0.1667 | 1698 | 0.8165 | 73.253 | 24.322 | False | 203.478 | 96.51 |
| oss_port_a16_tau1_g0.1 | oss_port | 16 | 1 | 0.1 | 0.1667 | 27 | 0.8165 | 49.142 | 42.74 | False | 1.7689 | 0.75 |
| tensorlbm_koerner_a16_tau1_g0.1 | tensorlbm_koerner | 16 | 1 | 0.1 | 0.1667 | 27 | 0.5674 | 34.184 | 31.142 | False | 10.3265 | 0.67 |

## Arm-to-arm (TensorLBM vs OSS oracle)

* a=16 τ=1: max 22.727 %, mean 16.971 %, pass15=False
* a=16 τ=1: max 63.636 %, mean 41.153 %, pass15=False
* a=32 τ=1: max 53.797 %, mean 33.898 %, pass15=False

## Controls

