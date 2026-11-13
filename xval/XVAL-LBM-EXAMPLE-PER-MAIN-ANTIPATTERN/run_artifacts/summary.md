# XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN -- machine referee

arms_complete=True all_pass=True

| check | value | threshold | pass |
|---|---|---|---|
| C1_channel_umax_within_owner_band | 0.007755102269816609 | 0.15 | True |
| C2_channel_umax_within_3pct_target | 0.007755102269816609 | 0.03 | True |
| C3_profile_l2_target_at_finest | 0.0012242133131070254 | 0.03 | True |
| C4_resolution_monotone_umax_err | [0.007755102269816609, 0.0016908573404970774, 0.0016601286193255705] | - | True |
| C5_mass_drift | 9.550138457825597e-13 | 1e-10 | True |
| C11_stability_rule | - | - | True |
| C6_tg_decay_target | 0.001127977207684141 | 0.03 | True |
| C7_harness_matches_independent_control | 6.124291821339015e-13 | 1e-08 | True |
| C8_determinism | - | - | True |
| C9_detector_discriminates | 0.6212 | 0.2 | True |
| C10_antipattern_present_in_repo | - | 0.2 | True |

## channel u_max vs closed form g H^2 / (8 nu)
| ny | u_max got | u_max ref | rel err | profile L2 | interior L2 | mass drift | steps |
|---|---|---|---|---|---|---|---|
| 16 | 0.00243100 | 0.00245000 | 0.7755% | 0.3633% | 0.3372% | -2.86e-13 | 4400 |
| 32 | 0.01123098 | 0.01125000 | 0.1691% | 0.0793% | 0.0766% | -7.51e-13 | 12000 |
| 48 | 0.02640609 | 0.02645000 | 0.1660% | 0.1224% | 0.1222% | -9.55e-13 | 15000 |

## Taylor-Green energy decay
| got | ref (4 nu k^2) | rel err |
|---|---|---|
| 1.543865e-02 | 1.542126e-02 | 0.1128% |

## independent example-main control (ny=32, torch.roll streaming)
| harness u_max | control u_max | rel diff | profile L2 diff |
|---|---|---|---|
| 0.0112309779 | 0.0112309779 | 6.124e-13 | 6.204e-13 |
