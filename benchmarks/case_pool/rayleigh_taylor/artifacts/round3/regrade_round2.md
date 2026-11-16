# Round-2 recorded velocities re-graded (both conventions)

| case | scheme | A | lam | u_measured | ref(C=3) | err(C=3) | pass | ref(C=1) | err(C=1) | pass | u_b/u_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A050_l064_g1e-4 | allen_cahn | 0.500 | 64 | 0.03024 | 0.01843 | +64.1% | FAIL | 0.03192 | -5.3% | PASS | 0.9998 |
| A050_l064_g2e-4 | allen_cahn | 0.500 | 64 | 0.04769 | 0.02606 | +83.0% | FAIL | 0.04514 | +5.7% | PASS | 1.0000 |
| A033_l064_g2e-4 | allen_cahn | 0.333 | 64 | 0.03664 | 0.02128 | +72.2% | FAIL | 0.03685 | -0.6% | PASS | 0.9998 |
| A050_l064_g2e-4_legacy | bgk | 0.500 | 64 | 0.05229 | 0.02606 | +100.7% | FAIL | 0.04514 | +15.9% | FAIL | 0.9958 |
