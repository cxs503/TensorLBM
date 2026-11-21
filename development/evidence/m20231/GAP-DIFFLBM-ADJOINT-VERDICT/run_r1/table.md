### Table 1 -- total vs frozen-field gradient (objective: lift, parameter: disk-centre ordinate y_c)

| resolution | y_c | Re | lift | g_AD_y | g_FD_y | AD-FD rel.err | g_frozen_y | |g_frozen/g_AD| | cos(2D) | cos(1D y_c) | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 14.00 | 7.0 | +1.3127e-02 | -2.2695e-03 | -2.2695e-03 | 1.57e-05 | +1.0706e-03 | 0.472 | -0.9950 | -1.0000 | blocked |
| R1 | 21.00 | 7.0 | -1.3127e-02 | -2.2695e-03 | -2.2695e-03 | 1.57e-05 | +1.0706e-03 | 0.472 | -0.9950 | -1.0000 | blocked |

### Table 2 -- streamwise component and the shipped proxy (tensorlbm.adjoint.adjoint_sensitivity)

| resolution | y_c | g_AD_x | g_frozen_x | g_shipped_x | g_shipped_y | proxy_flag |
|---|---|---|---|---|---|---|
| R1 | 14.00 | +1.043e-04 | -1.577e-04 | +1.689e-05 | -8.471e-04 | True |
| R1 | 21.00 | -1.043e-04 | +1.577e-04 | -1.689e-05 | -8.471e-04 | True |
