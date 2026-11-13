### Operator table (ref = OSS literal `average_surround`, float64)

| case | active nbrs | ref rho | ref ux | got rho (avg) | rel_f (avg) | got rho (fixed) | rel_f (fixed) |
|---|---|---|---|---|---|---|---|
| C1_homogeneous | 3 | 1.000 | -0.019 | 1.000 | 2.40e-08 | 1.000 | 2.40e-08 |
| C2_pressure_gradient | 3 | 1.002 | -0.019 | 1.002 | 2.40e-07 | 1.000 | 2.00e-03 |
| C3_density_contrast | 3 | 1.300 | -0.019 | 1.300 | 4.66e-08 | 1.000 | 0.23 |
| C4_multiphase_like | 3 | 0.500 | -0.049 | 0.500 | 2.43e-08 | 1.000 | 1.00 |

### Isolated-cell case (documented deviation)

active neighbours = 0; OSS ref rho = 0.000 (set_equi -> empty population); TensorLBM `average_surround` rho = 1.000 (population 1.000); TensorLBM `fixed_rho` rho = 1.000; historical zero fallback (`rho_init=None`) rho = 0.000 (population 0.000) = OSS state reproduced.

### Sensitivity tables

| u amplitude | rel_f (average_surround) | rel_f (fixed_rho) |
|---|---|---|
| 0.005 | 2.01e-08 | 2.00e-03 |
| 0.02 | 2.40e-07 | 2.00e-03 |
| 0.08 | 1.18e-07 | 2.00e-03 |

| d(rho)/dx per cell | rel_rho (average_surround) | rel_rho (fixed_rho) |
|---|---|---|
| 0.0 | 5.96e-08 | 5.96e-08 |
| 0.02 | 1.93e-07 | 0.04 |
| 0.05 | 1.30e-07 | 0.09 |
| 0.15 | 1.47e-07 | 0.23 |

### System table (2D dam break)

| grid | steps | mode | births (cells) | inventory | rel drift | front [cells] | iface | finite |
|---|---|---|---|---|---|---|---|---|
| 32x64 | 160 | legacy | 6 | 300.102677 | 3.42e-04 | 16 | 37 | True |
| 32x64 | 160 | fixed_rho | 6 | 300.102692 | 3.42e-04 | 16 | 37 | True |
| 32x64 | 160 | average_surround | 6 | 300.102677 | 3.42e-04 | 16 | 37 | True |
| 64x128 | 80 | legacy | 0 | 1271.284538 | 2.24e-04 | 31 | 72 | True |
| 64x128 | 80 | fixed_rho | 0 | 1271.284538 | 2.24e-04 | 31 | 72 | True |

### Mode divergence (legacy baseline vs selected mode)

| grid | mode | rel L-inf df | front legacy | front mode | front rel diff | inv rel diff | <15 % |
|---|---|---|---|---|---|---|---|
| 32x64 | fixed_rho | 1.04e-03 | 16 | 16 | 0.00% | 0.000% | yes |
| 32x64 | average_surround | 0.00 | 16 | 16 | 0.00% | 0.000% | yes |
| 64x128 | fixed_rho | 0.00 | 31 | 31 | 0.00% | 0.000% | yes |

### Verdict

- operator max rel_f, `average_surround`: **2.40e-07** (PASS vs 0.15)
- operator max rel_f, `fixed_rho`: **1.00** (FAIL vs 0.15)
- system front max rel diff across modes: **0.00%**
- system inventory max rel drift: **3.42e-04**
- all runs finite: **True**
- runtime: 20.7 s
