# Two-component SC layered Poiseuille — ref vs got (light matrix)

anchor: `XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM` · git `1a222a9cbd61` · torch 2.7.1 · wall 75.0 s

| case | KPI | ref (exact) | got | error | gate | verdict |
|---|---|---|---|---|---|---|
| `control_single_light` | u_peak | 0.060375 | 0.0601741 | 0.33 % | 3 % | PASS |
| `control_single_light` | eps_inf | — | — | 0.33 % | 3 % | PASS |
| `control_single_light` | Q_total | 0.885562 | 0.882258 | 0.37 % | 3 % | PASS |
| `layered_r167_light` | u_peak | 0.0231563 | 0.0241088 | 4.11 % | 15 % | PASS |
| `layered_r167_light` | Q_total | 0.337622 | 0.347945 | 3.06 % | 15 % | PASS |
| `layered_r167_light` | eps_l2_rel | — | — | 7.92 % | 15 % | PASS |
| `layered_r167_light` | eps_inf | — | — | 9.97 % | 15 % | PASS |
| `layered_r167_light` | y_iface | 11 | 11.4801 | 2.18 % | 5 % | PASS |
| `layered_r167_light` | nu_lower | — | — | skip | 15 % | SKIP |
| `layered_r167_light` | nu_upper | — | — | skip | 15 % | SKIP |
| `layered_r233_light` | u_peak | 0.0193375 | 0.017342 | 10.32 % | 15 % | PASS |
| `layered_r233_light` | Q_total | 0.278322 | 0.252036 | 9.44 % | 15 % | PASS |
| `layered_r233_light` | eps_l2_rel | — | — | 12.08 % | 15 % | PASS |
| `layered_r233_light` | eps_inf | — | — | 15.83 % | 15 % | FAIL |
| `layered_r233_light` | y_iface | 11 | 12.7827 | 8.10 % | 5 % | FAIL |
| `layered_r233_light` | nu_lower | — | — | skip | 15 % | SKIP |
| `layered_r233_light` | nu_upper | — | — | skip | 15 % | SKIP |
| `layered_static_light` | spurious_ratio | — | — | 0.00 % | 15 % | PASS |

KPI rows: 14 (12 pass / 2 fail), 4 skipped, diverged runs: 0.

```bash
PYTHONPATH=TensorLBM/src python TensorLBM/xval/two_component_sc_layered_poiseuille/run_benchmark.py --matrix light --out TensorLBM/artifacts/xval_two_component_sc_layered_poiseuille_edm
```
