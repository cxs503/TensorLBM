# R4 — Re-confound analysis of the a-ladder (programmatic dump)

* Re = sqrt(g h0) h0 / nu, h0 = 2a, nu = (tau-1/2)/3; Re reference (a=32, tau=1) = **21.7223**
* window T = [0.5, 4.0], primary metric = Z_sub0.05, gates = {'z_max_rel_err': 15.0, 'inventory_drift': 1.0, 'asymptotic_target': 3.0}
* reference: Martin & Moyce (1952), 17 digitised points

## Runs (signed error sign is reported; a '<' zero run lags the experiment)

| label | a | tau | Re | max \|err\| % | signed err @ max % | T@max | err @ T_end % | late mean % | wall clearance (a) |
|---|---|---|---|---|---|---|---|---|---|
| tensorlbm_koerner_a16_tau1 | 16 | 1 | 7.68 | 27.39 | -27.39 | 4.00 | -27.39 | -24.878 | 2.375 |
| tensorlbm_koerner_a24_tau1 | 24 | 1 | 14.11 | 15.88 | -15.88 | 0.50 | -11.55 | -8.44 | 1.917 |
| tensorlbm_koerner_a32_tau1 | 32 | 1 | 21.72 | 14.66 | -14.66 | 0.50 | +1.24 | 4.934 | 1.5 |
| tensorlbm_koerner_a48_tau1 | 48 | 1 | 39.91 | 27.37 | +27.37 | 2.25 | +19.86 | 24.489 | 0.771 |
| tensorlbm_koerner_a64_tau1 | 64 | 1 | 61.44 | 42.19 | +42.19 | 2.75 | +35.21 | 39.458 | 0.25 |

## E(Re) collapse (ladder tau=1 + a=32 tau sweep)

| Re | run | max \|err\| % | signed @ max % | late mean % |
|---|---|---|---|---|
| 7.68 | tensorlbm_koerner_a16_tau1 | 27.39 | -27.39 | -24.878 |
| 14.11 | tensorlbm_koerner_a24_tau1 | 15.88 | -15.88 | -8.44 |
| 21.72 | tensorlbm_koerner_a32_tau1 | 14.66 | -14.66 | 4.934 |
| 39.91 | tensorlbm_koerner_a48_tau1 | 27.37 | +27.37 | 24.489 |
| 61.44 | tensorlbm_koerner_a64_tau1 | 42.19 | +42.19 | 39.458 |

* zero crossing of the signed late-window error: Re* = 18.914 ([14.1091, 21.7223], [-8.44, 4.934])

## Fixed-Re trio (decisive resolution test)

| a | tau | Re | max \|err\| % | signed @ max % |
|---|---|---|---|---|
| 32 | 1 | 21.72 | 14.66 | -14.66 |

## Domain-width control

* (no widened run yet)

## Conclusions (computed)

* `all_runs_pass_15pct` = False
* `n_runs` = 5
* `best_run` = {'label': 'tensorlbm_koerner_a32_tau1', 're': 21.7223, 'max_abs_err_pct': 14.656}
