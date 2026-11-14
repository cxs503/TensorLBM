# R2 resolution study — numbers (programmatic dump, do not hand-edit)

* window T = [0.5, 4.0], dT = 0.25, g = 5e-05, tau = 1, h0 = 2a, nx = 5a, ny = 3a
* hypothesis: H1: E(a) = E_inf + C a^-p with E_inf < 3 % on Z_sub(f=0.05)
* thresholds: {'z_max_rel_err': 15.0, 'inventory_drift': 1.0, 'asymptotic_target': 3.0}
* references: M&M (experiment, T <= 4), Ritter (exact shallow water, T <= 2)

## Ladder (TensorLBM Körner, tau = 1) — max |rel err| % vs M&M / Ritter

| a | steps | u_max | inv drift % | Z_cell M&M | Z_cell Ritter | Z_sub.02 M&M | Z_sub.05 M&M | Z_sub.10 M&M | Z_sub.05 Ritter | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 3200 | 0.2377 | 0.0322 | 28.161 | 35.417 | 26.776 | 27.394 | 28.527 | 34.365 | 65.32 |

## Observed-order fits E(a) = E_inf + C a^-p — vs Martin & Moyce 1952

| metric | p | E_inf % | C | rms % | monotone | H1 (E_inf<3%) |
|---|---|---|---|---|---|---|
| Z_cell | — | — | — | — | — | fit failed: need >= 3 finite samples (got 1) |
| Z_cell_flag | — | — | — | — | — | fit failed: need >= 3 finite samples (got 1) |
| Z_sub0.02 | — | — | — | — | — | fit failed: need >= 3 finite samples (got 1) |
| Z_sub0.05 | — | — | — | — | — | fit failed: need >= 3 finite samples (got 1) |
| Z_sub0.1 | — | — | — | — | — | fit failed: need >= 3 finite samples (got 1) |

## Richardson extrapolation — vs Martin & Moyce 1952

* `Z_cell`: not available (need >= 3 grids)
* `Z_cell_flag`: not available (need >= 3 grids)
* `Z_sub0.02`: not available (need >= 3 grids)
* `Z_sub0.05`: not available (need >= 3 grids)
* `Z_sub0.1`: not available (need >= 3 grids)

## Observed-order fits E(a) = E_inf + C a^-p — vs Ritter 1892 exact SW

| metric | p | E_inf % | C | rms % | monotone | H1 (E_inf<3%) |
|---|---|---|---|---|---|---|
| Z_cell | — | — | — | — | — | fit failed: need >= 3 finite samples (got 1) |
| Z_cell_flag | — | — | — | — | — | fit failed: need >= 3 finite samples (got 1) |
| Z_sub0.02 | — | — | — | — | — | fit failed: need >= 3 finite samples (got 1) |
| Z_sub0.05 | — | — | — | — | — | fit failed: need >= 3 finite samples (got 1) |
| Z_sub0.1 | — | — | — | — | — | fit failed: need >= 3 finite samples (got 1) |

## Richardson extrapolation — vs Ritter 1892 exact SW

* `Z_cell`: not available (need >= 3 grids)
* `Z_cell_flag`: not available (need >= 3 grids)
* `Z_sub0.02`: not available (need >= 3 grids)
* `Z_sub0.05`: not available (need >= 3 grids)
* `Z_sub0.1`: not available (need >= 3 grids)

## tau sweep at a = 32

| tau | nu | Z_sub.05 M&M % | Z_sub.05 Ritter % | Z_cell M&M % | inv drift % | u_max |
|---|---|---|---|---|---|---|

## OSS port rungs (matrix completion)


## Hydrostatic control (declared in R1 design, never executed there)

| arm | steps | |u|_max | gate | pass |
|---|---|---|---|---|
