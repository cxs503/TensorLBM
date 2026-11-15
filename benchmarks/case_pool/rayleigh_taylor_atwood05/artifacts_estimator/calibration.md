# Front-estimator calibration (CASE-RT-INSTABILITY-ATWOOD, finite-density-ratio arm)

`sc_g = 2.8`, `tau = 1.0`, `64 x 256` box, `precondition_steps = 600`, `g = 0.0002` (lattice).

## K1 seed readback (relaxed profile displaced by a known single mode)

| a0/lam | column amp | column err | gradient amp | gradient err | lock spread [rows] | peak ratio |
|---|---|---|---|---|---|---|
| 0.005 | +0.00015 | -96.9% | +0.00580 | +16.0% | 1 | 0.974 |
| 0.010 | -0.00005 | -100.5% | +0.00969 | -3.1% | 1 | 0.974 |
| 0.020 | +0.00005 | -99.7% | +0.02011 | +0.6% | 3 | 0.973 |
| 0.050 | +0.00118 | -97.6% | +0.04980 | -0.4% | 6 | 0.975 |
| 0.100 | +0.00857 | -91.4% | +0.10041 | +0.4% | 13 | 0.974 |
| 0.200 | +0.03956 | -80.2% | +0.20017 | +0.1% | 26 | 0.974 |

## K3 hydrostatic common mode (profile x (1 + eps y/H))

| eps | column amp | column err | column common | gradient amp | gradient err | gradient common |
|---|---|---|---|---|---|---|
| +1.536e-01 | -0.00086 | -104.3% | -0.16522 | +0.02011 | +0.6% | +0.00003 |
| +4.608e-01 | -0.00152 | -107.6% | -0.25560 | +0.02058 | +2.9% | +0.00041 |
| -1.536e-01 | +0.00230 | -88.5% | +0.29047 | +0.02011 | +0.5% | -0.00015 |

## Relaxed-interface geometry (sharp-interface reference validity)

10-90 % transition width 9 rows = 0.140625 lam -> k delta = 0.8835729338221293; sharp interface (k delta < 0.2) ok: False. Measured bulks rho_low ~ 0.3470443839888961, rho_high ~ 1.1048327314417592 against the coexistence targets 0.2545369048466741 / 1.2078184525247444.


(K2 not requested)


Wall depletion layer (rows 1-4 set to 0.05): column amp -0.00371 / spike -0.00329; gradient amp +0.02011 / spike -0.01909.

## K4 coexistence convergence (sc_g = 2.8, tau = 1.0)

| ny | steps | rho_high | rho_low | ratio | A | wall time [s] |
|---|---|---|---|---|---|---|

A spread nan -> u_s reference changes +nan% across the spread.

## K5 physics attempt

| quantity | value |
|---|---|
