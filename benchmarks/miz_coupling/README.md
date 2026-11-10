# benchmarks/miz_coupling — P46W3-N04 MIZ coupling benchmark

Machine-checkable acceptance benchmark for the marginal-ice-zone resistance and
seakeeping coupling implemented in
`src/tensorlbm/miz_resistance.py` and `src/tensorlbm/miz_seakeeping.py`.

```bash
python benchmarks/miz_coupling/run_acceptance.py   # checks -> results/acceptance.json
python benchmarks/miz_coupling/make_figures.py     # -> figures/*.png
python benchmarks/miz_coupling/make_report.py      # -> /tmp/report.json (API report payload)
python -m pytest tests/test_miz_resistance_seakeeping.py -q
```

## Pre-declared design (fixed before looking at results)

| item | value |
|---|---|
| decision | does the semi-analytical MIZ coupling satisfy the five node criteria over a *sweep*, not one point |
| unit of analysis | one (C_floe, Hs, Tp, h_ice) environment state / one ship |
| baseline / control | `C_floe = 0` everywhere (must reduce exactly to open water); naive superposition (edge wave height) as the "no-coupling" control |
| repetitions | none needed — the models are deterministic closed form, so coverage comes from the grid below, not from repeats |
| thresholds | a: r ≥ 0.9 · b: a/λ ∈ [0.05, 0.20] · c: ≤ 10 % · d: ≥ 20 % · e: ≤ 5 % |
| environment | python 3.11, numpy 2.4, matplotlib Agg; no cluster/HPC needed (full sweep < 10 s) |
| analysis | max (not mean) of the error metrics over the grid — worst case decides |

## Grid

`C_floe ∈ {0.10 … 0.90}` (9), `Hs ∈ {1.0, 1.5, 2.0, 2.5}`, `Tp ∈ {6, 8, 10, 12}`,
`h_ice ∈ {0.3, 0.5, 1.0, 1.5}`, `x_ship = 250 m`, `L_strip = 300 m`.

## Result (from `results/acceptance.json`, `all_pass = true`)

| criterion | threshold | measured |
|---|---|---|
| (a) attenuation α vs C_floe | r ≥ 0.9 | 0.999942 |
| (b) a_floe/λ_peak window | [0.05, 0.20] | [0.0662, 0.1645], mean 0.102, 64 states |
| (c) superposition vs coupled | ≤ 10 % | max 2.01 % (60 states) |
| (d) roll RAO reduction | ≥ 20 % | 23.1 – 76.4 % |
| (e) energy ledger residual | ≤ 5 % | max 2.05 % (18 states) |

Reference-state resistance (C=0.5, Hs=1.5 m, Tp=8 s, h=0.5 m, x_ship=250 m):
`R_ow = 72.3 kN`, `R_wave = 12.8 kN`, `R_ice = 56.3 kN`
(break 14.4 / submergence 51.8 / friction 4.5 kN), `R_total = 141.4 kN`
superposed vs `142.4 kN` coupled (+0.68 %).

Known limitation recorded in the JSON: `mech_over_diss` (floe drag + fracture
power over the dissipative share of α) ranges 0.49–4.93, i.e. the constant-β
split of α into scattering and dissipation cannot be reconciled at low
concentration — see `docs/MIZ_COUPLING_P46W3_N04.md` §4.4.