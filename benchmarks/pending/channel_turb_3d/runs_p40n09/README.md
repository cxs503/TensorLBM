# P40-N09 run artifacts

Everything in this directory is produced by the campaign scripts one level up
(`run_p40n09.py`, `probe_stability_p40n09.py`, `analyze_p40n09.py`,
`build_p40n09_report.py`).  No file here was edited by hand; the two exception
files are marked as transcripts below.

| path | produced by | content |
|---|---|---|
| `D4_cs0_result.json` / `D4_cs0_meta.json` | `run_p40n09.py --case D4_cs0` | reference arm: box 4πδ × 4πδ/3, natural-KBC, C_s = 0, Re_τ = 180 |
| `D2_cs0_result.json` / `D2_cs0_meta.json` | `run_p40n09.py --case D2_cs0` | domain arm: box 2πδ × 4πδ/3, everything else identical |
| `SGS_cs0_result.json` / `SGS_cs0_meta.json` | `run_p40n09.py --case SGS_cs0` | SGS-off arm (cumulant, Re_τ = 30 -- see prereg amendment 1) |
| `SGS_cs01_result.json` / `SGS_cs01_meta.json` | `run_p40n09.py --case SGS_cs01` | SGS-on arm, C_s = 0.1, same grid/point |
| `gates_p40n09.json` | `analyze_p40n09.py` | the six gates with evidence triples `(value, source-field, ok)` |
| `p40n09_numbers.json` | `analyze_p40n09.py` | every number the report quotes, incl. per-batch profiles and CIs |
| `figures_p40n09/*.png` | `analyze_p40n09.py` | the three report figures |
| `stability_probe_retau180.json` | `probe_stability_p40n09.py` | cumulant vs natural-KBC stability at the production grid (fetched artifact) |
| `stability_probe_matrix1.json` | — (**transcript**) | the first probe matrix, transcribed from the job stdout because the job directory was not archived; the transcript keeps the raw numbers and is labelled as such |

## Raw fields the estimators consume

* `statistics.profile_batch_moment_means[b][component][row]` — consecutive batch
  means of the plane-averaged moments (component 0 = u, 6 = u′v′ raw);
* `statistics.profile_batch_counts[b]` — samples per batch (sums to
  `profile_samples`);
* `wall_exchange.shear_series` — wall-shear time series (stride
  `wall_exchange.shear_series_stride`), the input of the autocorrelation
  integral time.

## Timing / hardware

All arms ran on the 外协方 cluster (`n315`, `n330`; loong64, 128 cores each,
torch 2.13.0 CPU, float32).  Per-arm wall clock and `cell_updates_per_s` are in
each `*_meta.json`; the throughput quoted by the budget extrapolation is the
`D4_cs0` value.