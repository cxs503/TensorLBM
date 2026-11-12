# Validation matrix — three-level status (generated)

generated: 2026-11-12T06:10:38Z | matrix v1.1 | **passed 6 / failed 4 / pending 1**

Gate: every metric |err| <= its tolerance at every level AND the level series
converges (span <= 3%, monotone where >= 3 levels exist). `failed` keeps its root
cause; `pending` means evidence is withheld or not yet produced — never a hidden pass.

| # | slot | case | status | key metric (worst level) | convergence | evidence |
|---|------|------|--------|--------------------------|-------------|----------|
| 1 | cavity | `cavity_re100` | **passed** | max_abs_dev_pct +0.75% (tol 3%) | max_abs_dev_pct span 0.03% | `benchmarks/verified/cavity/re100/result.json` |
| 2 | cylinder | `cylinder_re100` | **passed** | cd_err_pct +2.55% (tol 3%) | cd_err_pct span 1.02%; st_err_pct span 1.70% | `benchmarks/verified/cylinder/re100/result.json` |
| 3 | ship | `ship_resistance_suboff` | **passed** | cd_tot_err_pct +3.17% (tol 3%) | cd_tot_err_pct span 2.02% | `benchmarks/verified/suboff_re1000/result.json` |
| 4 | sphere | `sphere_re100` | **passed** | cd +2.77% (tol 3%) | cd span 0.00% | `benchmarks/verified/sphere_re100/result.json` |
| 5 | static_droplet | `static_droplet_laplace_ci` | **passed** | laplace_max_dev_pct +0.82% (tol 3%) | laplace_max_dev_pct span 0.23%; sigma_pair_dev_pct span 0.00% | `benchmarks/validation_matrix/cases/static_droplet_laplace_ci/result.json` |
| 6 | taylor_green | `taylor_green_2d` | **passed** | gamma_e +0.07% (tol 1%) | gamma_e span 0.00% | `benchmarks/verified/taylor_green_2d/result.json` |
| 7 | channel | `channel_turb_re180` | **failed** | uplus_rms_err_pct +nan% (tol 3%) | uplus_rms_err_pct span nan% | `benchmarks/pending/channel_turb_3d/result.json` |
| 8 | cylinder | `cylinder_re100_dfg` | **failed** | cl_amp_err_pct +36.04% (tol 5%) | cd_err_pct span 3.72% | `benchmarks/verified/cylinder/re100_dfg/artifacts/result.json` |
| 9 | dam_break | `dam_break_ritter_ci` | **failed** | profile_l2_pct +9.82% (tol 5%) | front_err_pct span 1.95%; profile_l2_pct span 3.53% | `benchmarks/validation_matrix/cases/dam_break_ritter_ci/result.json` |
| 10 | rayleigh_taylor | `rayleigh_taylor_rt` | **failed** | — | — | `python benchmarks/pending/rayleigh_taylor/run.py` |
| 11 | ship | `ship_wave_michell` | **pending** | — | — | `python benchmarks/wave_wake_michell/run.py` |

## Root causes / blockers

- `channel_turb_re180` **failed** — TERMINAL honest FAIL (benchmarks/pending/channel_turb_3d/README.md): root cause = domain size too small + insufficient statistical sample; the u+ RMS error in the log layer never reaches the 3% gate. The tabular Moser DNS anchor is also not frozen in-repo, so the anchor itself is WITHHELD.
- `cylinder_re100_dfg` **failed** — one or more machine-checked criteria failed
- `dam_break_ritter_ci` **failed** — one or more machine-checked criteria failed
- `rayleigh_taylor_rt` **failed** — Honest FAIL with root cause (benchmarks/pending/rayleigh_taylor/README.md): over 10 (lambda, g) cases the SCMP pseudo-potential model has NO growth window inside the reachable parameter surface (sigma_sim <= 0 or NaN; lambda=280/g=6.5e-6 already at 91% of the vapor-spinodal gravity ceiling), and the earlier VOF module track failed structurally. Two independent model classes fail -> the slot stays failed until a model class with a demonstrable window exists.
- `ship_wave_michell` **pending** — WITHHELD (benchmarks/wave_wake_michell/REPORT.md): the primary criterion 'LBM free-surface C_W within 15% of Michell' was NOT obtained - the 3-D Koerner free-surface kernel measured 9.56 s/step at 4e5 cells, extrapolating to ~35 h for a lambda/dx>=20 Wigley case, outside the node budget. 2-D free-surface substitute: period error 8.8-13.8%. Reported as WITHHELD, not PASS.

## Reproduce

```bash
python benchmarks/validation_matrix/runner.py audit    # reference provenance (zero hand-copy)
python benchmarks/validation_matrix/runner.py check    # re-judge frozen evidence
python benchmarks/validation_matrix/runner.py run --tier ci   # CI-scale cases end-to-end
python benchmarks/validation_matrix/runner.py status --write  # regenerate this table
```

Per-case status + sha256 evidence chain: `benchmarks/{verified,pending}/<case>/validation_status.json`.
