# Validation matrix — reference vs got (generated, zero hand-copy)

generated: 2026-11-12T16:12:52Z | matrix v1.1 | **passed 6 / failed 4 / pending 1**

Gate: |err| <= tol at every gated level AND the level series converges (span <= declared max_span_pct; monotone where required)

Every number below is read from the case artifact and judged by `criteria.py`; the reference column carries its in-repo anchor and the sha256 of the file the anchor was read from.

## Per-level comparison

| case | slot | metric | level | reference | got | err % | tol % | pass | status |
|---|---|---|---|---|---|---|---|---|---|
| `cavity_re100` | cavity | max_abs_dev_pct | 128x128 | — | 0.751372 | 0.751372 | 3 | yes | **passed** |
| `cavity_re100` | cavity | max_abs_dev_pct | 192x192 | — | 0.726341 | 0.726341 | 3 | yes | **passed** |
| `cylinder_re100` | cylinder | cd_err_pct | D48 | — | -1.53 | -1.53 | 3 | yes | **passed** |
| `cylinder_re100` | cylinder | cd_err_pct | D64 | — | -2.55 | -2.55 | 3 | yes | **passed** |
| `cylinder_re100` | cylinder | st_err_pct | D48 | — | 0.7 | 0.7 | 3 | yes | **passed** |
| `cylinder_re100` | cylinder | st_err_pct | D64 | — | -2.4 | -2.4 | 3 | yes | **passed** |
| `sphere_re100` | sphere | cd | D40 | 1.09173 | 1.12198 | 2.77076 | 3 | yes | **passed** |
| `sphere_re100` | sphere | cd | D60 | 1.09173 | 1.11946 | 2.54009 | 3 | yes | **passed** |
| `channel_turb_re180` | channel | uplus_rms_err_pct | coarse | — | nan | nan | 3 | NO | **failed** |
| `channel_turb_re180` | channel | uplus_rms_err_pct | fine | — | nan | nan | 3 | NO | **failed** |
| `taylor_green_2d` | taylor_green | gamma_e | first-half | 0.00061685 | 0.000616395 | -0.0738252 | 1 | yes | **passed** |
| `taylor_green_2d` | taylor_green | gamma_e | second-half | 0.00061685 | 0.00061673 | -0.0195285 | 1 | yes | **passed** |
| `static_droplet_laplace_ci` | static_droplet | laplace_max_dev_pct | R=8/12/16 | — | 0.821534 | 0.821534 | 3 | yes | **passed** |
| `static_droplet_laplace_ci` | static_droplet | laplace_max_dev_pct | R=12/16/20 | — | 0.591709 | 0.591709 | 3 | yes | **passed** |
| `static_droplet_laplace_ci` | static_droplet | sigma_pair_dev_pct | R=8/12/16 | — | 0.258157 | 0.258157 | 3 | yes | **passed** |
| `static_droplet_laplace_ci` | static_droplet | sigma_pair_dev_pct | R=12/16/20 | — | 0.258157 | 0.258157 | 3 | yes | **passed** |
| `rayleigh_taylor_rt` | rayleigh_taylor | — | — | — | — | — | — | — | **failed** |
| `dam_break_ritter_ci` | dam_break | front_err_pct | nx=200 | — | 4.51248 | 4.51248 | 5 | yes | **failed** |
| `dam_break_ritter_ci` | dam_break | front_err_pct | nx=400 | — | 2.56249 | 2.56249 | 5 | yes | **failed** |
| `dam_break_ritter_ci` | dam_break | profile_l2_pct | nx=200 | — | 9.8248 | 9.8248 | 5 | NO | **failed** |
| `dam_break_ritter_ci` | dam_break | profile_l2_pct | nx=400 | — | 6.29158 | 6.29158 | 5 | NO | **failed** |
| `dam_break_ritter_ci` | dam_break | mass_drift_pct | nx=200 | — | 2.20934e-14 | 2.20934e-14 | 1 | yes | **failed** |
| `dam_break_ritter_ci` | dam_break | mass_drift_pct | nx=400 | — | 2.21489e-14 | 2.21489e-14 | 1 | yes | **failed** |
| `ship_resistance_suboff` | ship | cd_tot_err_pct | L=48 | — | -3.17 | -3.17 | 0 | NO | **passed** |
| `ship_resistance_suboff` | ship | cd_tot_err_pct | L=80 | — | -1.15 | -1.15 | 3 | yes | **passed** |
| `ship_wave_michell` | ship | — | — | — | — | — | — | — | **pending** |
| `cylinder_re100_dfg` | cylinder | cd_err_pct | D28 | — | 5.59677 | 5.59677 | 5 | NO | **failed** |
| `cylinder_re100_dfg` | cylinder | cd_err_pct | D40 | — | 1.87284 | 1.87284 | 5 | yes | **failed** |
| `cylinder_re100_dfg` | cylinder | st_err_pct | D28 | — | 1.11332 | 1.11332 | 5 | yes | **failed** |
| `cylinder_re100_dfg` | cylinder | st_err_pct | D40 | — | 0.843238 | 0.843238 | 5 | yes | **failed** |
| `cylinder_re100_dfg` | cylinder | cl_amp_err_pct | D28 | — | 36.0421 | 36.0421 | 5 | NO | **failed** |
| `cylinder_re100_dfg` | cylinder | cl_amp_err_pct | D40 | — | 35.0722 | 35.0722 | 5 | NO | **failed** |
| `cylinder_re100_dfg` | cylinder | dp_err_pct | D28 | — | 10.8745 | 10.8745 | 5 | NO | **failed** |
| `cylinder_re100_dfg` | cylinder | dp_err_pct | D40 | — | 5.14859 | 5.14859 | 5 | NO | **failed** |

## Convergence (span across levels)

| case | metric | errors | span % | max span % | monotone | pass |
|---|---|---|---|---|---|---|
| `cavity_re100` | max_abs_dev_pct | 0.751372 → 0.726341 | 0.0250312 | 3 | yes | yes |
| `cylinder_re100` | cd_err_pct | 1.53 → 2.55 | 1.02 | 3 | no | yes |
| `cylinder_re100` | st_err_pct | 0.7 → 2.4 | 1.7 | 3.5 | no | yes |
| `sphere_re100` | cd | 1.12198 → 1.11946 | 0.0025184 | 3 | yes | yes |
| `channel_turb_re180` | uplus_rms_err_pct | nan → nan | nan | 3 | no | NO |
| `taylor_green_2d` | gamma_e | 0.000616395 → 0.00061673 | 3.3493e-07 | 1 | no | yes |
| `static_droplet_laplace_ci` | laplace_max_dev_pct | 0.821534 → 0.591709 | 0.229825 | 3 | yes | yes |
| `static_droplet_laplace_ci` | sigma_pair_dev_pct | 0.258157 → 0.258157 | 1.2379e-14 | 3 | yes | yes |
| `dam_break_ritter_ci` | front_err_pct | 4.51248 → 2.56249 | 1.94999 | 5 | yes | yes |
| `dam_break_ritter_ci` | profile_l2_pct | 9.8248 → 6.29158 | 3.53322 | 5 | yes | yes |
| `ship_resistance_suboff` | cd_tot_err_pct | 3.17 → 1.15 | 2.02 | 3 | yes | yes |
| `cylinder_re100_dfg` | cd_err_pct | 5.59677 → 1.87284 | 3.72393 | 3 | yes | NO |

## Reference anchors

| case | anchor id | quantity | value / formula | origin | source (sha256) |
|---|---|---|---|---|---|
| `cavity_re100` | `ghia_re100_u_center` | u(0.5, 0.5) on the cavity vertical centreline, Re=100 | -0.20581 | Ghia, Ghia & Shin 1982, JCP 48:387, Table I (Re=100) | `benchmarks/verified/cavity/re100/README.md` (8753e67dc4638457) |
| `cylinder_re100` | `cylinder_cd_re100` | mean drag coefficient of a circular cylinder, Re=100 (2-D) | 1.35 | Braza, Chassaing & Ha Minh 1986, JFM 165 (experiment Cd=1.35+-0.05) | `benchmarks/verified/cylinder/re100/README.md` (150a774768337487) |
| `cylinder_re100` | `cylinder_st_re100` | Strouhal number of a circular cylinder, Re=100 | 0.165 | Braza et al. 1986 (St=0.164-0.165 band; 2-D literature 0.164-0.168) | `benchmarks/verified/cylinder/re100/README.md` (150a774768337487) |
| `sphere_re100` | `sphere_cd_re100` | drag coefficient of a sphere, Re=100 | 1.09173 | Schiller-Naumann correlation Cd=24/Re*(1+0.15 Re^0.687), cross-checked against Clift-Gauvi | `benchmarks/verified/sphere_re100/REFERENCE_AUDIT.md` (171b101e7959739b) |
| `channel_turb_re180` | `moser_channel_uplus` | channel-flow mean velocity profile u+(y+) at Re_tau=180 | DNS profile (tabular, not tabulated in-repo) | Moser, Kim & Mansour 1999, Phys. Fluids 11:943 (Re_tau=178.12) | `benchmarks/pending/channel_turb_3d/README.md` (b54b9ca16252f066) |
| `taylor_green_2d` | `tg2d_gamma_e_theory` | energy decay rate of the 2-D Taylor-Green vortex gamma_E = 4 nu k^2 | 0.00061685 | analytic Taylor-Green solution (energy decays twice as fast as velocity) | `benchmarks/verified/taylor_green_2d/README.md` (4bc2cdd7501418ee) |
| `static_droplet_laplace_ci` | `laplace_law_2d` | static-droplet Laplace pressure jump | Delta p = sigma / R (2-D circular droplet) | Young-Laplace law | `benchmarks/verified/laplace_droplet/README.md` (813960b42c2bb31d) |
| `rayleigh_taylor_rt` | `chandrasekhar_rt` | Rayleigh-Taylor linear growth rate sigma(k) for two viscous fluids | viscous RT dispersion relation (Chandrasekhar 5x5 determinant) | Chandrasekhar 1961, Hydrodynamic and Hydromagnetic Stability, ch. X | `benchmarks/pending/rayleigh_taylor/README.md` (32e4691f3e65d058) |
| `dam_break_ritter_ci` | `ritter_front_law` | dam-break front position x_f(t) | x_f(t) = 2 sqrt(g h0) t | Ritter 1892 self-similar shallow-water solution; implementation anchor in the library docs | `src/tensorlbm/shallow_water.py` (5abb26d72972569d) |
| `ship_resistance_suboff` | `blasius_cf_suboff` | Blasius laminar flat-plate friction coefficient at Re=1000 (wetted-area normalization) | 0.041995 | Blasius Cf = 1.328/sqrt(Re); frozen in the verified artifact as reference.Cf_ref | `benchmarks/verified/suboff_re1000/README.md` (946fa05154563277) |
| `ship_wave_michell` | `michell_cw` | thin-ship wave resistance coefficient C_W(Fn) | Michell (1898) thin-ship integral | Michell 1898; in-repo closed-form implementation src/tensorlbm/wave_wake_michell.py | `benchmarks/wave_wake_michell/REPORT.md` (0f74d6ba1177c67f) |
| `cylinder_re100_dfg` | `cylinder_dfg2d2_cd_re100` | mean drag coefficient of the DFG 2D-2 channel cylinder, Re=100 | 3.23 | Schäfer & Turek 1996, NNFM 48, pp. 547-566, Table 3 (2D-2) | `benchmarks/verified/cylinder/re100_dfg/REFERENCE.md` (c3bc60d7a89e94da) |
| `cylinder_re100_dfg` | `cylinder_dfg2d2_clamp_re100` | lift coefficient amplitude of the DFG 2D-2 channel cylinder, Re=100 | 1 | Schäfer & Turek 1996, NNFM 48, pp. 547-566, Table 3 (2D-2) | `benchmarks/verified/cylinder/re100_dfg/REFERENCE.md` (c3bc60d7a89e94da) |
| `cylinder_re100_dfg` | `cylinder_dfg2d2_st_re100` | Strouhal number of the DFG 2D-2 channel cylinder, Re=100 | 0.3 | Schäfer & Turek 1996, NNFM 48, pp. 547-566, Table 3 (2D-2) | `benchmarks/verified/cylinder/re100_dfg/REFERENCE.md` (c3bc60d7a89e94da) |
| `cylinder_re100_dfg` | `cylinder_dfg2d2_dp_re100` | pressure difference p(0.15,0.2)-p(0.25,0.2), DFG 2D-2, Re=100 (raw, rho=1,U=1) | 2.48 | Schäfer & Turek 1996, NNFM 48, pp. 547-566, Table 3 (2D-2) | `benchmarks/verified/cylinder/re100_dfg/REFERENCE.md` (c3bc60d7a89e94da) |

## Reproduce

- `cavity_re100` (passed): `python benchmarks/verified/cavity/re100/run.py`
- `cylinder_re100` (passed): `python benchmarks/verified/cylinder/re100/run.py`
- `sphere_re100` (passed): `python benchmarks/verified/sphere_re100/run.py`
- `channel_turb_re180` (failed): `python benchmarks/pending/channel_turb_3d/run.py`
- `taylor_green_2d` (passed): `python benchmarks/verified/taylor_green_2d/run.py`
- `static_droplet_laplace_ci` (passed): `python benchmarks/validation_matrix/cases/static_droplet_laplace_ci/run.py`
- `rayleigh_taylor_rt` (failed): `python benchmarks/pending/rayleigh_taylor/run.py`
- `dam_break_ritter_ci` (failed): `python benchmarks/validation_matrix/cases/dam_break_ritter_ci/run.py`
- `ship_resistance_suboff` (passed): `python benchmarks/verified/suboff_re1000/run.py`
- `ship_wave_michell` (pending): `python benchmarks/wave_wake_michell/run.py`
- `cylinder_re100_dfg` (failed): `python benchmarks/verified/cylinder/re100_dfg/run.py --levels 20 28 40 --steps 14000 20000 27200 --threads 8`

Artifacts and sha256: `benchmarks/validation_matrix/BENCHMARK_REF_VS_GOT.json`; run-level provenance: `artifacts/validation_matrix/runs/index.jsonl`.
