# `benchmarks/keel_drag` — ice-keel form/skin drag campaign (P46W3-N01)

Campaign driver and analysis for
[`tensorlbm.keel_drag`](../../src/tensorlbm/keel_drag.py): the skin/form
decomposition of the ice–water drag coefficient `C_d = C_d^skin + C_d^form` for a
keel field parameterised by depth `h_k`, base width `b_k` and spacing `s_k`.

## Run

```bash
# full campaign, one process (laminar channel: ~1-4 min/case on 8 threads)
python benchmarks/keel_drag/run_campaign.py --out benchmarks/keel_drag/campaign

# sharded (cluster / multi-core): 4 shards write into the same directory
for i in 0 1 2 3; do
  python benchmarks/keel_drag/run_campaign.py --out benchmarks/keel_drag/campaign \
      --shard $i --nshards 4 --threads 32 &
done

# criteria (a)-(e) + full-scale C_d + figures
python benchmarks/keel_drag/analyse.py --results benchmarks/keel_drag/campaign
```

Outputs: `campaign/cases/<case_id>.json` (one record per run, including the
failing ones), `results/cases.json` (merged), `results/verdicts.json` (the
machine-checked verdicts, the field-scale prediction and the figure list) and
`results/figures/*.png`.

## Case table (predeclared)

| family | geometry | purpose |
|---|---|---|
| `smooth` | level ice | negative control (e), smooth baseline for `ΔC_F` (d) |
| `depth` | `b_k = 16`, `s_k = 32`, `h_k = 1…8` | criterion (b), `C_D(h_k)`, criterion (e) intercept |
| `width` | `h_k = 4`, `s_k = 32`, `b_k = 8…24` | geometry parameterisation `b_k` |
| `spacing` | `h_k = 2`, `b_k = 4`, `s_k = 8/16/32` | geometry parameterisation `s_k` |
| `reynolds` | `h_k = 4`, `b_k = 16`, `s_k = 32`, `Re_tau = 10…80` + level-ice partners | criterion (c) and `ΔC_F` at matched `Re` |
| `streamwise` | `θ = 0` ridges | structural control: no x-normal faces ⇒ `C_d^form = 0` |

Grid `32×32×66`, `u_tau = 0.002`, `Re_tau = 20` unless stated, 12 000 steps with
3 000 warm-up, `Δt` = 20 samples. One-at-a-time sweeps so that each criterion is
attributed to a single parameter.

## Predeclared acceptance criteria (mission P46W3-N01)

| # | criterion | tolerance | where |
|---|---|---|---|
| a | `C_d^skin + C_d^form = C_d^total` (total from the imposed momentum balance `ρ g h_eff`) | ≤ 1 % | `criterion_a_closure` |
| b | `C_d^form` increases with `h_k` | Pearson `r ≥ 0.90` | `criterion_b_form_vs_depth` |
| c | `C_d^skin` decreases with `Re` | rank + log-log `r ≤ −0.80`, exponent < 0 | `criterion_c_skin_vs_re` |
| d | `ΔC_F` vs the ITTC-78 allowance | ≤ 15 % | `criterion_d_ittc78` |
| e | `h_k → 0` ⇒ `C_d^form → 0` ⇒ `C_d = C_d^skin` | 5 % | `criterion_e_negative_control` |

Reading notes that were fixed **before** the campaign ran:

* (a) uses the *imposed* momentum balance as the total, not the link-exchange sum,
  because `form + skin = measured` is an implementation identity; the criterion is
  therefore a momentum-budget test and cannot be satisfied by construction.
* (c) is read as a rank correlation plus a log–log power law, not as a raw
  Pearson coefficient: over the 60× Reynolds span of the sweep a perfect `1/Re`
  decay has a raw Pearson of only ≈ −0.4. Both numbers are reported.
* (d) is reported under four `(k_s, L)` conventions, each with an explicit
  validity flag for the ITTC-78 calibration window `1e-6 ≤ k_s/L ≤ 1e-4`.

## Laminar amplitude renormalisation

`ν` is pinned by `ν = u_tau h_eff / Re_tau` at `u_tau = 0.002`, so the viscous
spin-up time of the channel is `~3e5` steps and no affordable run is amplitude
converged. Because steady laminar flow is linear in the forcing amplitude, the
mismatch is measurable from the wall stress (`f = τ_measured/ρ g h_eff`) and is
removed from the *initial condition* by a fixed-point iteration
(`init_scale ← init_scale/f`). Nothing is rescaled after the run:
`|f − 1|` of the production pass **is** criterion (a).