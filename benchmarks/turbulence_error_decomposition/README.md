# Turbulence `u+` error decomposition + Taylor-Green AMR conservation

Two coupled studies that answer two different questions with the same discipline
("report the ledger, not the verdict alone"):

1. **How much of the `u+` gap is noise, how much is box-length bias, and how much
   is the sub-grid closure?**  -> `tensorlbm.statistics_error_budget`,
   `tensorlbm.uplus_error_decomposition`.
2. **Does the 2:1 block-AMR reflux correction actually conserve mass, momentum
   and interface flux?**  -> `tensorlbm.taylor_green_amr`.

## Pipeline

```bash
# 1. produce wall-normal time series (2-D LES channel, one case per invocation)
PYTHONPATH=src python benchmarks/turbulence_error_decomposition/run.py \
    --case base --box-factor 2.0 --cs 0.1 \
    --warmup 30000 --samples 34000 --sample-every 2 \
    --out runs/uplus_error_decomposition/base.npz

#    base   : Lx = 2*pi*delta, C_s = 0.1   (production)
#    long   : Lx = 4*pi*delta, C_s = 0.1   (domain-length arbitration)
#    nosgs  : Lx = 2*pi*delta, C_s = 0     (SGS negative control)

# 2. statistics + AMR study + figures + machine-readable ledger
PYTHONPATH=src python benchmarks/turbulence_error_decomposition/analyze.py \
    --case base=runs/uplus_error_decomposition/base.npz \
    --case long=runs/uplus_error_decomposition/long.npz \
    --case nosgs=runs/uplus_error_decomposition/nosgs.npz \
    --amr-steps 32 --amr-horizons 4 8 16 32

# 3. report (numbers injected from the JSON, never hand-copied)
PYTHONPATH=src python benchmarks/turbulence_error_decomposition/make_report.py
```

`runs/` is git-ignored by repo policy; every number in the committed artifacts is
regenerable from the commands recorded in the JSON's `provenance` block (which
also stores the sha256 of each input `.npz`).

## What the numbers mean

| Quantity | Where | Reading |
|---|---|---|
| `T_int` (Sokal window) | `statistics_error_budget.integrated_autocorrelation_time` | integration window of the time average, in samples; flagged `converged=False` when the window is not resolvable inside the record |
| `obs_over_t_int` | same ledger | how many correlation times the average actually covers; below ~50 a "converged" profile claim is not supported |
| `variance_inflation` | batch means | ratio of the correlation-aware to the iid standard error; equals `sqrt((1+phi)/(1-phi))` for AR(1) |
| `C2` domain arbitration | `uplus_error_decomposition.arbitrate_domain_length` | max relative `u+` difference between `2*pi*delta` and `4*pi*delta` boxes inside `30 <= y+ <= 0.5 Re_tau` |
| `C3` SGS negative control | `uplus_error_decomposition.sgs_negative_control` | the criterion *fails* when switching the closure off changes nothing |
| `interface_error_profile` | `taylor_green_amr` | rms(AMR - uniform fine)/max|u| versus depth from the interface; proves the AMR error is an interface boundary layer |
| `interface_residual_over_raw` | `taylor_green_amr` | post-reflux interface flux residual relative to the transported flux (machine precision when no correction is capped) |

## Baseline knowledge this benchmark encodes

* A 2-D channel at this parameterisation **relaminarises**: `u_max ~ u_tau*Re_tau/2`
  forces `Ma > 0.1` for any resolved `Re_tau` unless `u_tau << 1`, and a 2-D flow
  has no vortex stretching.  `parabolic_fit_relative_residual` in the ledger is
  the machine-readable symptom.  Treat C1-C3 numbers here as an end-to-end
  rehearsal of the estimators, not as wall-law physics validation.
* Refinement conservation and refinement accuracy are **different budgets**:
  mass/flux/momentum hold to `1e-15` in float64 while the AMR-vs-uniform-fine L2
  difference is a per-step interface error that accumulates with the horizon.
* The reflux correction is momentum-neutral by construction;
  `interface_momentum_applied_ratio_min == 1` is the check that it really is.