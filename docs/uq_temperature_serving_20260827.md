# UQ temperature knob for serving — landing of the #251 audit (2026-08-27)

Branch `exp/uq-temp`.  Module change: `src/tensorlbm/ai/inference_service.py`
(`ENV_UQ_TEMPERATURE`, `resolve_uq_temperature`, `DragSurrogateService(
uq_temperature=...)`); tests `tests/test_inference_service.py::
TestUqTemperature`.  Numbers below are from `docs/uq_calibration_20260827.md`
(PR #251, machine-readable audit in `/nfs/wangxi/runs/uq_calibration_20260827/`).

## Why

The #251 audit established that the served deep-ensemble `std` is
**conditional on what the query means**:

| semantics | evidence (linear space) | verdict on raw std |
|---|---|---|
| in-envelope (v4 fit/val/test, random holdout) | cov95 93.0 / 87.9 / 89.1 %, rms z 1.15–1.59; T(fit half) 1.15, test rms z 1.35 | near-calibrated — **keep T = 1.0** |
| new-design (out-of-fold halves, fam-350 retrain) | T = 2.21 (B→A): cov95 70.3 → 94.3 %; T = 2.50 (A→B): 84.0 → 96.0 % | ensemble σ under-disperses ~2× for designs not in training |

So a deployment that wants `std` to mean "expected error scale on a **new**
design" should report `2.3 × std`; a deployment answering in-corpus queries
should not scale at all.  The knob makes that a configuration choice instead
of a code fork — **explicit, default off** (T = 1.0 is bit-for-bit the
pre-knob behaviour; the multiply is skipped entirely).

## Usage

```python
# arg (highest priority)
DragSurrogateService(backend, guard, uq_temperature=2.3)
DragSurrogateService.from_run_dir(run_dir, uq_temperature=2.3)
DragSurrogateService.from_checkpoints(paths, cond, uq_temperature=2.3)

# env (used when the arg is None; read once at service construction,
# like TENSORLBM_DRAG_BACKEND)
export TENSORLBM_DRAG_UQ_TEMPERATURE=2.3
```

Precedence mirrors the #252 backend-knob convention: **arg >
`TENSORLBM_DRAG_UQ_TEMPERATURE` > 1.0**.  The applied value is reported in
`DragCurveResult.info["uq_temperature"]` (and therefore in the HTTP
response `info` of `app/backend/routers/drag_surrogate.py`, which whitelists
floats).  Values must be finite and positive; a blank env var counts as
unset.

## Semantics audit — what T touches and what it must never touch

Touched (scaled by T):

* `DragCurveResult.std` and `uq_dict()["std"]` / `["mean_std"]` — the
  reported ensemble σ (router `UQOut.std` / `UQOut.mean_std`).

Never touched (audited 2026-08-27, `grep -rnE 'std *[<>]|mean_std *[<>]'`
over `app/`, `src/tensorlbm/ai/` serving paths — zero σ-threshold
conditionals exist):

1. **Verdict (`ok` / `review` / `reject`)** — produced solely by
   `EnvelopeMahalanobisGuardrail.check`: per-dimension envelope bounds plus
   shrunk-covariance Mahalanobis distance against chi-square-calibrated
   cuts (4.49 / 5.13 at D = 8).  The guard never reads the ensemble std, so
   no temperature can shift a flag.  Pinned by
   `test_verdict_invariant_under_temperature`.
2. **`EchoResult.confident`** (`drag_echo` router) —
   `guard.flag == FLAG_OK and not unsupported_channels`; guard-driven, no σ.
3. **`lo` / `hi`** — the member min-max band, a different statistic from σ
   (an empirical member range, analysed as PICP separately in #251).
   Decision: **not scaled** — inflating a min-max range by a Gaussian
   calibration factor would misreport it.
4. **`cd` / member predictions** — the mean and every member value are
   unscaled; the knob is report-side only.

Known coverage boundary: the echo pipeline
(`src/tensorlbm/ai/geometry_pipeline.py` → `drag_echo` router) drives the
backend and calls `ensemble_stats` itself, i.e. it does not go through
`DragSurrogateService.predict` and is **not** affected by this knob.  That
surface belongs to the parallel geometry workflow; if it adopts the knob
later, the same sigma-only semantics apply.

## Warnings

* T = 2.3 is for **new-design semantics** (fresh geometry axis /
  out-of-fold discipline).  In-envelope serving stays at 1.0 — the audit
  explicitly recommends against deploying a temperature fit on training
  residuals (that gives T ≈ 1.15 and buys ~2 coverage points).
* Temperature does **not** repair extrapolation: LOHO/LOFO folds have
  rms z 3.4–27 and heavy tails; on `fam_slender` two members blow up by
  orders of magnitude.  A scalar cannot fix a biased mean — the guard
  (which caught 112/112 out-of-family points at ≥ 15 % error) is the right
  instrument there.
* z is non-Gaussian in the tails (excess kurtosis up to ~12): treat
  T-scaled 95/99 % bands as empirical, not exact Gaussian intervals.

## Test map (`tests/test_inference_service.py::TestUqTemperature`)

| guarantee | test |
|---|---|
| arg > env > 1.0, blank env = unset, bad values raise | `test_resolver_*` |
| default T = 1.0 bit-identical (std/cd/lo/hi/guard) | `test_default_is_bit_identical` |
| T = 2.3 reports std exactly × 2.3; cd/lo/hi untouched | `test_temperature_scales_reported_std_exactly` |
| verdict (flag, score, reasons) invariant under T | `test_verdict_invariant_under_temperature` |
| env reaches `from_run_dir` service; arg beats env | `test_env_reaches_service_and_arg_wins` |
