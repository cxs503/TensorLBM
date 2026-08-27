# UQ calibration + guardrail audit — B4 drag-surrogate service (2026-08-27)

Branch `exp/uq-calib` (new files only).  Machine-readable audit:
`/nfs/wangxi/runs/uq_calibration_20260827/` (`audit.json`, `audit_arrays.npz`,
`report.md`, driver `audit_uq.py`, renderer `make_report.py`).  Metric layer:
`src/tensorlbm/ai/uq_calibration.py`, tests `tests/test_uq_calibration.py`.

The B4 serving contract sells two honest numbers per query: the deep-ensemble
`std` and a guard `verdict` (`ok`/`review`/`reject`).  Neither had ever been
quantified — is `std` a calibrated sigma, and does `ok` imply small error?
This audit answers both, and the answer is conditional: **yes in-envelope,
no under extrapolation, and the guard knows which regime you are in**.

## Protocol

* **Deployed ensemble** — the 5-seed serving checkpoints
  `/nfs/wangxi/runs/b4_serve_20260824/ckpts/serve_cfull_s{0..4}.pt`
  (arm C_full, v4 random split fit=186/val=33/test=55 of the 274-point
  `/nfs/wangxi/runs/b4_v4_20260824/cache_v4.npz` corpus).
* **Fit / residual view** — deployed ensemble evaluated on the v4 corpus
  rows by split membership, and on the 4 geometry families of the
  350-point `/nfs/wangxi/runs/b4_fam_20260824/cache_fam.npz` corpus it
  never saw (112 points).
* **Leave-out / generalisation view** — (a) archived v4 LOHO folds
  (`preds_v4.npz`, C_full members: 5/3/5 seeds for bare_hull/with_sail/full),
  (b) archived B4-fam LOFO folds (`preds_lofo.npz`, C_geoM/F_geoM, 3 seeds,
  geoM condition — a different arm, reported as reference), (c) a fresh
  point-level grouped **5-fold x 5-seed retrain** on the fam-350 corpus with
  the verbatim b4_serve C_full protocol.  Chosen leave-out discipline: 5-fold
  (every corpus point gets exactly one out-of-fold prediction) with the #243
  fit-split rules applied inside every fold — param-key groups never straddle
  folds, val carved from train groups only, fit-stats from fit rows only.
  The two views are tabulated separately and never pooled.
* **Spaces** — linear `C_D` (what the service exposes) and `log10 C_D` (the
  space members regress); coverage at 1 / 1.96 / 2.5758 sigma is nominal
  68.3 / 95 / 99 %.
* **Guard** — `EnvelopeMahalanobisGuardrail` over `condition_v3`, fit on the
  serving fit split (186 rows; review 4.49 / reject 5.13, chi2-calibrated),
  i.e. the deployable configuration; LOHO pools refit the guard on each
  fold's fit rows.  Per-point verdicts via `uq_calibration.row_verdicts`
  (row-wise `guard.check`, exactly the semantics of a single-design query).

## 1. Is the served std a calibrated sigma?

Fit / residual view (deployed ensemble, linear space):

| group | n | MAPE | cov68 | cov95 | cov99 | rms z | KS p |
|---|---|---|---|---|---|---|---|
| v4 fit (trained) | 186 | 0.27% | 64.0% | 93.0% | 97.8% | 1.15 | 0.00 |
| v4 val (early-stop) | 33 | 0.42% | 60.6% | 87.9% | 90.9% | 1.59 | 0.38 |
| v4 test (random holdout) | 55 | 0.35% | 70.9% | 89.1% | 90.9% | 1.35 | 0.64 |

Leave-out view (out-of-fold retrained ensemble on fam-350, linear space):

| group | n | MAPE | cov95 | rms z |
|---|---|---|---|---|
| folds 0-4 (all 350 points) | 69-71 each | 0.34-0.45% | 66.7-88.6% | 1.7-3.8 |
| grouped half A / half B | 174 / 176 | ~0.4% | 70.3% / 84.0% | - |

Leave-out view (extrapolation folds, archived C_full members):

| fold | n | MAPE | cov95 | rms z | PICP(min-max) |
|---|---|---|---|---|---|
| LOHO bare_hull (5 members) | 14 | 2.13% | **0.0%** | 3.4 | 0.0% |
| LOHO with_sail (3 members) | 72 | 1.19% | 75.0% | 1.7 | 47.2% |
| LOHO full (5 members) | 188 | 5.44% | 76.6% | 4.4 | 61.2% |

Deployed ensemble on the 4 unseen geometry families (112 pts): errors
50-54% MAPE on three families; on `fam_slender` two of five members blow up
(predictions of ~3e3-5e4 against truth ~4-23, i.e. the served mean is off by
~3 orders of magnitude) while the band "covers" only because sigma itself is
~3.3 decades.  LOFO-archived 3-seed geoM ensembles on the same families:
cov95 = 0% for `fam_blunt` / `fam_long_nose` (rms z 18-27).

Reading:

* **In-envelope the std is close to a calibrated sigma** — coverage within a
  few points of nominal on trained and randomly held-out points, rms z
  1.15-1.59, and it slightly *under*-covers on held-out rows (test rms z 1.35).
* **Point-level leave-out breaks it by roughly a factor 2** — out-of-fold
  cov95 66-89%, rms z up to 3.8; the ensemble under-disperses for designs it
  was not trained on, exactly the regime a new query lives in.
* **Hull / family extrapolation breaks it completely** — LOHO bare_hull cov95
  = 0%, LOFO folds cov95 = 0% with rms z up to 27; on `fam_slender` the
  deployed mean itself is garbage.  No scalar rescaling fixes a wrong mean.
* **z is not Gaussian in the tails** (KS rejects at most groups, excess
  kurtosis up to ~12 even where coverage is fine) — treat 95/99% bands as
  empirical quantiles of a heavy-tailed z, not exact Gaussian intervals.

## 2. Guardrail ROC — does the verdict discriminate error?

`ok` vs `review`/`reject` as a classifier for |rel err| above an operating
threshold; AUC over the continuous Mahalanobis severity score (NaN = a class
is empty):

| pool | n | AUC@1% | AUC@2% | AUC@5% | AUC@10% |
|---|---|---|---|---|---|
| deployed, v4 fit rows | 186 | n/a (0 large) | n/a | n/a | n/a |
| deployed, v4 test rows | 55 | 0.434 | n/a | n/a | n/a |
| deployed, unseen families | 112 | n/a (0 small) | n/a | n/a | n/a |
| 5-fold OOF retrained | 350 | 0.712 | 0.582 | n/a | n/a |
| LOHO with_sail | 72 | 0.667 | 0.717 | n/a | n/a |
| LOHO full | 188 | 0.549 | 0.689 | 0.941 | 0.964 |

Operating points at the deployed cuts:

* Unseen families: all 112 points are `reject` and all 112 have |err| >= 15%
  — capture 100%, precision 100% at every threshold from 1% to 10%.
* LOHO full (guard refit without the hull): all 188 `reject`, capture 97.6%
  of >10% errors with 10.2% false alarms (precision 72.7%).
* In-envelope pools (v4 fit/val/test, OOF): flagged fractions 0.5-3%, but
  the flagged points are *not* systematically the erroneous ones — AUC at
  1-2% error thresholds is 0.43-0.72, i.e. near chance.

Conclusion: the guard is an **envelope boundary detector, not an error
ranker**.  It separates "corpus contains anything like this design" from
"it does not" essentially perfectly (the 4 geometry families are all caught),
but *within* the envelope its severity score carries almost no information
about which point will be worse — the manual condition space cannot see the
in-envelope failure modes (consistent with the #235/#241 latent-space
motivation and the AL-campaign blind spot).

## 3. What does `ok` actually mean?

Empirical `P(|err| <= t | verdict)` from the pools above:

| pool | verdict | P(err<=1%) | P(err<=2%) | P(err<=5%) |
|---|---|---|---|---|
| deployed, v4 fit | ok (n=183) | 100.0% | 100.0% | 100.0% |
| deployed, v4 test | ok (n=54) | 96.3% | 100.0% | 100.0% |
| deployed, 4 new families | reject (n=112) | 0% | 0% | 0% (all >=15%) |
| OOF 5-fold (fam-350) | ok (n=340) | 90.0% | 99.7% | 100.0% |
| LOHO bare_hull | ok (n=14) | 0% | 21.4% | 100.0% |
| LOHO with_sail | ok (n=42) | 59.5% | 92.9% | 100.0% |
| LOHO with_sail | reject (n=30) | 40.0% | 73.3% | 100.0% |
| LOHO full | reject (n=188) | 22.3% | 42.0% | 68.1% |

Honest contract statement: for a query the corpus genuinely covers, `ok`
means `P(|err| < 1%) ~= 90-100%` and `P(|err| < 5%) ~= 100%`.  For hull-form
configurations whose hull type was never trained (LOHO), `ok` degrades to
`P(|err| < 5%) ~= 100%` but `P(|err| < 2%)` can be as low as 21% — and the
verdict does not distinguish these because the manual channel space is
in-envelope there.  `reject` on out-of-family geometry is reliably
catastrophic error (all 112/112 >= 15%).

## 4. Temperature scaling — needed?

`fit_temperature` returns the closed-form NLL-optimal scalar `T`
(`T = rms z`), fit on one half and validated on the held-out half:

| experiment | space | T (fit half) | held-out cov95 raw -> cal | NLL raw -> cal |
|---|---|---|---|---|
| residual view (v4 fit -> v4 test) | linear | 1.15 | 89.1% -> 90.9% | -1.912 -> -1.992 |
| residual view | log10 | 1.15 | 89.1% -> 89.1% | -4.643 -> -4.722 |
| OOF halves (B -> A) | linear | 2.21 | 70.3% -> 94.3% | -0.100 -> -1.799 |
| OOF halves (A -> B) | linear | 2.50 | 84.0% -> 96.0% | -0.744 -> -1.872 |

* **In-corpus serving (residual view): no temperature needed** — raw std is
  already within ~2-6 coverage points of nominal and T ~ 1.15 buys ~2 points.
* **Generalisation interpretation: a global T ~= 2.2-2.5 is justified and
  transfers across halves** (70.3 -> 94.3% and 84.0 -> 96.0% cov95, large
  NLL gains, consistent between linear and log10 spaces).  If the service
  wants `std` to mean "expected error on a *new* design", serving
  `2.3 * std` (or refitting members with the OOF discipline) is the honest
  number; the current std is a residual-scale sigma, roughly half the
  leave-out error scale.
* Temperature does **not** repair extrapolation: LOHO/LOFO folds have rms z
  3.4-27 with heavy tails — a scalar cannot fix a biased mean or member
  blow-up, and the guard, not the sigma, is the right instrument there.

## 5. Actionable summary

1. Keep serving raw `std` for in-envelope queries, but document it as a
   **residual-scale sigma**: for unseen designs the expected error scale is
   ~2.2-2.5x larger (audited halves both land at 94-96% cov95 after T).
2. `verdict: ok` may be advertised as `P(|err| < 5%) ~= 100%` for
   corpus-covered designs; do **not** advertise sub-2% guarantees for
   hull-form variants whose hull was undertrained (LOHO rows).
3. `review`/`reject` on out-of-envelope geometry should be treated as
   "numbers not trustworthy" — on the 4 geometry families all points erred
   >= 15% and the slender family member blow-up makes mean/lo/hi all
   meaningless.  The guard caught 112/112.
4. Within-envelope error ranking needs a stronger feature space (SDF latents
   per #235/#247), not a threshold retune: AUC 0.43-0.72 at operating cuts.
5. If a global temperature is deployed, fit it on out-of-fold predictions
   (T ~= 2.3 here), never on training residuals (which give T ~= 1.15).

## Reproducibility

25 members retrained on GPU 3 (557 s total, best epochs 77-442), splits
`{v4: 186/33/55, fam-350 folds: 71/70/70/70/69}`; every number above is
mechanically derived from `audit.json` by `make_report.py` (this document is
the curated view of `/nfs/wangxi/runs/uq_calibration_20260827/report.md`).
Module `uq_calibration.py` is import-safe, scipy-free and CPU-testable;
`tests/test_uq_calibration.py` (23 tests) pins coverage/NLL/CRPS on known
Gaussians, the closed-form temperature, ROC tie handling and the protocol
fields.
