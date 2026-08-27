# B4-P3b · Active-learning loop — first slice (2026-08-25)

Closes (partially, and measurably) the loop that the #241 echo finding
opened: the serving ensemble answers `ok` in channel space on hull-form
variants while its C_D trend runs OPPOSITE to the B4-fam cache.  This
slice wires the full cycle — guardrail-flagged queries → acquisition of
new design points → labels → incremental retrain → verdicts — and
measures what actually improves.

New code (all on top of main `65cb33c`; nothing #241 touched is edited):

- `src/tensorlbm/ai/active_learning.py` — the loop module
- `tests/test_active_learning.py` — 29 synthetic CPU-only tests
- `benchmarks/b4_al_acquisition_bench.py` — acquisition latency bench
- this document

## The loop

```
             ┌──────────────────────────────────────────────────────────┐
             │  served ensemble (b4_serve, 5 seeds, v3 hand channels)   │
             └───────────────┬──────────────────────────┬───────────────┘
                             │ channel guard            │ C_D trend on
                             │ (Mahalanobis, D=8)       │ hull-form axes
             flag: ok 56/56 ─┘ score 0.61-1.66 ─────────┤ truth: OPPOSITE
                             │                          │ sign (#241 echo)
             honest verdict = channel ∧ axes-envelope → review 56/56
                             │
                             ▼
        propose_acquisition (3 strategies, budget 16, seeded)
                             │
        labels_from_cache (oracle: B4-fam 112 pts)   ← this slice
        [production path: scan_runner → same C_D extractor] ← proven on 3 pts
                             │
        augment_corpus (+16 rows → 290) + retrain_ensemble
        (verbatim b4_serve protocol, 5 seeds)
                             │
                             ▼
        eval_loop → family MAPE, verdict flips, trend-sign test,
                    mother MAPE guard, member std
```

## Module API (`tensorlbm.ai.active_learning`)

| name | role |
| --- | --- |
| `HULLFORM_AXES`, `hullform_component_counts`, `hullform_geo_block`, `hullform_condition_rows` | geometry frontend: params → `condition_v3` rows (bitwise-parity tested vs `geometry_pipeline.py` of #241) |
| `point_param_key`, `FlaggedQuery`, `AcquisitionPoint`, `AcquisitionLabel` | loop records |
| `propose_acquisition(strategy=...)` | `envelope_shell` / `max_disagreement` / `coverage` (all seeded, bitwise reproducible) |
| `labels_from_cache` | oracle labels from `cache_fam.npz` (shape ∧ Re matching) |
| `load_corpus_index`, `corpus_cond_v3`, `corpus_design_keys`, `augment_corpus` | corpus side (read-only inputs) |
| `axes_envelope`, `honest_verdict` | axes-envelope half of the honest verdict |
| `spearman_rho`, `trend_stat`, `TrendStat` | trend-sign test statistics |
| `ServiceSpec`, `nearest_field`, `predict_design` | serving one ensemble snapshot |
| `split_random`, `fit_stats`, `train_member`, `retrain_ensemble` | verbatim b4_serve training protocol |
| `eval_loop`, `LoopReport`, `write_loop_report` | before/after verdict of the whole loop |

## Demo (2026-08-25, run dir `/nfs/wangxi/runs/b4_al_20260825/`)

Inputs: corpus `b4_v4_20260824/cache_v4.npz` (274 rows), families
`b4_fam_20260824/cache_fam.npz` (4 × 28 rows), served checkpoints
`b4_serve_20260824/ckpts/serve_cfull_s{0..4}.pt`.  Roles per family
(stride 4): rows 0 → eval holdout (28), rows 1-2 → flagged queries (56),
row 3 → spare.  GPU 0, budget 16, seed 20260825.

**Premise reproduced.** Channel guard on the 56 flagged queries: `ok`
56/56 (scores 0.61-1.66 vs review threshold 4.49); honest verdict
(channel ∧ corpus axes envelope): `review` 56/56 — the exact #241
failure class.

**Strategy ablation** (acquire 16 → oracle-label → retrain 5 seeds):

| strategy | proposed | oracle-matched | retrain test MAPE % (5 seeds) | holdout family MAPE % |
| --- | --- | --- | --- | --- |
| envelope_shell | 16 | 0 | — (unlabelable) | — |
| max_disagreement | 16 | 5 | 1.71, 1.75, 1.44, 1.51, 1.46 | 16.72 |
| coverage | 16 | 16 | 1.31, 1.08, 1.14, 1.40, 1.54 | **16.07** |

`envelope_shell` proposes a continuous LHS shell — honest behaviour:
continuous hull-form shapes between the family anchors have no oracle
label.  `coverage` proposes exactly the family corners (one axis at a
design-box extreme × that family's own Re levels — cross-family Re
pairings are unlabelable because all 112 cache Re values are distinct),
so it matches 16/16 and wins the holdout.

**Loop verdict** (`loop_report.json`, before → after retrain):

| metric | before | after |
| --- | --- | --- |
| holdout family MAPE (28 pts) | 16.73 % | **16.07 %** |
| honest verdicts on holdout | review 28 | **ok 28** (flips: review→ok 28) |
| member std (mean) | 0.0265 | 0.0450 |
| mother MAPE (274 in-corpus rows) | 0.31 % | 0.51 % |

The verdict flip is legitimate, not guard fatigue: after augmentation the
corpus axes envelope genuinely contains the queried hull-form points, and
reject is never weakened — the flagged queries were `review` purely on
the axes-envelope clause.

**Trend-sign test** (the headline; sweep `l_over_d_mult` 0.75/1.0/1.3 at
Re 92/239/619, Spearman per Re column):

| | sign | mean rho | C_D at Re 92 (0.75/1.0/1.3) |
| --- | --- | --- | --- |
| cache truth | +1 | +1.00 | 9.18 / 11.83 / 17.22 |
| before | −1 | −1.00 | 12.07 / 11.84 / 11.30 |
| after (budget 16) | −1 | −1.00 | 11.81 / 11.74 / 11.62 |

Budget 16 damps the wrong trend — spread at Re 92 shrinks 0.78 → 0.19
(−76 %) — but does not flip the sign.  Budget diagnostics
(`trend_budget.json`, `trend_reps.json` in the run dir):

- the coverage candidate pool saturates at 17 labelable points
  (budgets 32 and 48 both propose 17);
- **17 labels flip the trend sign to +1** (rho +1.00; C_D at Re 92
  11.65 / 11.85 / 14.60, truth 9.18 / 11.83 / 17.22);
- the flip is attributable to the marginal label, not GPU-training
  noise: repeated 5-seed retrains of the SAME 16-label augmentation
  keep sign −1 (spread at Re 92 +0.186, +0.180) while repeated
  retrains of the 17-label augmentation keep sign +1 (spread −2.95,
  −2.96).  The marginal label is
  `with_sail|1|1|l_over_d=1.3|…|re=591.2` — the highest-Re slender
  anchor.

The margin (spread 3.1 vs truth 8.0 at Re 92) is still under-shot:
16-17 family rows against 274 mother rows is ~6 % of the training mass,
and the log10-C_D regression is dominated by the Re direction.  Next
slice should either up-weight family rows, add family rows to the
corpus faster than one loop turn, or move the hull-form signal out of
the 4-channel geo block (the SDF encoder track).

## Determinism

- `propose_acquisition`: bitwise reproducible (rerun yields identical
  point keys; `determinism.json: acquisition_bitwise = true`).
- `retrain_ensemble` on GPU: NOT bitwise — same seeds rerun gives test
  MAPE 1.139 vs 1.183 and different weights (`retrain_bitwise_weights =
  false`).  The b4_serve protocol does not pin
  `torch.use_deterministic_algorithms`; FNO FFT/conv backward kernels
  are nondeterministic on CUDA.  Consequence: trend/MAPE numbers carry
  ~±0.05 pp run-to-run jitter, which is why the budget-16-vs-17
  attribution above uses repeated retrains.

## Production labeling path (real scans)

`scan_launch.py` (run dir) takes acquired points to real simulations:
re-proposes the coverage acquisition deterministically, picks 3 points
spanning the trend axis (l_over_d 0.75, l_over_d 1.30) plus one other
hull-form axis, and runs them through the SAME production chain as the
B4-fam corpus — `suboff_n128`, cumulant, `u_in=0.1`, 4000 steps,
`DragSurveySpec(margin=4, interval=25)`, `ScanExecutor` on GPU 0 —
into `/nfs/wangxi/datasets/scan_suboff_al_demo_20260825/`.
`scan_extract.py` computes C_D with the fam_validate formula
(`2·mean(tail Fx)/(u²·A_proj)`, tail = last 25 %, `A_proj` from the
final-step solid mask) and compares against the oracle labels.

Result (2026-08-25, GPU 0, 65 s wall for all three points):

| point | acquired key | Re | scan C_D | oracle C_D | rel diff |
| --- | --- | --- | --- | --- | --- |
| al0000 | `with_sail\|1\|1\|0.75\|1\|1\|1\|66.68` | 66.7 | 11.7187 | 11.7187 | 0.0 |
| al0001 | `with_sail\|1\|1\|1.3\|1\|1\|1\|66.42` | 66.4 | 22.2322 | 22.2322 | 0.0 |
| al0002 | `with_sail\|1\|1\|1\|1.3\|1\|1\|70.71` | 70.7 | 14.1246 | 14.1246 | 0.0 |

All three completed the full 4000 steps (8 snapshots, 160 drag samples
each, tail drift < 5 %, fields finite).  The rel diff is exactly 0 —
the production labeling path reproduces the oracle labels bit-for-bit
(same chain, same device; the oracle cache was built by the same
machinery).  Cross-device reruns should be expected to differ at the
kernel-rounding level; that check is left to the next slice.

## Bench

`benchmarks/b4_al_acquisition_bench.py` times the acquisition step on
the demo inputs (real mode) or a synthetic corpus (no /nfs needed).

Real mode (demo inputs, GPU 0, budget 16, 512 candidates):

| strategy | latency | proposed | oracle-matched |
| --- | --- | --- | --- |
| envelope_shell | 16.1 s | 16 | 0 |
| max_disagreement | 0.75 s | 16 | 5 |
| coverage | < 0.01 s | 16 | 16 |

Synthetic fallback (no /nfs artifacts) exercises the same code paths
with a pseudo-ensemble std; its match counts are not comparable (the
synthetic queries are not cache rows).

## Production needs / next slice

1. Replace the oracle: route `labels_from_cache` misses to
   `scan_runner` automatically (the 3-point proof here is manual).
2. Family-row weighting or quota in `train_member` — 16-17 rows move
   the trend but under-shoot the margin by ~2.6×.
3. Deterministic GPU training (or CPU fallback) before any auto-loop;
   the current ±0.05 pp jitter is fine for verdicts, not for A/B.
4. Guardrail: the channel-space Mahalanobis guard alone said `ok`
   56/56 on exactly the points where the trend was wrong — the honest
   verdict (axes envelope) is the load-bearing check and should be
   promoted into `inference_service.py` serving proper.
