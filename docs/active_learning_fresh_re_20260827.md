# B4-P3b · Fresh-Re acquisition mode + exact-key dedup (2026-08-27)

Closes **G1** of the 2026-08-27 real-labeling campaign
(`/nfs/wangxi/runs/al_campaign_20260827/report.md` §5): the default
`propose_acquisition` cannot serve a real-labeling campaign, because

- (a) its candidate Re values are drawn ONLY from the flagged queries'
  Re values — and flagged queries are served (cached) rows, so every
  candidate duplicates a cached Re;
- (b) the Mahalanobis `_exclusion_floor` is a *duplicate* proxy and
  over-rejects fresh-Re corner points.

This slice adds a **fresh-Re mode** and **exact-key dedup** to
`src/tensorlbm/ai/active_learning.py` — additive, default-off, and the
default path is bit-identical (all 29 pre-existing tests pass unchanged).

## What changed (branch `exp/al-fresh-re`, base main `ce7ab98b`)

| API | role |
| --- | --- |
| `propose_acquisition(..., fresh_re=True)` | opt-in mode: candidate Re levels come from a GRID crossed with the existing geometry arm/corner machinery, not from the flagged (cached) rows |
| `fresh_re_grid: Sequence[float] \| None` | explicit Re grid; default is `default_fresh_re_grid(existing_cond, n_re_levels)` — log-spaced over the corpus Re window (recovered from `existing_cond` channel 0 = `log10 Re`, `.9g`-rounded so the log round-trip is bitwise) |
| `default_fresh_re_grid(existing_cond, n_levels=8)` | the default grid builder (public so a campaign can inspect/log the grid it will label on) |
| `labeled_keys: Sequence[str] \| None` | exact-key dedup in BOTH modes: candidate keys (`point_param_key` layout) already present in the corpus index or labeled in earlier rounds are never (re-)proposed |
| `corpus_point_keys(index)` | `point_param_key`-layout keys of every `cache_v4` corpus row (mother axes 1.0) — feed these plus earlier rounds' keys as `labeled_keys` |
| `AcquisitionPoint.guard_score` / `.floor_pass` | per-point Mahalanobis score and floor verdict, recorded on every returned point in both modes |
| floor semantics | old mode: hard filter (unchanged). Fresh-Re mode: ADVISORY — below-floor candidates are retained with `floor_pass=False` and their `guard_score` recorded |

In fresh-Re mode all three strategies take their Re from the grid:
`coverage` / `max_disagreement` cross every grid level with the corner /
anchor-combo machinery (the own-family Re restriction is dropped — it
exists only because the *oracle cache* never shares Re values across
families, which does not apply to fresh simulations), and
`envelope_shell` draws its Re from the grid values (same LHS axes, same
flagged-score shell band). Everything stays seeded and bitwise
reproducible.

## Why the floor had to become advisory (G1 evidence)

The campaign labeled 24 genuinely-new points (out-of-cache by
exact-key assertion) with real simulations:

- **10 of 24** new points scored BELOW the corpus Mahalanobis floor —
  proven new by exact-key, so the floor's "below floor ⇒ duplicate"
  reasoning is falsified in the fresh-Re corner region.
- The #243 pool itself had already lost the blunt family's
  142.4 / 198.0 / 222.4 spread levels to the floor, and the campaign's
  trend anchor (Re 239.19) sits inside the floor rejection band.
- The labels are real and worth acquiring: 3 same-point control
  re-scans reproduce the cache bitwise (**rel_diff = 0.0** — the solve
  chain has zero drift vs the cache), and on the 24 out-of-cache new
  points within-family log-log interpolation matches truth at
  **median 6.6e-5 / mean 8.6e-5 / max 2.5e-4** (≤ 0.03 %), while the
  nearest cached neighbor differs by median 1.9 % / max 5.7 % — the
  new points are genuine increments, not re-measurements.

Hence in fresh-Re mode the floor is recorded per point
(`guard_score`, `floor_pass` — the same field names the campaign stored
in `acquisition.json`) instead of dropping candidates; exact-key dedup
now does the duplicate exclusion the floor was a proxy for.

## Exact-key dedup in the old (cached-Re) mode

The dedup is a pure safety property and applies whenever `labeled_keys`
is passed, in both modes. Old-mode candidates collide with corpus keys
only in one constructible corner: `max_disagreement`'s all-mother combo
(all four axes 1.0) paired with a flagged Re that is a corpus Re — an
exact corpus duplicate that the floor drops only when it scores below
the corpus's own 10th percentile (i.e. not reliably). `coverage`
corners always carry a moved axis and `envelope_shell` samples
continuous axes, so their keys cannot equal a corpus key (corpus rows
key at mother axes). Nothing about old-mode OUTPUT changes: callers who
do not pass `labeled_keys` get bit-identical proposals (pinned by
`test_default_path_bitwise_unchanged` plus the 29 pre-existing tests).

## Usage: a real-labeling campaign

```python
from tensorlbm.ai.active_learning import (
    corpus_point_keys,
    default_fresh_re_grid,
    load_corpus_index,
    propose_acquisition,
)

index = load_corpus_index(".../cache_v4.npz")
labeled = set(corpus_point_keys(index))  # never re-propose the corpus
# ... after each labeling round: labeled |= {p.key for p in round_points}

grid = default_fresh_re_grid(existing_cond, n_levels=8)  # log-spaced corpus window
points = propose_acquisition(
    flagged_queries,
    strategy="coverage",
    budget=24,
    existing_cond=existing_cond,
    grid=grid_obj,
    fresh_re=True,  # Re from the grid, not the cache
    fresh_re_grid=grid,  # or pass campaign anchors explicitly
    labeled_keys=sorted(labeled),  # exact-key dedup, both modes
)
# every point carries guard_score + floor_pass for triage:
below_floor = [p for p in points if p.floor_pass is False]  # advisory, kept
```

Labels then come from real simulations (the campaign's `launch.py` /
`extract_labels.py` chain: `suboff_n128`, cumulant, `u_in=0.1`, 4000
steps, tail-25 % C_D extractor) and enter the corpus through the
unchanged `augment_corpus` payload interface (campaign finding D3: it
already accepts hand-built real-label payloads).

## Tests (`tests/test_active_learning.py`, 29 -> 37)

New class `TestFreshReAcquisition`:

1. `test_fresh_re_levels_come_from_grid_not_cache` — grid Re values
   absent from both the flagged set and the cached corpus Re;
2. `test_fresh_re_default_grid_spans_corpus_window` — default grid =
   `geomspace(corpus_re_min, corpus_re_max, n_re_levels)` bitwise
   (`.9g` round-trip), and proposals include genuinely fresh Re;
3. `test_fresh_re_all_strategies_on_grid` — `envelope_shell` and
   `max_disagreement` also take Re from the grid;
4. `test_exact_key_dedup_in_both_modes` — `labeled_keys` blocks
   re-proposal in the old mode (safety) and refills the full budget
   disjointly in fresh mode;
5. `test_corpus_point_keys_exclude_corpus_duplicates` — corpus keys
   exclude exact corpus duplicates (the all-mother combo), and only
   corpus keys are dropped;
6. `test_floor_advisory_in_fresh_mode_hard_in_old_mode` — the in-cloud
   blunt/Re-200 candidate is dropped in old mode, retained with
   `floor_pass=False` and `guard_score < floor` in fresh mode;
7. `test_default_path_bitwise_unchanged` — explicit
   `fresh_re=False, fresh_re_grid=None, labeled_keys=None` equals the
   implicit default (keys, scores, dataclass equality);
8. `test_fresh_re_arg_validation` — `fresh_re_grid` without
   `fresh_re`, non-positive grid entries, and a degenerate (single-Re)
   corpus window all raise.

All 37 pass in `/nfs/wangxi/venvs/tensorlbm` and in
`/nfs/wangxi/venvs/ci-cpu` (`OMP_NUM_THREADS=1`, CPU-only); ruff
check + format clean; mypy on `src/tensorlbm` unchanged at the
baseline error count.
