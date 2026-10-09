# Hull-form families: the LOFO campaign (Phase 2 second cut)

Date: 2026-08-24 · Branch: `exp/b4-fam` (base `ca9e0c4d`) ·
Dataset: `/nfs/wangxi/datasets/scan_suboff_hullform_20260824` (112 points) ·
Run dir: `/nfs/wangxi/runs/b4_fam_20260824`

## The question

B1–B4 so far varied **operating condition** (Re, u_in) and **appendage scale**
(sail/fin multipliers) around the DARPA SUBOFF mother lines. The hull itself —
the lines plan — never moved. The Phase 2 roadmap asks whether the surrogate
family generalises to *hull forms*, and if not, how much of the gap is

1. **handcrafted-feature blindness** — the mask-projection features
   (A_proj, A_side, A_top, V) literally cannot see some shape changes, or
2. a **representation problem** — the features could see it but a
   handcrafted linear/FiLM conditioning on 4 scalars is too weak, which is
   what the SDF encoder (roadmap L2) is meant to close.

The leave-one-family-out (LOFO) baseline below separates the two: a family
that is invisible to the geometry channels yet shifts C_D measures (1)
directly; the residual error on visible families after conditioning on the
channels measures (2).

## 1 · Hull-form axes (`suboff_cad` extension)

Four continuous affine deformations of the mother hull, all expressed as
multipliers on `SuboffConfig` (`l_over_d_mult`, `nose_len_mult`,
`stern_len_mult`, `sail_x_mult`, default 1.0):

| axis | multiplier | semantics | mother value |
|---|---|---|---|
| slenderness | `l_over_d_mult` | L/D scaled; diameter shrinks in lattice units (hull length pinned at 76.8 lu) | 8.573 |
| entrance | `nose_len_mult` | bow segment length; midbody absorbs the change (parallel-body insertion) | 1.0 |
| run | `stern_len_mult` | stern taper length; midbody absorbs | 1.0 |
| sail station | `sail_x_mult` | sail translated along the deck | 1.0 |

Implementation: **inverse-frame deformation** (the pattern already used by
`_sail_unscaled_frame`). Query coordinates are mapped back to the mother
ft-frame through a piecewise-linear axial node map
(`_variant_nodes_ft` / `_pw_map_torch`), and the untouched DARPA predicates
are evaluated there. Appendage rooting is preserved by construction (sail and
fin are evaluated in the mother frame, then the sail is shifted by
`_sail_x_shift_ft`), so no deformation can float an appendage.

**Default bit-identity.** Every multiplier at 1.0 takes an early-return path
that reproduces the mother geometry bit-for-bit. Verified against the actual
`origin/main` module loaded side-by-side (`git show origin/main:...` +
`spec_from_file_location`): masks `torch.equal`, statistics dict equal, STL
triangles bit-equal; solid-cell anchors 4093 / 4121 / 4157 (bare / with_sail
/ full) at n128. 21 new tests in `tests/test_suboff_cad_hullform.py`.

**Per-axis monotonicity** (n128, with_sail, mult ∈ {0.7, 0.85, 1.0, 1.15, 1.3}):

| axis ↑ | solid voxels | displacement | A_proj | L/D | centroid x |
|---|---|---|---|---|---|
| `l_over_d` | strictly ↓ | strictly ↓ | strictly ↓ | exactly linear ↑ | — |
| `nose_len` | strictly ↓ | strictly ↓ | — | — | ↑ |
| `stern_len` | strictly ↓ | strictly ↓ | A_side ↓ | — | — |
| `sail_x` | ≈ const (±8 cells) | ≈ const | — | — | sail centroid ft ↑ |

Vocabulary note: longer (finer) ends *replace full-radius midbody*, so
`nose_len`/`stern_len` ↑ strictly *decreases* volume — the axes are
"parallel-body insertion" multipliers, not length multipliers. Guards:
midbody collapse, sail leaving the deck window `[0.5·bow, mid_end]`, and
non-positive multipliers all raise. The joint axial map is **not** the
composition of single-axis maps (piecewise-linear maps do not commute);
deviation measured 0.13–1.23 ft (≤ 8.6 % L) and pinned as a bounded recorded
property, not forced to zero.

## 2 · Family corpus

Four families around the mother (all `with_sail`, appendage scale pinned 1 so
the family axis is the lines plan only) × 28 LHS re log-U [60, 700], seed
20260824 — the family *shape* is fixed; this corpus measures whole-family
held-out generalisation, not within-family jitter:

| family | axis | L/D | solid | A_proj | V_lu3 |
|---|---|---|---|---|---|
| mother | — | 8.57 | 4121 | 73 | 3836 |
| slender | l_over_d 1.30 | 11.14 | 2437 | 42 | 2408 |
| blunt | l_over_d 0.75 | 6.43 | 6140 | 112 | 6223 |
| long_nose | nose 1.30 | 8.57 | 3985 | 73 | 3741 |
| aft_sail | sail_x 1.30 | 8.57 | 4121 | 73 | 3836 |

Chain identical to G2: `suboff_n128`, cumulant, u_in = 0.1, 4 000 steps,
snapshot 500, `DragSurveySpec(margin=4, interval=25)`, τ = 0.5 + 23.04/Re
∈ [0.533, 0.865] (inside the proven envelope). Smoke (4 × 300 steps) ran
first and validated. Full scan: 112/112 completed in **757 s** on 6 GPUs
(0/1/2/5/6/7; 3–4 were occupied by another campaign), 71.8 MLUP/s per point.

**Preflight geometry-channel visibility** (the LOFO design hinge):

| family | dlog A_proj | dlog V | dlog A_side | dlog A_top |
|---|---|---|---|---|
| slender | −0.240 | −0.228 | −0.077 | −0.092 |
| blunt | +0.186 | +0.173 | +0.063 | +0.074 |
| long_nose | +0.000 | −0.015 | −0.009 | −0.006 |
| aft_sail | **0.000** | **0.000** | **0.000** | **0.000** |

aft_sail translates the sail; every projection and the volume are invariant
by construction. It is the designed probe of handcrafted-feature blindness.

**Corpus validation** (all 112 finite, tail drift 0.000, C_D vs the mother
`with_sail` power law, 14-point fit):

| family | C_D/C_D_mother median | range |
|---|---|---|
| slender | 1.424 | [1.405, 1.494] |
| blunt | 0.776 | [0.764, 0.820] |
| long_nose | 0.959 | [0.947, 0.995] |
| aft_sail | 0.985 | [0.973, 1.024] |

The family shift is a near-Re-independent multiplicative factor (spread ≤ 9 %
within each family). Directions check physically: slenderness at fixed length
scales wetted area ~L·D but frontal area ~D², so frontal-referenced C_D *rises*
for the slender hull; blunt falls; longer entrance trims volume slightly;
moving the sail aft changes hull–sail interaction by ~1.5 %.

## 3 · LOFO protocol

Corpus `cache_fam.npz` = 350 points = v2 corpus (238, datasets 0–5) + family
corpus (datasets 6–9). One geometry-block definition for every point —
**generalised mask-derived channels**, computable from any solid mask with no
CAD predicate:

    g0..g2 = log10(A_proj / A_side / A_top ÷ mother with_sail values)
    g3     = log10(V / V_mother)

(old-corpus points get theirs from the CAD predicates — 116 unique design
keys — with the stored-mask bit-agreement audit; family points are 112/112
bit-identical to CAD by construction).

Splits:

- `random` — group-stratified random over all 10 datasets (in-family
  interpolation reference);
- `lofo::<fam>` — test = one family (28 pts), fit/val = other three
  families + the 238-point mother corpus;
- `lofo_legacy::<fam>` — test = the family, fit/val = the mother corpus
  only (training has never seen a hull-form variant).

Arms (trainer protocol byte-identical to v3; only the geometry block
changes):

- `power_re` — [1, log re] floor;
- `power_geoM` — [1, lr, g0..g3, lr·g0..lr·g3]: the power-geo baseline with
  handcrafted features generalised to mask projections;
- `F_geoM` — `CondFNODrag`, cond = [log re, log u_in, log sail, log fin |
  g0..g3] (8-dim, v3 structure, plain sampling);
- `C_geoM` — + `QuotaSampler` + force-tail aux head λ=0.1 (the v3 C_full
  recipe);
- `M_geoM` — cond-only 2×64 MLP on the same 8 channels, no field input
  (added to attribute the failure: field branch vs condition pathway).

3 model seeds (0/1/2) per FNO/MLP arm per split; GPU 5. 54 FNO + 27 MLP
trainings, all in `lofo.log` / `metrics_lofo.json`.

## 4 · Results

Family × arm × train-config (MAPE % on the 28 held-out points; FNO/MLP
arms mean ± std over 3 model seeds; `power_*` are single deterministic
fits). `rest` = fit on the other three families + 238-point mother corpus;
`legacy` = fit on the mother corpus only. 99 result rows in
`metrics_lofo.json`; per-run lines in `lofo.log`.

| held family | config | power_re | power_geoM | M_geoM | F_geoM | C_geoM |
|---|---|---|---|---|---|---|
| slender (L/D 11.1) | rest | 36.11 | **4.77** | 33.70 ± 4.24 | 30.26 ± 3.98 | 33.66 ± 6.01 |
| slender | legacy | 35.89 | 27.80 | 23.38 ± 13.46 | 20.30 ± 13.41 | 25.50 ± 11.06 |
| blunt (L/D 6.4) | rest | 23.58 | **3.01** | 25.84 ± 4.05 | 25.48 ± 0.34 | 27.05 ± 0.99 |
| blunt | legacy | 17.57 | 26.66 | **13.76 ± 3.81** | 15.37 ± 9.78 | 16.70 ± 4.00 |
| long_nose | rest | 1.78 | 2.28 | 1.74 ± 0.10 | **1.58 ± 0.11** | 1.68 ± 0.07 |
| long_nose | legacy | 4.80 | 1.56 | **0.49 ± 0.19** | 1.43 ± 0.63 | 1.67 ± 0.54 |
| aft_sail | rest | 4.58 | 1.52 | **0.22 ± 0.05** | 0.40 ± 0.13 | 0.29 ± 0.08 |
| aft_sail | legacy | 7.30 | 1.70 | **0.44 ± 0.03** | 0.41 ± 0.19 | 0.43 ± 0.11 |
| *random* (in-family) | | 14.66 | 2.00 | 0.47 ± 0.01 | 0.46 ± 0.03 | **0.34 ± 0.03** |

Arms: `power_re` [1, log re]; `power_geoM` [1, lr, g0..g3, lr·g0..lr·g3]
(10-param linear); `M_geoM` cond-only 2×64 MLP (no field input);
`F_geoM` `CondFNODrag` + plain sampling; `C_geoM` + `QuotaSampler` + aux
head (the v3 C_full recipe). For scale: the v3 LOHO hull-out folds
(bare/with_sail/full) were 2.00 / 5.16 / 10.25 % — an order of magnitude
easier than slender/blunt LOFO.

Prediction signature on the hard families: every nonlinear arm
*collapses toward the mother manifold* — median predicted/true ≈ 0.64–0.74
for slender (true C_D = 1.42 × mother; models predict ≈ mother) and
≈ 1.25–1.28 for blunt (true 0.78 ×; models predict ≈ mother). The linear
`power_geoM` moves with the family and lands at 3–5 %.

## 5 · Reading

1. **The features are not the bottleneck — the conditioning is.** The four
   mask-projection channels carry enough signal for a *linear* map to
   extrapolate to an unseen family at 3–5 % whenever another family
   brackets it (`rest`). Every *nonlinear* consumer of the same channels
   — the cond-only MLP (no field at all) and both FiLM-FNO arms — fails
   identically at 25–34 %, collapsing to the mother. Since M_geoM does
   not see the field, the failure cannot be blamed on the field branch:
   **nonlinear encoders mis-extrapolate the geometry channels themselves**
   (slender/blunt g-vectors sit ±0.2 log-units outside the fit cloud;
   z-scored FiLM/MLP responses saturate instead of continuing the local
   linear trend). This is a representation/architecture problem, not
   handcrafted-feature blindness — and it defines the acceptance bar for
   the SDF-encoder round: any L2 candidate must beat `power_geoM`'s 3–5 %
   on slender/blunt LOFO, not just the in-family 0.3 %.
2. **The designed "blind" family is the easy one.** aft_sail is invisible
   to every mask channel (all four ≡ mother by construction) yet is
   predicted at 0.2–0.5 %, because moving the sail aft shifts C_D by only
   ≈ 1.5 % — within the model's mother-family accuracy. Feature
   blindness at this perturbation amplitude is *not* the binding
   constraint; it would become one for sail stations with a larger C_D
   effect (a larger-`sail_x_mult` sweep interacting with the stern taper
   is the natural follow-up).
3. **Cross-family training data is what makes the linear model work**
   (`rest` 4.77 vs `legacy` 27.80 on slender): without another family
   bracketing the held one, nothing extrapolates ±0.2-log-unit channel
   shifts well. Practical rule: keep at least one displaced family in
   every training mix, or add channel jitter/mixup along the family axes.
4. **Quota sampling + aux head do not help OOD** (C_geoM ≈ F_geoM within
   seed noise on the hard families, and *worse* on slender); their gains
   are in-family only. Seed variance explodes exactly where the point is
   (slender legacy: 5.7–38.1 % across seeds) — extrapolation is unstable,
   not just biased.

### Anomalies / caveats

- Old-corpus stored-mask bit-agreement 174/238 — the known vintage issue
  (re_drag/lhs40 predate an appendage predicate fix; same 64 points as
  the v3 audit). Family corpus is 112/112 bit-identical to CAD.
- One slender seed flipped sign of the family response (ratio_med 1.448
  vs 0.64–0.74 for the other seeds) — reported above via the ±std.
- G1 smoke at fixed re = 200 shows C_D falling 8.28 → 7.30 as u_in rises
  0.07 → 0.12 (−12 %): by exact lattice similarity C_D should be
  u-invariant at fixed Re, so this is a genuine compressibility/lattice
  artifact — precisely the u_in × drag interaction G1 was designed to
  expose.

## 6 · G1 supplement (u_in × geometry, gap ①)

`/nfs/wangxi/datasets/scan_suboff_uin_geo_g1_20260824` — 60 points:
u_in ∈ {0.07, 0.085, 0.1, 0.12, 0.14} × 12 LHS (re log-U inside the
per-level τ envelope clamp [0.533, 0.884], sail/fin U [0.4, 3.0]), mother
hull with_sail, same chain. 60/60 completed in 454 s, all finite, tail
drift 0.000. This closes the u_in axis at the data level (the corpus is
B4-ready); training on it is the next round.

## Reproduce

```bash
# corpus (6 idle GPUs; smoke first)
cd /nfs/wangxi/runs/b4_fam_20260824
PYTHONPATH=/nfs/wangxi/worktrees/b4_fam/src \
  /nfs/wangxi/venvs/tensorlbm/bin/python fam_launch.py --smoke --gpus=0,1,2,6
PYTHONPATH=/nfs/wangxi/worktrees/b4_fam/src \
  /nfs/wangxi/venvs/tensorlbm/bin/python fam_validate.py scan_suboff_hullform_smoke
PYTHONPATH=/nfs/wangxi/worktrees/b4_fam/src \
  /nfs/wangxi/venvs/tensorlbm/bin/python fam_launch.py --gpus=0,1,2,5,6,7
# cache + LOFO
PYTHONPATH=/nfs/wangxi/worktrees/b4_fam/src \
  /nfs/wangxi/venvs/tensorlbm/bin/python build_cache_fam.py
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/nfs/wangxi/worktrees/b4_fam/src \
  /nfs/wangxi/venvs/tensorlbm/bin/python lofo_v1.py
# tests (worktree)
cd /nfs/wangxi/worktrees/b4_fam && TMPDIR=/nfs/wangxi/tmp \
  /nfs/wangxi/venvs/tensorlbm/bin/python -m pytest tests/test_suboff_cad_hullform.py \
  --basetemp=/nfs/wangxi/tmp/pt_b4fam -q
```
