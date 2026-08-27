# SUBOFF bare-hull C_D–Re benchmark (v1.1)

> **v1.1 (2026-08-22)** — results replaced by the exact control-volume drag
> observer (`tensorlbm.scan_drag`, PR #204). The v1 wake-momentum-deficit
> estimator is **deprecated for absolute values on this configuration**: it
> underestimates C_D by a factor 2.9–7.9 that grows with Re and mis-states
> the scaling exponent (−0.888 measured vs −0.703 exact). The v1 table is
> retained below for the record — its shape and monotone trend remain
> correct; its absolute values and slope do not. Headline:
> **C_D = 2.813 @ Re = 800, C_D ∝ Re^−0.703** (v1 reported 0.355, −0.888).

**Dataset (v1.1, exact)**: `/nfs/wangxi/datasets/scan_suboff_re_drag_20260821`
(5090 server) · scan id `scan-suboff-re-drag` · 24 points · 48 catalog
products (fields at steps 2 000/4 000) · splits 17/4/3 points · 766 MB ·
run at code `7554bb8` (PR #204).
**Dataset (v1, wake)**: `/nfs/wangxi/datasets/scan_suboff_re_20260821` ·
scan id `scan-suboff-re-sweep` · 24 points · 192 catalog products · 3.1 GB.
**Campaign** (identical in both): `suboff_n128` (D3Q19, cumulant),
resolution 128 → grid 64×64×128, `u_in = 0.1`, bare hull, 4 000 steps, mass
correction every 10 steps, seed 0; 24 log-spaced Re levels 50 → 800
(τ = 0.5 + 23.04/Re ∈ [0.529, 0.961]). Exact rerun on 1× RTX 5090, ≈ 21
s/point, 448 s total; v1 ran on 8× RTX 5090, 124 s wall.
**Estimator (v1.1)**: `tensorlbm.scan_drag` — exact discrete-kinetic
control-volume observer, `DragSurveySpec(margin = 4, interval = 25)` → 160
force samples per point flushed to a `drag_history.json` sidecar (schema
`tensorlbm.drag-history/v1`).

## Exact method (v1.1)

One box control volume per point: the hull's solid bounding box expanded by
4 cells, clamped to the largest strictly interior window so every boundary
condition stays outside it. Every 25 steps the observer closes the
discrete-kinetic momentum balance over one complete solver step,

```
F_on_body = Σ CV faces streaming momentum import − fluid momentum change
```

in phase with the reporter (post-stream, post-BCs, post mass correction).
The impulse the periodic mass correction injects inside the CV is
compensated sample-by-sample; on the PR #204 validation run (Re = 148)
nested CVs (margin 2 vs 4) agree to ~3e-6 relative and the balance matches
the Ladd link momentum exchange to < 5e-4 per step past startup. Crucially,
**no far-field reference state enters the measurement** — ρ and u below
appear only in the normalisation.

Reduction: tail mean of `force_x` over the last 25% of samples (40 samples,
steps 3 025–4 000), then

```
C_D   = 2·F_x_tail / (ρ·u²·S_proj),   ρ = 1, u = 0.1, S_proj = 69 cells²
Cf_eq = C_D·S_proj / S_wet,           S_wet = 2 494 faces
```

Geometry identical to v1 (same case, same seed): hull L = 76.8 cells at
cx = 44.8, 4 093 solid cells. Normalising by the *nominal* ρu² instead of
the mass-corrected far-field value is now a ≤ 1.2% bookkeeping choice, not
a sign-flipping bias.

## Exact results (v1.1)

`F_x tail` in lattice units (momentum per step); `exact/wake` = C_D here
over the v1 wake value (see below).

| Re | F_x tail | C_D | Cf_eq | C_D·Re | exact/wake | drift |
|-----:|-------:|-------:|-------:|-------:|-------:|-------:|
| 50 | 6.8215 | 19.7726 | 0.54704 | 989 | 4.31 | 0.007% |
| 56.4 | 6.2011 | 17.9742 | 0.49728 | 1014 | 4.02 | 0.007% |
| 63.6 | 5.6433 | 16.3574 | 0.45255 | 1040 | 3.77 | 0.006% |
| 71.8 | 5.1359 | 14.8868 | 0.41186 | 1069 | 3.55 | 0.005% |
| 81.0 | 4.6814 | 13.5693 | 0.37541 | 1099 | 3.36 | 0.004% |
| 91.4 | 4.2707 | 12.3789 | 0.34248 | 1131 | 3.20 | 0.004% |
| 103.1 | 3.9011 | 11.3076 | 0.31284 | 1166 | 3.07 | 0.003% |
| 116.3 | 3.5671 | 10.3395 | 0.28606 | 1202 | 2.97 | 0.002% |
| 131.2 | 3.2649 | 9.4634 | 0.26182 | 1242 | 2.91 | 0.002% |
| 148.0 | 2.9914 | 8.6706 | 0.23988 | 1283 | 2.87 | 0.001% |
| 166.9 | 2.7441 | 7.9538 | 0.22005 | 1327 | 2.86 | 0.001% |
| 188.3 | 2.5188 | 7.3008 | 0.20199 | 1375 | 2.88 | 0.001% |
| 212.4 | 2.3145 | 6.7087 | 0.18561 | 1425 | 2.93 | 0.001% |
| 239.6 | 2.1287 | 6.1700 | 0.17070 | 1478 | 3.02 | 0.000% |
| 270.3 | 1.9595 | 5.6796 | 0.15713 | 1535 | 3.15 | 0.000% |
| 305.0 | 1.8051 | 5.2322 | 0.14476 | 1596 | 3.31 | 0.000% |
| 344.0 | 1.6649 | 4.8259 | 0.13352 | 1660 | 3.52 | 0.000% |
| 388.1 | 1.5368 | 4.4545 | 0.12324 | 1729 | 3.78 | 0.000% |
| 437.8 | 1.4200 | 4.1158 | 0.11387 | 1802 | 4.10 | 0.000% |
| 493.9 | 1.3132 | 3.8064 | 0.10531 | 1880 | 4.51 | 0.000% |
| 557.2 | 1.2157 | 3.5237 | 0.09749 | 1963 | 5.03 | 0.000% |
| 628.6 | 1.1265 | 3.2654 | 0.09034 | 2053 | 5.70 | 0.000% |
| 709.1 | 1.0451 | 3.0292 | 0.08381 | 2148 | 6.61 | 0.000% |
| 800.0 | 0.9705 | 2.8130 | 0.07782 | 2250 | 7.92 | 0.000% |

`drift` = relative `force_x` change between the last sample and three
samples back (75 steps) — ≤ 0.007% at every point, two orders of magnitude
tighter than v1's snapshot-convergence check (≤ 0.24%). The tail is flat:
the 40-sample tail mean differs from the final sample by < 0.01%
everywhere.

### Scaling (v1.1)

Log-log fit over the 24 points: **C_D ∝ Re^−0.703** (R² = 0.9988; the v1
wake estimator gave −0.888, R² = 0.93 — the exponent was wrong, not just
the level). `C_D·Re` rises monotonically 989 → 2 250: still between laminar
flat-plate friction (Blasius, Re^−0.5) and fully Stokes drag (Re^−1),
consistent with a friction-dominated laminar regime with a growing
pressure-drag fraction as Re rises — but materially shallower than v1
suggested.

### Reference lines (context only — outside validity range)

At Re = 800: ITTC-1957 `0.075/(log10 Re − 2)²` = **0.0920**, Blasius
`1.328/√Re` = **0.0470**, exact `Cf_eq` = **0.0778** — between the two
lines, where a streamlined body with a thick laminar boundary layer at
lattice Re plausibly sits (v1's 0.0098, below both, was an artifact of the
underestimated wake deficit). Both lines assume ship-/plate-scale Re
(≥ 10⁵) and are orientation only; no experimental SUBOFF data exists at
this Re. The benchmark's authority is internal: tail drift ≤ 0.007%, CV
invariance ~3e-6, and an estimator with no far-field reference left to
bias it.

## Quantified discrepancy: why the wake estimator fails here

The per-point `exact/wake` ratio is tabulated above: 4.31 at Re = 50,
falling to a minimum **2.86 at Re ≈ 150–167**, then rising **monotonically**
to **7.92 at Re = 800**. Three mechanisms, all visible in this dataset:

1. **Mass-correction far-field drift.** The periodic mass correction
   settles the far-field ≈ +0.6% above nominal; the border-ring reference
   removes only the first-order term and couples `u_∞` to the ring width
   (v1 limitation). The exact observer measures the force on the body
   directly, with the mass-correction impulse compensated, so this drops
   out entirely.
2. **Near-field planes, neglected pressure term.** The survey planes
   (x-offsets {4, 8, 16, 32} from the outlet) sit only 12–40 cells =
   **1.3–4.3 hull diameters** behind the tail; the static-pressure term
   has not decayed and the deficit is not a developed far-field profile.
3. **Confined, undeveloped wake.** Frontal blockage is small (S_proj/A =
   69/4 096 = 1.7%), but the lateral Dirichlet walls are only ≈ 2.9 hull
   diameters from the hull surface, and at the nearest plane the
   > 5%-deficit footprint covers **≈ 59% of the interior at Re = 50**
   (≈ 92% at the 1% level; ≈ 15% at Re = 800, 19% at 1%) — at low Re the
   8-cell reference ring is itself inside the viscous wake, so `u_∞` is
   deficit-contaminated and the deficit is doubly underestimated.

v1's internal-consistency checks (snapshot convergence ≤ 0.24%, plane
invariance ≤ 5% for Re ≥ 148, worst 6% at Re = 50) were real, but they
established the estimator's *self*-consistency, not its accuracy.

## v1 wake estimator (deprecated) — retained for the record

Post-hoc momentum deficit on cross-sections `x_w` downstream of the hull
(`tensorlbm.drag_survey.plane_drag`, planes at x-offsets {4, 8, 16, 32},
`u_∞` from an 8-cell border ring):

```
D(x_w) = Σ_{y,z} ρ·u_x·(u_∞ − u_x)      [lattice units]
C_D    = 2D / (ρ_∞·u_∞²·S_proj)
```

Full method notes (ring-reference rationale, nominal-vs-ring bias of
≈ 64%, plane-spread diagnostics) are in the v1 revision of this file —
`git log --follow -- docs/benchmarks/suboff_cd_re_20260821.md`.

| Re | C_D | Cf_eq | plane ΔD | drift₃ |
|-----:|-------:|-------:|-------:|-------:|
| 50 | 4.58461 | 0.12684 | 0.2655 | 0.002 |
| 56.4 | 4.46886 | 0.12364 | 0.2354 | 0.002 |
| 63.6 | 4.34146 | 0.12011 | 0.2038 | 0.002 |
| 71.8 | 4.19928 | 0.11618 | 0.1706 | 0.002 |
| 81.0 | 4.04301 | 0.11186 | 0.1372 | 0.002 |
| 91.4 | 3.87041 | 0.10708 | 0.1041 | 0.002 |
| 103.1 | 3.68176 | 0.10186 | 0.0722 | 0.002 |
| 116.3 | 3.47685 | 0.09619 | 0.0429 | 0.002 |
| 131.2 | 3.25682 | 0.09010 | 0.0184 | 0.002 |
| 148.0 | 3.02402 | 0.08366 | 0.0082 | 0.002 |
| 166.9 | 2.78233 | 0.07698 | 0.0251 | 0.002 |
| 188.3 | 2.53445 | 0.07012 | 0.0385 | 0.002 |
| 212.4 | 2.28629 | 0.06325 | 0.0471 | 0.002 |
| 239.6 | 2.04191 | 0.05649 | 0.0512 | 0.002 |
| 270.3 | 1.80572 | 0.04996 | 0.0514 | 0.002 |
| 305.0 | 1.58112 | 0.04374 | 0.0485 | 0.002 |
| 344.0 | 1.37201 | 0.03796 | 0.0435 | 0.001 |
| 388.1 | 1.17888 | 0.03262 | 0.0371 | 0.001 |
| 437.8 | 1.00306 | 0.02775 | 0.0302 | 0.001 |
| 493.9 | 0.84408 | 0.02335 | 0.0232 | 0.001 |
| 557.2 | 0.70113 | 0.01940 | 0.0166 | 0.001 |
| 628.6 | 0.57321 | 0.01586 | 0.0120 | 0.001 |
| 709.1 | 0.45826 | 0.01268 | 0.0099 | 0.002 |
| 800.0 | 0.35524 | 0.00983 | 0.0081 | 0.002 |

`drift₃` = relative C_D change across the last three snapshots (500-step
spacing). Treat the C_D/Cf_eq columns as qualitative trend only.

## Known limitations (v1.1)

1. **fp32 solver state**; sidecar forces are fp32 exports.
2. **Single seed, single realisation** per Re — no ensemble spread. At
   Re ≥ ~500 the wake is weakly unsteady; the tail mean averages 40
   samples over the last 1 000 steps of one realisation.
3. **Domain confinement not yet checked.** The exact observer removes the
   *measurement* bias, but the flow itself still sees lateral walls
   ≈ 2.9 hull diameters out; a wider-grid rerun is the natural next
   validation.
4. **Reference lines far outside validity range** (see above).

## Usage going forward

```python
from tensorlbm.scan_drag import DragSurveySpec
from tensorlbm.scan_runner import ScanPlan

plan = ScanPlan(..., drag_survey=DragSurveySpec(margin=4, interval=25))
```

One plan field enables the observer (#204, on `main`); each point directory
then carries `drag_history.json` (schema `tensorlbm.drag-history/v1`,
samples `{step, force_x, force_y, force_z, force_abs}` in lattice units,
flushed after every sample and resumable from checkpoints), and
`scan_summary.json` reports `drag_final` / `drag_mean_tail` per point.
`tensorlbm.drag_survey` remains available for post-hoc surveys of legacy
snapshot-only datasets — read its output as qualitative there (see the
caveat in its module docstring).

Reproduce the exact table:
`/nfs/wangxi/venvs/ci-cpu/bin/python /nfs/wangxi/tmp/extract_exact_cd.py`.

## Next steps

- **Domain-width independence**: rerun an Re subset on a wider lateral grid
  to separate wall-confinement effects on the flow (limitation 3) from the
  now-unbiased measurement.
- Re range extension needs either τ headroom (u_in sweep at fixed ν) or an
  SGS model; τ = 0.529 at Re = 800 is already near the practical floor for
  this lattice.
- v1 campaign launcher `/nfs/wangxi/tmp/b1_launch_scan.py`; the exact
  rerun adds only the single `drag_survey` plan field.

## Data pointers

- **Exact (v1.1)**: `/nfs/wangxi/datasets/scan_suboff_re_drag_20260821` —
  `points/p*/drag_history.json` (160 samples/point), `scan_summary.json`,
  `plan.json` (records the `drag_survey` spec).
- **Wake (v1, deprecated)**: `/nfs/wangxi/datasets/scan_suboff_re_20260821`
  — `drag_summary.json` (raw per-snapshot plane data).

---

*v1.1 generated 2026-08-22 from `main` @ `3be979f` (PR #204) — exact values
from the `drag_history.json` sidecars via `/nfs/wangxi/tmp/extract_exact_cd.py`.
v1 generated 2026-08-21 from commit `bb7ec40` + `drag_survey`.*
