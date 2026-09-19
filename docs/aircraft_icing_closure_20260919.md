# Aircraft-icing benchmark closure: survey, three benchmark tracks, module fixes, RG-15 acceptance (2026-09-17 → 2026-09-19)

This document folds the 2026-09 icing arc into the repo record: the
capability survey, the three parallel benchmark tracks (IC-A convergence,
IC-B polydisperse MVD, IC-C RG-15 adoption), the module fixes they forced
(IC-D1/D2/D3 → PRs #294/#295/#296, all merged; main @ 64a979de carries the
full set), the D2×D3 interaction fix found only in integration, and the
final RG-15 acceptance rerun with its honest external-anchor verdict.
Every decisive number below was rerun or recomputed by the controller from
the run artifacts (json / npz / masks / CSVs), independently of the run
scripts — including one instance where a verifier that copied the formula
under test verified a wrong formula green (IC-A sign error, §3).

## 1. Baseline at the start (survey, 2026-09-17)

- Internal consistency of `tensorlbm.aircraft_icing`: engineering-grade,
  tier-A/B gates CI-enforced at the 48-test level (main @ 332258e6).
- External anchoring: calibration-grade at best — exactly ONE external
  anchor (Shin & Bond 1992 NACA 0012), −22 % ice-area deviation, no test
  asserts.
- The RG-15/IPW-2 benchmark (reference cuts + official conditions) was on
  disk but NOT adopted: geometry hardcoded, β unnormalized, no Messinger
  feed consistency.

## 2. Benchmark inventory and current status

| Benchmark | Role | Status after this arc |
|---|---|---|
| NACA 0012 rime (internal; chord 0.5334, V 67, Re 2.5e6, LWC 0.5, MVD 20, −10 °C, 360 s, 4°) | production design point, grid-convergence anchor | **validated**: Richardson p = 4.594 (t_max), determinism bitwise, LWC linear R² = 0.999996 |
| NACA 0012 #187 fine point (nx 890) | accuracy cross-check | reproduced under Schiller–Naumann drag only (+9.8 %), see §3 |
| RG-15 / IPW2 cases 3.1–3.3 (glaze/mixed/rime; official conditions §5) | external anchor | **adopted + internally exact**; external magnitude gap quantified and attributed (§7) |

Module state on main @ 64a979de: 72 tests (69 passed + 3 xpassed), CI 4/4,
ruff clean; the merged tree is bit-identical (7f55487d) to the integration
branch on which every number in §6–§7 was produced.

## 3. IC-A — grid convergence and response structure (run-only)

Four-level ladder nx 320/640/890/1280 (dx 4.17/2.08/1.50/1.04 mm) + SN-drag
companion ladder + T×LWC matrix + determinism pair
(`/nfs/wangxi/runs/icing_conv_20260917/`, controller checker 122/122):

- t_max Richardson **p = 4.594**, f_ext = 10.508 mm; deviations at L2–L4 sit
  3–22× BELOW the dx/√12 quantization floor (plateau beyond L2).
- Frozen mass/span p = 3.698, f_ext = 0.330 kg/m — **quantization-limited**
  (below the conserved collected ≈ 0.40 kg/m; pending-water 19–45 %).
  Collected mass converged ±3 %, non-monotone (no real order).
- T×LWC: T-rows bitwise identical (Macklin capped at 917 at all T here —
  honest null: the rime path has no T sensitivity at these conditions);
  LWC slope 0.9857, R² = 0.999996, max dev 0.15 %. Determinism bitwise.
- **#187 provenance finding**: the #187 β table was produced with
  Schiller–Naumann drag (β_L1 0.7521 vs ref 0.7500 = +0.28 %); the
  module-default Stokes law gives +21.6 %. The #187 fine point is not
  reproduced under either law (SN +9.8 %).
- **Sign-error lesson**: the run script's extrapolation
  `f_ext = f_f + (f_m−f_f)/(r^p−1)` was wrong (must be −); its 73/73
  "verification" script contained a copy of the same formula, so the
  correlated error verified itself. A verifier must be an independent
  implementation, never a copy. (Both fixed in place; erratum in the
  run report.)

## 4. IC-B — polydisperse MVD bins (PR #293, merged)

`IcingConfig.mvd_bins`: per-bin Eulerian fields with mass-weighted credits;
official IPW-2 7-bin DSD (7.3…81.3 µm @ 5/10/20/30/20/10/5 %, mvd_eff
29.94 µm exact). Design point **bitwise unchanged** (15/15 arrays + 3 CSVs
vs pristine single-MVD); official-DSD demo: deposited +20.7 %, frozen
+26.7 %, per-bin closure ≤ 1.7e-8, wall 3.53× (Python bin loop).

## 5. IC-C — RG-15 adoption and the measured-cut audit (run-only)

Official IPW-2 conditions (workshop values = primary anchor): 3.1–3.3 =
AoA 4°, 25 m/s, T −2/−4/−10 °C, LWC 0.44 g/m³, MVD 24 µm monodisperse,
1200 s, chord 0.30 m; 7-bin DSD recorded for sensitivity, not anchored.
Reference cuts: MCCS v2/v3 per case (tunnel-frame; alignment = rigid
motion of raw CSV at 1e-14).

Controller independent recompute (own parser/rasterizer/KD/air props):
**29/29 PASS** — measured areas/thicknesses reproduced < 0.6 mm² / 0.05 mm.
**Erratum fixed in the agent report**: the full-collection limit was taken
as the flat-plate formula (48.2 mm); the exact 4°-projected dat extent is
**33.3 mm** — five of six cuts imply 1.30–1.43× full collection
(consistent only with soft rime ρ ≲ 650, i.e. the ρ≈650 reading); the
Mixed v2 cut (1395.6 mm²) implies 2.91× full collection at any density —
**physically impossible, flagged suspect**.

Seven module gaps came out of this audit; the four highest-leverage ones
became IC-D.

## 6. IC-D — module fixes (PRs #294/#295/#296, all merged)

**#294 geometry (IC-D1)**: `airfoil_dat` Selig-contour ingest + Kasa LE
circle fit + `airfoil_dat_mask_2d`. RG-15 LE diameter from the dat fit
2·r_le·c = 4.8588 mm (r_le 0.8098 %c) vs the NACA 4-digit formula 9.5204.
Default path bitwise unchanged; dat mask == IC-C monkeypatch cell-for-cell
(1018 cells, 0 diff).

**#295 render conservation (IC-D2)**: root cause REFUTED the per-shot drop
hypothesis — the legacy carries m_w across shots (ledger closes ≤ 1.3e-16);
the whole deficit was the **final sub-voxel remainder floor** (rounding
floor 3.7–6.6e-4 kg exceeds the frozen 4.3e-4 kg → exact whole-cell
conservation impossible by construction). Fix: `deposit_remainder="carry"`
(default) + exact per-cell `ice_mass` ledger + end-of-exposure flush
(tips ≥ half fill join the mask, the rest stays sub-voxel mass). RG-15 3.3
rendered/frozen 0.576 → **1.000000**; drop mode == legacy bit-identical;
NACA rime bitwise unchanged. Caveat: the binary mask cannot represent
sub-half-cell mass — quote `rendered_mass_kg`, never cells×m_cell.

**#296 β window + Messinger feed (IC-D3)**: four stacked β defects —
(a) the IC-C pilot's 0.305 was a script slip (un-pinned cfg.lwc_eff at 2×
the per-shot acceleration), but it exposed (b) `run_glaze_icing` driving
`beta_window_frac = 0` in trailing mode = an EMPTY differencing window →
module β ≡ 0 for exactly the glaze configs (scripts hand-rolled the
normalization), (c) the Eulerian cloud starts at zero slip and needs
~τ_d_lu steps → whole-shot ledgers under-collected by a shot-length-
dependent amount (the −24 % steps ladder 0.609/0.727/0.757), (d)
`surface_arc_length` sign rule annihilated s for every cell on the
stagnation row (7 cells to 0.65c @4°) → wake cells diluted the stagnation
bin 0.763 → 0.103. Fixes: whole-shot fallback, `droplet_warmup`,
`surface_arc_sign_fix` (both opt-in, default False → legacy byte-identical).
Messinger feed: panel area now the wetted-surface strip `n_sf` (the deposit
ring double-covers thin regions up to ~4×), `n_f_stag` sampled at the
impingement peak not the geometric LE (LE sampling alone had classified
every temperature as rime); `*_le` keys retained; **the h model was left
untouched (owner decision)**. Verified: β shot-structure md5-identical
5-shot vs single-shot; dt_shot 120/240/1200 β_pk spread 1.4e-6; n_f ladder
0.6465 (glaze, T_s = 0.00) / 1.000 (−0.98) / 1.000 (−7.08); independent
0-D Messinger bisection 0.5589 vs module 0.5591 (0.04 %); NACA bitwise.

**D2×D3 interaction (found only in integration)**: the wetted-strip panels
create sf-only panels downstream of the impingement limit (`n_sf > 0`,
`n_dep == 0`); runback water freezes there but the credit loop only
iterates `dep_p` → **2.26 % of frozen mass never reached the deposit** —
invisible on either branch alone (pre-D2 legacy drop strands sub-voxel mass
anyway), fatal against the D2 conservation ledger. Fix (in #296's merge
commit d2807ebd): credit sf-only panels uniformly over their surface cells;
the freezer's #84-4 cascade moves solid-cell water to the outward growth
frontier, so **runback ice now renders at all** (an IC-C gap). Panels with
deposit cells untouched → all D3-verified panel numbers unchanged.

## 7. RG-15 acceptance rerun — integrated branch, official conditions

Runs `/nfs/wangxi/runs/icing_accept_20260917/` (acc31/acc32/acc33; native
`airfoil_dat` geometry, `le_diameter` pinned to the IC-D3 value
0.005038170698008188 so the β cross-check is apples-to-apples); controller
checker `ctrl_acc_check.py` **9/9 PASS**; summary in the run dir.

Internal gates (all three cases):

| Gate | 3.1 glaze −2 °C | 3.2 mixed −4 °C | 3.3 rime −10 °C |
|---|---|---|---|
| audit closure_error | 3.1e-17 | 1.2e-16 | 7.6e-18 |
| rendered/frozen | 1.000000 | 1.000000 | 1.000000 |
| β_pk vs IC-D3 ctrl (bitwise) | == | == | == |
| n_f_stag (peak-sampled) | 0.6486 glaze | 1.000 | 1.000 |
| rendered mass [kg] | 4.181e-4 | 4.139e-4 | 4.288e-4 |

The β_curve arrays are bitwise identical to the IC-D3 runs — integration
changed nothing on the collection side; solid/ice_only differ only by the
carry-flush + runback rendering (intended).

External anchor vs MCCS cuts — the honest gap:

- Measured (controller recompute): 641.5/681.4, 1395.6/670.2 (v2 suspect),
  682.5/622.1 mm²; thickness 25.7–27.9 mm.
- Module: ice area 192–198 mm² (sim/meas 0.14–0.32), peak panel thickness
  1.27–2.33 mm (sim/meas 0.05–0.09; controller rerun on the artifacts).
- **∫βdy (rendered) 13.4–13.9 mm = 41 % of the full-collection height
  33.3 mm; the measured cuts imply ≈ full collection (ρ≈650 reading).**

Ranked causes (documented, nothing tuned):
1. No intra-shot ice-shape feedback: the real ~26 mm horns grow into the
   flow and self-buffer collection toward full; the module's 5 quasi-steady
   shots stay near the clean sharp LE (clean ballistic estimate ~4–5 mm
   ∫βdy — the module is already 3× above it via shot-wise geometry updates).
2. Grid resolution: 2.343 mm cells vs the 4.86 mm LE (β_pk 0.788 → 0.818
   at nx 320→640, ~0.85 extrapolated; a ~10 % peak effect).
3. ρ mismatch: module 917 (Macklin capped at these conditions) vs
   measured-implied ≈ 650; at ρ 650 the rendered-area ratio would be ≈ 0.41.

**Verdict**: benchmark *correctness* goal met — the full chain (geometry →
Eulerian collection → β → Messinger → runback → render) is internally
exact, shot- and acceleration-invariant, and mass-conserving to machine
precision, with runback ice rendering. The remaining anchor gap is physical
model class (quasi-steady collection without intra-shot horn feedback) +
resolution — not bookkeeping.

## 8. Open items

- Messinger h model: explicitly NOT changed (owner decision, preserved
  through #296); revisit only with a dedicated decision.
- MCCS Mixed v2 cut: physically impossible (2.91× full collection at any
  density) — recommend excluding from any quantitative anchor.
- If the 41 % collection gap must close: intra-shot shape feedback
  (continuous re-meshing / moving boundary) is the dominant lever; next
  are LE resolution and a soft-rime density model.
- β_stag = 0.788 at the production grid is resolution-limited (grid ladder
  0.788/0.805/0.818 at nx 320/480/640; an independent Lagrangian
  measurement on the same flow reads 0.578) — reported as-is.

## 9. Artifact map

| What | Where |
|---|---|
| IC-A convergence runs + 122/122 checker | `/nfs/wangxi/runs/icing_conv_20260917/` |
| IC-B A/B + official-DSD runs | `/nfs/wangxi/runs/icing_mvd_{ab,official}_20260917/` |
| IC-C RG-15 adoption + 29/29 checker | `/nfs/wangxi/runs/icing_rg15_20260917/` |
| IC-D1/D2/D3 runs + checkers (16/16, 14/14) | `/nfs/wangxi/runs/icing_d{1,2,3}_20260917/` |
| acceptance runs + 9/9 checker + summary | `/nfs/wangxi/runs/icing_accept_20260917/` |
| reference data (dat, PDF conditions, MCCS CSVs) | `/nfs/wangxi/rg15_ref/` |

Controller checkers and staging: `/nfs/wangxi/icing_conv_check/` (local
staging mirror — the 5090 does not see it; everything is scp'd).
