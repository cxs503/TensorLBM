# Ice-keel roughness → ice/water drag closure: three-lens cross-adjudication

**Mission P103W2C-N01** (TensorLBM).  Report date 2026-11-09.
Machine-generated tables: `artifacts/ice_rough_cw_c/REPORT.md` (+ `adjudication.json`,
`criteria.json`, `cw_table.json`, `propagation.json`, `figures/`).
Every number quoted here is read from those files (`tools/run_ice_rough_cw_adjudication.py
aggregate`), which in turn read the 25 per-case simulation records — nothing is transcribed by
hand.

---

## 1. What this node asked, and what it got

The node asked for the *cross-adjudication* step that the previous closure node
(P103W2B-N02) could not do: take the wall-resolved LBM channel data over rough ice-keel
geometries, require **three independent friction relations** — the hydraulically-smooth power
law (lens S), the Colebrook–White implicit relation (lens C) and the Nikuradse/von Kármán
fully-rough log law (lens R) — to each reproduce the data *in the zone where that relation is
supposed to hold*, with `κ = 0.41` entering only as a **check** quantity; then push the
resulting `C_w(k_s⁺, λ/k_s)` into the sea-ice drift model, adjudicate the Nansen angle against
the observed band, and prove the end-to-end hook in TensorDEM.

| criterion | requirement | measured | verdict |
|---|---|---|---|
| C1 | smooth-wall `C_f` within 3 % | max 77.1 % (`Re_τ = 4700`, `dy⁺ = 97.9`); best 4.4 % at `dy⁺ = 3.9` | **FAIL** |
| C2 | `κ = 0.41 ± 0.02` from the fully-rough slope | `κ = 2.02 ± 0.65` (bulk scope, 12 pts, 4 morphologies); no case had a usable log window | **FAIL** |
| C3 | smooth-zone slope `n = −0.22 ± 0.03` | `n = −0.047 ± 0.013` over the 5 smooth baselines | **FAIL** |
| C4 | GCI < 5 % at fixed physical conditions | GCI(`ΔU⁺`) = 16.2 % at `k_s⁺ = 90`, `λ/k_s = 5` (observed order 1.73) | **FAIL** |
| C5 | `ΔU/U` 10–40 %, `Δα` 5–15°, state inside the observed band | `ΔU/U = 17.8…40.7 %`, `Δα = 6.9…14.4°` **but** `α = 23…64°` (outside the 20–40° band) | **FAIL** (honest) |
| C6 | TensorDEM `C_w` callback switchable, exchange residual < 1e-10 | 4 drag models switch in one config; residual **0.0** (< 1e-10) for every one | **PASS** |
| D0 | campaign resolution adequate for a roughness function | `dy⁺ = 3.9…103.5`; 0 of 20 rough cases with a ≥3-node log window | **FAIL** |

The three-lens machinery, the propagation machinery and the DEM hook are delivered and tested
(37 new tests).  What the adjudication *found* is that the campaign data behind the closure —
the sweep inherited from P103W2B-N02 (`h = 48`, `k_s = 5` nodes, `u_τ = 0.012`, 20 rough + 5
smooth cases) — **cannot carry a roughness function**.  Sections 2–4 give the machinery,
section 5 the evidence for that verdict, section 6 the re-scope and its cost, section 7 the
limits.  The honest-FAIL is not softened anywhere.

---

## 2. The three lenses (what was built)

`src/tensorlbm/ice_roughness/three_zone.py`

* **Lens S — hydraulically smooth power law.**  `C_f,S = C_f,smooth(Re_τ)` from the DNS-backed
  channel correlation, plus the *local* exponent `d ln C_f / d ln Re_b` of that correlation
  (`smooth_power_law_exponent`) and the measured exponent of the baseline series
  (`fit_power_law`).  Criterion C3 is exactly the comparison of the two.
* **Lens C — Colebrook–White.**  `1/√λ = −2 log₁₀(k_s/(3.7 D_h) + 2.51/(Re_D √λ))` solved by
  fixed point (contraction factor < 0.3, Swamee–Jain seed), with the standard duct convention
  `D_h = 4h` for the plane channel and `Re_D = U_b D_h/ν`.  Colebrook was fitted to Moody's
  experiments, so it is *not* a restatement of the log law; its roughness enters through
  `k_s/D_h`.  The relation is also inverted (`ks_over_d_from_cf_colebrook`) to *measure* the
  equivalent sand-grain ratio `k_s,eff/k_s` of each simulated morphology.
* **Lens R — Nikuradse/von Kármán.**  `ΔU⁺ = (1/κ) ln(k_s⁺/k₀)` with `k₀ = 3.87` fixed by
  Nikuradse's additive constant, `U_b⁺ = U_b,smooth⁺ − ΔU⁺`, `C_f = 2/U_b⁺²`.  Its **slope**
  is `1/κ`; fitting it to the data with per-morphology intercepts
  (`fit_kappa_group_effects`, group fixed effects, common slope) is criterion C2.  `κ` never
  enters as an input.

Adjudication (`adjudicate`) computes all three predictions for every case, labels the case's
zone from its *measured* `k_s⁺`, and credits a lens in its zone only if its median absolute
error there is ≤ 15 % **and** it is the best of the three.

`src/tensorlbm/ice_roughness/propagation.py` turns the fitted law into
`C_w(k_s, λ/k_s)`, Nansen angle and wind factor, with two explicit back-fill forms:

* **absolute** — log law + measured roughness function, no constant enters (the form a drift
  model should consume, and the headline verdict);
* **amplification** — `C_w = C_w,m5682 · C_f,rough/C_f,smooth`, which double-counts roughness
  because the `m5682` constant already contains the roughness of real ice (reported as a
  sensitivity bound).

`src/tensorlbm/dem/cw_coupling.py` + `DEMConfig.cw_callback` give TensorDEM a pluggable
`C_w` (`ConstantCwCallback`, `RoughnessCwCallback`, `TabulatedCwCallback`) and a
`MomentumExchange` ledger whose water-side reaction is recomputed independently from the
recorded drag states — so the C6 residual measures the *implementation*, not Newton's third
law by construction (a 1 % corruption of one recorded `C_w` raises the residual to ~1e-2:
negative control in `tests/test_dem_cw_callback.py`).

---

## 3. Campaign executed

25 cases on the 702 cluster (two loong64 nodes, native C/OpenMP back-end), 180.6 G
cell-updates, ~40 min wall on 10 workers:

| block | cases | design |
|---|---|---|
| ribs `λ/k_s = 2, 5, 10` | 15 | `k_s = 5` nodes, pitch `= 5 λ/k_s`, `k_s⁺ = 20, 45, 90, 200, 500` |
| Nikuradse sand control | 5 | random grains, same `k_s⁺` targets |
| smooth baselines | 5 | exactly matched `Re_τ` |
| refinement trio (`k_s⁺ = 90`, `λ/k_s = 5`) | 3 | `h = 32, 48, 64` → `dy⁺ = 30, 18, 12.9` |

`dy⁺ = Re_τ/h_eff` is fixed by the design (`Re_τ = k_s⁺ h_eff / k_s`), which is the root of
the problem in §5.

---

## 4. What the three lenses said about the data

Zone adjudication (median |rel. error| of each lens in each Nikuradse zone, from
`three_zone.lenses.zone_adjudication`):

| zone | n | lens S (smooth) | lens C (Colebrook) | lens R (log law) | expected | best | passed |
|---|---|---|---|---|---|---|---|
| transition (5 < k_s⁺ < 70) | 8 | 0.544 | **0.425** | 0.621 | C | C | no (43 % > 15 %) |
| fully rough (k_s⁺ ≥ 70) | 12 | 2.106 | **0.213** | 0.900 | R | C | no |
| hydraulically smooth (k_s⁺ ≤ 5) | 0 | — | — | — | — | — | no case exists |

Two things are worth keeping from this table even though nothing passes:

1. **The lens ordering is the physically correct one.**  In the transitional zone Colebrook is
   the best of the three and the smooth lens is worse by 30 %; in the fully-rough zone the
   smooth lens is off by a factor 3 and the log law reduces the error by more than half.  The
   machinery discriminates between the relations as intended — the *data* are not good enough
   to certify any of them to 15 %.
2. **The fitted morphology factors are unphysical in direction and size** for the dense pitch:
   `λ/k_s = 2` returns `k₀ = 101.6` (`k_s,eff/k_s = 26`), `λ/k_s = 5` → `k₀ = 11.8` (3.06),
   `λ/k_s = 10` → `k₀ = 2.66` (0.69).  Published rib/sand data give `k_s,eff/k_s ≈ 0.2–1.5`;
   a factor 26 means the fit is absorbing the numerical wall layer, not the ribs.

---

## 5. Why it fails: three independent, quantified causes

**(a) The wall layer is unresolved (`dy⁺ = 3.9 … 103.5`).**  A wall-resolved channel needs
`dy⁺ ≈ 1–2`; this campaign has one lattice node per 4–100 wall units.  The smooth baselines
show the consequence as a clean, monotone discretisation law (`figures/fig5_resolution_audit.png`):

| Re_τ | dy⁺ | C_f | C_f reference | error |
|---|---|---|---|---|
| 188 | 3.9 | 0.00615 | 0.00589 | +4.4 % |
| 423 | 8.8 | 0.00546 | 0.00481 | +13.7 % |
| 846 | 17.6 | 0.00542 | 0.00410 | +32.3 % |
| 1880 | 39.2 | 0.00535 | 0.00346 | +54.6 % |
| 4700 | 97.9 | 0.00513 | 0.00290 | +77.1 % |

The measured `C_f` is essentially *flat* in `Re_τ` (0.0062 → 0.0051) while the DNS-backed law
falls by a factor 2: the numerical wall layer adds a Reynolds-independent drag floor.  This is
what makes C1 fail (77 %), C3 fail (`n = −0.05` instead of `−0.22`, because the flat floor
dominates the power law) and the bulk-difference `ΔU⁺` unreliable.

**(b) No log window exists.**  With `3 k_s⁺ < y⁺ < 0.35 Re_τ` and `k_s/h = 0.106` the window
collapses to 1–2 lattice nodes for every rough case (`max 2`, median 1, 0 of 20 usable).  The
profile-based roughness function is therefore not measurable at all, and `κ` can only be
estimated from bulk differences — which inherit the smooth-baseline error of (a).  The
measured `κ = 2.02 ± 0.65` is the direct consequence.

**(c) The geometry itself caps the roughness function.**  In the fully-rough limit nothing can
be slower than the log law at the centreline, so
`ΔU⁺ ≤ U_b,smooth⁺ − [(1/κ) ln(h_eff/k_s) + 8.5]`.  At `k_s/h = 0.106` that ceiling is
≈ 4.5–10 over the sweep, i.e. the sand-grain law becomes unreachable above roughly
`k_s⁺ ≈ 108` — and **5 of 20 cases sit at or above the ceiling** (median ceiling fraction 0.53,
`three_zone.geometric_cap`).  Real rough-wall DNS uses `k_s/h ≤ 0.05` precisely to avoid this.

Two further numerical margins: peak bulk Mach 0.28 (> 0.2, compressibility error no longer
negligible) and minimum LBM relaxation time `τ = 0.50035`, i.e. a 3.5e-4 margin to the 0.5
stability edge, both a by-product of fixing `u_τ = 0.012` while spanning `Re_τ = 188…4700`.

**Resolution study (`figures/fig4_gci.png`, criterion C4).**  The same physical case
(`k_s⁺ = 90`, `λ/k_s = 5`) on three grids: `C_f = 0.00695 / 0.01229 / 0.01774`
(`dy⁺ = 30 / 18.0 / 12.9`), i.e. `ΔU⁺ = 5.35 / 9.34 / 11.37` against Nikuradse's 7.78 —
monotone, and a **44 % change in `C_f` between the two finest grids**.  Roache's GCI at
observed order 1.73 is **16.2 %**, three times the 5 % threshold: the quantity the closure is made
of is not grid-converged, which is the strongest single piece of evidence here.

---

## 6. Propagation (C5) and the DEM hook (C6)

**C5, headline (absolute form, `k_s = 0.35 m`, `λ/k_s = 5`, `m5682` constant 5.5e-3):**
`C_w` rises from 5.5e-3 to the closure value, the drift speed changes by **17.8 … 40.7 %** and
the Nansen angle by **6.9 … 14.4°** — both *inside* the mission's target bands — but the
resulting drift state, `α = 23 … 64°` and wind factor 0.015–0.020, only partly overlaps the
observed `α ∈ [20, 40]°` band (the low-wind end runs out of it at 64°).  The amplification form
is worse (`ΔU/U = −48 … −35 %`, `α = 8 … 30°`) — as expected, since it double-counts roughness.
**Verdict: FAIL, honest.**  And even the numbers that fall inside the mission bands must be
read with §5 in mind: they are propagated from a `ΔU⁺(k_s⁺)` that fails C1–C4.

**C6, passed.**  One `DEMConfig` field switches the particle drag between
`schiller_naumann`, a constant `C_w`, the three-zone closure and the tabulated LBM `C_w`
table; the particle velocities differ between the four (so the switch is real), the water-side
reaction impulse is recomputed independently for every drag event, and the relative
particle/water momentum-exchange residual is **0.0** (machine zero, threshold 1e-10) in all
four runs.  `ks_m` and `lambda_over_ks` reach the callback from the config (asserted), and a
1 % corruption of one recorded `C_w` raises the residual to ~1e-2 (negative control).

---

## 7. Re-scope and its cost (what a valid campaign needs)

The design identities are `k_s⁺ = k_s_nodes · dy⁺`, `k_s/h ≤ 0.1` and
`Re_τ = k_s⁺/0.1` — the last one **independently of `dy⁺`**, so a tighter wall resolution does
not raise the Reynolds number, it only grows the grid.  `three_zone.wall_resolved_cost`:

| k_s⁺ max | dy⁺ | h nodes | Re_τ | cells | cell-updates | memory | feasible |
|---|---|---|---|---|---|---|---|
| 500 | 1 | 5 000 | 5 000 | 2.5e11 | 2e17 (2e5 T-upd) | 76 TB | no |
| 500 | 10 | 500 | 5 000 | 2.5e8 | 2e13 (20 T-upd) | 76 GB | no (single case ≈ 56 h) |
| 100 | 10 | 100 | 1 000 | 2.0e6 | 3.2e10 | 1 GB | **yes** (0.1 h) |
| 100 | 5 | 200 | 1 000 | 1.6e7 | 5.2e11 | 5 GB | **yes** (1.4 h) |
| 50 | 5 | 100 | 500 | 2.0e6 | 3.2e10 | 1 GB | **yes** |

Recommended re-scope, in order of value:

1. **`k_s⁺ ≤ 100` at `dy⁺ = 5`, `k_s/h ≤ 0.1`** — 15 cases ≈ 20 h of cluster time at
   100 MLUPS, *affordable*.  This covers the transitional zone and the onset of the fully rough
   zone, which is where the three-zone law is decided; the `450`-fold range to `k_s⁺ = 500`
   stays an explicit log-law extrapolation with `k₀` from the measured low-`k_s⁺` points.
2. Do **not** keep `u_τ = 0.012` fixed across the sweep: it drives `τ → 0.5` and Mach → 0.28 at
   the top of the range.  Set `u_τ` per case from a Mach budget (`u_τ ≈ 0.03–0.05`) and get
   `Re_τ` from `ν` instead.
3. Keep the Nikuradse sand control **and** report `k_s,eff/k_s` from the Colebrook inversion:
   it is the one morphology number the campaign can define without a log window.

A wall-resolved campaign at `k_s⁺ = 500` is not a matter of more cluster time: it needs
`≥ 2.5e11` cells (76 TB), which is why published rough-wall DNS stops around `k_s⁺ ~ 100`.

---

## 8. Limitations and honest statement

* The closure `C_w(k_s⁺, λ/k_s)` **is not certified** by this node: its inputs fail C1–C4.
  The `k₀ = 101.6` of the `λ/k_s = 2` group in particular must not be used.
* C5's band numbers are reported because they are what the fitted law gives; the verdict is
  FAIL on the observed-band clause, and the propagation inherits the data problem.
* The candidates for the *cause* of the coarse-grid drag floor (numerical wall layer, SGS
  viscosity at `cs = 0.10`, Mach 0.28 compressibility, `τ → 0.5`) are named but not separated
  experimentally — separating them needs the re-scoped campaign of §7.
* What is trustworthy from this node: the three-lens machinery (analytically tested), the
  GCI/diagnostic arithmetic, the DEM `C_w` callback with its exchange proof, and the
  *negative* result that the inherited campaign design cannot measure a roughness function.

## 9. Reproduce

```bash
# campaign (cluster, 10 workers x 12 threads):
PYTHONPATH=src python tools/run_ice_rough_cw_sweep.py worker --profile campaign \
    --workers 10 --worker-index N --threads 12 --out runs/ice_rough_cw/cases
# refinement trio (C4):
PYTHONPATH=src python tools/run_ice_rough_cw_adjudication.py refine --out runs/ice_rough_cw/refine
# adjudication, figures, report:
PYTHONPATH=src python tools/run_ice_rough_cw_adjudication.py aggregate \
    --run-dir runs/ice_rough_cw/cases --refine-dir runs/ice_rough_cw/refine \
    --out artifacts/ice_rough_cw_c
# tests:
PYTHONPATH=src python -m pytest tests/test_ice_roughness_three_zone.py \
    tests/test_dem_cw_callback.py -q
```