# Periodic-array (Hasimoto) drag reference — cross-module audit and repair

**Anchor**: `GAP-LBM-SPHERE` (in_progress) — the sphere case is judged by a *third,
correlation-free* force instrument whose confinement factor **is** the Hasimoto
periodic-array series; a defective judge silently moves the verdict. Second
beneficiary: `CASE-LBM-CUBIC-ARRAY-DRAG`, which uses the same series as its
analytic cross-check.
**Mission**: m16754 · branch `agent/m16754` · 2026-11-12.
**All numbers**: `scripts/array_drag_reference_audit.py` →
`artifacts/array_drag_reference/{audit.json,tables.md,fig.png}` (nothing typed
by hand here; the tables file is machine-generated).

---

## 1. What was found

The same physical reference — the drag enhancement
`K = F / (6 π μ a U)` of a sphere in a **simple-cubic** periodic array,
Hasimoto (1959, *JFM* 5:317) — existed **seven times** in the repository, in
three mutually inconsistent versions:

| # | site | form before the audit | verdict |
|---|---|---|---|
| 1 | `benchmarks/sphere_curved_drag/driver.py::hasimoto_K` | `1 − 2.837λ + 4.19λ³ − 27.4λ⁵` | **defect**: λ⁵ should be λ⁶ |
| 2 | `benchmarks/sphere_curved_drag/analysis_v2.py::K_hasimoto` | same λ⁵ form | **defect** (copy of #1) |
| 3 | `src/tensorlbm/force_control_surface.py::hasimoto_factor` | `1 − 2.10444λ + 2.08877λ³ − …` | **defect**: −17.1 % at λ=1/6, not the SC series |
| 4 | `src/tensorlbm/d3q27_recovery.py::hasimoto_correction` | `1 − 2.837297λ + 4.19λ³` | OK (documented truncation, −0.05 % at λ=1/6) |
| 5 | `src/tensorlbm/particle_suspension.py::hasimoto_drag_correction` | `1 − 1.7601φ^{1/3} + φ` | OK (documented O(φ²) truncation) |
| 6 | `src/tensorlbm/sc_sphere_array_drag.py::hasimoto_K` | 5-term φ series | OK (correct series, duplicated) |
| 7 | `benchmarks/case_pool/sc_sphere_array_drag/run.py::hasimoto_K` | 5-term φ series | OK (correct series, duplicated) |

The volume-fraction form is the one that is **two-source locked** in the
ecosystem (`benchmarks/verified/permeability3d/NOTES.md` §1.6):

```
U/U0 = 1 − 1.7601 φ^{1/3} + φ − 1.5593 φ² + 3.9799 φ^{8/3} − 3.0734 φ^{10/3}
K    = U0/U,      φ = 4π(a/L)³/3
```

so the λ-forms are truncations of *this* series, and the audit's job was to
prove which ones are and which are not.

## 2. How it was done

1. **Derivation instead of transcription.** `tensorlbm/array_drag_reference.py`
   now holds the series once, in φ. The λ coefficients are *produced* from it
   by `φ = (4π/3)λ³`
   (`lambda_series_coefficients()`), giving
   `−2.83727 λ + 4.18879 λ³ − 27.3594 λ⁶ + 181.46 λ⁸ − 364.12 λ¹⁰`.
   No coefficient is typed twice, so a future edit cannot split the family.
2. **Independent anchor.** The canonical series is compared with the locked
   Sangani–Acrivos / Zick–Homsy SC table already in the repo
   (`SC_TABLE`), which was *not* produced by this series.
3. **Call-site unification.** Sites #1, #2, #3, #6 delegate to the canonical
   module (sites #4/#5 keep their documented truncations and are now
   consistency-tested instead of duplicated). Site #7 delegates as well.
4. **Regression tests** (`tests/test_array_drag_reference.py`, 15 cases):
   locked-table agreement, λ↔φ identity, derived-coefficient checks against the
   constants quoted elsewhere in the repo, cross-site agreement, and two tests
   that *reproduce the replaced defective series* so they cannot come back.

## 3. Results

### 3.1 The canonical series is the locked one

`φ = 0.027 → K = 2.0077` (table 2.0077), `φ = 0.125 → 4.2894` (4.292, −0.06 %),
`φ = 0.216 → 7.3827` (7.4423, −0.80 %); at `φ = 0.343` it is −7.4 % low, i.e.
outside the dilute radius of convergence. The module therefore declares
`SERIES_MAX_PHI = 0.216`, and `benchmarks/sphere_curved_drag/driver.py` now
*raises* outside `λ ∈ [0, SERIES_MAX_LAMBDA = 0.3722]` instead of silently
returning an extrapolated number (`hasimoto_K(0.4)` still raises, so the
pre-existing contract is preserved).

### 3.2 Consequences at the geometry the sphere case actually runs

`GAP-LBM-SPHERE` runs `R = 6, L_cell = 36`, i.e. `λ = R/L = 1/6` (the
`prereg.md` convention `λ = 2R/L` is the separate defect already logged as
amendment A1 in `docs/GAP-LBM-SPHERE_v3_converged_control_verdict.md` §3.1):

| quantity | value |
|---|---|
| canonical `K(λ=1/6)` | **1.83140** |
| pre-audit driver `K` | 1.84148 (**+0.55 %**) |
| pre-audit `force_control_surface` `K` | 1.51795 (**−17.1 %**) |
| `K` if the prereg's `λ = 2R/L` is fed in | 5.17210 (**+182 %**) |
| criterion tolerance (`GAP-LBM-SPHERE`) | 8 % |
| share of that tolerance consumed by the +0.55 % bias | **6.9 %** |

So: the driver defect did **not** flip the GAP-LBM-SPHERE verdict (the observed
`F_bfl` deficits are 13–24 %, an order of magnitude larger), but it ate 7 % of
the criterion's error budget and it made the reported `K` values in the v3
table wrong in the 4th digit (1.8415 → 1.8314). The
`force_control_surface` series is a genuine outlier: any array-drag comparison
made with it was 17 % pessimistic before this audit.

### 3.3 Where the λ-truncations stop being usable

| form | λ=1/10 | λ=1/8 | λ=1/6 | λ=1/4 |
|---|---|---|---|---|
| truncation through λ⁶ | −0.00 % | +0.00 % | +0.02 % | +0.69 % |
| `d3q27_recovery` through λ³ | −0.00 % | −0.01 % | −0.09 % | −1.20 % |
| pre-audit driver (λ⁵) | +0.03 % | +0.11 % | **+0.55 %** | **+6.80 %** |
| pre-audit `force_control_surface` | −8.99 % | −11.82 % | **−17.12 %** | −30.37 % |

(values printed by `artifacts/array_drag_reference/tables.md`; the two OK
sites are within 0.1 % wherever the repository actually evaluates them).

![audit](../figures/array_drag_reference_audit.png)

## 4. 注意事项 / 局限

* The series is a **dilute expansion**. `φ > 0.216` (λ > 0.372) needs a
  tabulated reference (the locked `SC_TABLE`); the 5-term series is −7.4 % at
  `φ = 0.343` and must not be used there.
* The provenance chain is: Hasimoto (1959) → `permeability3d/NOTES.md` §1.6
  (two-source locked) → `SC_TABLE` (Sangani–Acrivos / Zick–Homsy). This audit
  could not open the original papers (offline); it verifies *internal
  consistency with the locked transcription* and the analytic limit
  `K(λ→0) → 1`, not the transcription itself.
* `SC_TABLE` values were not re-derived here; they are treated as the
  literature anchor they are documented to be.
* No simulation was re-run: the anchors are the already-committed
  `v3/three_instrument.json` records and pure arithmetic. Consequently the
  *simulated* side of GAP-LBM-SPHERE (the 13–24 % `F_bfl` deficit, `steady=False`
  on all three records) is untouched and still honestly unadjudicated.
* `force_control_surface.hasimoto_factor` had no live verdict consumer (only
  the module export and one test), so no historical pass/fail needs re-grading;
  that is why this is a `defect_fix` and not a re-verdict of a merged result.

## 5. 误差判定（关键量 vs 基准，逐项）

| 量 | 基准 | 自报误差 | < 3 % ? |
|---|---|---|---|
| canonical `K(φ=0.125)` | locked SC table 4.292 | −0.06 % | **是** |
| canonical `K(φ=0.216)` | locked SC table 7.4423 | −0.80 % | **是** |
| canonical `K(φ=0.343)` | locked SC table 15.402 | −7.4 % | 否（超出级数收敛半径，已声明禁用） |
| λ-form coefficients | derived from φ form | ≤ 1e-12 (identity) | **是** |
| derived `−27.3594 λ⁶` vs the repo's quoted `27.4` | rounding of literature | 0.15 % | **是** |
| `d3q27_recovery` truncation vs canonical | canonical | 0.05 % @ λ=1/6 | **是** |
| pre-audit driver vs canonical | canonical | +0.55 % @ λ=1/6 / +6.80 % @ λ=1/4 | 见 3.2（已修复） |

## 6. Test run (honest)

`python -m pytest tests/test_array_drag_reference.py tests/test_sphere_curved_drag.py
tests/test_force_control_surface.py tests/test_d3q27_recovery.py tests/test_fcm.py -q`
→ **64 passed, 1 failed**. The failure is
`tests/test_fcm.py::test_single_sphere_drag_near_stokes` (a drag *simulation*,
>120 s). It is independent of this diff by import graph — neither
`src/tensorlbm/fcm.py` nor that test imports `force_control_surface`,
`sc_sphere_array_drag` or the new module — but the pre-change baseline re-run
was **not** obtained (the attempt exceeded the shell timeout), so its status is
`not_run` rather than `passed`.

## 7. Reproduction

```
python scripts/array_drag_reference_audit.py               # regenerates §3 tables + figure
python -m pytest tests/test_array_drag_reference.py -q     # 15 passed
python -m pytest tests/test_sphere_curved_drag.py tests/test_force_control_surface.py -q
```