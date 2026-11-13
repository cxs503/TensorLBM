# Adversarial Audit: m17718 XVAL-LBM-BITWISE-REFDATA-REGRESSION-PLUS-BENCHMAR

**Auditor:** board-audit (mission m17734)  
**Date:** 2026-11-13  
**Subject:** m17718 — Bitwise reference-data regression lock + analytic TGV benchmark for TensorLBM D2Q9 primitives  
**Repo:** TensorLBM  
**Verdict:** **L4 — APPROVED** (benchmark-verified, all gates pass, defect found & fixed)

---

## 1. Scope and Independence

This audit independently verifies the three deliverables of m17718 against the benchmark criteria:
1. **run_artifacts** files — do they exist and contain real data?
2. **benchmark ref vs got** — are computed values within tolerance?
3. **repro_cmd** — does it reproduce the claimed result?

I re-executed the repro_cmd from a clean state, re-ran the full pytest suite, and cross-checked every number in result.json against the README and REPORT.md. I did **not** re-derive the analytic TGV solution (it is a standard textbook result) but verified the formula matches the stated reference.

**Limitations:** The `tensor_benchmark_cases` table was not found in any accessible database; benchmark criteria were taken from `result.json`'s `tolerance_pct=15.0%`, which is consistent with the README, test file, and owner's accuracy bar.

---

## 2. Deliverable Verification

### 2.1 run_artifacts — ✅ ALL EXIST AND CONTAIN REAL DATA

| artifact | path | size | verified |
|---|---|---|---|
| Machine-produced result | `benchmarks/xval_bitwise_refdata/result.json` | 39,613 B | ✅ JSON valid, all gates populated |
| Frozen golden data (9 cases + payloads) | `benchmarks/xval_bitwise_refdata/refdata/d2q9_bitwise_v1.json` | 244,990 B | ✅ 9 cases with sha256 + base64 payload |
| TGV benchmark figure | `benchmarks/xval_bitwise_refdata/figures/tgv_bitwise_bench.png` | 125,140 B | ✅ PNG readable |
| Defect root-cause probe | `benchmarks/xval_bitwise_refdata/probe_weight_bias.json` | 558 B | ✅ hypothesis_confirmed=true |
| Six-section report | `docs/xval_bitwise_refdata_m17718/REPORT.md` | 12,484 B | ✅ all 6 sections present |

### 2.2 Benchmark ref vs got — ✅ ALL WITHIN TOLERANCE

| case | ref | got | err% | <15%? | <3%? |
|---|---|---|---|---|---|
| 24×24 τ=0.8000 steps=56 | 1.64087301e-02 | 1.62870907e-02 | 0.7413 | ✅ | ✅ |
| 32×32 τ=0.8000 steps=100 | 1.63525945e-02 | 1.62810988e-02 | 0.4372 | ✅ | ✅ |
| 64×64 τ=0.8000 steps=400 | 1.63525945e-02 | 1.63350467e-02 | 0.1073 | ✅ | ✅ |
| 32×32 τ=0.7000 steps=150 | 1.66067266e-02 | 1.65314166e-02 | 0.4535 | ✅ | ✅ |
| 32×32 τ=0.9000 steps=75 | 1.63525945e-02 | 1.62903629e-02 | 0.3806 | ✅ | ✅ |

**Worst error: 0.7413%** — 20× inside the 15% bar, 4× inside the 3% bar.

### 2.3 repro_cmd — ✅ INDEPENDENTLY REPRODUCED

```
PYTHONPATH=src python3 benchmarks/xval_bitwise_refdata/run.py check
```

- Deleted result.json, re-ran from clean state → exit code 0
- All 9 regression cases: BITWISE_OK
- Negative controls: both_detected=True
- Benchmark: worst_err_pct=0.7413%, passed=True
- Fresh result.json written with identical numbers

### 2.4 Pytest suite — ✅ 14/14 PASS

```
tests/test_xval_bitwise_refdata.py — 14 passed in 12.62s
```

Including: bitwise reproduction, kernel drift detection, golden inconsistency rejection, env mismatch degradation, negative controls, mass conservation, stream permutation, collision mass, fused chain drift, float32 weight table invariance, cross-process digest reproducibility, TGV benchmark tolerance.

---

## 3. Regression Lock Deep Verification

| case | verdict | max_ulp | notes |
|---|---|---|---|
| feq_int_field_16 | BITWISE_OK | 0.0 | equilibrium from integer algebra |
| bgk_step_16 | BITWISE_OK | 0.0 | BGK collision + stream |
| trt_step_16 | BITWISE_OK | 0.0 | TRT collision + stream |
| mrt_step_16 | BITWISE_OK | 0.0 | MRT collision + stream |
| bgk_fused_step_16 | BITWISE_OK | 0.0 | fused BGK (after _w_for fix) |
| rlbm_step_16 | BITWISE_OK | 0.0 | regularized LBM |
| stream_periodic_16 | BITWISE_OK | 0.0 | streaming only |
| chain_trt_50_24 | BITWISE_OK | 0.0 | 50-step TRT chain on 24² |
| chain_bgk_50_f32_16 | BITWISE_OK | 0.0 | 50-step BGK chain float32 |

env_match=True, ulp_budget=0.0, n_regression=0.

**Negative controls:**
- 1-ulp tamper of one population → digest detects ✅
- τ bump (1e-7 relative) → digest detects ✅ (max_ulp=15,905,815)
- both_detected=True ✅

---

## 4. Defect Discovery Assessment

The lock discovered a real defect: `collide_bgk_fused` was casting float32 weights to float64, causing a mass drift of 9.313e-9 per step (1.86e-6 over 200 steps). The fix (`_w_on(device).to(dtype)` → `_w_for(dtype, device)`) is one line in `src/tensorlbm/solver.py:120`.

**Root-cause probe verification:**
- `probe_weight_bias.json`: predicted step drift = 9.313e-9, measured = 9.313e-9, prediction error = 0.0
- After fix: measured step drift = 0.0, 200-step drift = -1.38e-14 (round-off)
- Float32 path unchanged (test_float32_weight_table_is_unchanged_by_the_fix passes)

**Noted gap:** Same-class unfixed sites exist (`collide_rlbm`, `d3q27.py`, `solver3d.py`, `cylinder_cross_validation.py`) — recorded in REPORT.md §6.1 for follow-up. This is honest and appropriate.

---

## 5. Report Six-Section Completeness

| section | present | quality |
|---|---|---|
| ① Problem setting & reference | ✅ | Clear: bitwise lock motivation + analytic TGV reference |
| ② Method & implementation | ✅ | Detailed: integer algebra, sha256 digest, ulp metrics, negative controls |
| ③ Results | ✅ | Numeric tables with programmatic values, no hand-copying |
| ④ Negative controls & defect | ✅ | Both controls documented, defect root-caused with prediction vs measurement |
| ⑤ Convergence & sensitivity | ✅ | Resolution series (24→32→64) + τ series (0.7/0.8/0.9) |
| ⑥ Error judgement | ✅ | All errors <3%, explicit "yes" for both 3% and 15% bars |

---

## 6. Severity-Ranked Findings

| # | severity | finding | evidence |
|---|---|---|---|
| 1 | **non-blocking** | `tensor_benchmark_cases` table not found in DB; benchmark criteria from result.json | All DB files checked, table absent |
| 2 | **non-blocking** | Same-class unfixed `_w_on().to(dtype)` sites in 3D/RLBM paths | REPORT.md §6.1 lists 4 sites |
| 3 | **non-blocking** | `collide_bgk_matmul` excluded from bitwise lock (BLAS non-determinism) | README documents exclusion; appropriate |

No blocking issues found.

---

## 7. Maturity Level Determination

| level | definition | met? |
|---|---|---|
| L0 | Document output | Exceeded |
| L1 | Math/unit verification | Exceeded |
| L2 | Independent functional implementation | Exceeded |
| L3 | System integration | Exceeded |
| **L4** | **Benchmark verification (complete cases + error analysis + reproducible)** | **YES** |
| L5 | Research engineering output | Close but not claimed |

**L4 justification:**
- ✅ Complete benchmark cases: 5 parameter points with analytic reference
- ✅ Error analysis: per-point errors, convergence study, parameter sensitivity
- ✅ Reproducible: repro_cmd independently verified, 14 tests pass, bitwise lock
- ✅ Defect found and fixed with quantitative root-cause analysis
- ✅ Report six-section complete with programmatic numbers

---

## 8. Disposition

**APPROVE.** m17718 delivers a genuine L4 bitwise regression lock with analytic benchmark anchoring. All three deliverables verified, all gates pass, defect discovery is real and fixed, repro_cmd works independently. No blocking issues.

---

## 9. Checks Not Run

- Cross-platform bitwise reproduction (only loongarch64 available)
- 3D kernel bitwise lock (out of scope for this mission)
- `tensor_benchmark_cases` SQL query (table not found)
