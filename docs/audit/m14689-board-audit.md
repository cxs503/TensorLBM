# Board audit — m14689 `GAP-LBM-MCMP-POISEUILLE` (TensorLBM)

Auditor mission **m15044** · verdict_by `board-audit` · audited commit **`d0a945e2`**
("MCMP stratified Poiseuille R2: neutral-wetting SC wall, spurious-current diagnostic,
five-criteria closure"), delivered tree `/root/tp-agent/work/m14689`.

This is a **delivery audit, not a redo**: I read the delivered artifacts, re-derived every
acceptance number independently from the raw records, and re-ran the documented reproduction
commands. All numbers below are produced by `scripts/audit_m14689.py`
(`artifacts/audit_m14689/audit_summary.json`) — nothing hand-copied.

---

## 1. Scope and independence limitations

* Scope: the three-piece evidence set (run artifacts, benchmark reference-vs-got, `repro_cmd`),
  the five-criteria closure, the implementation surface (`src/tensorlbm/multiphase.py`), the test
  surface, and registration/integration of the benchmark.
* Independence: I did not write the audited code and had no access to the author's reasoning.
  I **did** share the machine and the same `tensorlbm` source tree, so "independent" here means
  *independent re-derivation and re-execution*, not an independent implementation of the solver.
* Checks **not run**: the full `run_r2_matrix.sh all` (20 cells; ~40 min) — I re-ran one acceptance
  cell end-to-end plus the wall-calibration cell plus the analysis/assembly stage, which is where
  every published number is produced. Multi-node/GPU paths untouched (the case is CPU float32).
* Not assessed: whether the physical model choice (SC linear pseudopotential) is the right
  multiphase model for the platform's use cases; only whether the benchmark measures it honestly.

## 2. What was checked, and how

| # | check | method | result |
|---|---|---|---|
| 1 | two-layer closed form is right | re-derived `u''=-G/ν` per layer with no-slip + velocity/shear-stress continuity; recomputed `Q_total` from the record configs | `max abs(ΔQ_ref) = 0.0` over 6 record×rule combos |
| 2 | reference stored in records is the reference used for verdicts | recomputed with `analytic.layer_viscosities/layered_reference/sample_profile` (`halfway` convention) and compared | identical to the stored blocks (`err_Q_mix` matches to 1e-16) |
| 3 | criterion ① (interface position < 5 %) | `y_int / (ny/2-0.5) - 1` | **−3.84 % / −2.95 % / +1.38 %** → PASS |
| 4 | criterion ② (flow rate < 10 %, both viscosity rules as pre-registered) | `Q_mix / Q_ref - 1` | mixture −3.97 / −2.71 / +0.29 %; component −6.46 / −5.24 / −2.30 % → PASS |
| 5 | criterion ③ (spurious current < 0.005·u_max) | `S_all/u_max` per acceptance record | **0.3968 / 0.2049 / 0.0861** → FAIL (gate missed 79×/41×/17×) |
| 6 | `Q_mix` is the profile integral, not a separate number | `|Q_mix − Σ u_profile_mix|` | 2.5e-7 (float32 round-off) |
| 7 | single-cell reproducibility | re-ran `run_sweep.py --ny 25 --steps 20000 --g12 -2.5 --u-eq viscous --force-scheme guo --wall-mode mirror` | bit-identical: `Q_mix 0.391216903924942`, `u_max_mix 0.02561205066740513`, `spurious_mix 0.010163000784814358`, `interface_crossing 11.539123946161155` |
| 8 | wall-convention calibration (verdict-critical) | re-ran `run_sweep.py --mode walls --ny 25` | `H=ny−1`: u_max −8.23 %, Q −12.17 %; `H=ny−2`: u_max −0.08 %, Q −0.20 % → halfway convention is justified by an **independent single-phase** measurement |
| 9 | tests | `pytest tests/test_mcmp_wall_neutral.py tests/test_mcmp_stratified_poiseuille.py -q` | **12 passed, 30.19 s** |
| 10 | "no hand-typed numbers" | re-ran `r2_benchmark.py` on the delivered tree and hashed the outputs | numbers regenerate, **but** the artifact hash does **not** reproduce (finding F2) |
| 11 | static-control evidence | inspected the 40k and (uncommitted) 80k `g25, ny=49, mirror` controls | the "floor" is a **limit cycle**, not a converged value (finding F3) |
| 12 | registration/integration | `grep` for the benchmark id in `benchmarks/matrix.md`, callers of `wall_mode` in `src/` | no matrix row; `wall_mode` used only by case scripts/tests (F6) |

## 3. Severity-ranked findings

### F1 (high, claim strength) — `verified` is admitted under thresholds looser than the repo's own rule
`benchmarks/matrix.md:4` states the admission rule for `verified/`: **error ≤ 3 %**. The R2 case is
labelled `status: verified` (`spec.json`, `REPORT.md:3`) with pre-registered thresholds 5 % (①) and
10 % (②); measured worst cases are **3.84 % (①, ny=25)** and **3.97 % (② mixture, ny=25)**,
6.46 % (② component). Under the house 3 % rule only the **ny=49 hold-out** qualifies (and that record
is `steady=False`, so it is *not* an acceptance cell). The apparatus is complete and honest; the
**label** overstates it relative to the repository norm.

### F2 (medium, reproducibility of the published artifact) — `repro_cmd` does not reproduce the committed `result_r2.json`
The published `repro_cmd` (`result_r2.json`, key `repro_cmd`) is
`run_r2_matrix.sh all && analyze.py && r2_benchmark.py`. Running the assembly stage on the delivered
tree rewrites `result_r2.json` (+2026 lines) and `r2_spurious_current.png` (sha256 differs), because
`run_r2_matrix.sh` includes the cell `sp_static_ny49_long` whose output
`artifacts/.../diag/sp_gx0_ny49_g25_mirror_long.json` is **present in the work tree but untracked**
(`git status`: `??`). The committed artifact was therefore assembled from a strictly smaller input
set than the documented command produces. Per-cell reproducibility is exact (check 7); the published
*aggregate* artifact is not hash-reproducible.

### F3 (medium, evidence validity) — the reported "static floor" is a limit cycle, not a converged floor
The untracked 80 000-step control (`sp_gx0_ny49_g25_mirror_long`) shows `S_ifc` oscillating between
**1.88e-3 and 9.20e-3** (4.3×) over its second half; the committed 40 000-step run's own second half
spans 1.20e-3–9.82e-3. The table value `S_ifc = 2.24e-3` (`ratio 0.0195`, REPORT §3.3) is a **phase
sample** of that cycle. Consequences: the "resolution-independent floor" argument is right in
magnitude, but (a) the per-value floor numbers are not converged, and (b) the round-2 candidate
`|G12|=2.0 → ratio 0.0019 < gate` is even less defensible than REPORT already admits. The ③ **FAIL**
verdict is unchanged (it is only strengthened).

### F4 (medium, external claim) — the platform report summary understates the ③ miss
Report id 13845 (m14689) summary says the spurious gate "still fails by 4-13x". Measured on the
acceptance cells: **79.4× (ny=25), 41.0× (ny=33), 17.2× (ny=49)** in ratio, i.e. 3.9×–79× reduction
in `S` would be needed. The in-repo REPORT says "4×–100×", which is closer. The public one-line
summary understates the gap by roughly an order of magnitude at ny=25.

### F5 (low, prose) — "8.7×" reduction is 8.45× in the committed numbers
REPORT §3.3: `0.745 → 0.086 = 8.7×`. With one denominator (`u_max_analytic = 0.115`) the committed
values give `0.7453 → 0.0882 = 8.45×`; 8.7 mixes the two denominators. Direction and conclusion
unaffected; the 3 % overstatement should be corrected.

### F6 (low, integration/registry) — the case is not registered
* no row in `benchmarks/matrix.md` for `mcmp_stratified_poiseuille` (row P22 is the older
  SC-MCMP two-phase Poiseuille, still `❌ FAIL` anti-convergence 26.82→33.17 %);
* the R2 verdict lives in the **non-canonical** `result_r2.json`; the canonical `result.json`
  (round-1) still carries the older cell set and its own `overall_verdict`;
* `spec.json` still describes the walls as "effective wall planes at the wall-cell centres", i.e. the
  convention R2 **rejected** — the pre-registration was not annotated after calibration;
* `wall_mode="mirror"` is reachable only through `sc_two_component_force` /
  `collide_sc_two_component` (public, exported at `src/tensorlbm/__init__.py:1333`); no production
  solver or wettability path (`apply_wall_wettability_sc`) consumes it. That is a library-API
  integration, not a solver integration.

### F7 (low, evidence plumbing) — the verdict-critical calibration is missing from the R2 artifact chain
`artifacts/.../analysis.json` has `"wall_calibration": []`, and `repro_cmd` contains no
`--mode walls` cell, although the `H=ny−2` choice decides ② (−3.97 % vs −17.7 % under `H=ny−1`).
The calibration **is** recorded in round-1 `result.json` and I reproduced it (check 8), so the claim
is verifiable — but a reader of the R2 chain cannot see it.

### F8 (note) — duplicated controls and a misleading runner summary
`sp_gx0_ny25_dry.json` and `sp_gx0_ny25_g25_dry.json` (and the two `ny=25` mirror files) are the same
run committed twice; `sp_static_floor` consequently repeats rows. Separately, `run_sweep.py`'s stdout
`SUMMARY.errors` block prints the **cell-centre** reference (`Q_ref 0.4752`, `err_Q_mix −17.7 %`) while
the delivered record carries the calibrated reference (−3.97 %) — the same script therefore shows two
different "errors" for the same run.

## 4. Checks that passed without qualification

* the two-layer analytic reference is correct and is the one actually used (checks 1–2);
* criteria ① and ② pass as delivered, against **both** pre-registered viscosity rules (checks 3–4);
* the honest **FAIL** on criterion ③ is real, with `fail_stage`/`failure_reason` recorded (check 5);
* `Q_mix` is the profile integral, not an independent hand-typed number (check 6);
* a single acceptance cell re-runs **bit-identically** (check 7);
* the wall convention was chosen by an independent measurement, reproduced here (check 8);
* the 12-test suite is green, including the bit-identity assertion for the historical `dry` path (check 9);
* the no-fabrication claim holds: every number in `result_r2.json` is derivable from committed
  records — the only regeneration difference traces to the single extra (untracked) input of F2;
* REPORT §3.3 and §5 disclose the honest FAIL and the post-hoc discovery of the `|G12|=2.0` window —
  no retro-fitted pass.

## 5. Level

**L4 — benchmark validation** (complete case + error analysis + reproducible), *not* L5:

* L0 (documents) — exceeded.
* L1 (math/unit) — the closed form and the mirror-wall zero-force identity are verified by tests.
* L2 (independent implementation) — real code: input → solve → numbers, ~1 s/1 k steps, no stub.
* L3 (integration) — **partial**: public library API (`wall_mode` kwarg, default unchanged) but no
  solver/wettability-path caller and no benchmark-registry row (F6).
* **L4** — complete case (ny = 25/33/49, dry vs mirror, 14 diagnostics), error analysis across 2
  viscosity rules × 2 wall conventions, bit-reproducible cell, honest FAIL with a stated next-method
  list.
* L5 — not claimed: the physical validation is **partial** (③ fails, and by the house ≤3 % rule only
  the non-steady hold-out passes), and there is no production consumer.

## 6. Disposition

**Approve with notes** for the *delivered* verdicts (① PASS, ② PASS, ③ honest FAIL, ④ PASS at the
reported thresholds); the audit's independent numbers agree with the delivered ones to the last
printed digit. The notes F1–F4 are claim-strength defects, not arithmetic defects: the case should be
labelled *verified against pre-registered 5 %/10 % thresholds* rather than *verified* under the house
3 % rule, the untracked 80 k control must be committed and the static-floor values re-quoted as a
cycle range, the platform summary's "4-13×" must be corrected to 17–79×, and the benchmark needs a
`matrix.md` row + canonical `result.json` before it is discoverable by the fleet.

## 7. Artifacts of this audit

| path | content |
|---|---|
| `scripts/audit_m14689.py` | the whole audit (re-derivation, re-run parsing, figures) |
| `artifacts/audit_m14689/audit_summary.json` | every checked number in machine form |
| `artifacts/audit_m14689/audit_m14689_criteria.png` | ①② error vs ny with gates, ③ ratio with the 0.005 gate |
| `artifacts/audit_m14689/audit_m14689_static_control.png` | `S_ifc(t)` of the static control: limit cycle, not a floor |