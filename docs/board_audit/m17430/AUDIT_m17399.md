# Board cross-audit (交差核验) — m17399

**Anchor**: `XVAL-LBM-AA-PATTERN-EVEN-ODD-TIMESTEP-TRACKER`
**Audited mission**: m17399 (repo TensorLBM, branch `agent/m17399`, HEAD `d951110e`, status `succeeded`)
**Report under audit**: `tensor_reports.id = 16394`
**Audit mission**: m17430 (`board-audit`), branch `agent/m17430`
**Verdict**: **L2** (independent functional implementation), `gate_state` machine-adjudicated.

Verification, not re-do. Nothing below is taken from the mission report or from the
mission's own `result` field: every number is re-derived from the delivered bytes, from a
fresh execution, or from an independent from-scratch implementation in this audit worktree.

---

## 0. Registered referee check (the sole judge)

```sql
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug = 'XVAL-LBM-AA-PATTERN-EVEN-ODD-TIMESTEP-TRACKER';
-- (0 rows)     table total = 53 rows
```

**There is no registered referee row for this anchor.** The 15 % gate in
`benchmarks/xval_aa_pattern_even_odd.py:58` (`GATE_PCT = 15.0`) and the acceptance
thresholds in `result.json["criteria"]` are therefore *self-declared by the author*, not
platform-registered. Under the "registered referee is the sole judge" rule this alone
blocks L4 (see §5, F4).

---

## 1. Three acceptance pieces — do they really exist?

| piece | pointer | state |
|---|---|---|
| impl | `/root/tp-agent/work/m17399/src/tensorlbm/aa_pattern.py` (15752 B, sha256 `ae82b0ff…dce`) — `AAPatternStepper` :111, `_stream_half` :199 (in-place per-direction gather at :206), `_kernels_for_timestep` :234 (parity→kernel order), `step`/`_complete_timestep` :244/:248 (parity flip at :276), `parity_blind_step` :284 (defect arm), `_require_complete` :301 (gate), `parity_report` :319 | **REAL**, tracked in git |
| bench entry | `/root/tp-agent/work/m17399/benchmarks/xval_aa_pattern_even_odd.py` (38716 B, sha256 `af85d137…bd0a`) — `run_arm` :144 drives `st.step()` for every one of 1000 steps for the AA arm (:161-207), `ab_step` :242 for the AB arm, `hazard_probe` :262, `analyse` :291 | **REAL**, tracked |
| tests | `/root/tp-agent/work/m17399/tests/test_aa_pattern_parity.py` (15 `test_*`, sha256 `cf5dbf96…6633`) | **REAL**, tracked |
| run_artifacts | `/root/tp-agent/work/m17399/artifacts/xval_aa_pattern_even_odd/result.json` (`f87c2f02…8ac3`) + 9 × `k_history_<arm>_<case>.csv` + `/root/tp-agent/work/m17399/figures/xval_aa_pattern_even_odd/{tg_decay_aa_vs_ab_vs_exact,tg_field_aa_vs_exact,parity_hazard}.png`; duplicate copy under `runs/xval_aa_pattern_even_odd/` | **exists, git-tracked (durable)** |

Byte copies + `SHA256SUMS` of everything audited: `artifacts/audit_m17430/bytecopies/`,
`artifacts/audit_m17430/SHA256SUMS`.

Durability caveat: `agent/m17399` is **not merged into main** — main (`c05d7af3`, the base
of this audit worktree) has no `src/tensorlbm/aa_pattern.py`. The deliverable lives only on
the mission branch.

---

## 2. `repro_cmd` — executed, not trusted

Documented command: `python benchmarks/xval_aa_pattern_even_odd.py && pytest tests/test_aa_pattern_parity.py -q`.

Executed in a throw-away tree (`/tmp/audit_m17430/repo` = delivered `src/`, `tests/`,
`benchmarks/`, `pyproject.toml` copied out of m17399; `PYTHONPATH=src`):

| check | result |
|---|---|
| benchmark exit | 0, `result.json["overall"] == "PASS"`, 10/10 criteria `true` |
| 9 history CSVs | **byte-identical** to the delivered ones (sha256 match on all 9, e.g. `k_history_aa_tg_n64_re100.csv = bda086ec…2c0e`) |
| 242 physics/derived numeric fields of `result.json` (all except `wall_s`/`s_per_step`) | **max abs delta = 1.8e-12** (float64 reduction-order noise; single outlier `half_probe.mass` 1.64e4 × 1e-12) |
| wall time | delivered 5.9 s/case vs audit 82 s/case — sandbox is heavily loaded; **not** a physics difference |
| `pytest tests/test_aa_pattern_parity.py -q` | **15 passed in 6.19 s** (report claims 15 passed — confirmed) |

So the repro command is executable and deterministic; the delivered artefacts are exactly
what the delivered code produces.

---

## 3. Independent reconstruction (own code, zero repo imports)

`artifacts/audit_m17430/independent_recompute.py` re-implements, from the textbook
definitions and **without importing `tensorlbm.aa_pattern` or `tensorlbm.solver`**: D2Q9
equilibria, BGK collision, the two-field AB reference, and a single-field in-place scheme
that moves the velocity halves `S={1,2,5,8}` / `S'={3,4,7,6}` in alternating order. Output:
`artifacts/audit_m17430/independent_recompute.json` (1000 steps, `U0=0.05`, `Ma=0.0866`).

| case | ν | γ_K exact = 4νk² | audit γ_K err | delivered γ_K err | audit field L2 | delivered field L2 | max&#124;f_AA−f_AB&#124; (audit) |
|---|---|---|---|---|---|---|---|
| tg_n64_re100 | 0.032 | 1.23370e-3 | **−0.0756 %** | −0.0593 % | **0.3044 %** | 0.3044 % | **0** |
| tg_n128_re100 | 0.064 | 6.16850e-4 | **−0.1014 %** | −0.0749 % | **0.3003 %** | 0.3003 % | **0** |
| tg_n128_re320 | 0.020 | 1.92766e-4 | **−0.1353 %** | −0.0715 % | **0.3651 %** | 0.3651 % | **0** |

* The two headline claims survive independent reconstruction: **field L2 agrees to all
  printed digits**, and the **single-field in-place scheme is bitwise identical to the
  two-field AB reference** (`max|Δf| = 0.0` exactly, on all three cases) — the permutation
  argument the module rests on is correct.
* The γ_K error differs by 0.02–0.06 pp because the audit fits `log K(t)` by least squares
  over the **whole** history while the author's `analyse()` (:291) fits a window that is not
  documented. Both are ≪ 3 %. Reported `γ_K_r2 = 0.999995` is reproduced independently
  (0.999995).
* Half-streamed-window hazard, independently measured on the smooth TG field: **3.29 %**
  (delivered 3.33 %, N=64). On white noise the delivered figure is 119.08 %.
* Spec self-consistency re-derived: `ν = U0·N/Re`, `τ = 3ν+0.5` → 0.596 / 0.692 / 0.560,
  matching `result.json["specs"]`; `k = 2π/N`; `γ_K = 4νk²` matches to 1e-9 relative.

---

## 4. Report six-section check (report id 16394)

① case intro (TG physics, exact solution, geometry/params) ✓ · ② method (AA single field,
`e`/`o` halves, three arms, parity gate) ✓ · ③ numbers table, programmatically injected ✓ ·
④ three embedded figures, all three PNGs present and tracked ✓ · ⑤ caveats ✓ · ⑥ explicit
error judgement with an answered "is it <3 %?" ✓. No hand-copied numbers were found: the
report table values match `result.json` (`0.00123297` ← `0.0012329685`, `−0.0749 %`,
`0.3651 %`, `9.91e-14`, `2.63e-16`, `164.99 %`).

Honesty check: the report *itself* discloses the two things an auditor would otherwise
flag — the upstream `ND = SPEEDS + PAD` dual-slot A-A indexing is **not** implemented
(single-slot + row buffer instead, declared round-2), and upstream `aa-pattern-e/o` source
could not be fetched (no egress) so no byte-level identity is claimed. That disclosure is
**accurate**: verified, the module really is single-slot (`_stream_half`, :199-218), and the
tree contains no vendored upstream source.

---

## 5. Findings (severity-ranked)

**F1 — benchmark not registered (blocks L4, author-actionable).**
0 rows in `tensor_benchmark_cases` for this anchor → no platform tolerance judge; the 15 %
gate is local to the script. Recommended registration: `metric = "AA vs AB bitwise max|Δf|
(max over 3 cases) AND γ_K err vs 4νk²"`, `tolerance = "=0 AND <3 %"`, which is machine
checkable and cannot be passed by a parity-blind implementation.

**F2 — `kernel_order_immaterial` is vacuous, i.e. structurally true (overstatement, not a
bug).** `_stream_half` (:206) gathers direction `i` out of **its own plane only**
(`self.f[i][idx]`), so the two halves are fully decoupled: *any* order of *any* partition
works. Consequence: this build does not reproduce the upstream A-A hazard the anchor is
named after (read-old/write-new on dual slots, where order genuinely matters). The
alternation is load bearing only in the weaker sense that losing it also drops a half from
streaming (`parity_blind_step` :284-299, streams one half twice). Independently confirmed:
`max|f|` difference between `kernel_order=("even","odd")` and `("odd","even")` after 20
steps is exactly 0 — because it *must* be. So the capability delivered is truthfully
"single-field in-place streaming + phase/parity gate", not "A-A dual-slot pattern".

**F3 — the 15 % gate cannot catch parity loss on a smooth field (design, disclosed).**
Half-window read on the smooth TG field is 3.29–3.33 %, i.e. it *passes* a 15 % tolerance;
mass/momentum drift also stays at 1e-13 / 1e-16 (conservation is blind to it). The
discriminating evidence in the bundle is the defect arm (147.6 / 148.1 / 165.0 % field L2
and γ_K error turning into +17.3 / +32.8 % at N=128). The report states this in ⑤ —
correct practice, but it means the acceptance criterion that actually carries the claim is
the injected-defect arm, not the tolerance.

**F4 — no convergence order (blocks L4's "error analysis" completeness).** At fixed Re=100
the field L2 goes 0.3044 % (N=64) → 0.3003 % (N=128): an error floor, not a rate. A
two-point comparison at one Re cannot establish an order of accuracy; the γ_K error is
likewise flat (−0.06 % … −0.14 %). Claim warranted: "agrees with the closed form to <0.4 %";
claim not warranted: "converged / resolved".

**F5 — fit window undocumented (minor, reproducibility of a derived number).** `analyse()`
:291 fits a window that is not stated in the doc or the report; the audit's full-history fit
gives −0.0756 / −0.1014 / −0.1353 % instead of −0.0593 / −0.0749 / −0.0715 %. Both are far
inside tolerance, but the reported γ_K error should state the fit window so it is
reproducible to the printed 4th digit.

Non-blocking positives: artefacts are git-tracked (durable, not gitignored); the probe is
proven non-perturbing (`test_probe_does_not_perturb_the_run`, and the 1.8e-12 rerun delta
confirms it end-to-end); the gate really raises (`AAPatternParityError`, 1 hit in
`result.json["criteria"]["parity_gate_refuses_half_state"]`); the initial state is not
aliased (`test_stepper_does_not_alias_the_caller_buffer`).

---

## 6. Level judgement

| level | requirement | verdict |
|---|---|---|
| L0 documentation | — | exceeded |
| L1 mathematical/unit verification | tests exist and pass | **met** (15 tests: bitwise AB equivalence, mass/momentum at 1e-13, gate refusal, defect-arm detectability, TG decay) |
| L2 independent functional implementation | real code, input → compute → output | **met** — `AAPatternStepper(f0, tau).step()` → complete state → `macroscopic()`; reproducible artefacts; independently reconstructed here |
| L3 system integration | wired into main solver / platform | **NOT met** — `grep -rn aa_pattern --include=*.py` outside the three delivered files = **0 hits**; `src/tensorlbm/__init__.py` mentions `aa_pattern` **0** times (not exported); `src/tensorlbm/solver.py` has no pattern/AA hook; the `TensorLBM/` sub-package does not reference it; branch not merged into main. The module is a satellite that *consumes* `collide_bgk`/`stream` (`aa_pattern.py:73`) but is not callable from any solver path |
| L4 benchmark verification | complete case + error analysis + reproducible, judged by a registered referee | **NOT met** — 0 registered referee rows (§0); no convergence order (F4); the scheme's *ordering* hazard is not the upstream one (F2); upstream source unverifiable (self-declared) |
| L5 research-engineering result | — | not claimed and not supported |

**Verdict: L2.** The implementation is genuine, deterministic, independently reproducible,
and its central claim (single-field in-place streaming ≡ two-field AB streaming, bitwise) is
confirmed by this audit from scratch. It stops at L2 because it is integrated nowhere and
has no registered referee. Scope-honest, no inflated claims found — the two limitations a
hostile reader would raise were already disclosed by the author.

---

## 7. Checks performed / not run

Performed: registry query; byte-copy + sha256 of impl/entry/tests/artefacts/figures;
fresh execution of the documented repro command with field-by-field diff against the
delivered `result.json`; pytest; from-scratch reimplementation of BGK + AB + in-place
single-field streaming with γ_K / field-L2 / bitwise / half-window metrics; integration grep
across the mission tree; report six-section and number-provenance check.

Not run (stated, not passed): upstream walberla `aa-pattern-e/o` source comparison
(no network egress in this sandbox — same limitation the author declares); grid-refinement
study at more than two resolutions; non-periodic boundaries (out of claimed scope);
performance/bandwidth measurement of the AA storage claim (the `nbytes_persistent` figure in
`result.json["tracker_budget"]` is structural arithmetic, not measured).

Reproduce this audit:

```bash
cd /root/tp-agent/work/m17430
OMP_NUM_THREADS=1 python3 artifacts/audit_m17430/independent_recompute.py --steps 1000
# delivered-side re-run (throw-away tree, do not write into m17399):
#   cp -a <m17399>/{src,tests,benchmarks,pyproject.toml} /tmp/audit_m17430/repo
#   cd /tmp/audit_m17430/repo && PYTHONPATH=src python3 benchmarks/xval_aa_pattern_even_odd.py
#   OMP_NUM_THREADS=2 python3 -m pytest tests/test_aa_pattern_parity.py -q   # 15 passed
```