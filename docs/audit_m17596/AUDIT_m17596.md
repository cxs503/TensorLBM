# Board audit — m17596 `XVAL-LBM-BLOCKFOREST-METIS-MULTISTREAM-OVERLAP`

| item | value |
|---|---|
| audited mission | `tensor_dev_missions.id = 17596` (status `succeeded`, repo `TensorLBM`) |
| audited worktree | `/root/tp-agent/work/m17596` @ commit `10d568a6` ("TensorLBM: block-forest + block-graph load balancing + multistream comm/compute overlap, cross-validated vs walberla practice"), 13 files, 1 082 insertions of source |
| audited report | `tensor_reports.id = 16590` (19 838 chars, 3 embedded figures) + `docs/XVAL_BLOCKFOREST_METIS_MULTISTREAM.md` (259 lines, six sections) |
| audit mission / worktree | `m17665`, `/root/tp-agent/work/m17665` (this document) |
| machine referee | `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-BLOCKFOREST-METIS-MULTISTREAM-OVERLAP'` → **0 rows** ⇒ no formal referee row; judgement uses the mission's own predeclared gates (`result.json:predeclared`, `run_xval.py:65`) plus independent recomputation |
| verdict | **L2** (functional implementation, real code, tests, reproducible artifacts). L3 **not** met (no production caller), therefore the L4-format evidence (benchmark + repro_cmd) cannot promote the case under `maturity_gate`'s cumulative scheme |
| disposition | **approve with notes** at L2; **request changes** for L3/L4 promotion (see §7) |

Method: adversarial re-derivation, not a re-read of the author's conclusions. Everything below was
re-executed or recomputed in this audit worktree against the audited worktree at `10d568a6`; no file
of the audited worktree was modified (only read / imported).

---

## 1. What was actually delivered (verified, not claimed)

| element | pointer | status |
|---|---|---|
| implementation | `src/tensorlbm/blockforest.py:95` `BlockForest` (1-irregular forest, level-1 refinement, face/edge neighbour split, per-block workload), `:251` `BlockGraph` (CSR `xadj/adjncy/adjwgt`), `:578` `partition_blockforest` (geometric seed → recursive bisection → `:333` `_kl_refine` → `:483` `_kway_refine`), `:693` `exact_bisection_cut` (exhaustive optimum for k=2), `:639` `slab_partition` / `:663` `weighted_slab_partition` (controls), `:768` `MultistreamBlockSim` (`:992` `step`, `:949` `_pump`) | **REAL** — 1 081 lines, 0 placeholder markers (`NotImplementedError` / `raise TODO` / bare `pass` bodies absent) |
| entry point | `benchmarks/xval_blockforest_metis_multistream/run_xval.py:634` `main()`; runs E1…E6 and writes `result.json`, `tables.md`, 5 PNGs | **REAL**, executes end-to-end |
| run artifacts | `artifacts/xval_blockforest_metis_multistream/result.json` (832 lines), `tables.md`, `fig1…fig5*.png` | present, non-empty, internally consistent with the report tables |
| tests | `tests/test_blockforest.py` (10 tests, `:37`,`:52`,`:69`,`:79`,`:89`,`:107`,`:122`,`:145`,`:158`,`:181`); **this audit re-ran: 10 passed in 13.60 s** | **REAL invariants**, not smoke tests (see §3) |
| integration | `blockforest` is imported **only** by `benchmarks/xval_blockforest_metis_multistream/run_xval.py:51` and `tests/test_blockforest.py:17`; `src/tensorlbm/__init__.py` does not export it; no solver/runner/script calls it | **inbound only** (the new stepper calls the production kernels `collide_bgk3d`, `stream3d_roll` from `src/tensorlbm/solver3d.py`); no production caller ⇒ L3 open |
| report | `docs/XVAL_BLOCKFOREST_METIS_MULTISTREAM.md` §1 problem+reference, §2 method (repo-vs-OSS difference list §6), §3 parameter table (τ=0.8, ν=0.1, k=2π/64, A=0.05, Ma=0.0866, Re≈5.09 defined per amplitude/2π, α grid), §4 ref-vs-got table with ticks/crosses, §5 sensitivity (block size, part count, α) + limits, §6 failure modes/frictions | six elements **complete**; figures present |

## 2. Machine referee check (the only judge)

`tensor_benchmark_cases` has **no row** for this anchor (0 rows, also 0 rows for `ILIKE '%blockforest%'`).
The formal judge therefore degrades to the mission's predeclared gates, which this audit recomputed:

| gate | quantity | predeclared cap | shipped got | re-run by audit | verdict |
|---|---|---|---|---|---|
| D1a | cut rel. error vs **exhaustive optimum**, 6 instances | ≤15 % | 0.00 % (6/6, committed artifact) | **worst +4.23 %** from HEAD code (see F0) | **PASS** (≤15 %), but the committed 0.00 % is **not reproducible** |
| D1b | cut rel. error vs **networkx Kernighan–Lin**, 6 instances | ≤15 % | max +11.46 % (3×4×6, k=8) | identical | **PASS** |
| D1c | cut rel. error vs best-of-64 restarts, 9 instances | ≤15 % | 0.00 % (9/9) | identical | **PASS** |
| D2 | workload imbalance `max/mean−1` | ≤5 % | **22.03 % (refC k=4/8)** | 22.03 % | **FAIL** (shipped verdict `PARTIAL`, correctly not `PASS`) |
| D3 | `max|f_dec−f_mono|` | ≤1e-12 | **0.0** (bitwise) | **0.0** | **PASS** |
| D4 | shear-wave decay rate vs νk² | ≤15 % | +0.0514 % | +0.0514 % | **PASS** |
| D5 | hidden comm fraction, α≤0.5, blocks≥16³ | ≥85 % | 92.6 % | not re-run (declared ±10 % timing spread) | PASS with caveat (§5 F4) |

Reproducibility of the deterministic subset (E1/E2/E3/E4/E6, `independent_check.py --repro`):
**6 numeric diffs out of the whole tree, all of them `exact_enumeration_s` wall-clock fields**
(`docs/audit_m17596/repro_diff_m17596.json`: `n_number_diffs=6`, `all_diffs_are_wall_clock=true`) —
but this comparison was made against the **working-tree** copy of `result.json`, which a re-execution
of the driver had already overwritten (see F0). Against the **committed** artifact the same re-run
**does not** reproduce (E1 row 4), and that is the decisive reproducibility result.

## 3. Are the tests real? (invariant vs execution)

| test | invariant asserted |
|---|---|
| `:69` | partition cut ≤ exhaustive optimum × 1.15 **and** imbalance ≤ 5 % on a 2×2×2 forest |
| `:79` | graph partition beats the shipped z-slab cut, both balanced |
| `:89` | on 4×4×4 with every third block refined: graph partition balanced, count-slab **provably** not (`slab.imbalance > 0.05`), every block assigned exactly once, all parts non-empty, weight sum conserved |
| `:122` | decomposed streaming == production `collide_bgk3d`+`stream3d_roll`, `atol=1e-13`, mass `rel=1e-14` |
| `:145` | overlapped (latency 1e-3) == serialized, **`torch.equal`** (bitwise), 8 steps |
| `:158` | mass conservation `1e-13` over 10 overlapped steps + periodic ghost-plane sums (wrap check) |
| `:181` | decay rate vs analytic νk² on a 1×4×1 forest |
| `:52` | refined block workload = 8×, interface weight of a refined face = 4× |

These are invariants with independent references (exhaustive optimum, production kernel, analytic
νk²), not "runs without error". E4's `max_field_diff = 0.0` is a genuine two-implementation
comparison: the decomposed path uses hand-written per-face indexing (`_stream_core`/`_stream_shell`,
`blockforest.py:962/975`) while the monolithic reference uses `stream3d_roll` — two code paths,
bitwise identical result, including edge/corner cells.

## 4. Severity-ranked findings

### F1 — no production integration (blocking for L3/L4 promotion; blocks the level, not the merge)
`grep -rn 'blockforest' --include=*.py src/ tests/ benchmarks/` → only the module itself, its test, and
its benchmark driver. Not exported in `src/tensorlbm/__init__.py`; no simulation, script or runner starts
a `MultistreamBlockSim`. Everything runs single-process (no MPI, no rank halo exchange).
**Effect:** under `maturity_gate` `SCHEME[3]`, `integration_callers` is empty ⇒ L3 unreachable ⇒ the L4-format
evidence (`benchmark` + `repro_cmd`, both genuinely present) cannot promote the case. This is exactly the
m17013/m17591 failure mode in this repo; graded L2 here for the same reason, **despite** better evidence
quality than those cases.

### F2 — predeclared D2 gate fails, and neither side of the failure is analysed (major)
Shipped `ours_imbalance = 22.03 %` on `refC` k=4/8 (27 blocks, 13 refined) against a 5 % cap; the report
says `PARTIAL` honestly. But the report never establishes **whether 5 % is even attainable** on that
instance class. This audit computed the exact quantized floor (blocks indivisible, workload 1× coarse /
8× level-1; `independent_check.py --floor`, `imbalance_floor.json`):

| instance | k | shipped imbalance | achievable floor | cap 5 % feasible? | excess over floor |
|---|---|---|---|---|---|
| refA | 8 | 2.78 % | 0.00 % | yes | +2.78 pp |
| refB | 8 | 3.45 % | 0.00 % | yes | +3.45 pp |
| refC | 2 | 3.39 % | 0.00 % | yes | +3.39 pp |
| **refC** | **4** | **22.03 %** | **8.47 %** | **no** | **+13.56 pp** |
| **refC** | **8** | **22.03 %** | **8.47 %** | **no** | **+13.56 pp** |

Two distinct facts, neither in the report: (a) the 5 % cap is **mathematically infeasible** for
refC (13 heavy + 14 light blocks over 4–8 parts ⇒ min-max 32/29.5−1 = 8.47 %), so the gate is
mis-calibrated for that class; (b) ours is nevertheless **13.56 pp worse than the achievable floor** —
the k-way repair trades balance for cut (cut is optimal, 0.00 % vs best-of-64 in all 9 E3 instances).
So the "METIS-style load balancing" capability is demonstrated on uniform and 64-block refined forests,
and is demonstrably weak on small, strongly heterogeneous forests.

### F3 — predeclared sample vs executed sample (major, integrity not accuracy)
`result.json:predeclared.sample.E4` = *"shear wave 8x64x8, **600 steps**, 1 configuration"*, but the shipped
run executed **400** steps (`run_xval.py:646` `e4_physics(steps=150 if fast else 400)`, while the function
default at `:313` is 600). The report, line 75, calls the 400 steps "(predeclared)". Thresholds were not
tuned afterwards, and this audit reproduces the 400-step decay error exactly (+5.144e-4), so nothing is
numerically wrong — but the predeclaration text is falsified by the artifact and the deviation is not
logged, which breaks the mission's own experimental-design discipline.

### F4 — D5 "hidden fraction ≥85 %" is a pass inside the timing noise (medium)
E5 rows contain physically impossible values: `hidden_fraction = 242.4 %` (20³, α=0), 166.4 %,
124.3 %, 103.6 %, and negative exposed times (e.g. 20³ α=0.25: `exposed_overlapped_s = −11.8 ms`) —
i.e. `t_overlapped < t0` beyond the noise floor, so the metric's denominator/numerator are noise-dominated.
The 16³ α=0.5 gate row (92.6 % vs 85 % cap) has only 7.6 pp of margin inside the report's own declared
"±10 % run-to-run spread" (§5). The defensible statement is *"exposed exchange time is indistinguishable
from zero for α≤0.5 at 16³/20³ blocks"*; the precise 92.6 % figure is not supportable.

### F5 — "multistream" is emulated, not the reference mechanism (medium; honestly declared)
`MultistreamBlockSim.step` spawns a Python `threading.Thread` (`blockforest.py:949`) with
`time.sleep(latency)` as the injected network delay; `torch.cuda.is_available() == False`
(`result.json:predeclared.environment`), single process, no MPI. The report §6 states this in the
repo-vs-OSS table ("CUDA streams → queue + worker thread"), so it is not concealed — but the ported
capability is the *schedule* (push-on-collide → interior → join → boundary layer), not CUDA stream/MPI
overlap. Absolute timings are not transferable to the reference setting.

### F6 — external anchor is documentary, not verified or executed (medium)
`docs/...md:24-31` states the worktree had no network (`curl` → exit 35), no waLBerla checkout and no
METIS installation; the reference practices are cited "at module/function granularity from the public
documentation/source layout" — i.e. from model knowledge, not from source read in this run. The platform's
own source-derived inventory does contain the practice (`oss_solutions` id=365 `lssfau/walberla`
feature `lbm-gpu-mpi-overlap`: "Metis/ParMetis 按块负载迁移；每层每块绑定 CUDA stream，块级 halo 通信与计算
在不同流上重叠"), which the mission neither cited nor reconciled with.
Substitute references are legitimate for the cut criterion: within *x* % of the exhaustive optimum
**implies** within *x* % of METIS on the same instance (`cut_METIS ≥ cut_opt`) — the argument is sound.
But the headline "cross-validated **vs waLBerla**" is, strictly, "port of the documented practice +
validation against independent offline references"; no waLBerla number, binary, or source line was
compared.

### F7 — no benchmark registry row (minor, platform-level)
0 rows in `tensor_benchmark_cases` for the anchor, so the "基准裁判" is empty and the case cannot be
machine-judged later. Recommendation: register `('XVAL-LBM-BLOCKFOREST-METIS-MULTISTREAM-OVERLAP',
'cut_rel_error_vs_exhaustive_optimum_k2', 0.15)` and `(…, 'shear_wave_decay_rate_rel_error', 0.15)`.

### F8 — report linkage gap (minor, platform-level)
`tensor_dev_missions.report_id` is NULL for m17596 although `tensor_reports.id=16590` exists for
`mission_id=17596` and `report_owed=f`. Same defect class as reported for m17593.

### F9 — mandated pre-work checks degraded (informational)
`docs/...md` "Ecosystem frictions" 1: knowledge-base API returned `engine_offline` (8021 refused), so the
three mandatory pre-work queries could not be run; negative-knowledge search answered 0 hits. Honest
friction logging, no capability impact; platform-side issue.

## 5. Overstated claims (report vs evidence)

| claim in report | evidence | audit reading |
|---|---|---|
| "cross-validated vs waLBerla practice" (title, §1) | no waLBerla execution, no source read, no METIS run (§F6) | **overstated**: design-level port + independent offline references; the cut argument is sound but it is not a cross-implementation comparison |
| "D5 hidden 93–166 %" (§4, commit message) | values >100 % are noise artefacts; 7.6 pp margin inside ±10 % spread (§F4) | **overstated precision**; direction is supported, number is not |
| "600 steps … predeclared" vs "(predeclared)" 400 steps (§3 line 75) | `run_xval.py:646` runs 400; `predeclared` text says 600 (§F3) | **inconsistent predeclaration**; no numerical impact |
| "D2 is PARTIAL and it is reported as such" (§4 reading) | true, but no feasibility/floor analysis (§F2) | **incomplete** rather than false: the capability gap (13.56 pp above floor) is not surfaced |
| commit message "cross-validated vs walberla practice … D5 latency hidden 93-166 %" | as above | keep, but annotate with F4/F6 |

## 6. Checks not run / independence limitations

* Not run: real MPI multi-rank exchange; CUDA-stream overlap (no GPU on this node); a METIS comparison
  (`pymetis`/`metis` not installed, no network); mesh/two-grid convergence for the E4 shear wave;
  full E5 re-run (declared non-reproducible by construction). These are marked **not_run**, not "passed".
* Independence: this audit is performed by a peer agent inside the same ecosystem, on the same node,
  re-using the mission's own code paths. It independently re-derives references (exhaustive enumeration,
  analytic νk², quantized imbalance floor) and independently recomputes the D1/D2 numbers, but it cannot
  constitute external (third-party) review, and it cannot verify the waLBerla source claims offline.
* Machine referee: **absent** (`tensor_benchmark_cases` 0 rows) — the arithmetical judge here is the
  predeclared gate set + this audit's recomputation, stated explicitly rather than implied.

## 7. Disposition and remediation

**Disposition: approve with notes.** The delivery is a real, tested, reproducible functional
implementation with honest PARTIAL self-reporting; it is **L2** by functional substance, and the
"verified capability" framing (the XVAL anchor) is only partially earned.

Remediation to reach L3 / L4 (in priority order):

1. **Integration (L3):** make a production entry point drive `MultistreamBlockSim` (e.g. an
   overlapped/distributed solver runner or an `src/tensorlbm/__init__.py` export used by a simulation
   script), or explicitly scope the deliverable as a *library module* so L2 is the ledger level by design.
2. **D2:** publish the quantized-floor analysis (this audit's `imbalance_floor.json`), then either relax the
   cap per instance class or add a cut-constrained balance pass to close the 13.56 pp gap on refC;
   a criterion of the form `imbalance ≤ floor + 2 pp` is the correct one.
3. **Predeclaration:** align `PREDECLARED.sample.E4` with the executed step count (or run 600 steps) and log
   deviations explicitly; never label a post-hoc sample "predeclared".
4. **D5:** replace the hidden-fraction point estimate with a paired-difference estimator over ≥20 repeats and
   report a confidence interval; drop rows whose denominator is inside the noise floor (`NaN`/`>100 %` values
   must fail the gate rather than be explained away).
5. **Anchor:** cite the platform inventory (`oss_solutions` id 365) and record the offline sourcing limitation
   in `external_anchor`, so the XVAL claim is graded as documentary; add the METIS comparison once installable.
6. **Registry:** insert the two `tensor_benchmark_cases` rows (F7) and fix `tensor_dev_missions.report_id` (F8).

## 8. Verdict record

`tensor_maturity_verdicts` row (see `verdict_m17596.sql`, `verdict_by='board-audit'`):

* scope: `m17596 XVAL-LBM-BLOCKFOREST-METIS-MULTISTREAM-OVERLAP`
* level: **2**
* evidence keys: `impl_files`, `entry`, `run_artifacts`, `tests`, `integration_callers` (empty by finding),
  `benchmark` (ref vs got numbers), `repro_cmd`, `external_anchor` (documentary), plus `findings`,
  `audit_artifacts`, `gate_notes`.

Artifacts of this audit (this worktree):

* `docs/audit_m17596/AUDIT_m17596.md` (this file)
* `docs/audit_m17596/independent_check.py` (`--repro` / `--floor` / `--figure`)
* `docs/audit_m17596/repro_diff_m17596.json` (6 wall-clock diffs, everything else bit-identical)
* `docs/audit_m17596/imbalance_floor.json` (exact quantized imbalance floors)
* `docs/audit_m17596/fig_audit_m17596.png` (imbalance vs floor vs cap; cut error vs 15 % cap)
* `docs/audit_m17596/verdict_m17596.sql`

## 9. ADDENDUM (decisive finding F0) — the committed artifacts are stale w.r.t. the committed code

**F0 [blocking for the reproducibility claim, severity highest].** `run_xval.py` at HEAD (`10d568a6`,
code unmodified: `git status` shows only `artifacts/...` dirty) emits, deterministically across two
consecutive calls, `exact_cut = 4544.0`, `ours_cut = 4736.0`, `ours_rel_error = +0.04225` for the E1
instance "refined 2x3x3 (checker)", whereas the **committed** `artifacts/.../result.json`
(`git show HEAD:artifacts/xval_blockforest_metis_multistream/result.json`) records
`exact_cut = 1280.0`, `ours_cut = 1280.0`, `ours_rel_error = 0.0` for the same instance, and the
committed report table (line 117) repeats the 1280 / +0.00 % numbers.

Consequences:

1. `result.json`, `tables.md`, `fig1…fig3` in the delivered commit **cannot be regenerated from the
   delivered code** — the run recorded in the artifact (07:26:48–07:30:01Z) predates a code change made
   before the commit (07:31:50Z) and was never re-run. The stale numbers are the *more favourable* ones
   (0.00 % instead of +4.23 %), which is exactly what a machine referee must not accept silently.
2. The correct D1a statement from HEAD code: **+4.23 % worst-case cut error vs the exhaustive optimum**
   (5/6 instances exact, one refined checker forest at 4736 vs 4544), still inside the owner's 15 % cap,
   but **not** inside the 3 % band and not the "0.00 % on 6/6" claimed in the commit message and report.
3. `max_field_diff = 0.0` (D3), the E2/E3 cut numbers, the E3 imbalance numbers and the decay rate
   (+5.144e-4) **do** reproduce from HEAD code bit-identically, so the verdict (L2) is unaffected; what
   fails is the artifact-provenance property the mission's "reproducible" requirement demands.
4. Self-disclosure: this audit's own re-execution of `run_xval.py` (needed to test reproducibility)
   overwrote the working-tree artifacts (`M artifacts/xval_blockforest_metis_multistream/{result.json,
   tables.md,fig1..fig3}`). The comparison in §2/§3 is therefore stated twice — against the committed
   artifact (F0, mismatch) and against the fresh run (bit-identical, `repro_diff_m17596.json`).

Remediation (adds to §7): regenerate and commit the artifacts from HEAD code (or revert the late code
change), add a CI check that a clean re-run of `run_xval.py` diffs clean against the committed
`result.json` (gate on `exact_enumeration_s`/timing fields only), and never commit an artifact produced
by a different code revision.
