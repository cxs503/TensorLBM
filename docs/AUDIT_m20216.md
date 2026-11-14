# AUDIT m20216 — GAP-DIFFLBM-SOLID-LIMIT (TensorLBM)

Board-audit verdict for mission **m20216** (`succeeded`), anchor
**GAP-DIFFLBM-SOLID-LIMIT** (“volumetric solid limit vs hard wall: vanishing-
penalty convergence and hard-wall drag consistency”).  This is a cross-check
(交差核验) ticket, not a rebuild: nothing in the audited tree was modified.

| item | value |
|---|---|
| audited commit | `4ed08062e84f679cefc106aa2f96a54c75b93348` |
| commit date | 2026-11-14 12:08:39 +0000 |
| branches containing it | agent/m20216 |
| ancestor of mainline HEAD (b9939a426) | **no** |
| `wall_closure` occurrences in mainline `src/tensorlbm/cylinder_st2d1.py` | 0 |
| evidence bundle | `artifacts/m20216_audit/evidence.json` (+ `MANIFEST.json`, sha256 per file) |
| audited snapshot | `artifacts/m20216_audit/audited_commit/` (verbatim copy of the audited files) |

## 1. Scope, independence and limits of this audit

* Read-only audit of the m20216 work tree `/root/tp-agent/work/m20216` and of the
  platform DB tables `tensor_benchmark_cases` / `tensor_dev_missions`.
* The auditor is a different mission (m20388) and did not author the audited code,
  so the review is independent of the authoring session.  It is **not** independent
  of the same platform, toolchain and reference data: the only baseline available is
  the repository's own pinned `benchmarks/verified/cylinder/re20_st/result.json`
  (`ref_cd = 5.57953523384`, Schäfer–Turek DFG 2D-1, Re = 20).
* **Not run:** an independent re-simulation of the delivered arm.  The delivered run
  took 123 s; the audit shell is capped at 120 s and the rerun was killed
  (`not_run`).  No independent reproduction of the delivered
  `Cd` was obtained — this is recorded as *not run*, not as passed.
* All other checks below were executed; commands are in
  `artifacts/m20216_audit/build_evidence.py` and reproduce `evidence.json`.

## 2. What the delivery actually contains

| element | state | pointer |
|---|---|---|
| implementation | real, non-placeholder (366 lines: VOF reconstruction, partial /
  Bouzidi bounce-back, two momentum-exchange estimators) | `src/tensorlbm/volume_solid.py:95,149,165,189,213,233,254,305` |
| entry points | `run_case.py` CLI (`--arm/--D/--scale/--max-steps`), `campaign.py --index` |
  `cases/GAP-DIFFLBM-SOLID-LIMIT/run_case.py:454`, `campaign.py:43` |
| unit + integration tests | **15 passed** in this audit |
  `tests/test_volume_solid.py` (12), `tests/test_volume_solid_solver_integration.py` (3); `15 passed, 1 warning in 65.92s (0:01:05)` |
| solver-side dispatch | implemented **on the audited commit only** |
  `src/tensorlbm/cylinder_st2d1.py:178,259,312-317` (`wall_closure=`) |
| run artifacts | 1 record of a 14-combo campaign, and that one **off-grid** |
  `artifacts/GAP-DIFFLBM-SOLID-LIMIT/runs/amplitude_D16_s1.json` |
| ref-vs-got table | **empty, 0 bytes** | `artifacts/GAP-DIFFLBM-SOLID-LIMIT/table.csv` |
| platform referee | **absent** | `tensor_benchmark_cases` has 0 rows for this anchor |

## 3. Checks performed

1. **Three-piece receipt.** run_artifacts exist but are 1/14; the ref-vs-got table is
   empty; `repro.sh` is executable and its smoke path runs the tests, but it cannot
   reproduce any delivered number (§5 F4).
2. **Campaign completeness.** `campaign.py --list` declares
   14 combos over D ∈ [20, 40] ×
   3 arms × 3 mask scales.  Delivered: 1 record,
   `delivered_run_on_declared_grid = [False]`
   (D = 16 is not in the grid) →
   **14 of 14 declared combos have no**
   **record**, and the one record that exists is not one of them (`delivered_on_grid_count = 0`).
3. **Criteria replay.** Re-running the delivery's own `run_case.evaluate()` on the
   delivered artifact set gives:
   `verdict = INCOMPLETE`, `n_table_rows = 0`, `passes_all_resolutions = {"amplitude": true, "interface": true}`.
   The empty-set control gives the *identical* output
   (`{"amplitude": true, "interface": true}`), i.e. the two
   PASS lines in the report are vacuous, not measurements (F2).
4. **Benchmark referee.** `SELECT case_name, metric, tolerance FROM
   tensor_benchmark_cases WHERE anchor_slug='GAP-DIFFLBM-SOLID-LIMIT'` returns
   **0 rows**.  The three `GAP-DIFFLBM*` rows that do
   exist belong to other anchors (3D-COST, ADJOINT-VERDICT), so no tolerance could be
   machine-adjudicated here.
5. **Got vs baseline, error analysis.**

   | quantity | value |
   |---|---|
   | delivered got Cd (control-volume, arm `amplitude`, D=16, s=1.0) | 6.446459 |
   | delivered got Cd (Ladd momentum exchange) | 6.446455 |
   | baseline ref Cd (Schäfer–Turek 2D-1 Re=20) | 5.579535 |
   | error reported in artifact | +15.5376 % |
   | error recomputed by auditor | +15.5376 % (agree: True) |
   | owner bar | 15 % → **above**, margin +0.5376 pp |
   | 3 % ecosystem caliper | **not met** (error is 15.54 %) |

   Answer to the mandatory question *“is the key quantity within 3 %?”*:
   **NO — 15.54 %**, and it is also above the
   15 % owner bar.  The comparison is against a real external anchor
   (Schäfer–Turek), not self-referential.
6. **Six-section report check.** The delivered `REPORT.md` contains criteria, verdict,
   per-resolution table and raw records, but no case/physics/geometry section, no
   method section, no error-judgement section, and the per-resolution table body is
   empty (0 rows).  `docs/GAP-DIFFLBM-SOLID-LIMIT_RESULTS.md`, which does have the six
   sections, is **untracked** in the audited tree (never committed) and its results
   table body is empty as well.
7. **Lineage / landing.** The delivered commit is contained only in `agent/m20216`
   and is **not** an ancestor of the mainline HEAD; mainline
   `src/tensorlbm/cylinder_st2d1.py` contains 0 occurrences of `wall_closure`, i.e. the
   solver-side dispatch never landed.  Mainline instead carries a parallel m20071
   version of `src/tensorlbm/volume_solid.py` (301 lines, no solver dispatch).

## 4. Maturity level

**Verdict: L2** (independent functional implementation with real run artifact),
with the following ceilings:

* **L2 — met.** Real, non-placeholder algorithm (`volume_solid.py`), a real
  input→compute→output entry (`run_case.py`), real run artifact (1 json record),
  15 passed tests green in this audit, including
  invariant-style assertions (binary-limit equivalence to half-way bounce-back,
  `torch.equal` on the fluid set, mass and momentum bookkeeping to 1e-12,
  plane-wall force equals the cut-link sum).
* **L3 — code exists at the audited commit, not counted.** `run_cylinder_st2d1`
  gains `solid_fraction=`/`wall_closure=` and there is an integration test driving the
  main solver (`tests/test_volume_solid_solver_integration.py:37`), but the commit is
  unmerged and mainline has no caller (`wall_closure` count = 0).  A capability that
  exists only on an unpushed branch is not system integration.
* **Why the gate's own L3 grep is not used as evidence here.** A dry run of
  `maturity_gate.judge(3, ...)` on this evidence returns 3 because
  `callers_verified()` greps the ecosystem repo for the literal
  `src/tensorlbm/cylinder_st2d1.py`; that string is matched by audit/report scripts
  (`benchmarks/case_pool/cylinder_drag_re20/make_report.py`,
  `docs/board_audit/m17742/*.py`), not by a solver caller.  Machine-passable is not
  the same as integrated, so the level is capped at the substantively defensible one.
* **L4 — not met, five independent reasons:** (i)
  14/14 declared combos never ran and the
  single delivered record is not among them;
  (ii) the single delivered run is off the declared grid, so D1/D2 (which are
  *same-D* comparisons) are undefined; (iii) `table.csv` is empty, no ref-vs-got row
  exists; (iv) no `tensor_benchmark_cases` referee row for the anchor; (v) the one
  measured error is 15.54 % > 15 % owner bar
  and `provenance.git_commit` in the artifact is `null`, so the number is not bound to
  a commit at all.
* **L5 — not reachable** without L4.

## 5. Findings (severity-ranked)

**F1 — blocking (evidence): the study that the anchor is about was never run.**
  all 14 declared combos are unmeasured (0 on-grid
  records) and the one
  delivered run sits at D = 16, outside the declared D ∈ {20, 40}.  The anchor's
  question (“does the volumetric closure have a solid limit?”) is therefore
  unanswered by data.  Reproduce: `campaign.py --list` +
  `evaluate()` replay (`artifacts/m20216_audit/evidence.json →
  audit_checks.campaign_grid / criteria_replay`).

**F2 — blocking (correctness of the report): vacuous PASS lines.**
  `run_case.py:436` computes `all(_ok(r) for r in table if ...)`; an empty table makes
  every `all([])` true, and `summarize.py:212-213` / `publish.py:127-128` print
  “`amplitude`: PASS at every resolution” anyway.  The `verdict` field is guarded to
  `INCOMPLETE`, but the per-arm PASS lines are not — a reader of REPORT.md sees two
  PASS lines above an empty table.  Reproduce: the empty-set control in the evidence
  bundle returns `passes_all_resolutions = {amplitude: true, interface: true}` for a
  run set with **zero** records.  Suggested minimal fix: when `table` is empty emit
  `NOT_EVALUATED` / `n/a` instead of `PASS`.

**F3 — blocking (accuracy): the only delivered number misses the bar.**
  Cd = 6.446459 vs ref 5.579535 →
  +15.54 % (owner bar 15 %, ecosystem caliper 3 %).
  Independently recomputed by the auditor from the artifact and the artifact's own
  reported value agrees exactly.  The artifact cannot even attribute the error: there
  is no `hardwall` arm at D = 16, so closure error and discretisation error are
  confounded, and `provenance.git_commit = null`.

**F4 — blocking (reproducibility): three different grids, one stale command.**
  `campaign.py --list` → D ∈ [20, 40];
  `repro.sh --full` hardcodes `--resolutions 20 32`; the delivered artifact is D = 16.
  No delivered number is reproducible by the shipped `repro_cmd`, and no delivered
  number can be re-derived from the declared grid.

**F5 — major (landing): the deliverable never reached the repository.**
  Commit `4ed08062e` is contained only in `agent/m20216`; mainline HEAD does not have
  it, mainline `cylinder_st2d1.py` has no `wall_closure`, and mainline already carries
  an overlapping m20071 implementation of `volume_solid.py`.  The solver-side
  integration that would justify L3 exists only on the unmerged branch.

**F6 — major (overstated claims).**
  `cases/GAP-DIFFLBM-SOLID-LIMIT/publish.py:143` states “The reference Cd is 13 % above
  Schäfer–Turek at D = 20” — there is no D = 20 run in any artifact.  The delivered
  figure caption (`summarize.py:figure_convergence`) asserts “Discretisation error
  against the reference shrinks with D for all three arms” while the figure is drawn
  from a single D = 16 point.

**F7 — minor (diagnostic power).** In the delivered record the control-volume drag and
  the Ladd momentum-exchange drag agree to 5.95e-05 %; the
  two-estimator cross-check the harness advertises therefore has no discriminating
  power on the only measurement that exists.

**F8 — process (consistent with the delivery's own admission).** The mission result
  JSON for m20216 already records `fail_stage = run_artifacts`, “D1/D2 判据数字缺失”
  and “平台报告未实际 POST（仅 dry-run）”.  The audit confirms that admission with
  machine checks instead of trusting it — the difference being that the *report* still
  shows PASS lines and a convergence figure.

## 6. Checks not run

* Independent re-simulation of the delivered `Cd` (killed at the 120 s shell cap;
  recorded as `not_run`).
* Any D1/D2 evaluation — impossible: the required same-D arm/scale pairs do not exist.
* Tolerance adjudication against `tensor_benchmark_cases` — no row exists for the
  anchor; the external anchor used instead is the repo-pinned Schäfer–Turek value.

## 7. Disposition

**Request changes (re-opened at L2).**  What is solid: the closure mathematics, the
two estimators, the 15 green tests, the CLI harness.  What is missing is exactly the
measurement the anchor is named after.  Remediation, in order:

1. Run the declared campaign (at minimum D ∈ {20, 40} × 3 arms × s ∈ {1.0, 0.75, 0.5});
   shard it on the compute queue, 13 combos outstanding.
2. Fill `table.csv` from `evaluate()`; replace the vacuous PASS lines with
   `NOT_EVALUATED` when the table is empty (F2).
3. Commit the artifact provenance (`git_commit`, seed, grid) into every run json (F3).
4. Register the anchor as a `tensor_benchmark_cases` row (ref = 5.57953523384,
   tolerance = the owner 15 % bar, or 3 % if the D = 40 arm is claimed) so the referee
   exists (F4).
5. Make one grid authoritative across `campaign.py`, `repro.sh` and the artifact, and
   either merge the solver-side dispatch to mainline or drop the L3 claim (F4/F5).
6. Delete or qualify the D = 20 claim and the convergence-caption claim (F6).

Artifacts: `artifacts/m20216_audit/{evidence.json, MANIFEST.json, build_evidence.py,
make_figures.py, make_doc.py, figures/*.png, audited_commit/**}`; verdict row in
`tensor_maturity_verdicts` (repo `TensorLBM`, scope `m20216 GAP-DIFFLBM-SOLID-LIMIT`,
`verdict_by = board-audit`).
