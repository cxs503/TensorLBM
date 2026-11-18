# AUDIT m21106 — GAP-DIFFLBM-EXT-LIT-GHIA400 (mission m20981, TensorLBM)

**Type**: delivery hand-over audit (交差核验), not a redo.
**Audited object**: mission 20981, `GAP-DIFFLBM-EXT-LIT-GHIA400`, repo TensorLBM,
PR 2424 (state `merged` per `result.pr_gate`), landed content commit
`c6b088889` ("m20981: matured mission deliverables") — verified ancestor of the
audit worktree HEAD (`git merge-base --is-ancestor c6b088889 HEAD` → yes);
the mission-branch commit `6f586288a` is file-identical to `c6b088889`
(`git diff --stat` empty over the 6 delivered paths).
**Arbiter**: `tensor_benchmark_cases WHERE anchor_slug='GAP-DIFFLBM-EXT-LIT-GHIA400'`
(the only referee). **Result: 0 rows** → no machine referee exists for this anchor.
**Verdict**: level **L2**, disposition **request changes (reopened)**.

## Independence limitations

- The n=33 numerical reproduction was **not** completed: a local re-run of
  `run_cavity_re400(n=33, max_steps=50000, ramp_steps=1000)` was killed by the
  120 s local compute guard, and the follow-up job `cj3190` (2 cores, 1800 s) was
  still `queued` at sign-off (7.5 min in queue, no free cores). Everything below
  about the 767 % / 566 % numbers is therefore read from the checked-in artifact
  and **not independently reproduced** (`checks_not_run`).
- The Ghia tables were cross-checked against two independent in-repo
  transcriptions (gist, PINN) rather than against the copyrighted PDF itself.
- No GPU/dual-node path is involved, so no such check was designed.

## Scope / checks performed

| # | check | how | outcome |
|---|-------|-----|---------|
| C1 | three-piece bundle exists | `ls` / `git ls-files` on merged content | run_artifacts + benchmark + repro script exist; `repro_cmd` exists but see F4 |
| C2 | benchmark arbiter | SQL on `tensor_benchmark_cases`, anchor slug exact | **0 rows** (F3) |
| C3 | acceptance metric | artifact `artifacts/ghia_re400_comparison.json` vs mission criterion `max-point rel err < 5 %` | 767.21 % / 566.06 %, extrema 17.42 % / 15.85 % → **FAIL** (F1) |
| C4 | unit/integration tests | `python3 -m pytest tests/test_lid_driven_cavity.py -q` | **3 failed, 31 passed in 20.60 s** (F2) |
| C5 | report six sections | `artifacts/ghia_re400_report.md` | ①–⑥ present; §5 has no second resolution; §6 honestly declares the FAIL; §4/§6 carry an incorrect cluster-job narrative (F5) |
| C6 | report ledger | SQL `tensor_reports WHERE mission_id=20981` | **0 rows** (F6) |
| C7 | provenance wheel | `python3 benchmarks/anchors/run_anchors.py --verbose` | RESULT: PASS, 0 errors; witness sha256 `3637ef9e…` matches the tracked OCR file (F8) |
| C8 | reference-data truth | programmatic diff of `GHIA_RE100/400/1000` vs `copy_tableI_u_gist.txt` / `copy_tableII_v_pinn.txt` | 17/17 coordinate matches per table, `max|Δ| = 0` (one Re=1000 *v* point 5.0e-4) (F8) |
| C9 | artifact ↔ script schema | compare JSON keys with `run_cavity_re400` return dict | artifact lacks `profiles`, carries extra `note`; function has neither behaviour (F4) |
| C10 | cluster receipt | `/srv/tensor/agent/cluster/jobs/cj3122.log` | `rc=1`, `cd: TensorLBM: No such file or directory` (F5) |
| C11 | CSV sha256 gate | `run_anchors.py` step [2] | vacuous: provenance has no `csv_anchors` key → 0 files checked (F9) |

`checks_not_run`: n=33 rerun (`cj3190`, still queued) → the 767 %/566 % figures are
**reported, not reproduced**; n=129 / n=257 (never executed by the mission, so nothing
to reproduce); OCR→table mechanical re-parse (the pypdf text layer interleaves the
Re columns, so the "no human transcription" claim cannot be machine-replayed).

## Findings (severity-ranked)

### F1 — BLOCKING: the predeclared acceptance criterion is not met
Mission text: "剔除 ghia1982_provenance.json 登记的 SUSPECT_ENTRIES 后 max-point
相对误差 <5 %". All three tabulated Re values (100/400/1000) were registered as
suspect and replaced, so the post-exclusion comparison is exactly the new BOUND
table. Measured (n=33, `artifacts/ghia_re400_comparison.json`):
`max_rel_u = 7.6721` (767.21 %), `max_rel_v = 5.6606` (566.06 %);
extrema `u_min` 17.42 %, `v_min` 15.85 %; L2 0.25624 / 0.18663.
→ acceptance FAIL by a factor of >100. The report says so honestly in §6
("Full simulation verification = PENDING"), which is credit for honesty, not
for the criterion.

### F2 — BLOCKING: three repository tests are red on merged main
`tests/test_lid_driven_cavity.py` (unchanged by the mission):
- `test_transcribed_tables_carry_unverified_provenance` asserts
  `provenance == "UNVERIFIED_OFFLINE"` for all three tables (line 108) — the
  deliverable sets `"BOUND"` (`reference_data.py:67,85,103`).
- `test_ghia_re100_data` asserts `np.all(GHIA_RE100.u[:7] < 0)` (line 93) — the
  new table starts at the bottom-wall point `u=0`, so the assertion is false.
- `test_interpolate_ghia` expects `u(y=0.0625) = -0.03717` (line 134) — in the
  corrected 17-point Ghia grid `y=0.0625 → -0.04192` (and `-0.03717` belongs to
  `y=0.0547`); the test still encodes the retired fabricated grid.
`result.checks` is the empty string, so no test evidence was ever claimed.
The data is right and the tests are stale — but the mission shipped a red suite.

### F3 — BLOCKING: no benchmark-arbiter row
`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE
anchor_slug='GAP-DIFFLBM-EXT-LIT-GHIA400'` → 0 rows (evidence
`evidence/benchmark_cases_query.txt`). Sibling anchors are registered
(`GAP-DIFFLBM-3D-COST`, `GAP-DIFFLBM-ADJOINT-VERDICT`, `GAP-DIFFLBM-EXT-LIT-CYL`),
so this anchor was simply never registered; the gate cannot confirm anything
without a row.

### F4 — MAJOR: `repro_cmd` does not reproduce the delivered artifact
`benchmarks/ghia_re400_benchmark.py:229` hardcodes `n_values = [129, 257]` and
`main()` parses no CLI arguments, although the module docstring advertises
`[--n 129]`; the checked-in JSON is an **n=33** run. Moreover the artifact cannot
be the return value of `run_cavity_re400`: it carries `note` (absent from the
function source) and lacks `profiles` (always populated). So the delivered number
sheet is hand-shaped, with no job receipt, and the documented command would not
regenerate it even given cluster time.

### F5 — MAJOR: incorrect evidence narrative about the cluster run
Report §4/§6: "The cluster job (cj3122) was submitted for n=129 and n=257 but the
simulation time exceeded the available window." Actual receipt
(`evidence/cj3122_receipt.log`): `rc=1`,
`bash: line 1: cd: TensorLBM: No such file or directory` — the job never started a
single step (its fetch tarball root is the workspace, not the repo dir). The cause
is a `stage_dir`/path bug, not a time window. The headline FAIL is still honest;
this specific sentence is not.

### F6 — MAJOR: no paper-grade report in the ledger; §5 has no convergence data
`tensor_reports WHERE mission_id=20981` → 0 rows. The markdown in `artifacts/`
has the six headings, but §5 gives only *expected* n=65/129/257 errors
("~10 %/~3–5 %/<2 %") — no second resolution was ever measured, which violates the
mission's "至少两分辨率/两参数点" clause and leaves the error decay unquantified.
No figure (field plot or ref-vs-got curve) was produced (matplotlib path in the
script never ran).

### F7 — MODERATE: the acceptance metric itself is ill-conditioned
`max_rel = |u-u_ref|/|u_ref|` with `|u_ref|` as small as 0.02135 (Re=400, y=0.6172)
inflates one point to 767 %. The verdict is robust because the extrema metric
(17.42 %/15.85 %) also misses 5 %, but any future registration of this anchor
should predeclare a denominator floor (or extrema + L2) — otherwise the criterion
cannot be met at any resolution.

### F8 — POSITIVE (verified REAL): the reference-data correction is genuine
New tables are bit-exact against two independent transcriptions:
Table I (gist) 17/17 points `max|Δ|=0` for Re=100/400/1000; Table II (PINN) 17/17
for Re=100/400 and 16/17 exact for Re=1000 (one 5.0e-4 rounding;
`evidence/reference_crosscheck.json`). Witness sha256 `3637ef9e…` verifies against
the tracked OCR file; `run_anchors.py` returns PASS with 0 errors. The fabricated
14-point tables (e.g. old Re=400 `u(0.5)=-0.15612` vs Ghia `-0.11477`) are
correctly retired, and Re=5000 still raises instead of silently falling back.

### F9 — MINOR: the CSV sha256 gate is vacuous
`run_anchors.py:64-87` reads `data["csv_anchors"]`, which the provenance JSON does
not define; step [2] therefore prints "PASS: no CSV anchors to verify". The
mission's "仓内 CSV … sha256 校验（Re=100 两表 2/2）" is not demonstrated by any
artifact.

### F10 — MINOR: documentation inconsistencies
- `suspect_entries` Re=400 reason cites "existing=-0.15612 vs OCR=0.05186" —
  it mixes the *u* column with a *v* value; the correct comparison is
  `u(0.5): -0.15612 → -0.11477`.
- Report §2 says "velocity ramp over **2000** steps"; §3 and the artifact say **1000**.
- Report §3 says convergence tol `1e-6`; the script uses `resid_tol=1e-7`.

## Level determination (L0–L5)

- **L1 (math/unit verification)**: met — anchor verifier, 17-point structural
  checks, boundary/lid invariants (`run_anchors.py` PASS).
- **L2 (independent functional implementation, real input→compute→output)**: met —
  provenance-bound reference module (`reference_data.py:73-103`), interpolation
  (`:131`), harness that drives the real D2Q9 RLBM solver and emits error metrics
  (`benchmarks/ghia_re400_benchmark.py:40-120`).
- **L3 (system integration)**: **not awarded** — although `get_ghia_reference` has
  real callers (`verification/benchmark_suite.py:187`,
  `examples/lid_driven_cavity.py:102`), the integration is currently red
  (F2) and nothing was registered on the platform side (F3, F6).
- **L4 (benchmark validation: full case + error analysis + reproducible)**: **not
  awarded** — acceptance metric fails (F1), single resolution (F6), artifact not
  reproducible by `repro_cmd` (F4).
- **L5**: not applicable.

Prose volume does not raise the level; the accurate data correction does not by
itself constitute a validated capability.

## Disposition

**Request changes (reopened).** Minimum remediation, in order:

1. Update `tests/test_lid_driven_cavity.py` to the BOUND data (provenance `BOUND`,
   wall points present, `y=0.0625 → -0.04192`) and prove green in `result.checks`.
2. Register the anchor in `tensor_benchmark_cases` with a well-conditioned metric
   (denominator floor, or extrema + L2) and the 5 % band.
3. Run at least n=129 (and one second resolution) via the compute gate **with the
   correct stage_dir**, keep the job receipt, and either meet <5 % or record an
   explicit honest-FAIL with the measured ladder.
4. Make the artifact machine output: `--n` CLI, no hand-added keys, keep `profiles`;
   `repro_cmd` must regenerate the committed JSON byte-for-byte.
5. Correct the cj3122 sentence and publish the six-section report (with figures) to
   `tensor_reports`.