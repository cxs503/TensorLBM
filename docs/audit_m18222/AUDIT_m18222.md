# AUDIT m18222 — board cross-check of m18129 (TensorLBM)

**Anchor** `XVAL-LBM-PER-MARGIN-ARRAYS-POINTER-SHIFT-STREAMIN` · **object** mission 18129
(`succeeded`, finished 2026-11-13T12:14:52Z) · **auditor** mission 18222, `verdict_by=board-audit`
· **independence**: the mission work tree was **not** executed in place; everything below was re-derived
in a throw-away stage (`/tmp/audit_m18222/stage`) built from this worktree's `src/tensorlbm` plus the
**frozen** mission files (`docs/audit_m18222/evidence_snapshot/`, sha256 pinned below).

## 1. What was delivered (case introduction)

Round-1 open-source cross-validation (XVAL) of the CFD-GO/TCLB transport formulation "one population
array per margin block + moved base pointer instead of per-cell index arithmetic" against TensorLBM's
production gather transport. TCLB is an R-template + MPI/CUDA project: `src/*.Rt` cannot be built here
(`R` absent: R_available=False, nvcc=False), so the OSS side is
**source-derived** and the "OSS binary vs TensorLBM" comparison is honestly declared **WITHHELD**.

| item | value |
|---|---|
| implementation | `docs/audit_m18222/evidence_snapshot/per_margin_pointer_shift.py` (394 lines: plan builder, sliced executor, flat-offset executor, invariant checker) |
| driver | `docs/audit_m18222/evidence_snapshot/xval_tclb_pointer_shift/run_xval.py` (570 lines, `--quick` supported) |
| tests | `docs/audit_m18222/evidence_snapshot/test_xval_tclb_pointer_shift.py` (17 collected) |
| artefacts | `docs/audit_m18222/evidence_snapshot/result.json` (40.6 kB), `docs/audit_m18222/evidence_snapshot/figures/` (3 png), OSS excerpts `docs/audit_m18222/evidence_snapshot/xval_tclb_pointer_shift/oss_ref/` |
| report | tensor_reports #17028 + `docs/audit_m18222/evidence_snapshot/XVAL-LBM-PER-MARGIN-ARRAYS-POINTER-SHIFT-STREAMING.md` (§1-§6 present) |
| referee | `tensor_benchmark_cases` rows for this anchor = **0** |

## 2. Method (what the auditor actually ran)

`docs/audit_m18222/audit_run.py`, probes A1-A7:

* **A2 provenance** — re-hash every OSS file/excerpt against the pinned clone `/tmp/TCLB`.
* **A3 tests** — pytest on the frozen test file in the stage.
* **A4 mutation** — three sabotaged plans (drop wrapped strip, `flat_offset+1`, de-periodise) to measure
  whether the invariant checker *and* the bitwise test can fail.
* **A5 fresh run** — full driver end-to-end in the stage (88.1 s), diffed field-by-field against the
  shipped `result.json`.
* **A6 cost** — independent timing (warm-up + 9 reps) and a thread-count sweep (1/8/32/128).
* **A7 referee** — the machine-referee row set for the anchor (psql).

## 3. Results

### 3.1 Provenance (zero hand-transcription)

| check | result |
|---|---|
| OSS files re-hashed vs `/tmp/TCLB` | 8/8 match, mismatches 0 |
| excerpts found verbatim in clone at the recorded start line | 9/9 |
| excerpt sha256 == recorded `excerpt_sha256` | 9/9 |
| pinned revision == `.git/refs/heads/master` | True (`ded67cd768cf7e727bd078af139e3ec7895076e5`) |

Every quoted TCLB line (e.g. `move.pocket` → `Pocket$… += delta`, `src/LatticeAccess.inc.cpp.Rt:69-79`)
is byte-present in the clone. The provenance face of this deliverable is **real**.

### 3.2 Reproducibility (A5, full driver rerun 88.1 s)

| quantity | arm A/B | shipped | auditor rerun | rel. delta |
|---|---|---|---|---|
| TG2D n=32, 300 steps rel-L2 | A = B | 6.209416e-03 | 6.209416e-03 | 0.0e+00 |
| TG2D n=64, 400 steps rel-L2 | A = B | 9.150289e-04 | 9.150289e-04 | 0.0e+00 |
| Poiseuille u_max | A = B | 5.776989e-04 | 5.776989e-04 | 0.0e+00 |
| Poiseuille profile rel-L2 | A = B | 5.945090e-04 | 5.945090e-04 | 0.0e+00 |
| bitwise equal A vs B (14 cases) | — | True | True | — |

`max relative delta over all 15 physics scalars = 0.0e+00` — the physics face is **exactly
reproducible**. The only shipped/rerun field drift is `equivalence.short_axis_rejected`
(False shipped → True on rerun)
and the wall-clock numbers, see F2/F3.

### 3.3 Error judgement against the anchors (is it <3%?)

| quantity | reference | error | <15%? | <3%? |
|---|---|---|---|---|
| TG2D velocity rel-L2 (worst, n=32) | closed-form decay | 0.621% | yes | yes |
| TG2D velocity rel-L2 (n=64) | closed-form decay | 0.092% | yes | yes |
| Poiseuille u_max | analytic parabola | 0.058% | yes | yes |
| transport A vs B (14 shapes/dtypes) | production operator | 0 (bitwise) | yes | yes |
| **vs TCLB binary** | — | **WITHHELD** (no R/nvcc ⇒ OSS side not runnable) | n/a | n/a |

The `<3 %` question is answered **yes** on the analytic anchors; for the OSS *binary* it is
`WITHHELD`, which the mission states itself. **No tolerance in `tensor_benchmark_cases` exists for this
anchor (0 rows)** — the 15 % gate is self-applied.

### 3.4 Cost face and the thread-count confound (A6)

| grid | shipped A [ms] | shipped B [ms] | shipped slow | auditor slow @128 thr | auditor slow @1 thr |
|---|---|---|---|---|---|
| 2d/128 | 4.98 | 1.33 | 0.27x | 0.41x | 0.74x |
| 2d/256 | 8.99 | 70.16 | 7.80x | 7.44x | 0.27x |
| 3d/32 | 5.26 | 6.19 | 1.18x | 1.07x | nanx |
| 3d/48 | 9.12 | 155.86 | 17.10x | 17.88x | 0.18x |

Independent medians reproduce the shipped ratios at 128 threads (17.88x worst), but the
sign flips when the thread pool is small: at **1 thread** the pointer-shift arm is faster on every grid
(3d/48 `0.18x`, 2d/256 `0.27x`, 2d/128 `0.74x`).

### 3.5 Detection power (A4 — can the tests fail?)

| mutant | partition | flat_offset error | bitwise diff | caught? |
|---|---|---|---|---|
| M0 control (unmodified plan) | True | 0 | 0.0 |  (control) |
| M1 drop the wrapped x-strip blocks | False | 0 | 0.975969249859544 | invariant+bitwise |
| M2 flat_offset +1 on block #3 | True | 1 | 0.0 | invariant |
| M3 de-periodise the wrap (copy src onto itself) | True | 0 | 0.7864419624573327 | bitwise |

3/3 mutants detected ⇒ the suite has real
detection power, not just green ticks. 3D control also clean (`gather_max_abs_diff = 0.0`).

## 4. Figures

`fig_m18222_evidence.png` — left: Taylor-Green error of both arms against the 15 %/3 % lines (shipped vs
auditor rerun); right: the cost ratio as a function of torch thread count, which is where the shipped
"17x slower" claim lives.

## 5. Findings, ranked

* **F1 medium (claim overstated)** — decision text "the OSS formulation's advantage does not survive the
  port to torch" is not supported as stated: it holds only at `torch.set_num_threads(128)`; at 1 thread the
  port is **faster** on all four grids (0.18-0.74x). The measured penalty scales with thread count
  (2d/256: 1.12x@8, 3.93x@32, 9.13x@128), i.e. it is a per-block kernel-launch/thread-pool tax of the
  torch port, not a property of the per-margin formulation. D3 (do not adopt on this box) survives; the
  generalisation does not.
* **F2 medium (artifact stale vs code)** — shipped `result.json` records
  `equivalence.short_axis_rejected = False` while the delivered module raises on `(2, 8)`
  (`auditor rerun = True`): the artefact predates the `n>=3` guard. Every other
  field reproduces bit-exactly, so this is a regeneration miss, not a fake number.
* **F3 medium (report self-contradicts its artifact)** — doc §"Tests" states
  "`2 failed, 15 passed`"; the delivered test file passes **17/17** (auditor stage 7.89 s,
  rc=0; in the mission's own tree 17 passed / 6.41 s). A report line that reports failures
  that do not exist is as bad as hiding failures — it must be regenerated.
* **F4 low (guarded quantity is narrow)** — the banked "constant block offset" property is only exercised
  by the flat executor and by `check_plan_invariant`; a corrupted `flat_offset` (mutant M2) is invisible to
  `torch.equal(production, pointer_shift)`. The suite does catch it through the flat path, but the invariant
  is not what the headline bitwise test checks.
* **F5 low (durability)** — the five deliverable paths are **untracked** in the mission work tree
  (`git status`: `?? src/tensorlbm/xval/per_margin_pointer_shift.py`, `?? tests/…`, `?? benchmarks/…`,
  `?? docs/…md`, `?? figures/…`; HEAD `7d6cd8c7`, zero commits for this mission) ⇒ nothing is in any branch
  or in `/root/tensor-ecosystem`, so a push-on-accept cannot carry it. Auditor froze the bytes into
  `docs/audit_m18222/evidence_snapshot/` (this worktree) so the evidence survives.
* **F6 info** — `tensor_benchmark_cases` has **0 rows** for the anchor (and for variants: no
  `%POINTER%`/`%MARGIN%` case anywhere) ⇒ no machine referee; the gate cannot verify the <15 % claim
  independently of this audit.
* **F7 info** — doc line citations are sub/super-ranges of the hashed excerpts (e.g. `Global.h.Rt:64-68`
  vs excerpt `58-74`), still pointing at the claimed code; not auto-generated, not wrong.

## 6. Level decision

**L2** (independent functional implementation). Met: real code (input → plan → two executors → verified
output), non-placeholder body, runnable entry, reproducible run artefacts, and the operator is bitwise
identical to the production transport plus 0.62 %/0.058 % from closed-form physics. Not met: **L3** — no
production caller (`solver.stream` never calls the module; the mission itself decides *not* to adopt it,
D3=False) ⇒ by the ladder L4 (needs L3) is not reachable either, even though the benchmark face
(complete cases + error analysis + `repro_cmd` + independent rerun) is substantively present. **L5** not
claimed: formulation port + verification, no new science, OSS binary and GPU paths WITHHELD.
