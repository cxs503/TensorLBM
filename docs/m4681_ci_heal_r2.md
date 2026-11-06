# m4681 — PR #673 (branch `agent/m4644`) CI heal, round 2

Status: **gate-green locally, CI verdict still pending** — the two CI runs on
`48ff6ea8` were cancelled by infrastructure before any job produced a log, so no
green/red conclusion was ever recorded for this commit (see §1). What this note
proves is that every gate the workflow runs now passes on a clean checkout of the
branch, and that the two real defects found while doing so are fixed.

## 1. What the red actually was (evidence, not narrative)

`GET /repos/cxs503/TensorLBM/commits/48ff6ea8/check-runs` returns 14 check runs =
2 workflow runs × 7 jobs:

| run | event | ref | created | all jobs conclude | jobs |
|---|---|---|---|---|---|
| 42146309851 | `push` | `refs/heads/agent/m4644` | 2026-11-06T01:43:40Z | **cancelled** 02:04:37Z | lint, test (3.11), test (3.12), package-build, architecture-audit, repository-layout, platform-test |
| (PR run) | `pull_request` | `refs/pull/673/merge` | 2026-11-06T01:48:46Z | **cancelled** 02:04:35Z | same 7 |

Every job of both runs stops within the same 3-second window while the two runs
were still in progress (`lint` normally finishes in ~2 min, `test` in ~40 min —
they all end together), and the same instant also cancelled the two unrelated CI
runs of branch `agent/m4645` (01:49:00Z / 01:55:00Z). A single repository-wide
cancellation event therefore killed all in-flight runs; **no job ever reported a
step-level failure and no log exists**. Repo Actions history contains only these
runs (`total_count = 2` for the whole repository), so there is no earlier green
run to compare against either.

Consequence: the checklist in the task (`test=…, lint=…, … = cancelled`) carries
no diagnostic content. The only way to "fix the red" is to re-derive every gate
locally against the branch tree, which is what §2/§3 do.

## 2. Defects found by re-running the gates locally (both fixed here)

### D1 — `lint` / mypy core gate was genuinely red

The workflow's static gate is

```bash
mypy --follow-imports=silent src/tensorlbm/d2q9.py src/tensorlbm/d3q19.py \
     src/tensorlbm/solver.py src/tensorlbm/utils.py --ignore-missing-imports
```

On the branch tree (and on `main`, `src/` is byte-identical between the two) this
reports one error:

```
src/tensorlbm/solver.py:437: error: Returning Any from function declared to return "Tensor"  [no-any-return]
Found 1 error in 1 file (checked 4 source files)
```

Root cause: `correct_mass` ended with `return f * (target_mass / current)`. With
`--follow-imports=silent` plus torch's stub set, mypy infers the scalar
`float / Tensor` factor as `Any` (the `Tensor.__rtruediv__` overloads do not
narrow for a bare `float` left operand), so the product is `Any` and `--strict`'s
`no-any-return` fires. This is inherited from `main` commit `bfaf6d06` ("drop
redundant Tensor cast in rescale_mass — expression is already Tensor with torch
stubs"), whose premise is false for this mypy/torch combination: dropping the
cast is precisely what re-exposed `Any`.

Fix (one statement, `src/tensorlbm/solver.py`): name the factor with an explicit
`Tensor` annotation before multiplying.

```python
scale: torch.Tensor = target_mass / current
return f * scale
```

Gate after the fix: `Success: no issues found in 4 source files`.

The wider `mypy --follow-imports=normal` noise in `src/tensorlbm/backends/*` and
`src/tensorlbm/ai/*` is untouched by this: it is the *advisory* step
(`continue-on-error: true`) and is out of scope for this heal (C1 in
`docs/m4644_ci_heal_r1.md`).

### D2 — the branch still carried dangling provenance

`docs/lbm_analytic_verification_r1.md` §3/§7 and `docs/m4644_ci_heal_r1.md` §2/§5
pointed at `artifacts/m3634/lbm_analytic_convergence.{json,png}` and
`convergence.log` as *committed* files. They are not in the tree and cannot be:
the submission gate that publishes agent branches drops every path under
`artifacts/` (run-record directory). The round-1 note therefore claimed a fix that
had not happened — the same dangling-provenance defect it set out to remove.

Fix (root cause, not symptom):

* `scripts/verify_lbm_analytic_convergence.py` now writes the two **committed**
  copies to `docs/assets/` (`lbm_analytic_convergence.json`,
  `fig_lbm_analytic_convergence.png`) and mirrors a scratch copy plus stdout into
  `artifacts/m3634/` as a local run record.
* `docs/lbm_analytic_verification_r1.md` §3/§7 now cite the `docs/assets/` paths
  and state explicitly why the `artifacts/` copies are not committed.
* `docs/m4644_ci_heal_r1.md` gains §6, an errata marking the round-1 claim false.
* The two regenerated provenance files are committed with this change.

## 3. Gate-by-gate reproduction on the fixed tree

Commands copied from `.github/workflows/ci.yml`; run at repo root, Python 3.11.6,
`torch 2.7.1` (CPU), `ruff 0.15.0`, `mypy 1.18.2`.

| CI job | Command | Result |
|---|---|---|
| `lint` — ruff | `ruff check src tests examples benchmarks` | **All checks passed!** |
| `lint` — format | `ruff format --check src tests examples benchmarks` | **1453 files already formatted** |
| `lint` — mypy core | see D1 | **Success: no issues found in 4 source files** (was: 1 error) |
| `lint` — docs consistency | `python scripts/check_platform_docs_consistency.py` | `[docs-consistency] OK` |
| `repository-layout` | `python tests/test_repository_layout.py` / `tests/test_repository_report_migration.py` / `tools/check_repository_layout.py` | 12 tests OK / 6 tests OK / `Repository layout OK (5143 candidate paths)` |
| `architecture-audit` | `python tests/test_audit_import_graph.py` | 5 tests OK |
| `test (3.11/3.12)` — PR-relevant | `PYTHONPATH=src pytest -q tests/test_lbm_verification.py` | **7 passed** (29.2 s) — the four cases un-quarantined by this PR |
| `package-build` | `python -m build` | not re-run this round (round 1 recorded a successful sdist+wheel; no build input changed since except `docs/` + one function body) |

## 4. Numeric regression check of the regenerated provenance

`OMP_NUM_THREADS`-free run of `python scripts/verify_lbm_analytic_convergence.py`
on the fixed tree reproduces the R1 tables digit-for-digit:

| Case | published (R1 §3) | this run | match |
|---|---|---|---|
| A shear ν, nx = 16/32/64 | 4.465e-03 / 2.762e-04 / 1.722e-05 % | 4.465e-03 / 2.762e-04 / 1.722e-05 % | exact |
| C peak velocity, n_y = 8/12/16 | 0.9524 / 0.3367 / 0.1709 % | 0.9524 / 0.3367 / 0.1709 % | exact |
| D wall shear τ_w, n_y = 8/12/16 | 1.7588 / 1.0553 / 0.7538 % | 1.7588 / 1.0553 / 0.7538 % | exact |
| B Couette profile, n_y = 8/12 | 7.1e-13 / 2.4e-12 % | 7.112e-13 / 2.411e-12 % | exact |

So the solver-side change in D1 is behaviour-neutral (`correct_mass` computes the
same tensor; only the static type of the intermediate is named).

## 5. Caveats — what is *not* proven here

* **C1 — no green CI run was obtained by this mission.** The cancellation is
  repository-wide and instant (both runs, all jobs, plus unrelated branches), so
  it cannot be caused by the branch content and cannot be worked around from
  inside the repository. A re-run of the workflow on the healed commit is required
  for the PR to be recorded green.
* **C2 — Python 3.12 leg not reproduced locally** (3.11.6 only). The change is a
  single statement with no version-conditional behaviour.
* **C3 — mypy/torch versions.** D1 was diagnosed with `mypy 1.18.2` + `torch 2.7.1`
  stubs, which is what `pip install -e ".[dev]"` yields today (both are declared
  without upper bounds in `pyproject.toml`). The fix is version-robust by
  construction: naming the factor as `Tensor` satisfies the declared type in every
  configuration, whereas the removed expression relied on an inference that
  differs between mypy/torch releases.
* **C4 — `package-build` / `platform-test` / the whole `tests/` sweep** were not
  re-run this round; they are untouched by a `docs/` + one-statement change, and
  round 1 recorded them green on the same `src/` tree.

## 6. Files in this change set

| Path | Change |
|---|---|
| `src/tensorlbm/solver.py` | D1 fix: explicit `Tensor` annotation for the mass-correction factor (+3 comment lines) |
| `scripts/verify_lbm_analytic_convergence.py` | writes committed provenance to `docs/assets/`, scratch copy to `artifacts/` |
| `docs/assets/lbm_analytic_convergence.json` | regenerated provenance (machine-written numbers) |
| `docs/assets/fig_lbm_analytic_convergence.png` | regenerated convergence figure |
| `docs/lbm_analytic_verification_r1.md` | §3/§7 now cite the committed `docs/assets/` provenance |
| `docs/m4644_ci_heal_r1.md` | §6 errata: the round-1 "artifacts committed" claim was false |
| `docs/m4681_ci_heal_r2.md` | this note |