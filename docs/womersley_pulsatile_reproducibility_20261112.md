# Womersley pulsatile verification: stranded package, missing artifacts, now reproducible

Date 2026-11-12 · repo TensorLBM · ledger anchor `CASE-LBM-WOMERSLEY-PULSATILE` (was: open)
Artifacts `artifacts/womersley_pulsatile_alpha5/` · tests `tests/test_womersley_pulsatile.py`

## 0. Summary

The `CASE-LBM-WOMERSLEY-PULSATILE` deliverable (2D channel Sexl + 3D pipe Womersley with
BFL curved-wall bounce-back) was committed into a **stale duplicate source tree**
(`TensorLBM/src/tensorlbm/...`), its test into `TensorLBM/tests/`, and its report declared
"every number is read from `artifacts/womersley_pulsatile_alpha5/*.json` (no hand-typed values)"
for artifacts that **were never committed**, with a builder script
(`docs/benchmarks/build_womersley_pulsatile_report.py`) that **did not exist**. The capability was
therefore neither importable from the shipped package nor reproducible, and its 10 tests had never
been collected by the root suite (916 files).

This ticket relocates the package and test to the canonical paths, restores the missing
pieces, and re-derives every number in the report from a fresh run.

## 1. Defects found (all verified by execution, not by reading)

| id | defect | how it shows |
|---|---|---|
| D1 | verification package tracked under `TensorLBM/src/tensorlbm/verification/womersley_pulsatile/`, not `src/tensorlbm/...` | `PYTHONPATH=src python3 -c "import tensorlbm.verification.womersley_pulsatile"` -> `ModuleNotFoundError` |
| D2 | test file tracked under `TensorLBM/tests/` | root `pytest tests` (916 files) never collected it |
| D3 | `benchmarks/verified/womersley_pulsatile/run.py` did `sys.path.insert(0, REPO_ROOT/"TensorLBM"/"src")` | every run silently used the stale kernel copy, so the numbers could drift from the shipped `tensorlbm.d2q9`/`d3q19` without anybody noticing |
| D4 | report pointed at `artifacts/womersley_pulsatile_alpha5/*.json`; the directory did not exist in the repo | zero-hand-copy claim unverifiable |
| D5 | `docs/benchmarks/build_womersley_pulsatile_report.py` (the declared reproduction entry point) did not exist | report not regenerable |
| D6 | `pipe3d.run_steady()` result dict had no `finite` key although `test_pipe_steady_poiseuille_wall_treatment` asserts `r["finite"]` | `KeyError: 'finite'` on the first honest run of the relocated test |
| D7 | shipped `tensorlbm.d2q9` no longer exports `stream` (removed in the kernel refactor), but `channel2d.py` imported it | `ImportError` once the package moved into the real package path |

## 2. Fixes

1. `git mv TensorLBM/src/tensorlbm/verification/womersley_pulsatile src/tensorlbm/verification/womersley_pulsatile`
   and `git mv TensorLBM/tests/test_womersley_pulsatile.py tests/` — pure relocation, history kept.
2. `channel2d.py`: carries its own five-line periodic `stream()` kernel (equivalent to the removed
   `d2q9.stream`, driven off the library's own `C` table via `_c_on`), and the lattice tables are now
   imported once at module scope instead of a function-local import of removed entries.
3. `pipe3d.py::run_steady()` now reports `"finite"` (the invariant the test asks for), so the wall
   treatment audit fails loudly on NaN instead of raising `KeyError`.
4. `run.py`: `sys.path` now points at the shipped `src/`, so the benchmark and the test suite exercise
   the same kernels.
5. New `docs/benchmarks/build_womersley_pulsatile_report.py`: reads the artifact JSONs and rewrites
   the report tables **and** three matplotlib figures (centreline waveform, whole-profile complex
   phasor, error vs resolution). Figures are also emitted base64 for platform reports.

## 3. Verification

* `PYTHONPATH=src python3 -m pytest tests/test_womersley_pulsatile.py -q` -> **10 passed**
  (analytic reference values, quasi-steady limits, phasor estimator on synthetic signals, gate
  semantics, channel and pipe smoke runs, steady pipe wall audit).
* Fresh scan into `artifacts/womersley_pulsatile_alpha5/` reproduces the previously published
  alpha=5 channel point **digit for digit**
  (`amp 0.0386 %`, `phase 0.0871 %`, `profile L2 0.130 %`), i.e. the relocation and the local
  stream kernel are numerically identical to the stranded implementation.
* Regression value of the mass-drift assertion: when the local `stream()` was first written with the
  x/y components of the lattice table transposed, the channel smoke case drifted
  `1.41e-3 %` instead of `1.68e-4 %` and `test_channel_smoke_alpha5` failed. The assertion is
  therefore a genuine directional-streaming guard, not decoration.

## 4. Remaining, out of scope here

The duplicate tree `TensorLBM/` still holds ~165 other tracked stale files (`src/`, `tests/`,
`docs/`, `artifacts/` of an older snapshot). Deleting or re-homing them is a repository-hygiene
ticket of its own; this ticket only removed the files it owned, so no other mission's deliverables
were touched. Until that cleanup happens, any agent writing to `TensorLBM/...` paths will keep
creating invisible capabilities — a lint/CI guard (fail if a second `tensorlbm` package exists
outside `src/`) is the suggested follow-up.