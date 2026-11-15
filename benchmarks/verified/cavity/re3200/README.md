# Lid-driven square cavity, Re = 3200 — anchor `REPRO-LBM-HOU1995`

Reproduction of the Re = 3200 steady cavity that Hou, Zou, Chen & Doolen report in
*Simulation of cavity flow by the lattice Boltzmann method*, J. Comput. Phys.
**118** (1995) 329–347, against the Re = 3200 column of Ghia, Ghia & Shin (1982)
Table I/II.

This directory is the **executable + scored** half of the reproduction. The
previous round shipped only the scorer, so every grid came out `WITHHELD` — there
was nothing to score. What is here now:

| file | role |
|------|------|
| `run.py` | **entry point** — drives the library solver, writes `out/re3200_n<n>.json` + sha256 sidecar + `out/run_manifest.json` (`repro_cmd`, environment, reference hash, artefact hashes) |
| `benchmark.py` | **ref vs got bundle** — `benchmark.json` / `benchmark.csv` / `benchmark.md` with per-row `[coord, ref, got, err%]`, tolerance, verdict |
| `analyze.py` | coordinate-convention analysis + figures (`figures/re3200_centerlines.png`, `figures/re3200_error.png`) |
| `reference/ghia_re3200_reference.json` | admitted Ghia (1982) Re = 3200 column, one OCR cell quarantined |
| `out/` | run records (`re3200_n128.json`, …), manifest, per-file sha256 |

Implementation lives in the library, not in the scripts:
`TensorLBM/src/tensorlbm/verification/cavity_re3200/` (`runner.py`, `scoring.py`,
`reference.py`, `__main__.py`), plus a documented re-export shim at
`src/tensorlbm/verification/cavity_re3200/__init__.py` because this repository
carries two `tensorlbm` trees (pytest binds the `src/` one, the cavity solver
lives in the `TensorLBM/src` one).

## Case definition (programmatic, no hand-typed numbers)

| quantity | value |
|---|---|
| domain | unit square, `n × n` lattice, `L = n - 1` |
| boundary | half-way bounce-back walls, moving lid (momentum transfer) |
| `U_lid` | 0.1 (lattice) |
| `Re` | `U_lid · L / ν = 3200` |
| `ν(n=256)` | `0.1 · 255 / 3200 = 7.96875e-3` |
| `τ(n=256)` | `3ν + 1/2 = 0.52391` (regularised BGK / RLBM, Latt & Chopard 2006) |
| coordinates | `wall = (j-0.5)/(n-2)` (half-way walls at 0 and 1) and legacy `node = j/(n-1)` |
| tolerance | **5.0 %** of the lid speed, pointwise (pre-registered in `scoring.TOL_PCT`) |

Reference admission: the OCR cell `y/L = 0.4531` lost its decimal point
(`-0.86636` for `-0.06636`); it is bracketed by `u(0.5) = -0.04272` and
`u(0.2813) = -0.24427`, so it is *quarantined* — exported with its admissible
interval, excluded from scoring, and the verdict is shown not to depend on it
(`quarantine_sensitivity` in `benchmark.json`).

## Reproduce

```bash
# one grid (fast smoke)
python benchmarks/verified/cavity/re3200/run.py --n 128 --steps 2000 --verbose

# the production grid set
python benchmarks/verified/cavity/re3200/run.py --n 128 192 256 \
    --steps 150000 --monitor-every 5000 --resid-tol 2e-06 --save-fields
python benchmarks/verified/cavity/re3200/benchmark.py     # ref vs got bundle
python benchmarks/verified/cavity/re3200/analyze.py       # figures

# tests (contract + physics invariants, no long runs)
python -m pytest tests/test_cavity_re3200_runner.py -q
```

`run.py` exits `0` when every grid produced a finite solution and `2` when one
diverged / hit a config error; a *score* failure does not change the exit code —
it is recorded in the bundle, so "ran" and "passed" stay distinguishable.

## Current result and what is NOT yet demonstrated

The three production grids (`n = 128/192/256`, 150 000 steps, `resid_tol = 2e-6`)
were submitted to the cluster as jobs **cj3716 / cj3717 / cj3718** and were still
queued when this round closed (the cluster had ~85 jobs in flight).

What *is* committed at the time of writing is a short single-grid record produced
by the same entry point on the 8-core local budget (`n = 128`, 6 000 steps, no
`resid_tol`):

| n | steps | converged | `max|err|` vs Ghia Re=3200 (wall) | pass (5 %) |
|---|-------|-----------|-----------------------------------|------------|
| 128 | 6 000 | **false** | 96.44 % | FAIL |
| 192 | — | — | `WITHHELD` (record not yet produced) | — |
| 256 | — | — | `WITHHELD` (record not yet produced) | — |

That 96 % is a **transient, not a physics result**: 6 000 steps is roughly a
twelfth of the O(10^5) steps the Re = 3200 steady state needs, and the record says
`converged: false` precisely so the number cannot be quoted as a steady solution.
**The 5 % acceptance criterion has therefore not been met in this round** — and it
is not claimed. What is now in place is the missing machinery that makes the claim
testable: a real runner on the library solver, artefacts with hashes, a
`ref vs got` bundle, and tests that pin the scoring contract.

Known limits, stated plainly:

* `n = 256` at `τ ≈ 0.524` is close to the stability limit; RLBM + a 1 000-step
  lid ramp is used to survive the start-up acoustic transient.
* Convergence of a cold start is expensive (`≈ 0.035 s/step` at `n = 256` on four
  threads ⇒ ≈ 1.3 h for 130 000 steps), which is why `--init-from` can prolongate a
  converged coarse field onto the finer grid.
* `benchmark.py --check` exits non-zero unless the finest scored grid passes, so a
  CI job cannot silently absorb a regression.