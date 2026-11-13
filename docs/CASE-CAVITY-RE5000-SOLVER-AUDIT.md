# CASE-CAVITY-RE5000 — solver boundary-condition audit (anchor blocker)

Mission 17804 · branch `agent/m17804` · repo TensorLBM · anchor `CASE-CAVITY-RE5000`
(lid-driven square cavity, Re = 5000, tolerance 8 % of the lid speed vs Ghia 1982).

**Bottom line.** The anchor cannot be scored yet, and the reason is not resolution or
step count: the in-library cavity solver `tensorlbm.verification.lid_driven_cavity.solver`
fails the boundary-condition invariants at Re = 100 on a 64² grid. It injects a spurious
wall-normal velocity on the moving lid (measured `u_y = -0.032` where 0 is required, i.e.
32 % of the lid speed) and goes non-finite within ~50 steps. Every archive number for this
case (e.g. `out_local/re5000_n256.json`, max error 39.70 %) was produced on a *different*
worktree (`m17072`) and only shows how far the current code is from the documented
behaviour. This document records the defects, the machine-checkable reproductions and the
hardened harness that will score the case the moment the solver is repaired.

## ① Case, reference and parameters

2-D lid-driven square cavity, `Re = U L / ν = 5000`, incompressible steady state.
Reference: U. Ghia, K. N. Ghia, C. T. Shin, *J. Comput. Phys.* **48** (1982) 387-411,
Table III (u along `x/L = 0.5`) / Table IV (v along `y/L = 0.5`), Re = 5000 column; the
machine-read, gate-verified copy lives in
`benchmarks/verified/cavity/re5000/reference/ghia_re5000_reference.json`
(17 + 17 rows, sha256 recorded in every artefact this harness writes).

| quantity | value |
|---|---|
| lattice / collision | D2Q9, RLBM (regularised BGK, Latt & Chopard 2006) |
| boundary | half-way bounce-back (3 stationary walls) + Zou/He velocity wall (lid) |
| `U_lid` (lattice) | 0.1 → `Ma = 0.173`, compressibility error `O(Ma²) ≈ 3 %` |
| `τ` | `3 U_lid (n-1)/Re + 0.5`: n=256 → 0.51532, n=320 → 0.51927 |
| tolerance (pre-registered) | pointwise `|ref − got| ≤ 8 % U_lid` on the tabulated rows |

## ② Method

1. **Harness hardening.** `benchmarks/verified/cavity/re5000/run.py` was regressed relative
   to the contract pinned by `tests/test_cavity_re5000_runner.py` (`--monitor-every`,
   `--threads`, `--ref-json` were gone, `torch` was used but only imported inside `main`, so
   the module-level `run_case` raised `NameError`). It is rewritten to satisfy the contract
   *and* to produce convergence evidence:
   * `resid_history = [[step, max|Δu|, max|Δu|/U_lid, max_err_%, u_min, v_max], …]`;
   * `--resid-tol` stops on a residual threshold; a run that never reaches it is reported
     `not_converged` — a truncated run can no longer be quoted as a steady solution;
   * divergence/non-finite overrides everything: exit 2, `run_status="failed"`,
     `comparison={"status":"VOID"}` with `u_min`/`v_max` written as `null`, never as `NaN`;
   * a missing reference yields `WITHHELD` (no library fallback — an unverified table cannot
     be told apart from a verified one);
   * centreline sampling is the mean of the two central lattice planes (the geometric
     centre of an even grid lies between them), and the coordinate convention
     `node` (`y = j/(n−1)`) vs `wall` (`y = (j−0.5)/(n−2)`, half-way walls at 0 and 1) is
     selectable, so its effect on the near-wall error is measurable.
2. **Independent invariant probe.** `benchmarks/verified/cavity/re5000/diagnose_solver_bc.py`
   checks, on a case far easier than Re = 5000 (Re = 100, n = 64, `U_lid` = 0.1):
   I1 lid velocity imposed, I2 finiteness, I3 primary-vortex strength vs Ghia Table I,
   I4 fluid-mass conservation. Any correct cavity solver must pass all four; the script
   exits non-zero otherwise, so a future fix is machine-verifiable.
3. **Source review** of `solver.py` (`_stream`, `_apply_bounce_back`, `_apply_zou_he_lid`,
   `_collide_rlbm`).

## ③ Results (all numbers from run artefacts / script stdout, none hand-copied)

### 3.1 In-library solver, Re = 100, n = 64 (`diagnose_solver_bc.py`)

| check | measurement | verdict |
|---|---|---|
| I1 lid velocity `(U_lid, 0)` | `u = +0.000000` (target `+0.100`), std`(v) = 0.00e+00` at the first step, then `u_y = -0.032` (32 % of `U_lid`) | **FAIL** |
| I2 finiteness over 400 steps | non-finite at step 50 | **FAIL** |
| I3 vortex strength | `min u/U = nan` (Ghia Re = 100: −0.2109), `max v/U = nan` (0.1753) | **FAIL** |
| I4 fluid mass drift | n/a (solution non-finite) | **FAIL** |

The same failure occurs with plain BGK and at n = 128 for Re = 100 and Re = 1000 — it is
the boundary treatment, not the collision model, and not the grid.

### 3.2 Archived Re = 5000 artefact (worktree `m17072`, n = 256, 30 000 steps)

Re-scored from its stored centreline profiles
(`out_local/re5000_n256.json`, reference sha256 `1db7dce5…`):

| run | n | steps | final `max|Δu|` | status | max abs err [%] | RMSE [%] | vs 8 % |
|---|---|---|---|---|---|---|---|
| archived `m17072` | 256 | 30 000 | 3.21e-02 | ok (finite) | 39.70 | 24.04 | FAIL |

Worst rows: u at `y/L = 0.0703` (+39.7 %), v at `x/L = 0.0781` (−37.2 %); the error is
concentrated in the *lower* half of the cavity, i.e. the return flow is far too weak
(`u_min/U = −0.278` vs Ghia −0.436) — the signature of an unconverged run, yet the residual
`max|Δu| = 3.2e-2` is 32 % of `U_lid`, so 30 000 steps is nowhere near the steady state.
Coordinate-convention re-scoring (`node` vs `wall`) changes the maximum error by less than
0.1 percentage points, i.e. the half-cell mapping is *not* what limits this case.

### 3.3 Defects identified by source review

1. **Zou/He lid solves for the wrong population group** (`_apply_zou_he_lid`). At the top row
   the populations that arrive from the fluid are `3 (0,1)`, `5 (1,1)`, `6 (−1,1)`; the
   unknowns are `4 (0,−1)`, `7 (1,−1)`, `8 (−1,−1)`. The code overwrites `3, 7, 8` and leaves
   the periodically wrapped `f4` (which streamed in from the *bottom* wall) untouched, so the
   wall-normal momentum balance is never enforced. Correctly solving mass + both momenta + the
   normal non-equilibrium bounce-back `f4 − f4^eq = f3 − f3^eq` with `(u,v) = (U_lid, 0)`
   gives, exactly,
   `ρ = f0+f1+f2+2(f3+f5+f6)`, `f4 = f3`,
   `f7 = f5 + ½(f1−f2) − ½ρU_lid`, `f8 = f6 − ½(f1−f2) + ½ρU_lid`
   (the shipped code uses `f6/f5` with `±½(f1−f2)` and a `ρU_lid/6` term, i.e. a different,
   inconsistent closure).
2. **Half-way bounce-back reflects the wrong time level** (`_apply_bounce_back`). It uses
   `f_pre_stream[opposite]` at the wall node; the population that must be reflected is the one
   that just *streamed into* the wall node (`f_streamed[opposite]`). The pre-streaming value at
   a wall node is itself a wrapped value from the opposite wall, so an O(1) error is injected
   every step.
3. **Streaming translates populations against their momentum** (`_shifts = (−c_y, −c_x)`).
   `torch.roll(f, shifts=+c)` already implements `f(x+c, t+1) = f(x, t)`; negating the shift
   transports each population opposite to the velocity it carries, which no boundary condition
   can compensate for.
4. **RLBM does not regularise** (`_collide_rlbm`): `f − ω·f^neq,reg` keeps the
   un-regularised part `f^neq − f^neq,reg` — exactly the ghost modes regularisation exists to
   remove — undamped. The Latt & Chopard form is `f^eq + (1−ω)·f^neq,reg`.

Partial fixes were applied and measured during the audit, then reverted (a half-corrected BC
set must not be committed): with (1) and (2) fixed alone the lid row does impose
`(u, v) = (0.1, 0)` exactly and Re = 100 at n = 64 stays finite, but the vortex strength is
then `u_min/U = −3.30` instead of −0.21 — proof that the three defects interact and that the
runtime assertion, not intuition, is the arbiter. Fixing (3) on top makes the (partially
fixed) BC set diverge immediately. A correct rewrite is a solver-level mission of its own.

## ④ Figures

* `benchmarks/verified/cavity/re5000/figures/cavity_re5000_centerline.png` — u(0.5, y) and
  v(x, 0.5) of the archived run vs Ghia, with the ±8 % band.
* `benchmarks/verified/cavity/re5000/figures/cavity_re5000_error.png` — pointwise error per
  tabulated row against the ±8 % tolerance lines.
* `benchmarks/verified/cavity/re5000/figures/cavity_re5000_convergence.png` — stored residual
  history (empty for the archived run: it had no monitor; that is itself part of the defect
  history).
* `benchmarks/verified/cavity/re5000/figures/cavity_re5000_convention.png` —
  node- vs wall-coordinate re-scoring of the same profiles.

## ⑤ Notes, scope, and what this does **not** claim

* The Re = 5000 anchor stays **open**. No benchmark number is reported for it; the archived
  artefact is quoted only to show that its 30 000 steps are a truncation (`max|Δu|/U_lid = 0.32`),
  not a steady state.
* The archived artefact was produced by a different checkout (`environment.tensorlbm` =
  `…/work/m17072/…`), so it is *not* evidence about the current tree in either direction.
* Fixing the four defects will not by itself pass the 8 % target: Re = 5000 at `τ ≈ 0.515`
  also needs a converged fine grid (the lid boundary layer is `O(L/√Re) ≈ 3.6` cells at
  n = 256) and a compressibility budget `O(Ma²) ≈ 3 %`. The harness now reports
  `not_converged` unless `--resid-tol` is reached, so no future run can silently quote a
  truncated state.
* `CavityConfig` rejects `n < 17`; the contract tests therefore use `n = 32` (grid-size
  independent assertions). The library guard was not touched.

## ⑥ Error verdict

**Is the key quantity within 3 %?** Cannot be answered, and the honest answer is *worse than
unknown*: no current-tree run of this case is a solution at all — the solver fails at Re = 100
(I1–I4 above), so no error percentage for Re = 5000 can be quoted. The archived 39.70 % (node
and wall convention alike) is an unconverged artefact of a different checkout, listed for
provenance only and explicitly **not** an accuracy claim about `main`.

Reproduce:

```bash
python benchmarks/verified/cavity/re5000/diagnose_solver_bc.py --n 64 --re 100 --steps 400
pytest tests/test_cavity_solver_bc_invariants.py tests/test_cavity_re5000_runner.py -q
```