# Adversarial review — `tensorlbm.inverse` (GAP-PP-4-opt-loop)

Reviewer: verifier seat (tp-ver-1), mission m23029.  Date 2026-11-16.
Object of review: the inverse-problem infrastructure claimed for anchor
`GAP-PP-4-opt-loop` — `src/tensorlbm/inverse/{optimizer,losses,convergence,seed}.py`,
its documented reference consumer `src/tensorlbm/opt_loop_tau_recovery.py`,
the acceptance claims in `docs/inverse_infra.md`, and the evidence bundle
(artefacts, tests).

Independence limitations: the same worktree that carries the fix was used for
the review (no separate checkout); the full 18-job production suite
(6 tracks × 3 seeds) was still queued on the cluster at the time of writing
(`cj4540`; pre-refactor baseline `cj4539` also queued), so the n = 32 arm and
the complete 3-seed × 6-track table are **not_run**, not "passed".  Everything
below that is marked *reproduced* was re-executed from a clean interpreter on
this machine.

Acceptance criterion under audit: *τ/Re recovery demo runs end-to-end and
produces a machine-readable convergence JSON; Taylor-Green decay Re recovery
< 3 %; 3 seeds reproducible.*

## Findings (severity ranked)

### F1 — blocking (claim), fixed: the "reference consumer" never imported the package
`docs/inverse_infra.md` stated that `tensorlbm.opt_loop_tau_recovery` "是本基础
设施的参考消费者".  It was not: the module imported only `tensorlbm.autograd_path`
and `tensorlbm.d3q19` and carried its own private copy of the Adam + cosine +
box-clamp + L-BFGS loop (`recover()`), so `optimise_loop`, `LossComposition`,
`ConvergenceLog`, `seed_everything` and `jittered_init` were dead code with
respect to the only problem that was supposed to exercise them.  Two loops also
drift silently — and indeed they had already drifted (see F4).
Reproduction: `grep -n "^from\|^import" src/tensorlbm/opt_loop_tau_recovery.py`.
Fix: `recover()` now delegates to `inverse.optimise_loop` with a one-component
`LossComposition("rel_mse")`, `seed_everything` + `jittered_init` for seeding,
and an optional `ConvergenceLog`.  Pinned by
`tests/test_opt_loop_tau_recovery.py::test_recover_drives_tensorlbm_inverse`
(monkeypatch counts the delegation; asserts phases `adam … lbfgs_eval … final`).

### F2 — blocking (evidence), fixed: no convergence JSON in the infrastructure schema existed
The consumer wrote only its own bespoke schema
(`tensorlbm/opt-loop-tau-recovery/v1`).  Nothing ever validated that the
`tensorlbm/inverse/convergence/v1` schema could be emitted by a real problem,
so the acceptance clause "机读收敛日志" was untested end-to-end.
Fix: `opt_loop_tau_recovery --convergence-dir DIR` writes one
`tensorlbm/inverse/convergence/v1` file per (track, seed); new
`inverse.validate_convergence_payload` / `read_convergence_json` machine-check
them (schema tag, required keys, per-row types, `n_iterations` vs history
length, `final_loss` vs first loss).  4 logs produced by the n = 16 arms
validate with zero problems (`artifacts/opt_loop_verify_summary.json`).

### F3 — major (evidence), fixed: documented artefacts did not exist
`docs/inverse_infra.md` listed `artifacts/opt_loop_tau_recovery.json` and
`artifacts/opt_loop_seed{0,1,2}.json` as produced evidence.  They are absent
from the worktree and were **never tracked**: `git log --diff-filter=D --
artifacts/opt_loop_tau_recovery.json` is empty.  A claim of the form "the demo
passes" therefore had no committed artefact behind it.
Remediation: `docs/inverse_infra.md` corrected (artefact list now points at the
files actually produced by this mission, with the pending production run called
out); n = 16 arms regenerated here.

### F4 — major (correctness of the reported flag), fixed: `converged` was fabricated
The consumer set `converged = True` unconditionally after *running* the L-BFGS
polish, without looking at whether the polish reduced the loss; the shared loop
conversely never set it after a successful polish.  With the production
configuration (`iters = 2`, `tol = 1e-10`) neither behaviour describes the run:
the demo would report `converged=True` for a stalled polish, and the same run
through the infrastructure would report `converged=False` at an optimum.
Fix: `optimise_loop` now records `stop_reason ∈ {adam_tol, lbfgs_polish,
polish_stalled, max_iters}` and sets `converged` iff the Adam tolerance was met
or a completed polish did not worsen the objective
(`post_polish ≤ pre_polish·(1+1e-12)`).  The summary JSON carries
`stop_reason`, `polished`, `n_lbfgs_evaluations`.  Tests:
`test_stop_reason_max_iters_without_polish` (honest False),
`test_stop_reason_lbfgs_polish_and_converged_flag`.

### F5 — moderate, fixed: `seed_everything` did not seed the stdlib RNG
Its docstring promised that "all random sources … must flow through these
utilities", but only torch/CUDA/NumPy were seeded; a forward model sampling with
`random.random()` was not reproducible across restarts.  Counterexample
(reproduced before the fix: two calls after `seed_everything(1234)` differ).
Fix: `random.seed(seed)`; regression test
`test_seed_everything_seeds_stdlib_random`.  The same docstring claimed
`PYTHONHASHSEED` gives reproducibility — it cannot, for the *running*
interpreter; the note is now honest (exported for child processes only).

### F6 — moderate (reporting), fixed: `iters` counted L-BFGS line-search evaluations
`RecoverResult.iters` was set to `len(history)`, i.e. Adam iterations **plus**
every line-search evaluation plus the final record (12 for a 2-Adam-iteration
run).  Fix: `iters`/`adam_iterations` = Adam iterations (2), `n_records` = total
records (12); both are logged.

### F7 — moderate, open (documented limitation): the box constraint is not enforced inside the polish
`optimise_loop` projects the parameters back into the box only after
`opt_l.step()`, so strong-Wolfe line-search evaluations can visit infeasible
points (τ ∉ [0.5005, 1.9]) and those infeasible losses appear in the history.
Not changed here: clamping inside a quasi-Newton iteration breaks its curvature
pairs and the numbers would drift away from the reviewed baseline.  Documented
in the module docstring; the returned point is always clamped.

### F8 — minor, open: losses swallow mis-wired inputs
`relative_mse_loss` / `l1_loss` / `weighted_mse_loss` `zip()` the simulated and
observed sequences (silently truncating a length mismatch) and return a *zero*
loss for an empty sequence — a broken forward model then reads as
"converged at loss 0".  Not changed (truncation may be intentional for partial
overlap); the hazard is recorded here and in the docstrings' scope.

### F9 — minor: `jittered_init` is multiplicative
`jittered_init(0.0, seed=s)` returns 0 for every seed (no relative jitter at
zero).  Fine for the ν-recovery use, misleading for parameters that may be 0.

## What was actually executed (this machine, this worktree)

```
PYTHONPATH=src pytest tests/test_inverse_infra.py tests/test_opt_loop_tau_recovery.py -q
# 39 passed (29 infra + 10 demo), 4 new infra invariants, 1 new demo wiring test

python -m tensorlbm.opt_loop_tau_recovery --tracks twin16 analytic16 --seeds 0 \
    --convergence-dir artifacts/inverse_convergence --out artifacts/opt_loop_verify_n16_seed0.json
```

| arm | τ_true | τ_hat | Re_true | Re_hat | rel. error | gate < 3 % |
|---|---|---|---|---|---|---|
| twin16, seed 0 | 0.55 | 0.5499999999872 | 48.0 | 48.0000000123 | −2.6 × 10⁻⁶ % | pass |
| analytic16, seed 0 | 0.55 | 0.5500897791698 | 48.0 | 47.9139664774 | +0.1796 % | pass |
| analytic16, seeds 1, 2 | 0.55 | same optimum (log ν differs by ≤ 3 × 10⁻⁹ between seeds) | 48.0 | 47.9140 | +0.1796 % | pass |

The twin arm is the optimiser check (the recoverable optimum is exact: loss
7.3 × 10⁻²¹), the analytic arm is the physics check against the exact
Navier-Stokes solution (residual +0.18 %, dominated by the O(Ma²) compressibility
offset, absorbed into the nuisance amplitude 1.0277).

## Checks not run (unknown, not passed)

- full production suite, 6 tracks × 3 seeds (`cj4540`) and the pre-refactor
  baseline (`cj4539`): still queued after 45 min, cluster backlog ~110 jobs;
- the n = 32 analytic arm and the observation-noise arms;
- bitwise refactor-fidelity comparison against the pre-refactor loop (baseline
  job pending) — covered indirectly by the demo regression tests
  (`test_lbfgs_polish_is_what_converges_the_loop` asserts < 10⁻³ % at n = 8,
  `test_same_seed_reproduces_bit_for_bit`);
- GPU / multi-thread determinism.

## Disposition

**Request changes → addressed in this commit, approve with notes.**
F1–F6 are fixed with regression tests; F7–F9 are documented limitations, not
silent behaviour.  The acceptance claim is verified only for the n = 16 arms;
the n = 32 arm and the complete 3-seed production table must be read from
`cj4540`'s artefacts when the queue drains, and the artefact list in
`docs/inverse_infra.md` must be updated then.
