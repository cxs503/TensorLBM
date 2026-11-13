# Board audit — m17726 `XVAL-LBM-DIFFUSE-INTERFACE-WETTING-VIA-WALL-ORDER` (TensorLBM)

Audit mission: **m17853** · auditor seat: `board-audit` · subject: **m17726** (status `succeeded`,
finished 2026-11-13 09:10:46Z, model `deepseek-v4-flash`, report id 16715).
Repo `TensorLBM`, subject branch `agent/m17726` @ `9bcc6db6`.
Independence limitation: I audited the subject worktree as found
(`/root/tp-agent/work/m17726`) — I did **not** re-implement the physics; independent verification
is limited to (a) re-executing the delivered code bit-for-bit, (b) recomputing the predeclared
gates from the raw `result.json` files, (c) machine-checking repo landing state.

## 1. What was checked (scope)

| check | method | artefact |
|---|---|---|
| three-piece evidence (impl / benchmark ref-vs-got / repro_cmd) really exists | file walk + sha256 of every deliverable, plus `git log --all -- <path>` / `git ls-tree agent/m17726` | `artifacts/snapshot_manifest.json` |
| benchmark judge | `SELECT * FROM tensor_benchmark_cases WHERE anchor_slug like '%WALL-ORDER%'` → **0 rows before this audit**; auditor therefore reconstructed the judge from `predeclared.json` (`accept_pct = 15.0`) and re-derived every error from the run files | `register_and_verdict.py`, `artifacts/rerun_compare.json` |
| reproducibility | re-ran the delivered `run_xval.py port` for the 6 `psi_w` sweep points at `xi = 3` in a separate scratch copy (`refs/rerun`, deliverable code untouched, `PYTHONPATH=refs/src`) | `rerun_all.sh`, `artifacts/rerun_compare.json` |
| unit tests | `python3 -m pytest tests/test_wall_order_parameter_wetting.py -q` | 15 passed, 1 xfailed in 63 s |
| report六段 | read `tensor_reports.id = 16715` markdown + figures array | see finding F5 |
| external anchor | cluster job `cj1656` state in `cluster_jobs` | `queued` (never ran) |

## 2. Evidence grade of the deliverable

| capability | grade | evidence |
|---|---|---|
| `grad_laplacian` / `free_energy_parameters` / `analytic_contact_angle` | **REAL** | stencils exact on quadratics/linear fields at `atol=1e-12` (`tests/…py:41-99`); `kappa = 3/4 σξ`, `a = 2κ/ξ²` recombined to `σ` (`:81-99`) |
| `relax_to_equilibrium` (mass-conserving Allen–Cahn descent) | **REAL** | `src/tensorlbm/wetting/wall_order_parameter_2d.py:379-429`; mass error `1.6e-16` in every `result.json`; converges toward `mu = const` |
| `contact_angle_from_field` / `theta_from_cap_area` | **REAL** | Laplace-cap inversion, bisection on `A/h² = (θ − sinθcosθ)/(1−cosθ)²` (`:489-517`); round-trip on a synthetic cap at `abs=1e-4` (`tests:114-122`) |
| `step` (two-distribution LBM + Guo force + halfway bounce-back) | **REAL but falsified as a wetting carrier** | `:339-378`; the `lbm` arm dissolves the cap to 5.4 % of its peak in 1200 steps (`dissociation_ratio = 0.0541`) — the report's own defect finding, correctly reported and not hidden |
| OSS anchor arm (`ebinan92/Fingering_dynamics`) | **WITHHELD** | `oss_psi+0.00/result.json: "finite": false`, `psi_min/max = NaN` at step 200; the 200×250 retry (`cj1656`) is still `queued` → **no external reference number exists** |

## 3. Benchmark verdict against the predeclared 15 % gate

Re-derived from the run files (not from the report table): 10 port points.

| `psi_w` | `xi` | θ port [deg] | θ analytic [deg] | err | gate 15 % | converged |
|---|---|---|---|---|---|---|
| −0.50 | 3 | 133.182 | 133.433 | −0.19 % | pass | no (30000 it cap) |
| −0.35 | 3 | 118.605 | 120.236 | −1.36 % | pass | no |
| 0.00 | 3 | 88.109 | 90.000 | −2.10 % | pass | no |
| +0.35 | 3 | 59.856 | 59.764 | +0.15 % | pass | no |
| +0.50 | 3 | 47.167 | 46.567 | +1.29 % | pass | no |
| **+0.70** | 3 | **20.115** | **28.538** | **−29.52 %** | **FAIL** | no |
| 0.00 | 2 | 89.041 | 90.000 | −1.07 % | pass | no |
| 0.00 | 4 | 86.696 | 90.000 | −3.67 % | pass | no |
| +0.35 | 2 | 60.868 | 59.764 | +1.85 % | pass | no |
| +0.35 | 4 | 58.548 | 59.764 | −2.03 % | pass | no |

median |err| **1.60 %**, 9/10 inside the gate, **max 29.52 %** → the anchor's acceptance gate is
**not** satisfied (`summary.json: gates.all_within_accept = false`, `all_converged = false`).
The reference itself is a *self-derived* Young law from the same free energy
(`cosθ = (3ψ_w − ψ_w³)/2`), i.e. a free-energy consistency check, not an independent
experiment/literature anchor; the only intended independent arm produced NaN.

**Reproduction (auditor):** 6/10 points re-run, **all 6 bit-identical** to the shipped numbers
(`max |Δθ| = 0.0` deg, exact equality of the float), including the failing
`psi_w = +0.70` point (20.115° vs 20.115°) → the numbers are real and deterministic, not MOCK.
4 points (`xi = 2/4` resolution sweep) were **not** re-run in this audit (time budget) → unknown,
not passed.

## 4. Level

* **L1 math/unit** — met: exact stencil identities, `tanh` profile, Young law values, synthetic-cap
  inversion, mass conservation, 15 passed / 1 xfailed.
* **L2 independent implementation** — met *as substance*: real code, input → compute → output,
  reproduced bit-identically by an independent re-execution.
* **L3 system integration** — **not met**: the only reference to the module in the tree is the
  re-export `src/tensorlbm/wetting/__init__.py:42`; no solver/benchmark caller
  (`grep -rn wall_order_parameter` → `__init__.py` + docs only).
* **L4 benchmark validation** — **not met**: the primary acceptance gate fails at `psi_w = +0.70`
  (29.52 % > 15 %), the declared convergence tolerance (1e-8) is never reached
  (`converged = false` on 10/10 rows, 30000-iteration cap), and the external comparison arm is
  WITHHELD (NaN; `cj1656` still queued). Resolution independence (2.7 % over `xi = 2→4`) is fine.
* **L5** — not met (no independent experimental/literature anchor at all).

**Declared level = 2** (functional substance), **ledger level = 1** because of F1 below: nothing of
this mission is in the repository, so no repo-level L3/L4 evidence can be pointed at.

## 5. Findings (severity-ranked)

**F1 — BLOCKING (process + durability): the mission landed nothing in git.** Branch `agent/m17726`
holds no mission commit (`git log` HEAD = merge of m17533); the module, the tests, the doc and the
whole `xval/LBM_WALL_ORDER_PARAMETER_WETTING/` tree are **untracked** working-tree files.
`git log --all -- src/tensorlbm/wetting/wall_order_parameter_2d.py` → empty, also in
`/root/tensor-ecosystem/TensorLBM`; `tensor_dev_missions.result.evidence.branch = null`.
Consequence: a fresh clone of TensorLBM, the cluster stage-dir runs, and the maturity gate all see
nothing; `tensor_dev_missions.worktree_recyclable` may delete the only copy. This audit therefore
snapshots every file with sha256 under `refs/` (bit-identical: `artifacts/snapshot_manifest.json`).
*Remediation:* rework mission must commit the deliverables (or the capability dies with the worktree).

**F2 — BLOCKING (benchmark): the anchor had no judge row until now.** `tensor_benchmark_cases`
had **0** rows for this anchor (`anchor_slug` drift: the deliverable's `predeclared.json` says
`…-WALL-ORDER-PARAMETER`, the board表 says `…-WALL-ORDER`) → registered by this audit with the
measured numbers and the 15 % bar, status `pending` because 1/10 points fails.

**F3 — MAJOR (accuracy): `psi_w = +0.70` is 29.52 % out of gate.** Honestly disclosed in the
deliverable (`summary.json`, doc §(4),(8): "the strong-wetting branch ψ_w ≥ 0.7 is reported as out
of the validated range"), so not a misstatement — but the validated window is `|ψ_w| ≤ 0.5` only,
and no mechanism was given for the breakdown beyond "cap degenerates / both estimators
ill-conditioned" (the two estimators differ by >5° there, which is itself evidence of a measurement
defect rather than only of a model limit).

**F4 — MAJOR (validation depth): no independent anchor number.** The "XVAL against OSS" premise is
not satisfied numerically: the vendored reference goes NaN by step 200 and its published-box retry
`cj1656` has been `queued` since submission (never ran, rc NULL). The gate therefore rests on a law
derived from the *same* free energy — a consistency check, not a cross-validation.

**F5 — MINOR (report discipline):** report id 16715 has the six sections (1)–(6) plus (7) diff and
(8) decision, but (a) the three figure objects carry **empty `png_base64`** (`len == 0`, only
name+caption) although the PNGs exist in the worktree → figures are not actually embedded;
(b) it never explicitly answers the required "是否 < 3 %" — the data do answer it
(5 of 6 sweep points are < 2.2 %, `psi_w = +0.70` is 29.5 %), (c) no convergence statement ties
`converged = false` on 10/10 rows to the "stationary well before tolerance" claim
(that claim is supported only by one restart check quoted as "< 0.05 deg").

**F6 — MINOR (licensing/compliance):** `xval/…/oss_src/validation_ebinan92.py` vendors source from a
repository that ships **no LICENSE** (the report states this). Verbatim vendoring into TensorLBM is
a redistribution risk if it is committed; recommend keeping only the derived diff/description, or an
explicit provenance note in the rework.

**F7 — MINOR (doc/deliverable naming):** anchor slug drift between `predeclared.json`
(`…-PARAMETER`), the doc filename, and the board anchor (`…-WALL-ORDER`); also `summary.json`
`n_oss = 1` counts a NaN run as a row.

## 6. Overstated claims (what the report says vs raw evidence)

1. "10 port rows … (median |err| 1.60 %, max 29.5 %)" — **accurate** (verified: 1.6017 %, 29.5158 %).
2. "9/10 rows inside the 15 % gate" — **accurate**.
3. "ported … into TensorLBM" — **overstated** in the repo sense: nothing is committed (F1).
4. "cross-validation against the OSS implementation" — **overstated**: the OSS arm yields no number;
   what exists is a free-energy self-consistency check plus a *defect* found in the OSS closure
   (that defect claim is itself well supported: `dissociation_ratio = 0.0541`, mass conserved to
   1e-12, `fig3`).
5. "tests (15 passed, 1 xfailed)" — **accurate** (re-ran: 15 passed, 1 xfailed, 63 s).

## 7. Checks not run (explicitly unknown)

* the 4 resolution-sweep points (`xi = 2/4`) were not re-executed → their reproducibility is
  **not verified** (only the 6 `xi = 3` points are).
* the vendored OSS code was not re-executed at its published box (`cj1656` still queued; 200×250×1200
  steps not attempted in this audit).
* no independent literature contact-angle value (e.g. a measured water/air sessile drop or a
  molecular-dynamics Young angle) was compared — out of this audit's scope, but it is exactly what
  L5 would require.

## 8. Disposition

**Request changes → `reopened` (level 2 declared, ledger level 1).** Required of the successor:

1. commit the deliverable (module + tests + xval artefacts) so the capability is in the repo;
2. add a real integration caller (a benchmark/`run.py` entry or solver path) so L3 is machine-checkable;
3. either fix the `psi_w = +0.70` measurement (two estimators differ by >5° → measurement defect
   hypothesis is untested) or shrink the predeclared validated window to `|ψ_w| ≤ 0.5` in
   `predeclared.json` **and** in the gate row, and state the out-of-window branch as a limitation;
4. reach or re-declare the convergence criterion (10/10 rows hit the 30000-iteration cap with
   `converged = false`);
5. produce the missing independent number (`cj1656`) or downgrade the "cross-validation" framing to
   "free-energy consistency + OSS defect report";
6. embed the figures (`png_base64`) and answer the "是否 < 3 %" question explicitly.

## 9. Audit artefacts

```
docs/board_audit/m17853/
  AUDIT.md                     this file
  verify_snapshot.py           sha256 + git-landing-state + gate recomputation
  compare_rerun.py             shipped vs auditor re-run, per point
  rerun_all.sh                 the re-run driver (deliverable code verbatim)
  register_and_verdict.py      benchmark-case registration + maturity verdict INSERT
  artifacts/snapshot_manifest.json, artifacts/rerun_compare.json
  refs/src/…, refs/tests/…, refs/xval/…   bit-identical snapshot of the (unlanded) deliverable
  refs/rerun/runs/…            auditor re-run outputs
```