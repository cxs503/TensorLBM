# Board audit — GAP-DIFFLBM-ADJOINT-VERDICT (mission m20177)

**Verdict: maturity level L2 (independent functional implementation).** The claim "the
frozen-field adjoint proxy is not a total derivative and must not enter the optimisation
path" is **substantiated**: a real differentiable LBM solver produces the flow-response
gradient, an independent finite-difference check agrees to 1.6e-05, the mirror-symmetry
invariant holds to 1e-9, the admission gate is wired into the platform API and is tested.
It is **not** L3/L4: the anchor's demanded artefact — a *flow-response adjoint in
`adjoint.py`* — is not delivered (the shipped API still returns a labelled frozen-field
gradient), there is no external anchor for the gradient, and the recorded repro command does
not regenerate the delivered artefact set.

Audit mission m20231 · `verdict_by=board-audit` · reviewed commit `c48cacd31`
(branch `agent/m20177`, **not** in `origin/main` at audit time = `668b346af`).

---

## ① Case and audit scope

2-D plane channel (streamwise-periodic, constant body force) with an immersed circular disk
whose centre ordinate `y_c` is the optimisation parameter; objective = lift (momentum
exchange along +y); reference gradient = expanded AD (back-propagation through the whole
600-step window), cross-checked by central finite differences of the same finite-horizon
problem. Lattice units, D2Q9 BGK, `u*`-shifted-equilibrium forcing, partially saturated
bounce-back (Nobel & Torczynski 1998), momentum exchange (Ladd 1994 / Mei 2002).
R1 = 36×64, D = 6 lu, τ = 0.65, ν = 0.05, measured Re = 6.99; R2 = 42×74, D = 7 lu.

This is a **cross-audit (交差核验), not a redo**. Checks: (1) is the three-piece evidence
set real (run artefacts / reference-vs-got pair / executable repro command); (2) do the
report numbers survive independent recomputation from the raw artefacts; (3) does the
registered benchmark case pass; (4) what maturity level the delivered *function* reaches.

Independence limitations: I reused the mission's forward solver as the physics (a redo was
out of scope), so the *absolute* physical correctness of the penalised-disk lift is **not**
independently anchored. Independently checked here: the gradient reference against finite
differences, every reported number against the raw artefacts, the code path of the claim,
the test invariants, and replay from a staged checkout of the same commit on other nodes.

## ② What the delivery is (method / pointers)

| element | pointer | what it is |
|---|---|---|
| differentiable forward solver | `src/tensorlbm/adjoint_verdict.py:239` `simulate` | D2Q9 channel + smooth penalised disk + momentum exchange, differentiable in `(x_c, y_c)` |
| steady warm start | `src/tensorlbm/adjoint_verdict.py:213` `steady_state` | 4500 forward-only steps, parameter-independent initial condition |
| horizon study | `src/tensorlbm/adjoint_verdict.py:178` `run_horizon_study` | AD gradient vs BPTT horizon T |
| total derivative | `src/tensorlbm/adjoint_verdict.py:330` `ad_gradient` | `autograd.grad(lift, (xc, yc))` through the whole window |
| independent check | `src/tensorlbm/adjoint_verdict.py:348` `fd_gradient` | central differences, same warm start |
| shape term only | `src/tensorlbm/adjoint_verdict.py:371` `frozen_field_gradient` | `dJ/dp` with momentum fields held constant |
| verdict | `src/tensorlbm/adjoint_verdict.py:433` `verdict` | cosine / amplitude / sign / gate decision |
| shipped-proxy label + gate | `src/tensorlbm/adjoint.py:245-247`, `:314`, `:346` `optimization_gate`, `:402` `require_optimizer_ready` | flags `shape_derivative_only` / `flow_response_included` / `gradient_valid_for_optimization` + hard admission gate |
| platform wiring | `app/backend/routers/postprocess.py:3213-3249` | `POST /postprocess/adjoint-sensitivity/{job_id}` attaches `optimization_gate` to the response |
| entry | `benchmarks/adjoint_verdict_bench.py` (flags `--res --points --fd-steps --stage`) | cases, extras (lift sweep, horizon, trajectory), render (fields/tables/figures) |

## ③ Results — recomputed from the raw artefacts

| resolution | y_c | Re (measured) | lift | g_AD,y (reference) | g_FD,y (check) | |AD-FD|/|FD| | g_frozen,y (proxy) | |g_frozen|/|g_AD| | cos 1-D y_c | cos 2-D | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 14.00 | 6.994 | +1.31437e-02 | -2.26952e-03 | -2.26948e-03 | 1.57e-05 | +1.07065e-03 | 0.476 | -1.0000 | -0.9950 | blocked |
| R1 | 21.00 | 6.994 | -1.31437e-02 | -2.26952e-03 | -2.26948e-03 | 1.57e-05 | +1.07065e-03 | 0.476 | -1.0000 | -0.9950 | blocked |

Streamwise component (by streamwise symmetry the true component is ~0; the frozen-field
proxy invents a non-zero one):

| resolution | y_c | g_AD,x | g_FD,x | g_frozen,x | AD-FD rel.err (x) |
|---|---|---|---|---|---|
| R1 | 14.00 | +1.0428e-04 | +1.0428e-04 | -1.5772e-04 | 5.72e-06 |
| R1 | 21.00 | -1.0428e-04 | -1.0428e-04 | +1.5772e-04 | 5.72e-06 |

R2 — delivered as log lines only (`run_full/run.log`), plus the audit replay:

| R2 case | source | y_c | rel.err AD-FD (y) | g_frozen,y | cos 1-D | gate |
|---|---|---|---|---|---|---|
| R2 | delivered `run_full/run.log` | 18.833 | 3.19e-05 | +4.2627e-03 | -1.0000 | blocked |
| R2 | delivered `run_full/run.log` | 28.167 | 3.19e-05 | +4.2627e-03 | -1.0000 | blocked |
| R2 | audit replay cj2577 | 16.417 / 24.583 | 3.38e-05 | +2.3414e-03 | -1.0000 | blocked |

Field artefact `runs/GAP-DIFFLBM-ADJOINT-VERDICT/run_r1/fields.npz`: keys ['rho', 'solid', 'ux', 'uy', 'yc'],
shape [36, 64], `y_c = 21.0`, |u|_max = 0.05829 at row
10, 271 cells with solid fraction > 0.5, ρ ∈
[0.994477, 1.005787], all finite = True,
mean density 1.000000000000 (mass conserved to 5e-13).

**Report-vs-raw agreement:** max |Δcos| = 2.22e-16
between my recomputation and the mission's own `results.json`; `table.md` rel.err column
[1.57e-05, 1.57e-05] vs recomputed
['1.565e-05', '1.565e-05'] → consistent within printing precision. No
hand-copied number found in the shipped tables.

## ④ Reproduction (two replay jobs, same commit, different nodes/torch builds)

**A. Reconstructed staged flags** — `python benchmarks/adjoint_verdict_bench.py --out runs/repro_m20231_cases --threads 6 --stage cases --res R1 --points 14,21 --fd-steps 0.05 && python -m pytest tests/test_adjoint_verdict.py -q` (job cj2591, succeeded rc=0 on n315):

| case | g_AD,y delivered | g_AD,y reproduced | rel.diff | cos 1-D deliv / repro | gate |
|---|---|---|---|---|---|
| R1 y_c=14.0 | -2.269518958e-03 | -2.269405393e-03 | 5.00e-05 | -1.0000 / -1.0000 | blocked |
| R1 y_c=21.0 | -2.269518958e-03 | -2.269405393e-03 | 5.00e-05 | -1.0000 / -1.0000 | blocked |

Max relative difference of the reference gradient, staged replay:
5.00e-05 (delivered torch 2.7.1 vs node torch
2.13.0 on node315).

**B. The documented command** — `python benchmarks/adjoint_verdict_bench.py --out runs/repro_m20231 --threads 6 && python -m pytest tests/test_adjoint_verdict.py -q` (job cj2577, rc=1 on n330):

| case | g_AD,y delivered | g_AD,y reproduced | rel.diff | cos 1-D deliv / repro | gate |
|---|---|---|---|---|---|
| R1 y_c=14.0 | -2.269518958e-03 | -2.269405393e-03 | 5.00e-05 | -1.0000 / -1.0000 | blocked |
| R1 y_c=21.0 | -2.269518958e-03 | -2.269405393e-03 | 5.00e-05 | -1.0000 / -1.0000 | blocked |

Max relative difference of the reference gradient, documented replay:
5.00e-05. The verdict is **cross-version / cross-node robust**; it is
not bit-reproducible.

## ⑤ Figures

* `artifacts/m20231/fig_audit_gradients.png` — (a) g_AD vs g_FD vs g_frozen on y_c;
  (b) |AD−FD|/|FD| and |g_frozen|/|g_AD| (log); (c) cosine gate vs the 0.99 threshold.
* `artifacts/m20231/fig_audit_field.png` — |u| and u_y replotted from the delivered
  `fields.npz` with the 0.5 solid-fraction contour: the vertically asymmetric,
  lift-generating wake the verdict rests on.

## ⑥ Checks run / not run

Run:

* recomputation of every gradient, error, cosine and gate decision from `results.json`
  (`artifacts/m20231/recompute_verdict.py` → `recompute.json`);
* delivered-vs-replay comparison (`artifacts/m20231/repro_compare.py` → `repro_compare.json`);
* `python -m pytest tests/test_adjoint_verdict.py -q` → **8 passed, 1 warning** in
  18.7 s on worktree m20177 @ c48cacd31 (python 3.11.6 / torch 2.7.1, agent container), and **8 passed, 2 warnings** in 12.2 s on
  node315 via job cj2591 (torch 2.13.0). These tests assert invariants, not just execution: AD-vs-FD < 0.5 %
  (`test_ad_matches_finite_difference:59`), FD step independence (`:72`), mirror symmetry
  lift-odd / gradient-even to 1e-9 (`:86`), gate thresholds incl. the anti-parallel case
  (`:108`), flow-response term non-negligible (`:131`), shipped proxy labelled
  shape-derivative-only (`:155`, `:176`, `:184`);
* two replays of the delivery on the cluster (cj2591 and cj2577), plus
  parsing of the R2 numbers out of `run_full/run.log`;
* field-artefact sanity (shape, finiteness, mass, solid-cell count).

Not run (unknown, **not** passed):

* no external anchor for **d(lift)/dy_c** — no analytic or experimental value exists for a
  penalised disk at Re ≈ 7 in lattice units, and the delivery says so openly. AD-vs-FD is
  *same-model consistency* (very tight), not an independent physics benchmark;
* grid convergence / Richardson extrapolation — R2 numbers exist (log + replay) but no
  resolution trend or extrapolated value is quoted; convergence order **not_run**;
* the **delivered** artefact set contains no horizon data and only FD step 0.05, although the
  code computes both; the replay produced them (`[0.05, 0.1]`, horizon present) but the
  report's ⑤ conclusions are backed only by the un-shipped replay — **partially checked**;
* the platform endpoint `POST /postprocess/adjoint-sensitivity/{job_id}` was not exercised
  end-to-end (no live job with a checkpoint); only the gate function and its unit test were;
* generality beyond 2-D / one Reynolds number: the delivery itself states the *sign* is
  case-dependent; no second case was audited.

## ⑦ Errors vs benchmark and the explicit "< 3 %?" answer

| KPI (registered case) | reference | got | error | threshold | pass? |
|---|---|---|---|---|---|
| `CASE-ADJOINT-VERDICT-REFERENCE-GRADIENT` (id 135) | central FD of the same forward problem | |g_AD-g_FD|/|g_FD| = 1.565e-05 (delivered R1 y_c=14.00/21.00), reproduced 1.565e-05 (job cj2591); R2 3.19e-05 (run_full log) / 3.38e-05 (job cj2577); mirror symmetry lift-odd/gradient-even to 1e-9 | 1.57e-05 = 1.6e-03 % of the reference | <1e-3 band for a usable gradient reference (owner 15 %, ecosystem 3 %) | passed |
| `CASE-ADJOINT-VERDICT-FROZEN-PROXY-GATE` (id 136) | cos ≥ 0.99 required for admission | cos 1-D in y_c = -1.0000 (both mirrored points, delivered AND both replays), 2-D = -0.99497; |g_frozen|/|g_AD| = 0.476; sign agreement false; gate_decision = blocked everywhere incl. R2; shipped proxy returns shape_derivative_only=True with a spurious g_x | 100 % direction error (anti-parallel) and 53 % magnitude error vs the 3 % ecosystem bar | cos >= 0.99 required for admission (COSINE_GATE, src/tensorlbm/adjoint.py:51) | failed (by design: this failure IS the verdict; the frozen-field proxy is rejected) |

**Explicit answer:** the *gradient reference* is far inside 3 % (1.57e-05 ≈ 1.6e-03 %), so the
arbiter used for the verdict is trustworthy. The *frozen-field proxy* is **not** within 3 %:
anti-parallel to the total derivative (cos = −1.0000 in the parameter direction, −0.99497 in
2-D) with amplitude ratio 0.472–0.476 → 100 % direction error and ~53 % magnitude error. That
failure is the intended verdict, not a regression. The capability the anchor demands (a
flow-response adjoint exposed by `adjoint.py`) therefore remains **open**.

## Findings (severity-ranked, with evidence)

**F1 — blocking (level-limiting): the required rewrite of `adjoint.py` is not delivered.**
`adjoint.py` still returns a frozen-field gradient (`src/tensorlbm/adjoint.py:245-247`,
`:314`); the total derivative lives in `adjoint_verdict.py`, used only by a benchmark script
and carrying its own inline D2Q9 kernels (no reuse of the main solver). The platform endpoint
(`postprocess.py:3237`) therefore still returns a labelled *non-optimisable* gradient →
functional substance L2, not L3.

**F2 — high: the recorded repro command cannot regenerate the delivered artefact set.**
`results.json.repro_cmd` = `python benchmarks/adjoint_verdict_bench.py --out runs/GAP-DIFFLBM-ADJOINT-VERDICT/run_r1 --threads 6`. Replaying it (job cj2577)
produced ['R1 y_c=14.0', 'R1 y_c=21.0', 'R2 y_c=16.417', 'R2 y_c=24.583'] — a superset of the delivered ['R1 y_c=14.0', 'R1 y_c=21.0'] (R2 points
16.417/24.583, whereas the delivered `run_full/run.log` has 18.833/28.167) — with FD steps
[0.05, 0.1] instead of [0.05], then **exited rc=1** in
`make_figures` (`adjoint_verdict_bench.py:244`/`:549`): `ModuleNotFoundError: No module named
'matplotlib'` on the node image, although `requirements.txt:3` declares it. So no figures, no
`table.md`, no `fields.npz`, no `elapsed_seconds` on a clean node. The delivered `run_r1` was
assembled from a three-stage sequence (`run.log`: `stage=cases` → `stage=extras` →
`stage=render`) with `--res/--points/--fd-steps` flags that are recorded nowhere. With the
reconstructed flags (job cj2591) the delivered cases reproduce to
5.0e-05 and the verdict reproduces exactly (cos −1.0000, gate blocked).

**F3 — high: the optimisation-trajectory evidence is broken.** Lift becomes `NaN` at
iteration 2 for **both** the AD and the frozen trajectory
(`results.json.trajectories`); the doc still concludes "trajectories adverse (opposite
directions): True" from one finite step and two NaN-terminated curves. The replay reproduces
the same NaN, so this is a code defect, not a stale artefact.

**F4 — medium: predeclared checks were executed but dropped from the delivery.**
`study.fd_steps = [0.05, 0.1]`, `study.horizons = [300, 600]`, `"predeclared": true`,
yet the delivered `results.json` carries only `fd_steps =
[0.05]` and `fd_step_sensitivity_y = null`, and no horizon block; the
replay shows both run fine (FD step sensitivity 4.09e-05, horizon study present). Packaging
defect, but as delivered it must count as not shipped.

**F5 — medium: the self-reported wall time is understated ~8×.**
`results.json.elapsed_seconds = 22.02` is the render stage only
(the same field is written by whichever stage runs last); the delivered run log shows cases
82.9 s + extras 72.6 s + render
22.0 s ≈ 178 s, and the report's environment block quotes 22.0 s.

**F6 — medium: the delivery was not in the trunk at audit time.** `c48cacd31` is reachable
only from `agent/m20177`; `origin/main` = `668b346af` has neither
`src/tensorlbm/adjoint_verdict.py` nor the doc, so any gate verifying pointers from `main`
fails until the merge. Raw artefacts are snapshotted in [m20231](../development/evidence/m20231/).

**F7 — medium (registry): the anchor had no benchmark case at delivery time.**
`SELECT … WHERE anchor_slug='GAP-DIFFLBM-ADJOINT-VERDICT'` returned 0 rows, so the
"succeeded" verdict was machine-unverifiable. This audit registered ids 135
(reference-gradient check, `passed`) and 136 (proxy gate, `failed` by design).

**F8 — low: comparability mixing.** `grad_shipped_proxy` uses a different lift functional
(pressure difference on surface cells), so its scale is not comparable with `g_AD`; the report
says so in prose but puts `|g_frozen|/|g_AD|` (comparable, 0.472) next to `g_shipped` (not
comparable) in adjacent columns.

**F9 — low: figure / typography defects.** ⑤ renders "step )" with the value missing (empty
f-string); the `run_r1` field figure shows the last case (`y_c = 21.0`), not the
`y_c = 14` case of the headline table; the `run_full` R2 points are not mirror-symmetric about
the centreline (18.833 vs 28.167, centre 21).

## Level determination (L0–L5)

| level | requirement | met? | evidence |
|---|---|---|---|
| L0 | document | yes (exceeded) | `docs/GAP-DIFFLBM-ADJOINT-VERDICT.md` (6 sections); `src/tensorlbm/adjoint.py:11-31` |
| L1 | maths / unit verification | yes | `tests/test_adjoint_verdict.py` 8/8 invariants (AD-vs-FD, FD-step, mirror symmetry, gate, labels) |
| L2 | independent functional implementation | **yes — the level** | `adjoint_verdict.py:239/330/348/371/433`: input → LBM → AD/FD gradients → verdict; gate `adjoint.py:346` |
| L3 | system integration (main solver / platform) | no | only the *guard* reaches the platform (`postprocess.py:3247`); the flow-response gradient does not, and `adjoint_verdict.py` re-implements the LBM kernels instead of using the main solver |
| L4 | benchmark validation (full case + error analysis + reproducible) | no | no external anchor (declared analytic/numerical only); canonical artefacts = 1 resolution without horizon/FD-step data; the recorded command cannot regenerate them (F2) |
| L5 | research/engineering result | no | — |

**Level = L2.** The negative result itself is solid and belongs in the ledger; what is missing
for L4 is packaging (one command → the shipped artefacts, R2 promoted, matplotlib on the node
image) plus an independent anchor — not physics.

## Disposition

**Request changes** — claim closed as *negative result substantiated at L2*, capability gap
kept open:

1. expose the flow-response (total-derivative) gradient through
   `tensorlbm.adjoint.adjoint_sensitivity` / the platform endpoint, or mark the endpoint
   NOT_SUPPORTED for optimisation and point at the differentiable solver (F1);
2. make one recorded command regenerate exactly the shipped artefact set, ship R2 + horizon +
   FD-step 0.1 or delete the corresponding claims, and add matplotlib to the compute-node env
   (F2, F4);
3. fix the trajectory `NaN` or drop the trajectory conclusion (F3);
4. report total wall time instead of the last-stage time (F5); merge to `main` (F6);
5. keep registered cases 135/136 as the machine-checkable anchor (F7).

Honest summary: the *verdict* is real, reproduced on two other nodes/torch build
(5.0e-05 on the reference gradient, cos −1.0000 and gate blocked at
every case) and guarded by a tested gate — but it is a well-documented negative result plus a
fail-safe, not the flow-response adjoint the anchor called for.
