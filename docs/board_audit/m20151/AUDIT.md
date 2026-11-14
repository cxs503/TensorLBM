# m20151 board audit — GAP-DIFFLBM-BOUNDARY-GRAD (subject: m20070, TensorLBM)

Auditor mission: m20151 (`gap-difflbm-boundary-grad-audit`, board-audit).
Subject: m20070 `GAP-DIFFLBM-BOUNDARY-GRAD` (status `succeeded`), work tree
`/root/tp-agent/work/m20070`, branch `agent/m20070`.
Audit type: **交差核验 (reconciliation), not a redo.** No solver code was modified.

**Disposition: capability confirmed at L2; request changes (documentation + landing).**

| level | claim | gate outcome |
|---|---|---|
| L0 documentation | yes (215-line report) | — |
| L1 math / unit | yes (7 pytest, exact Couette/Poiseuille anchors) | — |
| **L2 independent functional** | **yes** — real module, entry, run artifacts, verified AD-vs-FD | claimed |
| L3 system integration | **no** — no caller anywhere in the ecosystem `grep -rl -F diff_boundary_grad /root/tensor-ecosystem/TensorLBM` = 0 files | not claimed |
| L4 benchmark validation | **no** — the one external case (DFG cylinder) FAILS; the gradient claim is a self-consistency (AD vs central FD) claim, plus exact-solution anchors for the *channel* probes | not claimed |
| L5 research/engineering | no | not claimed |

Evidence bundle (this directory):

| file | content |
|---|---|
| `delivered_snapshot/` | byte-identical copy of the subject deliverable (`sha256` in `evidence/sha256_source.txt`) — needed because the subject delivery is **not committed** (see F6) |
| `evidence/verification.json` | independent re-derivation of every reported number from the artifact JSON |
| `evidence/repro_compare.txt` | audit re-run of the 3 documented matrix repro commands vs artifact (18/18 rows) |
| `evidence/repro_anchors.txt` | audit re-run of the documented anchor command vs artifact |
| `evidence/pytest_m20151.log` | audit re-run of the delivered test suite: `7 passed in 47.55 s` |
| `figures/fig1..3` | AD-vs-FD matrix, DFG failure, exact anchors |
| `verify_audit.py`, `make_figures.py` | audit code (arithmetic recomputation only) |

---

## 1. What was checked, and the verified result

### 1.1 The three-piece set really exists

* **run_artifacts** — `artifacts/GAP-DIFFLBM-BOUNDARY-GRAD/diff_boundary_grad_full.json`
  (157 KB: 18 matrix rows in 3 groups, 3 anchor groups, closure-sensitivity scan),
  `diff_boundary_grad_quick.json`, and 4 PNGs. `env` records loongarch64, torch 2.7.1,
  float64, 6 threads, `2026-11-14T08:55:57Z`, block clocks 79.7/62.0/42.5/36.3 s
  (cluster job `cj2489` per the report).
* **implementation** — `src/tensorlbm/diff_boundary_grad.py` (339 lines): `DiffChannel2D`
  (BGK collision with graph-carrying τ, gather streaming, three boundary closures,
  Zou/He inlet, zero-gradient copy outlet, Ladd momentum-exchange probe, penalisation
  probe, gradient-checkpointed rollout). Read at lines 138–339: no stub, no in-place
  overwrite on the gradient path.
* **tests** — `tests/test_diff_boundary_grad.py`: 7 tests, **7 passed in 47.55 s** on the
  audit re-run. They assert numbers, not execution: relative error ≤ 1e-2 per closure,
  non-degeneracy (`|FD| > 1e-3`), drag ≠ lift gradient, penalisation gradient not
  detached from `y_c`, closure-magnitude sanity 0.5 < drag_pen/drag_bb < 2.

### 1.2 Numbers re-derived from the artifact (not re-read from the report)

Gradient matrix, all 3 groups × 2 objectives × 2 FD step sizes = 36 comparisons:

| group | rows | worst rel err, coarse FD step | worst rel err, fine FD step | > 1e-2 | degenerate rows |
|---|---|---|---|---|---|
| `gradient_matrix` (main, 30×96, D=8, 200 steps) | 6 | 1.770e-03 | 4.430e-04 | 0 | 0 |
| `resolution_scan` (22×73, D=5.25, τ=0.6) | 6 | 5.202e-03 | 1.309e-03 | 0 | 0 |
| `horizon_scan` (30×96, D=8, 100 steps) | 6 | 2.140e-04 | 5.300e-05 | 0 | 0 |

Non-degeneracy: every AD gradient and every FD difference is > 1e-3 in magnitude,
and `|dD/dθ − dL/dθ| > 1e-3` in every row (a zero-vs-zero comparison would have been
vacuous — it is not).

Exact-solution anchors of the force probes (these are what stop AD-vs-FD from being
trivially self-consistent):

| anchor | quantity | reference | got | rel err |
|---|---|---|---|---|
| plane Couette (moving wall, 22×8, 10 000 steps) | max rel err of u(y) | exact linear profile | — | 1.182e-11 |
| plane Couette | wall force | μU L/H = 0.002 | −0.0020000000001 | 3.853e-11 |
| plane Poiseuille (static BB, 22×8, 10 000 steps) | max rel err of u(y) | exact parabola | — | 2.306e-03 |
| plane Poiseuille | per-wall force | F H L/2 = 0.0016 | 0.0016000000000 | 1.479e-11 |

Sign conventions are asserted in code (Couette: fluid resists the moving wall;
Poiseuille: fluid drags both walls downstream), both `sign_consistent: true`.

### 1.3 `repro_cmd` really reproduces (audit re-run, independent process)

All four documented repro commands were executed by the auditor against the subject
work tree with `--out /tmp/aud20151{a,b,c,d}`; every block reproduces the delivered
JSON **bit-for-bit to double precision**:

| documented command | rows compared | result |
|---|---|---|
| `--part grad_main --grad-ny 30 --grad-nx 96 --grad-dcells 8 --grad-steps 200` | 6 | all `MATCH` (Δ = 0.0) |
| `--part grad_coarse --grad-ny 26 --grad-nx 88 --grad-dcells 7 --grad-tau 0.6 --grad-steps 200` | 6 | all `MATCH` |
| `--part grad_horizon --grad-ny 30 --grad-nx 96 --grad-dcells 8 --grad-steps 200` | 6 | all `MATCH` |
| `--part anchors --anchor-ny 22 --anchor-steps 10000 --dfg-dcells 8 --dfg-re 20 --dfg-steps 10000 --dfg-sample-from 7000` | 12 scalars | all `MATCH` |

The `AD ≈ FD` claim is therefore reproducible and is **not** an artifact of a single
lattice, horizon or seed.

### 1.4 The external benchmark fails — and the failure is disclosed

DFG cylinder (`anchors.dfg`, executed as **2-D-1**: Re = 20, U_mean = 0.2, D = 8,
35×176, 24.2 % blockage, 10 000 steps):

| quantity | reference | got | rel err |
|---|---|---|---|
| C_D | 5.57–5.59 (2-D-1, Re=20) | 74.83 | **1241 %** |
| C_L | 0.0108 (2-D-1) | 9.68e-14 | 100 % |

C_D grows monotonically over the sampled window 51.19 → 104.70 (60 samples,
steps 7000–10000): the run has not reached a steady state and the force is not an
equilibrium force. The report states this openly as **honest FAIL** — that is correct
behaviour, and it is why L4 is *not* claimed.

---

## 2. Findings (severity-ranked)

### F1 — MEDIUM (report correctness, two sub-items): the DFG section is not self-consistent and its diagnosis is unbacked

1. The report's DFG lines label the run as **"DFG 2-D-2 … Re=100"** with reference
   **3.22–3.24**, but the executed run is **2-D-1 (Re = 20)** — `anchors.dfg.params.re
   = 20.0, um = 0.2, D_cells = 8` in the delivered artifact. The artefact's own
   reference is 5.57–5.59, and the printed **"rel err 1241.08 %"** is 74.83/5.58−1, i.e.
   the percentage belongs to the 2-D-1 range it is *not* printed next to; against the
   printed 3.23 the same data gives 2217 %. Likewise the C_L row compares against
   "~1.0" (the 2-D-2 amplitude), not against the 2-D-1 value 0.0108 that the artifact
   uses.
2. The stated root cause — *"density inside the channel grows to `rho_max = nan` at
   `u_max = nan`"* — is **not supported by the delivered artifact**: the artifact has
   no `rho_max`/`u_max` field, and its last-step field snapshot (`field_snapshot.ux/uy`,
   18×88) contains **zero NaNs** with |u|_max = 0.467 / 0.195. Classification of that
   claim: **WITHHELD** (unbacked by the shipped evidence), while the FAIL conclusion
   itself is sound.

Impact: no change to pass/fail of the gradient matrix; both defects are in the
absolute-force section. A reader cannot reconcile §4's numbers as printed.

### F2 — MEDIUM (report correctness): stale parameter/step labels in §3/§5

Header labels contradict the delivered artifact and the documented commands, which the
auditor re-ran bit-for-bit: §5 says *"main resolution, 1200 steps"* (actual 200),
*"second lattice resolution (D = 8 cells)"* (actual D = 5.25, 22×73),
*"second time horizon (800 steps)"* (actual 100). §3 appends a second row-set
(*Couette/Poiseuille 34×8, 20 000 steps*; *DFG 2-D-2 43×220, D=10, Re=100, τ=0.53,
60 000 steps*) that was **never executed** — the artifact's anchors are 22×8 / 10 000
steps, and the DFG run is Re=20 / 35×176 / 10 000 steps. The results tables themselves
agree with the artifact; the labels do not.

### F3 — LOW-MEDIUM (artifact provenance defect in the driver)

`payload.setdefault("lattice", …)` (run_gap_diff_boundary_grad.py:~470) records only
the lattice of the **first** `--part` invocation that created the JSON. Consequence:
the delivered `lattice.main` says 44×140 / D=10 / u0=0.2, while the `gradient_matrix`
in the same file is 30×96 / D=8 / u0=0.25 (reproduced). Any downstream consumer that
reads `lattice` to describe the matrix inherits wrong geometry. Minimal fix: assign
(rather than `setdefault`) `lattice[block]` inside each block, or store the lattice on
each row (rows of `resolution_scan` already carry `ny/nx/d_cells/u0/tau/re`).

### F4 — LOW (reporting scope)

The headline *"worst relative error over the whole matrix is 1.31e-03"* is the maximum
over the **finer** FD step only (the driver's `summary` aggregates `rel_err_*`, which
is the h=1e-3 value). Over both declared step sizes the worst is **5.20e-03**
(`resolution_scan`) / 1.77e-03 (`gradient_matrix`). The 1e-2 gate still passes with a
margin ≥ 1.9×, so this is an under-stated uncertainty, not a false claim.

### F5 — MEDIUM (substantive limit, partially disclosed)

The *absolute* force level is anchored only for the static/moving bounce-back probes in
plane channels. The **volume-penalisation probe has no external anchor at all**: its
only cross-check is against bounce-back on the same geometry
(`closure_sensitivity` ratios 0.885 / 1.020 / 1.222 for interface widths 0.4 / 0.8 /
1.2 cells), i.e. a self-comparison between two closures. Combined with the DFG FAIL,
no statement of the form "penalised-body drag is accurate to x %" is supportable, and
the report correctly does not make one — but the gap should be stated as *"no external
anchor for the penalisation force"*, not only as a DFG outlet problem.

### F6 — MEDIUM (delivery, blocking for any downstream use)

* `git status` in `/root/tp-agent/work/m20070`: all six deliverables **untracked**
  (`docs/GAP-DIFFLBM-BOUNDARY-GRAD.md`, `src/tensorlbm/diff_boundary_grad.py`,
  `tests/test_diff_boundary_grad.py`, 3 scripts, `artifacts/GAP-DIFFLBM-BOUNDARY-GRAD/`).
* `git cat-file -e origin/main:src/tensorlbm/diff_boundary_grad.py` → **absent**;
  no `agent/m20070` branch on the remote. The capability exists only in that work tree.
* `SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='GAP-DIFFLBM-BOUNDARY-GRAD'`
  → **0**; `tensor_xval_cases` for this anchor → **0**. The anchor has no registered
  benchmark row, so the "唯一裁判" table has nothing to judge — this audit is the
  substitute, and the numeric judgement above is therefore recorded in the verdict
  evidence itself.

This is why the audit evidence bundle carries a `sha256`-identified snapshot: without
it, the pointers cannot be machine-resolved, and the capability is not delivered even
though it is real.

---

## 3. Independence limitations

* The auditor re-ran the subject's own scripts and re-derived the subject's own
  artifact arithmetic. That is an **independent execution**, not an independent
  re-implementation: a shared bug in both the module and its FD reference (e.g. a
  wrong but consistent force probe) would survive this check. Only the exact-solution
  anchors (V1.2) mitigate this, and they cover the bounce-back probe in channels — not
  the penalisation probe, not a body in a channel, not transient-force gradients.
* The DFG divergence was **not** repaired or root-caused by the auditor (out of audit
  scope): the "mass accumulation at the copy outlet" mechanism is plausible (the outlet
  literally copies the neighbouring populations, `_inlet_outlet`, lines 254–263) but is
  not proven here; F1.2 classifies the report's NaN diagnosis as unbacked, which is not
  the same as proving a different mechanism.
* The two "resolution" groups are not a pure refinement study: D and u0 change together
  with Re held at 40, and Re on the coarse lattice is met by rescaling u0 — so the
  agreement is a robustness check, not an observed convergence order.

## 4. Checks not run (must not be read as passed)

* No re-run of a 1200-step / D=10 matrix (that configuration exists only as stale
  metadata, F3); the delivered matrix is a 200-step matrix and was reproduced as such.
* No independent implementation of the LBM operator (see §3).
* No pressure/anti-reflecting outlet, no attempt to recover DFG C_D.
* No gradient convergence study in lattice resolution or FD step order
  (`O(h²)` behaviour is *assumed*, only two step sizes are reported).
* No verification of the report's stability claim ("stable ≥ 3000 steps at every
  interface width") — the artifact's `closure_sensitivity` block is a 300-step scan.
* No platform/API integration check exists to run: there is no caller (F6).

## 5. Accuracy statement against the 15 % / 3 % standard

* Exact-solution anchors: **3.85e-11 / 1.48e-11** on the wall force, 1.18e-11 /
  2.31e-03 on the profile → ≪ 3 % (machine-precision level).
* AD-vs-FD (the actual deliverable): worst **5.20e-03** → ≪ 3 %.
* External literature benchmark (DFG C_D): **FAIL at 1241 %** → **not** < 15 %.
  Therefore: *the delivered capability is a verified differentiable-boundary-gradient
  implementation (L2), not a validated body-force accuracy capability.*

## 6. Recommendation to the original mission (for the next round, not blocking L2)

1. Fix §3/§4/§5 labels and the DFG case identity (2-D-1, Re=20, refs 5.57–5.59 /
   0.0108); restate the §4 percentages accordingly; delete or mark the never-executed
   parameter rows.
2. Either back the NaN diagnosis with a recorded `rho_max/u_max` series or replace it
   with what the artifact shows (C_D 51 → 105 monotone, no steady state).
3. Record `lattice` per block (F3); report `worst_rel_err` over both FD steps (F4).
4. **Commit and land the deliverable**, and register one `tensor_benchmark_cases` row
   for the anchor (AD-vs-FD worst ≤ 1e-2, plus the exact-anchor numbers) so the gate
   has a machine-readable reference.
5. Add a pressure/anti-reflecting outlet (or a shorter domain with a sponge) and one
   external anchor for the penalisation force — that is the only route from L2 to L4.