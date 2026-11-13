# XVAL-LBM-ROUND2-METHOD-SELECTION - round 2 (open-source cross-benchmark, method selection)

**Anchor:** `XVAL-LBM-ROUND2-METHOD-SELECTION` . mission 18241
**OSS reference:** `Maarten-vd-Sande/lbm @ 573a01c4` (source at `/root/oss-cache/Maarten-vd-Sande_lbm`,
sha256 per file in `xval/benchmark_method_selection/oss_crosscheck.json` and in
`predeclared_round2.json::round1_reference::oss_source_sha256`; the source was never modified)
**Repo arm:** TensorLBM `src/tensorlbm/solver.py`, `src/tensorlbm/boundary/boundaries.py`,
`xval/benchmark_method_selection/run_arm.py`
**Predeclaration (written before any round-2 run):** `xval/benchmark_method_selection/predeclared_round2.json`
**Machine-readable results:** `table_round2.csv`, `summary_round2.json`, `oss_outlet_bc_xval.json`

> **Honest status line first.** The predeclared round-2 question (does the round-1
> selection gap close once the grid is resolved?) is **NOT ANSWERED in this
> session**: the two new resolution rungs (D=14, D=20) and the missing factorial
> cell (A6 at D=10) were designed, instrumented and submitted, but they did not
> execute - see the failure ledger in section 6. Everything that *did* execute is
> below, with numbers taken from artifacts, and the "not_run" items are listed
> rather than filled in.

## 1 Problem and reference provenance

Round 1 (`docs/XVAL_LBM_BENCHMARK_METHOD_SELECTION.md`) compared TensorLBM's recorded
method selection (**MRT collision + 6D downstream absorbing sponge**, A1) against the
OSS-practice selection (**single-relaxation BGK + plain zero-gradient outlet**, A2) on a
square cylinder at Re = 200 and reported:

| round-1 finding | value | source |
|---|---|---|
| dCd (A2 vs A1, D=10) | -9.18 % | `summary.json::table.A2_D10.delta_vs_A1_pct` |
| dSt | +4.52 % | same |
| dCl_amp | **-46.57 %** | same |
| D=14 rung | **never ran** (container background-process lifetime) | round-1 section 5.1 |
| 2x2 factorial | **incomplete** (BGK+sponge cell missing) | round-1 section 5.3 |

So the round-1 verdict ("H1 not established") is confounded: it mixes a *scheme*
difference with an *unconverged grid*, and it cannot separate "collision" from
"sponge". Round 2 attacks exactly that: same case, same harness, the *completed* 2x2
factorial and a three-rung resolution ladder.

**What the OSS side actually does** (read from source, not from memory):

| OSS fact | where (file:lines at the pinned commit) |
|---|---|
| collision = single relaxation `fout = fin + omega*(equi-fin)`, one global `omega = [1.8]` in the example (tau = 1/omega = 0.5556) | `lbm/streamcollide.py:147-157`; `examples/cylinder_flow.py:26` |
| outflow = **zero-gradient copy** `fin[n, ix, iy] = fin[n, ix-1, iy]` for n in 3..5, applied **inside the collision loop (pre-stream)** | `lbm/streamcollide.py:211-216` |
| inflow = Zou-He density closure `rho = (sum(0..2) + 2*sum(3..5)) / (1-u_x)` | `lbm/streamcollide.py:219-234` |
| no force / no drag / no Strouhal post-processing anywhere | grep over `lbm/ examples/` -> no hit |
| the example places **5 random circular obstacles** and animates them; no seed, no statistical convergence, no verification case | `examples/cylinder_flow.py:34-41` |
| no stability monitor, no sponge, no absorbing layer | no hit in source |

That is the porting question in one line: the OSS repo is a *demo*, and its
"method selection" is "one global BGK omega, plain outlet". The recorded TensorLBM
selection adds two knobs (MRT + sponge). Round 2 asks which of the two knobs is
responsible for the round-1 gap.

## 2 Method and implementation (reproducible)

```
# predeclaration (committed before the runs; nothing below changed a threshold)
cat xval/benchmark_method_selection/predeclared_round2.json
# one rung, all arms of that rung in parallel inside one job
THREADS=8  bash xval/benchmark_method_selection/run_round2.sh 14 A1 A2 A4 A6 A3 A5
THREADS=16 bash xval/benchmark_method_selection/run_round2.sh 20 A1 A2 A4 A6
THREADS=8  bash xval/benchmark_method_selection/run_round2.sh 10 A6 A2rep   # missing cell + determinism
# aggregate + figures (gated only by predeclared_round2.json)
python xval/benchmark_method_selection/analyze_round2.py
# outlet-BC instrument (round-1 gap: boundaries were excluded from the kernel XVAL)
python xval/benchmark_method_selection/oss_outlet_bc_xval.py
# tests (no cluster needed)
python -m pytest tests/test_xval_round2_method_selection.py -q
```

**Arms (only the selection knobs change; BC, force sampler, post-processing shared):**

| arm | collision | sponge | role |
|---|---|---|---|
| A1 | MRT | on | reference = recorded TensorLBM selection |
| A4 | MRT | off | sponge isolated |
| **A6 (new)** | BGK | on | factorial completion - collision isolated from the other side |
| A2 | BGK | off | OSS single-relaxation selection |
| A3 / A5 | TRT lambda=3/16 / lambda re-derived | off | TRT selection diagnostics |

A6 required a harness capability that did not exist: `collide_bgk` accepted only a
scalar tau, so a BGK arm could not carry a sponge. It now takes the same optional
`tau_field` as `collide_mrt` (`src/tensorlbm/solver.py::collide_bgk`), pinned by
`tests/test_xval_round2_method_selection.py` to be **bit-identical** to the uniform
path when `tau_field is None`, and exact against cell-wise explicit relaxation when
it is given.

## 3 Parameter table (complete dimensionless set)

| quantity | value | note |
|---|---|---|
| Re = U*D/nu | 200 | fixed, all arms, all rungs |
| blockage D/H | 1/20 (domain 20D x 20D, body centre 6D) | fixed |
| Ma = U/c_s | 0.0866 (U = 0.05) | fixed |
| tau = 0.5 + 3nu | 0.5075 | same for MRT/BGK/TRT (matched viscosity) |
| TRT lambda | 3/16 (A3) -> tau_minus = 25.5 ; min(3/16, 1.5(tau-0.5)) (A5) -> tau_minus = 2.0 | selection diagnostic |
| sponge | alpha = 10, length 6D, quadratic profile | only when the arm says "on" |
| seeding | 2 periods of St_seed = 0.13 at 10 % U, then 15 shedding periods | all arms, all rungs |
| steps (predeclared) | 21259 (D=10), 29763 (D=14), 42518 (D=20) | `scheduling()`, pinned by test |
| dtype / device | float64 / CPU | - |
| outlet-BC instrument | nx=240, ny=40, periodic y, tau=0.55, pulse A=1e-2, w=4 at x=60, probe x=100, u0=0 | `oss_outlet_bc_xval.json::config` |

## 4 Comparison tables (ref vs got, error marked)

### 4-A Selection comparison, D=10 (ref = A1; numbers read programmatically from `table_round2.csv`)

| arm | collision | sponge | Cd (ref 1.6143) | dCd % | St (ref 0.1690) | dSt % | Cl_amp (ref 0.6579) | dCl_amp % | gate (<15 %) |
|---|---|---|---|---|---|---|---|---|---|
| A1 | mrt | on | 1.6143 | +0.00 | 0.1690 | +0.00 | 0.6579 | +0.00 | - |
| A2 | bgk | off | 1.4661 | -9.18 | 0.1766 | +4.52 | 0.3516 | -46.57 | FAIL |
| A4 | mrt | off | 1.6424 | +1.74 | 0.1718 | +1.64 | 0.5566 | -15.40 | FAIL |

Verdict at D=10 (unchanged from round 1, recomputed by the round-2 aggregator):
A2 fails the gate on `cl_amp` (-46.57 %), A4 fails it too (-15.40 %) - so at D=10
**neither the collision knob alone nor the sponge knob alone is exonerated**: A4 is
inside 15 % on Cd and St but 15.4 % outside on Cl_amp.

**A6 (BGK+sponge) at D=10: NOT RUN** (`fail_stage = not_run`, see section 6). Without
it the round-1 question "is the 46 % a collision effect or a sponge effect?" stays
formally open; the round-1 section 5.3 differencing (collision effect = A4->A2 =
-10.73 % Cd, sponge effect = A1->A4 = +1.74 % Cd) is a *one-at-a-time* contrast,
valid only if the interaction is zero - and A6 exists precisely to test that.

### 4-B Code-level comparison K1/K2 (ref = executed OSS source; round 1, re-verified by the test suite)

| check | rel-L2 | verdict |
|---|---|---|
| K1 collision (OSS `lbm_collision` float32 vs TensorLBM `collide_bgk` float64, n=48, tau=0.5075) | 1.18e-07 | PASS <3 % (rounding level) |
| K2 streaming (interior, 1-cell frame excluded) | 2.71e-08 | PASS <3 % |
| K0 direction map | all 9 directions agree | PASS |

### 4-C Code-level comparison K3 (outlet boundary - new this round)

Instrument: an acoustic pulse in a quiescent box, identical quiet-inlet sponge in
every configuration, so **only the outlet operator differs**; R = reflected peak /
incident peak at the probe (x=100). Source: `oss_outlet_bc_xval.json`.

| outlet operator | incident peak | reflected peak | R |
|---|---|---|---|
| oss_zero_gradient | 8.391e-04 | 2.730e-04 | 0.3254 |
| far_field | 4.897e-04 | 1.224e-06 | 0.0025 |
| far_field_sponge | 4.897e-04 | 7.288e-07 | 0.0015 |
| absorbing_both_ends | 4.897e-04 | 7.622e-07 | 0.0016 |
| wall | 8.391e-04 | 5.408e-04 | 0.6445 |

Controls: **FAILED** (`controls_ok = false`, incident-peak
spread 1.713x, rigid-wall control R = 0.645 < 1).
The incident peak is *not* identical across configurations even though the outlet
cannot physically influence it before the wave arrives: the initial condition (a pure
density pulse at zero velocity) is not in local equilibrium and emits a startup
transient that reaches the outlet column within a few steps and is fed back
differently by each operator. The R column is therefore **indicative only, not
adjudicating**; the stable part is the ordering
(OSS zero-gradient 0.325 >> far-field 0.0025 ~
far-field+sponge 0.0015). This is recorded as an **instrument
failure** - an honest negative result, not evidence.

## 5 Convergence / parameter sensitivity (at least two)

1. **Grid convergence: NOT RUN.** D=14 and D=20 for A1/A2/A4/A6 (plus A3/A5 at D=14)
   were predeclared, instrumented and submitted; none executed (section 6).
   Consequence, stated plainly: **every KPI in 4-A is resolution-unconverged** and the
   round-1 caveat is not lifted. The ladder code path is pinned by
   `tests/test_xval_round2_method_selection.py::test_schedule_is_arm_independent_and_monotone_in_D`
   (step counts 21259 / 29763 / 42518), so a relay run reproduces it exactly.
2. **Sponge vs collision separability: PARTIAL.** At D=10 the one-at-a-time contrast
   gives collision effect -10.73 % Cd and sponge effect +1.74 % Cd (round-1 section
   5.3, recomputed identically by `analyze_round2.py::factorial` as soon as the A6 row
   exists). The interaction term - the quantity that decides whether the two knobs may
   be discussed separately - **needs the A6 row and is therefore not estimated**.
3. **TRT lambda sensitivity (round 1, still the strongest measured sensitivity):**
   literal OSS magic lambda = 3/16 at tau_plus = 0.5075 implies tau_minus = 25.5 ->
   divergence at step 600/21259 (A3, `runs/A3_D10/fail.json`); re-derived lambda with
   tau_minus = 2.0 diverges at 5800/21259 (A5, `runs/A5_D10/fail.json`). The literal
   OSS-style TRT transplant is **not usable at this lattice viscosity** - a porting
   result that generalises beyond this case.
4. **Outlet absorption (new, indicative only):** see 4-C - the zero-gradient outlet
   reflects an order of magnitude more than the far-field outlet, consistent with the
   recorded selection keeping a sponge, but the instrument controls failed so this is
   *not* used to justify the selection.

## 6 Failure modes and artifacts (honest-FAIL ledger)

| item | status | measured cause | artifact |
|---|---|---|---|
| A6 @ D=10 (missing factorial cell) | **not_run** | control node's `run_shell` hard-caps a command at **120 s** (process group reaped: `exit=-9, TIMEOUT after 120s`); one D=10 arm needs about 676 s at 32 threads on this heavily loaded node (load ~67 of 128) | `runs/A6_D10/` absent; `predeclared_round2.json` |
| D=14 rung (A1,A2,A4,A6,A3,A5) | **not_run** | same 120 s cap locally; submitted to the cluster as `cj1848` (48 cores, 7200 s) - still `queued` after 10 min with **310 jobs ahead**, nothing dispatched on either node | job id recorded here; no artifact |
| D=20 rung (A1,A2,A4,A6) | **not_run** | as above (`cj1849`, 64 cores) | - |
| A6 + A2rep @ D=10 (determinism re-run) | **not_run** | as above (`cj1847`, 32 cores) | - |
| K3 outlet instrument | **instrument FAILED** | startup transient contaminates the incident normalisation; controls not satisfied (4-C) | `oss_outlet_bc_xval.json` (`controls_ok: false`, deviations block) |
| A3 @ D=10 | **blow-up (measured)** | literal lambda = 3/16 => tau_minus = 25.5 | `runs/A3_D10/fail.json` |
| A5 @ D=10 | **blow-up (measured)** | tau_minus = 2.0 still outside the stable window at tau_plus = 0.5075 | `runs/A5_D10/fail.json` |

**Ecosystem friction (recorded, not worked around):** the compute path this anchor
depends on is structurally starved in this window - the control node allows at most
120 s per command, and the cluster queue has ~310 jobs ahead while both 128-core nodes
report load ~43 yet dispatch nothing. A 25-minute single-arm run therefore has no
execution surface. Round-2 artifacts are consequently: the predeclaration, the harness
(A6 + BGK sponge support), the aggregation pipeline, the outlet instrument with its
honest failure, and the tests - a *ready-to-execute* round 2, not a concluded one.

## 7 Repo vs OSS difference list + porting trade-offs

Full table: `xval/DIFF_OSS_VS_TENSORLBM_ROUND2.md`. Summary:

| dimension | OSS (`Maarten-vd-Sande/lbm`) | TensorLBM (recorded selection) | porting verdict |
|---|---|---|---|
| collision | one global BGK omega (tau=0.5556 in the example) | MRT with s_e=1.64, s_eps=1.54, s_q=1.7; BGK/TRT available | **keep MRT**: same viscosity contract, extra stability dials; BGK is the degenerate case (arm A2) |
| outlet | zero-gradient copy, pre-stream | far-field outlet (+ 6D sponge in A1) | **keep far-field + sponge**: K3 ordering (indicative) says the OSS outlet reflects about an order of magnitude more |
| TRT magic parameter | not implemented | lambda = 3/16 transplant diverges at tau = 0.5075 | **do not port literally**; re-derived lambda still diverges -> stay with MRT here |
| force / KPI | none | post-stream momentum exchange (sign defect already recorded) | **cannot port**; OSS has no measurable KPI |
| workflow | random obstacles, animation, no seed, no convergence criterion | fixed case, seeded, predeclared schedule, stability monitor | **do not port**: the OSS example is a demo, not a benchmark |
| verification | none in repo | round-1 kernel XVAL <1e-7 vs OSS source | **keep** |

**Top-line recommendation (unchanged by round 2, now with the missing cell
instrumented but unexecuted):** keep MRT + sponge; the OSS selection is defensible
only as the degenerate BGK case, and the round-1 46 % Cl_amp gap must not be
attributed to either knob until A6 (BGK+sponge) runs. **Two-round relay:** fetch
`cj1847`/`cj1848`/`cj1849` (or re-run `run_round2.sh` when the queue drains) then
`python analyze_round2.py` - it fills the A6 row in 4-A, the ladder in 5 and the
interaction term without changing any threshold.

## 8 Error adjudication (itemised, explicit)

1. **Selection XVAL vs OSS (15 % criterion)** - answer: **not adjudicated**. At D=10
   only A2 (-46.57 % Cl_amp) and A4 (-15.40 % Cl_amp) exist; A6, D=14 and D=20 are
   not_run. Among the rows that did run, Cd and St are inside 15 % (A2: -9.18 % /
   +4.52 %; A4: +1.74 % / +1.64 %) and Cl_amp is outside.
2. **Code-level XVAL (ref = executed OSS source)** - answer: **yes, <3 %** (measured
   rel-L2 1.18e-07 / 2.71e-08, i.e. float32 rounding level).
3. **Outlet boundary XVAL** - answer: **no (instrument failed)**, see 4-C; R values
   are not used as evidence.
4. **Convergence** - answer: **not demonstrated** (D=14/20 not_run), so no
   convergence claim is made.
