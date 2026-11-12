# Board audit — m15904 auditing mission **m15659** (`GAP-LBM-T5-ICE-COUPLING`, TensorLBM)

* **Audited mission**: m15659 · repo TensorLBM · status `succeeded`, `accepted=true`, PR **#1842** merged (merge commit `79126b06`)
* **Mission branch head**: `77045799` — "T5 ice coupling round 1: 3-D LBM-FEM-DEM load-exchange interface + invariant gates" (15 files, +2611)
* **Auditor**: m15904 (board-audit), worktree `/root/tp-agent/work/m15904` on `agent/m15904`, based on main `53046bcc`
* **Task type**: cross-check / evidence audit — *not* a redo. Nothing in the mission's deliverable was modified by this audit.
* **Nature of the claim**: round 1 of a two-round node whose contract is explicitly *"dependency audit + coupling-interface design only; a P1 periodic single-phase prototype does not count as validation"*.

---

## 1. What the mission actually delivered (inventory, verified on main)

| artefact | path | size | present on main |
|---|---|---|---|
| interface reference implementation | `src/tensorlbm/three_way/interface3d.py` (`Surface3D` L77, `HexGrid3D` L310, `distribute_point_consistent` L386, `distribute_surface` L400, `gather_point_velocity` L409, `project_to_beam` L443, `CouplingClock` L473, `added_mass_ratio` L508, `explicit_staggered_stable` L516, `relaxation_bound` L521, `staggered_spectral_radius` L548, `march_one_dof` L560, `InterfaceLedger` L577, `exchange_cycle` L610) | 26808 B | ✅ |
| gate driver (8 gates, JSON+CSV+5 figures) | `scripts/three_way_interface3d_gates.py` (`gate_i1` L78 … `gate_clock` L423, `main` L614) | 21337 B | ✅ |
| report generator (programmatic number injection) | `scripts/three_way_interface3d_report.py` | 289 L | ✅ |
| invariant tests | `tests/test_three_way_interface3d.py` (14 tests, L57–L237) | 243 L | ✅ |
| dependency audit | `docs/GAP-LBM-T5-ICE-COUPLING_dependency_audit.md` (gaps G5-01…G5-08, L130–L137) | 163 L | ✅ |
| interface design | `docs/GAP-LBM-T5-ICE-COUPLING_coupling_interface_design.md` | 203 L | ✅ |
| generated gate tables | `docs/generated/GAP-LBM-T5-ICE-COUPLING_gate_tables.md` | 81 L | ✅ |
| **run artefacts mirror** (`gates.json`, `gates.csv`, 5 PNG, pytest log) | `artifacts/three_way_interface3d/**` | 8 files | ❌ **absent on main** (see F1) |

Everything the mission *claims to have coded* is on main. Everything it claims to have *produced as a run artefact* is on the mission branch, survives in the mission worktree, and is regenerated bit-for-bit by the auditor (F1).

---

## 2. Reproduction (this is the core evidence, not the report prose)

Run in this auditor worktree, single-threaded BLAS, on the merged main tree:

| command | result | auditor wall clock | mission claim |
|---|---|---|---|
| `OMP_NUM_THREADS=1 PYTHONPATH=src python3 scripts/three_way_interface3d_gates.py` | **8/8 gates PASS** | 3.6 s | 8/8, ~4 s ✅ |
| `python3 -m pytest tests/test_three_way_interface3d.py -q` | **14 passed** | 5.75 s | 14 passed, 5.2 s ✅ |
| `python3 -m pytest tests/test_three_way_core.py -q` | **13 passed** | 61.50 s | 13 passed, 62.9 s ✅ |
| `python3 -m pytest tests/test_three_way_gates.py -q` | **6 passed** | 60.80 s | 6 passed, 66.2 s ✅ |

**Byte-level cross-check of the numbers.** The mission's artefact was extracted from the branch
object (`git show 77045799:artifacts/three_way_interface3d/gates.json`,
sha256 `346570164b0eccbafb627d4adc374014ffec87e4410c150a21195df168ddbd64`) and structurally
diffed against the auditor's reproduction
(sha256 `722496b7bd14dec7c950df0220d7b7bd33bb975ab7b39d3c7aa883ba7690999a`):
**173 numeric leaves compared, 0 differences** (`artifacts/audit/m15904/evidence.json`
→ `repro.numeric_diffs = []`). The only changed text in the tree after re-running is the
runtime stamp in the generated table (`3.9 s → 3.6 s`). The artefacts are real and the run is
deterministic.

---

## 3. Does the reported physics match the code? (claim-by-claim)

Baseline referee = closed-form / analytic values. The auditor re-derived each reference
**independently** of the delivered code (separate arithmetic in
`artifacts/audit/m15904/make_audit_evidence.py`, section 4) instead of trusting `ref` fields:

| quantity | mission `ref` | auditor independent value | agreement |
|---|---|---|---|
| `EI` (E=2.1e11, b=0.1, h=0.01) | 1750 N·m² | `b h³/12·E` = 1750.0 | exact |
| cantilever tip load `FL³/3EI` (F=100 N) | 1.9047619e-2 m | 1.9047619e-2 m | 0 rel. diff |
| cantilever UDL `qL⁴/8EI` (q=100 N/m) | 7.1428571e-3 m | 7.1428571e-3 m | 0 rel. diff |
| cantilever `f₁` (`1.875104²/(2πL²)·√(EI/ρA)`) | 8.355165 Hz | 8.355164 Hz | 1e-7 |
| added-mass threshold `μ_crit` | 1.0 | 1.0 (Causin–Gerbeau–Nobile 2005) | bisection gives 0.999975 |
| relaxation bound `2/(1+μ)`, μ=2 | 0.666667 | 0.666667 | exact |
| hemisphere pressure resultant `pπa²` | π | π | exact |
| closed surface `∮n dA` | 0 | 0 | measured 7.6e-18 |

Gate-by-gate (values from the mission artefact, **reproduced 0-ULP** by the auditor):

| gate | invariant it claims to gate | measured (worst / finest) | pass | falsifiable? |
|---|---|---|---|---|
| I1 | surface quadrature → `pπa²` | 2.55 % → 0.641 % → 0.1606 %, observed order **1.99 / 2.00** | ✅ | yes — three resolutions, order reported |
| I2 | consistent Q1 transfer conserves resultant **and** moment | resultant rel 0 / 1.1e-16, moment norm 5.7e-17 / 1.6e-17 | ✅ | yes — moment is not implied by resultant |
| I3 | work conjugacy `Σfᵢuᵢ = ∫t·u dA` for affine fields | identity err ≤ 1.72e-16; quadrature 4.04e-4 → 1.01e-4 → 2.53e-5 (order 2.0) | ✅ | yes — two independent routes |
| I4 | consistent projection vs nearest-node control | consistent moment err ≤ 1.12e-16 vs lumped 1.0395 / 0.5154 / 0.1145 (order 1.01) | ✅ | yes — **negative control implemented**, which is what makes the design claim non-tautological |
| I5 | added-mass guard: explicit staggered stable iff μ<1 | marched growth 0.99946 (μ=0.5) / 2.000044 (μ=2) vs closed form 0.99999 / 2.000044; `μ_crit` bisection 0.999975; relaxation rates 0.8 / 1.2 vs `1−ω(1+μ)` | ✅ | partly — see F4 |
| I6 | hull-girder leg fed **through the 3-D interface** | tip load 5.05e-12; `f₁` 1.26e-7; panel UDL 0.174 % → 0.0434 % (order 2.0) | ✅ | yes — the physical leg is a real Hermite FEM solve (`torch.linalg.solve` on `K_global`) |
| I7 | interface work ledger closes (`W = ΔE`) | closure 5.73e-5 → 1.43e-5, order **2.000** | ✅ | yes — second order is the predicted signature of a real scheme, not of a fudge |
| clock | sub-cycling consistency | consistent mismatch 0.0 / inconsistent 0.6 / invalid raises | ✅ | yes |

Nothing in the eight gates is decorative: each has a reference, a resolution or control
comparison, and an explicit `pass` predicate (`scripts/three_way_interface3d_gates.py`
L78–L460). The nearest-node control (I4) and the analytic-vs-FEM pairs (I6) are the two checks
that make the design claim falsifiable rather than self-affirming.

**Overstatement screen.** The mission summary says conservation holds to `≤3e-17`. Measured
maxima: `I2` moment norm **5.72e-17**, `I4` consistent moment err **1.12e-16**, `I3` identity
err **1.72e-16**, resultant rel **1.11e-16**. The correct statement is `≤1.8e-16` — see F2.
Conclusions are unaffected (all are machine precision), but the quoted bound is not what was
measured.

---

## 4. Benchmark referee — the part that is *not* satisfied

| check | result |
|---|---|
| `tensor_benchmark_cases` rows for `anchor_slug='GAP-LBM-T5-ICE-COUPLING'` | **0** (table has 25 rows; no ICE / T5 / coupling row) |
| `tensorbench_results` rows for mission 15659 | **0** |
| `GET /api/v1/tensor/bench/l4-verdict/15659` | `CLAIMED_NOT_PROVEN`, `l4_benchmark_pass=false`, failed_checks = `execution_success, source_cluster_job, cluster_job_bound, ref_got_error_present, within_tolerance` |

So the node's "benchmark ref vs got" criterion has **no machine referee at all**: every
comparison in this mission is against closed-form analytic values (correctly re-derived above,
section 3), computed inside the same script that reports the error. That is legitimate
*verification* evidence but it is not a platform-recomputed benchmark row, and it cannot
constitute L4 by itself. Round 1 did not claim otherwise: the report's §⑥ and
`docs/GAP-LBM-T5-ICE-COUPLING_dependency_audit.md` L130–L137 list `G5-04` (T1 multiphase not at
criterion), `G5-05` (no 3-D shell/solid FEM), `G5-06` (no 3-D breakable DEM), `G5-07`
(interface not driven by a real 3-D LBM field), and `G5-03` (2-D gate (a) FAIL at
7.7/4.8/8.8 %, sourced in `docs/THREE_WAY_LBM_DEM_FSI.md` L104) as **open**, i.e. the mission
itself declares the flagship unverified.

---

## 5. Integration (L3) check

`grep -rn 'tensorlbm.three_way.interface3d'` over the merged tree yields exactly two consumers:
`scripts/three_way_interface3d_gates.py` L43 and `tests/test_three_way_interface3d.py` L31.
`src/tensorlbm/three_way/__init__.py` does **not** export the module, and no solver, scenario
driver, `ice_chain3d` path or platform caller references it. The interface is a standalone
kernel today — L3 is refuted by construction, and the mission does not claim it.

---

## 6. Level determination (L0–L5)

| level | requirement | verdict for m15659 |
|---|---|---|
| L0 | documents only | exceeded — real code, not prose |
| L1 | math / unit verification | **met and exceeded**: 8 analytic gates + 14 invariant tests, all reproduced |
| L2 | independent functional implementation (real code: input → compute → output) | **MET** — `interface3d.py` is a usable library module: `HexGrid3D.distribute_surface(surface, traction) → nodal force vector`, `project_to_beam(x, f, L, n_el) → Hermite DOF vector` followed by a real `K⁻¹F` solve, `march_one_dof(μ) → [x,v,a]` history, `exchange_cycle(...) → {nodal, resultant, moment}`. Auditor exercised these through both the gate driver and the 14 tests; output is numeric field data, not a print |
| L3 | system integration (into main solver / platform) | **REFUTED** — no caller (§5) |
| L4 | benchmark validation (complete case + error analysis + reproducibility) | **REFUTED** — 0 rows in `tensor_benchmark_cases` / `tensorbench_results`, platform referee `CLAIMED_NOT_PROVEN`; no converged 3-D LBM→FEM→DEM case exists; the physical load path is still analytic tractions (G5-07) |
| L5 | research/engineering achievement | not claimed, not reachable this round |

**Level = 2.** It is not L1 (the deliverable is an executable kernel with a public API and a
real FEM solve behind it, and the auditor reproduced input→compute→output), and it is not L3/L4
for the reasons above. 600 lines of report plus 5 figures do not move this — the level is the
functional substance: an interface primitives library, verified against closed form, with the
flagship chain explicitly and honestly left open.

---

## 7. Findings

**F1 · MEDIUM · run artefacts are not on main.** The mission deliberately mirrored its
gitignored `runs/` output into tracked `artifacts/three_way_interface3d/` (8 files: `gates.json`,
`gates.csv`, 5 PNG, pytest log), and did commit them on `77045799`. After merge `79126b06`
(`Merge #1842 (m15659) via merge-master`), `git ls-tree -r 79126b06 | grep artifacts/three_way_interface3d`
returns **0 files**, and main has no such directory. The evidence chain on main is therefore
broken by a platform-side merge filter, not by the mission's negligence (it is the same class of
loss recorded in sibling merges' "N 个文件被门禁滤除"). Mitigation available and used here:
`git show 77045799:…` regenerates the artefact, and the auditor's own run reproduces it 0-ULP in
3.6 s. **Recommendation:** whitelist `artifacts/**/gates.json|csv|figures/*.png` in the merge
gate, or make the mission's `run_artifacts` pointer resolution fall back to the branch object.

**F2 · LOW · precision overstatement in the mission summary.** "守恒至机器精度（≤3e-17）" vs
measured max **1.72e-16** (`I3` identity), `I4` 1.12e-16, `I2` 5.72e-17. Claim is tighter than
reality by ~6×. Outcome unchanged (all machine precision), but the number in the summary does not
come from the artefact. Evidence: `artifacts/audit/m15904/evidence.json` → `claim_vs_measured`,
`bound_claim_holds=false`.

**F3 · MEDIUM · anchor has no machine referee (platform ledger gap).** 0 rows in
`tensor_benchmark_cases` for `GAP-LBM-T5-ICE-COUPLING` (25-row table), 0 `tensorbench_results`
rows for 15659. The task-board form of this node ("benchmark 数值对照 ref vs got") cannot be
machine-adjudicated until a case row exists. **Recommendation:** register the interface
invariants + the `FL³/3EI`, `qL⁴/8EI`, `f₁`, `pπa²`, `μ_crit=1` keys with tolerances
(`≤3 %` is comfortably met — measured 0.16 %/0.043 %/1.3e-7) so that round 2's coupled case has a
referee to be measured against.

**F4 · LOW · I5's "closed form" is the spectral radius of the same discretised scheme.**
`march_one_dof` and `staggered_spectral_radius` share `_staggered_matrix`
(`interface3d.py` L526–L575), so their 6.7e-16 agreement validates the marching *implementation*,
not the continuum threshold. The continuum claim (`μ_crit → 1`) is nonetheless separately
gated by the bisection (0.999975) and the relaxation-rate branch (0.8 / 1.2 vs `1−ω(1+μ)`), so the
gate is not circular — but the wording in the design doc should say "spectrum of the 3×3
recursion", not "closed form" without qualification.

**F5 · LOW · thresholds are in-code, not pre-registered in a design artefact of record.**
`pass_i*` tolerances (`<5e-2`, `<1e-10`, `<1e-6`, order `>1.8`) live inside
`scripts/three_way_interface3d_gates.py`. For an S3-grade claim they should be anchored to the
design document/benchmark case row (F3), otherwise the tolerance can be moved after seeing the
result on a later round. No evidence of tuning was found (the observed order 2.0 is the *design*
signature, not a fitted value), so this is a process note, not a correctness defect.

---

## 8. Checks not run (explicitly unknown, not "passed")

* The dependency audit's *background* citations were spot-checked, not re-run: `sphere_re100`
  C_D = 1.12198 / 1.11946 vs Schiller–Naumann 1.09173 (`G5-07` cheapest next step) — the cited
  artefacts were not opened; 2-D gate (a) FAIL numbers 7.7/4.8/8.8 % were verified to exist in
  `docs/THREE_WAY_LBM_DEM_FSI.md` L104/L117–L119 but the 2-D campaign was not re-executed.
* No cluster job was submitted for this audit (all runs ≤ 62 s); therefore no
  `cluster_job_bound` evidence exists for m15659 — consistent with the platform's
  `CLAIMED_NOT_PROVEN`.
* The `torch` version axis (mission used 2.7.1, auditor 2.7.1) was not varied; reproducibility
  across versions/dtypes is untested.
* Anti-cheat: the auditor did **not** inspect whether `docs/generated/GAP-LBM-T5-ICE-COUPLING_gate_tables.md`
  numbers were produced by the script (they match to 0 ULP and the generator `three_way_interface3d_report.py`
  reads `gates.json`, so hand-copying is implausible) — but this was not proven line-by-line.

---

## 9. Independence limitations

The audit ran on the same host and the same merged tree as the mission's own verification, with
the same `torch 2.7.1`; it therefore shares the environment. The one genuine independence lever
used was **re-derivation of every closed-form reference outside the delivered code** and
**structural diff of the mission artefact against a fresh run** (173 leaves, 0 diffs). The
`scipy`-free bisection, the FEM solve and the negative control were re-executed by the auditor,
not merely re-read.

---

## 10. Disposition

**Approve for L2, with notes; L3 and L4 refused (mission did not claim them; the platform
referee agrees).** The functional substance is real, the numbers are reproducible to the last
bit, the negative control exists, and the mission's own documentation declares the flagship
chain unverified in the exact places where it is unverified — this is the honest-FAIL discipline
working as intended, applied to *scope* rather than to a defect. Nothing here warrants reopening:
`maturity_gate` should **confirm** L2 and leave round 2 to close `G5-03/G5-04/G5-05/G5-06/G5-07`
and to register the anchor's benchmark row (F3) so that round 2 can be machine-judged.

Evidence bundle: `artifacts/audit/m15904/{evidence.json, mission_gates_77045799.json,
audit_repro_gates.json, audit_gate_errors.png, make_audit_evidence.py}`.