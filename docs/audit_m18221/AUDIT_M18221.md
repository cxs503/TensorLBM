# Board audit m18221 — m18128 `XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL` (TensorLBM)

Auditor: `board-audit` (mission 18221) · type: **cross-check (交差核验)**, not a redo ·
audited mission 18128 (status `succeeded`, runner `deepseek-v4-flash`,
`claimed_by=tp-merge-1`) · audited commit **`6c55b46c`** (*"XVAL m18128: P-R
pseudopotential vs openLBMPM (formula conformance 2.2e-16; Maxwell 9.8pct at
ratio-10, 24.9pct at ratio-18.6 honest FAIL)"*, 19 files, +8491 lines; present in
`/root/tensor-ecosystem/TensorLBM` refs, **not** merged into `main` HEAD `0c7a9b15`).

**Verdict: L3 (system integration / first-class exercise of the production SCMP
path). Declared level 3, gate keys complete. L4 not granted** — the delivery's own
preregistered refinement arm never ran and no benchmark case is registered in
`tensor_benchmark_cases` for this anchor (see F1/F2).

---

## 1. What was checked (evidence, with pointers)

| # | check | how (independent of the delivery's helpers) | result |
|---|---|---|---|
| A | OSS side really contains the P-R kernel and it is dead code | auditor's own parse of `/tmp/openLBMPM/ShanChen2D/OptimizedD2Q9GPU.py`; own whole-repo call-site count | kernel `calFluidPotentialGPUPR` **exists**, sha256 `cccd2b38…`, repo HEAD `3d841891daaf`, **0 call sites** (only the `def` line) → *kernel-only dead code* **as claimed** |
| B | E1 conformance 2.22e-16 | re-derived ψ by de-continuation + identifier mapping of the **verbatim CUDA text**, evaluated at 800 densities × 4 `T_r`, compared with `tensorlbm.multiphase.psi_eos_peng_robinson` | max rel. dev **2.220e-16** (= float64 eps), gate 1e-6 → **PASS**, bit-equal to the claim |
| C | E2 reference (PR Maxwell) is real, not tuned | own equal-area construction (unknown `ρ_v` → `p_sat=p(ρ_v)` → liquid root → `∫(p−p_sat)dv=0`, Brent on the vapour root) at 3 `T_r` | `ρ_l/ρ_v/p_sat` agree with the delivery's N5 to **≤2.4e-6 rel.** (ratio10 5.8867 / 0.5881 / 10.0096 / 0.02482) |
| D | reproducibility of the *got* numbers | re-ran `run_one.py --L 48 --tr 0.8266 --steps 20000` (75.0 s) | **bit-identical**: ratio `10.918366420659497`, err `0.09079017058552016`, Ma `0.11543023569630247` |
| E | zero-hand-copy of report numbers | cross-read `REPORT.md` ④/⑧ against `artifacts/results.json` / `cells.jsonl` | E2/E1 numbers match the artefacts; **one dangling pointer** (F4) |
| F | pre-existing capability + regression guard | `git log -- src/tensorlbm/multiphase.py` → last touched `5885134d` (merge #1781 / m15063, *before* this mission); `pytest tests/test_multiphase_psi_eos.py` | ψ-PR is **pre-existing production code**, 3 tests pass — the mission adds evidence, not code |

Artefacts of this audit: `audit_m18221/audit_pr_xval.py` (independent
reconstruction), `audit_m18221/audit_pr_xval.json` (raw), `audit_m18221/cells_audit.jsonl`
(reproduction cell), `audit_m18221/fig_m18221_audit.png`,
`docs/audit_m18221/evidence_snapshot/` (+ `SHA256SUMS.txt`, byte-frozen copies of
the audited deliverables).

## 2. Findings (severity-ranked)

**F1 — MODERATE (blocks L4): the benchmark judge has no row for this anchor.**
`SELECT case_name, metric, tolerance FROM tensor_benchmark_cases WHERE
anchor_slug='XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL'` → **0 rows** (table holds 94
rows overall; sibling anchors such as
`XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE` have 3). So there is no
registered ref/tolerance for the pass/fail to be judged against: the 15 % band
lives only in the mission goal + the self-written `prereg.md`. The audit therefore
reconstructed the reference itself (check C) — the number survives, but the
*registration* gap is real and is repeated across this anchor family.

**F2 — MODERATE (blocks L4): the preregistered refinement arm never ran.**
`prereg.md` §7 predeclares a refinement step; `results.json.verdict.note` says
"finest planar grid"; the delivery's own REPORT ⑥ admits L=96/L=128 were **not
executed** (cluster queue `cj1806` still queued after 7.6 min, killed). Consequence
visible in the delivered data: the grid series is **non-monotone** — ratio-10 error
9.079 % (L=48) → 9.788 % (L=64), i.e. **refining makes it slightly worse**, and the
pass margin to the 15 % band is only 5.2 pp. The pass is real but *not* demonstrated
converged. Honest disclosure (`honest FAIL` at the ratio-18.6 tier, 24.9 %) is
correctly reported and is **not** a defect — it is evidence discipline working.

**F3 — MODERATE (claim-scope): the "cross-implementation" arm is formula-level only.**
The OSS P-R kernel has **0 call sites** and cannot be executed (228 `@cuda.jit`,
`numba.cuda`, no GPU on the host); additionally the repo ships no LICENSE. So E1
compares *two transcriptions of the same algebraic ψ* (both sides use the same
P–R `p_EOS` and α≡1), not two running implementations. This is disclosed in
REPORT ①/⑥ (branch A3 of the prereg), so it is a **scope** limitation rather than
an overstatement — but any downstream reader who reads "XVAL vs openLBMPM" as a
numeric OSS comparison would be misled. The residual 2.22e-16 is a
tautology-check of the mapping `c₀ = c_s²`, valuable exactly because it quantifies
the `c₀=3` porting hazard (ψ ratio 3.0000 ± 2.8e-16 → force factor 9).

**F4 — MINOR (artefact pointer): REPORT ④ E3 cites "N7/图 4" but `NUMBERS.md` has no N7.**
The machine-written table file goes N1–N6 → N8 → N9. The droplet spurious-current
number `max Ma = 0.115` is only in `artifacts/snapshot.npz` meta
(`ma_max = 0.11538146851703832`, L=96, R=20, 4000 steps, `wall_s = 20.8`) and in
REPORT ③/④. So the droplet run **did** happen and fig4 is backed by a real field,
but that single number is hand-carried in the report — the "zero hand-copy"
claim is 99 %, not 100 %.

**F5 — MINOR (empty arm): the Laplace/surface-tension arm is empty.**
`results.json.laplace = {"points": []}`, `laplace_sigma = null`. The prereg does
not gate it and REPORT ⑧ explicitly refuses to call it "passed" (no external
reference) — consistent, but it means the delivery has **no surface-tension
benchmark**, which is the usual second leg of an EOS-pseudopotential validation.

**F6 — MINOR (no test added).** The mission adds no `tests/` file; the audited
capability is guarded by the *pre-existing* `tests/test_multiphase_psi_eos.py`
(3 passed: `test_corrected_psi_reconstructs_maxwell_pressure_lock`,
`test_forcing_velocity_shift_branch_is_bit_identical_to_legacy`,
`test_corrected_sign_attractive_and_isotherm_loops`). The XVAL driver itself has no
regression guard, so a future `ψ` change would silently invalidate the 9.8 %
number.

**F7 — INFO: the mission's delta is evidence, not capability.** No `src/` change;
`psi_eos_peng_robinson` (`src/tensorlbm/multiphase.py:249`) pre-exists and is wired
(`make_psi_eos`:271 → `:294`; consumers `multiphase3d.py:32`,
`multiphase3d_d3q27.py:35`). The mission *exercised* it through the production
SCMP path (`benchmarks/eos_maxwell/probe_lib.py:157 psi_fn = make_psi_eos(eos, T)`,
`:390 run_probe` → `collide_sc_single_component` + `solver.stream`, 12000–20000
steps) — that is why L3 is granted and L2 alone would understate it.

## 3. Score sheet (L0–L5)

| level | requirement | met? | why |
|---|---|---|---|
| L0 docs | — | ✔ | 248-line report + 9 sections + 4 figures |
| L1 math/unit | ψ identity, Maxwell construction | ✔ | 2.22e-16 conformance; independent Maxwell agreement ≤2.4e-6 |
| L2 independent functional impl. | real code input→compute→output | ✔ | `xval_driver.py:156-193 run_cell`, `run_one.py:41`, drives 20000-step LBM; artefacts real, reproduced |
| L3 system integration | exercised through main solver | ✔ | `xval_driver → probe_lib.run_probe → collide_sc_single_component + stream` (production D2Q9 SCMP), 3 production cells rerun bit-identically |
| L4 benchmark validation | complete case + error analysis + reproducible | ✘ | reproducible ✔, error analysis ✔, but **case set incomplete vs its own prereg** (F2: no L=96/128 refinement ⇒ no demonstrated grid convergence; non-monotone 9.08→9.79 %) and **no registered benchmark row** (F1); no surface-tension leg (F5) |
| L5 research/engineering result | new validated capability | ✘ | no new capability; the ψ is pre-existing and untested-against-OSS-at-runtime (F3, F7) |

**Level = 3.** What would lift it to L4 (all cheap, no new physics): run
`run_one.py` at L=96/128 for the ratio-10 tier and report the convergence trend;
add a Laplace arm (`snapshot.py` with R ∈ {12,16,20}) with σ vs R; register
`tensor_benchmark_cases` rows (metric = ratio/ρ_l/ρ_v, tolerance = 15 %) so the
judge — not the prereg — is the referee; add one pytest that re-derives the 9.79 %
figure in <60 s (the auditor's `cells_audit.jsonl` run took 75 s, so a reduced-step
guard is sufficient).

## 4. Checks NOT run (mark as unknown, not passed)

* No run of the OSS side at all — impossible: `numba.cuda`, no GPU, kernel unwired,
  no LICENSE (would in any case be reference-only per mission 8283).
* No convergence study (L=96/128) and no cluster rerun: I did not re-submit
  `cj1806`; the audit reproduces only the delivered grid points.
* D3Q19/D3Q27 SCMP paths and `forcing="standard"` were not cross-checked against
  the PR ψ (out of the delivery's scope, out of mine).
* No literature (non-OSS) anchor for the P-R coexistence numbers was consulted;
  the referee is the analytic Maxwell construction, rebuilt here.

## 5. Independence limitations

I read the delivery's code and report before writing my own reconstruction (unavoidable
in a cross-check), but `audit_pr_xval.py` imports **no** delivery helper: the OSS ψ is
re-derived from the verbatim kernel text with my own de-continuation/mapping, the
Maxwell reference is my own solver, and the reproduction ran the delivered entry point
into a separate output file (`audit_m18221/cells_audit.jsonl`). The one shared
dependency is `tensorlbm` itself — which is the object under test, not a helper.

## 6. Disposition

**Approve with notes.** The three load-bearing claims survive independent
reconstruction (formula conformance 2.22e-16; ratio-10 tier 9.08–9.79 % < 15 %;
ratio-18.6 tier 24.9 % honest FAIL), the artefacts exist and the production run
reproduces bit-identically. Graded **L3**; the L4 gap is registration + convergence
(F1/F2), not fabrication. Remediation list in §3 is small and mechanical.