# Board audit m17884 — m17839 `XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN` (TensorLBM)

Auditor mission: **17884** · verdict_by `board-audit` · audited commit **eee0988a6de3c7122fe4a4cf30b3f185d87277a7**
(`XVAL: cross-validate fused push-pull (upstream OSS ch.13) against OSS port and AB reference`,
worktree `/root/tp-agent/work/m17839`, mission 17839 status `succeeded`, platform report id 16786).

This is a **交差核验单**: re-derive the delivered numbers, decide the maturity level, land the verdict row.
No re-implementation and no changes to the audited branch (`git status` there still clean after the audit).

---

## 1. What was asserted, and against what judge

| claim (from report 16786 / doc §4) | asserted value |
|---|---|
| `u_x`,`u_y` rel. L2 vs closed-form Taylor–Green | 0.00192 (0.19 %), owner gate 15 %, target 3 % |
| `rho` rel. L2 vs analytic | 0.02885, "PASS", explained as forced acoustic 2ω wave |
| `nu` inverted from E(t) decay | 0.16680 vs 0.166667 → 7.84e-4 |
| cross-arm (OSS NumPy port vs `torch_fused`) | rel. L2 of the 9 populations = 0, **bitwise equal** |
| cross-arm (repo AB vs fused) | `u_x` rel. L2 8.35e-14 |
| negative controls can fail | `nu_bias_25pct` 0.476 FAIL, `mirror_x` 0.785 FAIL, `flip_all` PASS = documented blind spot |
| resolution | NX = 32/64/128, observed orders 2.08 / 2.39 |

Board rule for this audit: **the registered benchmark row is the sole judge** —

```sql
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug = 'XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN';
-- 0 rows (65 cases registered platform-wide, none for this anchor)
```

So "within tolerance" cannot be adjudicated by the registry; the audit therefore grades the
deliverable against its own *predeclared* criterion (owner 15 % band, 3 % target, stated in the
mission goal before execution) **and** records the missing judge as the decisive L4 blocker.

## 2. Independent re-execution

```
cd <checkout of eee0988a> && python xval/fused_push_pull_aa_pattern/run.py --out /tmp/audit_m17839_result.json
# finished in < 2 min (single host, OMP_NUM_THREADS=4)
python -m pytest tests/test_fused_push_pull_xval.py -q      # 12 passed in 6.21s
```

Fresh artifact vs shipped `xval/fused_push_pull_aa_pattern/result.json`
(structural compare of all 2 532 leaves, script `docs/board_audit/m17884/audit_m17884.py`):

* **4 differences, all of them `wall_s` timing fields**;
  **0 non-timing differences** → every number in the shipped artifact is reproduced exactly.
* Wall-clock outlier worth recording: shipped `torch_ab.wall_s = 117.1 s`, auditor 1.43 s (81×).
  The original run was evidently taken under heavy host load. No timing/Mlups claim is made in the
  deliverable (`storage` block is explicitly structural: 17/17/9/18 doubles per node), so this is a
  **non-blocking** remark, not a defect in the science.
* Re-rendering the document (`python xval/fused_push_pull_aa_pattern/make_report.py`) reproduced
  `docs/XVAL_m17839_FUSED_PUSH_PULL_AA_PATTERN.md` **byte-identically** (`git status` clean afterwards)
  ⇒ the doc's numbers really are injected from `result.json`, not hand-copied.

## 3. Re-derived grades (auditor numbers, not quoted)

| quantity | reference | got | rel. error | < 15 % | < 3 % |
|---|---|---|---|---|---|
| `u_x` rel. L2 @T=800 | analytic TG | 0.00191946 (oss port, torch_fused 0.00191946 identical) | 1.92e-03 | ✓ | ✓ |
| `u_y` rel. L2 @T=800 | analytic TG | 0.00191946 | 1.92e-03 | ✓ | ✓ |
| `rho` rel. L2 @T=800 | analytic TG | 0.02885291 | 2.89e-02 | ✓ | ✓* |
| `nu` from E(t) fit | 0.16666667 | 0.16679741 | 7.84e-04 | ✓ | ✓ |
| mass (ρ₀·NX·NY) | 4096 | 4095.999999999817 | 4.5e-14 | ✓ | ✓ |
| repo AB vs fused, `u_x` | — | 8.35e-14 | — | ✓ | ✓ |

\* with the caveat of finding **A4** below: `rho_lu2` follows upstream's convention of normalising by the
`O(u²)` fluctuation (`rhoa − rho0`). Auditor's own magnitude check at t = 801:
absolute density error **max 9.07e-08**, rms 4.03e-08, against an analytic fluctuation rms of 1.75e-06
(and 0.0231 relative → the number swings between adjacent steps, consistent with a standing acoustic
mode). So the "2.89 %" headline is a *relative* diagnostic on a 10⁻⁶-scale field, not a 3 % physical
error. The arrow of the claim is correct; the headline magnitude is easy to over-read.

Re-derived convergence (not copied): `p = log(e₃₂/e₆₄)/log2 = 2.0815`, `p = log(e₆₄/e₁₂₈)/log2 = 2.3856`
— **exactly** the recorded `[2.0814771219688715, 2.3856190152805503]`. Density residual barely moves with
h (0.1126 → 0.0289 → 0.0207), which supports the doc's "does not refine away" statement.

Negative controls (auditor re-read of `result.json.controls`, all produced by the run, gate can fail):

| control | analytic error | gate |
|---|---|---|
| `nu_bias_25pct` (kernel ν·1.25 graded vs true ν) | 0.4756 | **FAIL** |
| `mirror_x` (West/East swapped) | 0.7847 (`u_y`), and 0.7843 `u_x` vs the OSS arm | **FAIL** |
| `flip_all` (all velocities negated) | 0.00192 | PASS — acknowledged blind spot |

## 4. Findings (severity-ranked)

**A1 — blocking for L4: no machine judge.** `tensor_benchmark_cases` has **0 rows** for this anchor
(65 rows platform-wide). Per the board rule the registered case row is the sole judge, so "benchmark
validation" cannot be confirmed no matter how good the deliverable-internal numbers are. Also missing:
the pipeline never registers `result.json` as an immutable artifact of a case row.

**A2 — blocking for L3: the new capability is not integrated.** `src/tensorlbm/fused_push_pull.py:122
class FusedPushPullStepper` is referenced only by `tests/test_fused_push_pull_xval.py` and
`xval/fused_push_pull_aa_pattern/{run.py,make_report.py}`; `src/tensorlbm/__init__.py` exports it **0**
times and no module under `src/`, `benchmarks/`, `examples/` imports it. The doc is explicit that the
fused path is periodic-only and walls stay in `solver` — but then nothing ships as a usable stepper:
no boundary hooks, no `Step`/`solver` wiring, no API entry. Library-level consistency is proven
(AB arm uses the shipped `solver.stream`/`collide_bgk`, agreement 8.35e-14), yet no *caller* exists,
so this is a verified standalone module, not a system integration.

**A3 — non-blocking, claim wording: the "OSS arm" is a transliteration, not executed OSS.**
`xval/.../ref/{LBM.cu,LBM.h,main.cu}` are genuine upstream chapter-13 sources (sha256 in
`ref/PROVENANCE.json` **re-verified by the auditor**: `0c9854fa…`, `0bc95be6…`, `ebbb847c…` all match).
The auditor also line-matched the transliteration against the CUDA kernel: pull indices
`(xm1,y,1)…(xm1,yp1,8)`, moments, `tauinv = 2/(6ν+1)` folding and the write-to-the-other-buffer
pattern all correspond to `LBM.cu:181-252`. But **`nvcc`/GPU execution was never attempted** — the
"bitwise equal" claim proves *the author's NumPy transliteration is reproduced exactly by the torch
port*, i.e. reproducibility of a convention, not equality with the upstream binary. The doc labels the
arm `oss_numpy` correctly (§4.2, §6.1); the report title ("arm-to-arm bit-identical") invites the
stronger reading.

**A4 — non-blocking, metric amplification.** See §3 footnote: `rho rel. L2 = 0.0289` is 2.9 % of a
1.75e-06 fluctuation; absolute max error is 9.07e-08. The doc's §5.4 explanation (2k pressure mode,
envelope period 28.57 steps vs acoustic prediction π/(c_s·2k) = 27.71 steps, 3.1 % mismatch — auditor
re-derived 27.7128, rel. err 3.10 %) is sound, but the absolute magnitude is never quoted.

**A5 — non-blocking: negative controls bite the transliteration, not the shipped stepper.**
`run.py:176 _variant_step` injects the `nu_bias`/`mirror_x`/`flip_all` faults into
`OSSFusedStepper`; `FusedPushPullStepper` itself is never run with a fault. The falsifiability evidence
therefore covers the metric + the port, not the delivered class.

**A6 — informational: documented blind spot was audited and is real.** `flip_all` reproduces the TG
diagnostics to 0.00192 → a global velocity sign flip is invisible to both the analytic gate and the
population-level arm comparison. Honest disclosure (§6.2), keeps the claim set honest; round 2 should
add a non-symmetric case.

## 5. Level verdict — **L2**

| level | judgement | reason |
|---|---|---|
| L0 documentation | pass | 190-line doc, but level is not documentation-bound |
| L1 math/unit verification | **pass** | `tests/test_fused_push_pull_xval.py`: τ↔ν maps, D2Q9 direction table, pull index vs `torch.roll`, equilibrium, gate values, 12/12 passed in 6.21 s (auditor run) |
| L2 independent functional implementation | **pass** | real code `src/tensorlbm/fused_push_pull.py:122-258` (input→compute→output, 17 doubles/node, single-buffer variant 9), executable entry `xval/.../run.py`, artifacts reproduced bit-exactly (A→§2), tests present |
| L3 system integration | **fail** | A2: zero callers, zero exports, no boundary/solver wiring |
| L4 benchmark validation | **fail** | A1: no `tensor_benchmark_cases` row ⇒ no adjudicable judge; physical anchor is analytic (manufactured) verification only, upstream binary never executed (A3) |
| L5 research/engineering result | **fail** | inherits L4; no adjudicable external reference |

`gap_to_claim`: the mission's numeric claims (0.19 % velocity, 7.8e-4 ν inversion, bit-exact
reproducibility, failing controls) are **all true as stated** — this is not an inflated-evidence case.
What does not exist is (i) a registered judge row, (ii) any caller of the new module, (iii) execution of
the upstream implementation. Hence L2, and the deliverable is *reopened* for those three items, not for
its arithmetic.

## 6. Checks performed / not run

* **run**: `run.py` full (auditor, <2 min), `pytest tests/test_fused_push_pull_xval.py` (12 passed),
  `make_report.py` re-render (byte-identical), independent magnitude recomputation of ρ/u error.
* **read**: `result.json` (2 532 leaves), `docs/XVAL_m17839_…md`, `run.py`, `oss_port.py`,
  `src/tensorlbm/fused_push_pull.py`, `ref/LBM.cu` (cross-read against `oss_port.py`),
  `ref/PROVENANCE.json` + sha256, `tests/test_fused_push_pull_xval.py`.
* **not run**: upstream CUDA build/execution (no GPU/nvcc in this environment) → A3 cannot be closed
  here; no independent reference data other than the closed-form TG solution; NX=128 cross-arm
  comparison (not produced by the deliverable either); no bandwidth/Mlups measurement (not claimed).
* Auditor artifacts: `docs/board_audit/m17884/audit_numbers.json`,
  `docs/board_audit/m17884/figures/fig{1,2,3}*.png`,
  `docs/board_audit/m17884/verdict_m17884.sql`.

Figures: fig1 grades against the analytic solution with gate/target bands; fig2 resolution ladder with
re-derived slopes; fig3 negative controls plus the reproduction-diff scope (4 wall-clock entries, 0
numeric).