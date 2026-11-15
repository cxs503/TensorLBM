# REVIEW_ROUND3 — independent review of the round-2 PASS of the Zou–He channel xval

**Anchor** `XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTION` · **repo** `TensorLBM`
· **reviewer branch** `agent/m21990` · **reviewed artifact** commit `60ac3409e`
(merge #2341 / `m18793`) — `xval/XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTION/`
(`PREREG.md`, `REPORT.md`, `run_xval.py`, `artifacts/zou_he_channel_results.json`)

**Disposition: APPROVE WITH NOTES — PASS stands, but not for the reason round 2 gave.**
Two blocking *claims* are withdrawn (the steady gate was violated; the headline
"worst 4.15 %" comes from an arm that is not in the pre-registered case table).
The accuracy claim itself survives re-execution: worst registered arm 2.38 %
(≪ 15 %), and convergence pushes it *down*, not up.

---

## 0. Independence and scope

* **Independence.** The re-execution used the frozen code under test (nothing in
  `src/` was modified). New evidence comes from (a) a re-run of every round-2 arm
  with round-2's own call arguments (`repro_round2.py`), (b) a march that honours
  the pre-registered steady criterion (`run_xval_r3.py`), and (c) a **from-scratch
  re-derivation** of the boundary value problem solved as a 4×4 linear system
  (`independent_zouhe.py`), which shares no algebra with the module under test —
  only the D2Q9 lattice table, which is the definition of the lattice, not the BC.
* **Limitations.** The re-run to steady state covers the two *coarse* registered
  arms (C1 velocity, C3 pressure) up to 2.2–2.5 × 10⁴ steps; the fine pair
  (C2/C4) and the unregistered profile arm were **not** re-run under the gate
  (120 s wall-clock cap on the review host; the cluster job `cj3830` covering all
  five arms was still queued at position > 50 after 12 min). Conclusion about the
  *fine* arms is therefore inferred, not measured — marked `not_run` below.
* **Checks performed.** Reproduction of all five arms; steady-gate march + drift
  history; independent linear-system closure check on all three implemented layers
  (west velocity, west pressure, east pressure); regression suite
  (`pytest tests/test_zou_he_channel_reconstruction.py
  tests/test_zou_he_independent_closure.py
  tests/test_xval_zouhe_verdict_steady_gate.py` → **12 passed**); figure
  regeneration (round 2 shipped a manifest with **no** PNGs).

## 1. What round 2 claimed, and what survives

| # | round-2 claim | status after review | evidence |
|---|---|---|---|
| 1 | "PASS, worst predeclared `dp_max` = 4.15 %" | **PASS holds**; the "worst" attribution does not (F2) | `artifacts/repro_round2.json`, `artifacts/r3_steady_gate.json` |
| 2 | numbers are reproducible | **confirmed, bit-exactly** on all five arms | `repro_round2.py`: `profile_bitwise_equal = true`, `|Δ dp_max| = 0.00e+00` for C1/C2/C3/C4/P |
| 3 | "closure residual 5.33e-15 (round-off)" | **confirmed and strengthened**: an independent solve reproduces all three layers to `1.1e-15` | `independent_zouhe.py`, `tests/test_zou_he_independent_closure.py` |
| 4 | every arm satisfies the pre-registered steady gate `drift < 1e-6` | **FALSE** — blocking claim, see F1 | round-2 artifact: drifts 1.6e-4 … 2.6e-3 |
| 5 | "verdict on the primary metric `dp_max` at the measurement column" | true for `dp_max`; the second declared gate is not evaluated at all (F3) | `run_xval.py` verdict block |

## 2. Findings, severity-ranked

### F1 (blocking claim, no measurable impact on the verdict) — the run violated its own pre-registered steady gate

`PREREG.md` declares, in the same metric table as `dp_max`, the last row:
`drift = max|ux(t) − ux(t−1000)| / max ux  < 1e-6 (steady state)`, and the case
table declares `≤ 40 000 steps, steady-stop`. The published artifact reports

| arm | steps run | reported `drift` | over the declared bound |
|---|---|---|---|
| C1_coarse | 5 000 | 1.64e-4 | 164× |
| C3_coarse | 5 000 | 6.71e-5 | 67× |
| P_inlet_profile | 5 000 | 6.89e-5 | 69× |
| C2_fine | 8 000 | 2.55e-3 | **2 553×** |
| C4_fine | 8 000 | 3.57e-4 | 357× |

All five arms were stopped 5–8× short of the declared march budget with the
steady gate unmet, so `dp_max` was measured on a flow that was still relaxing.
`REPORT.md` §⑦ labels this "conservative"; that label is *asserted*, not shown.

**Review action taken (measured, not asserted).** C1 and C3 were marched with the
gate enforced (`drift < 1e-6`, 1 000-step window as declared):

| arm | 5 000 steps (round 2) | 22 000–25 000 steps (this review) | `drift` reached | gate met? |
|---|---|---|---|---|
| C1_coarse | 1.506 % | **1.364 %** | 4.9e-6 | **no** |
| C3_coarse | 2.443 % | **2.383 %** | 1.0e-5 | **no** |

Two consequences, both material:

1. the "conservative" claim is **supported for the two registered coarse arms** —
   `dp_max` moved *down* by 0.06–0.14 pp while `drift` fell by ~40×, so the
   transient inflated the metric, it did not hide an error;
2. the gate as declared is **not attainable within the declared 40 000-step
   budget at the fine resolution** (C2's drift was 2.55e-3 at 8 000 steps and the
   coarse arms are still at 5–10e-6 after 2.2–2.5 × 10⁴ steps; `τ_diff = H²/ν`
   is 2 890 (coarse) and 10 890 (fine) steps, and round-off-level convergence of a
   diffusive problem needs ≈ 10 `τ_diff`). The pre-registration is therefore
   *internally inconsistent* (F4), and no honest round of this xval can print
   `steady_ok = true` without either a longer budget or a different (residual-based)
   gate. `figures/fig5_drift_gate.png` shows the measured decay against the gate.

### F2 (blocking for the narrative) — the headline "worst 4.15 %, outside the 3 % stretch band" comes from an arm that is not pre-registered

`PREREG.md` fixes cases `C1…C5`. `P_inlet_profile` — the parabolic-inlet variant —
is **not** in that table, yet its `dp_max` (4.15 %) is the one folded into
`verdict.worst`, and `REPORT.md` then discusses "outside the 3 % stretch band".
Every *registered* arm is ≤ 2.45 % (`C3_coarse`). The honest headline is
"registered arms 0.88 % – 2.45 %, worst 2.45 %; supplementary profile-inlet arm
4.15 %, and 3.79 pp of those 4.15 % are an *amplitude* offset (the injected
parabola is not the developed solution at 5 000 steps), not a shape error".

Measured sensitivity of this number to the metric definition: with the *same*
velocity field, `dp_max` is **1.59 %** under the flux-matched normalisation round 2
used for C1/C2 and **4.15 %** under the a-priori one used for PR. The metric is
thus normalisation-sensitive by a factor 2.6 — a reason to report both, which
round 2 now does for PR but not for C1/C2.

### F3 (defect, fixed here) — the verdict logic could not see the second declared gate

`run_xval.py`'s verdict block only looked at `dp_max`; `PASS` was
`max(dp_max) < 0.15` and nothing else. A pre-registered metric could fail
without affecting the verdict — this is exactly why F1 went unnoticed. Fixed by
extracting a pure `decide_verdict(dp_max, drifts, ...)`: `PASS`/`STRETCH` now
require *both* gates, the metric-only outcome is preserved as `PASS_dp_max`, and
`not_steady` names the offending arms. Pinned by
`tests/test_xval_zouhe_verdict_steady_gate.py` (4 tests, incl.
"missing drift is not treated as converged").

### F4 (defect, fixed here) — the `drift` metric did not measure what was declared

`drift` is computed between *consecutive samples*, so it silently equals
`|ux(t) − ux(t − sample_every)|`. Round 2 used the default `sample_every = 500`,
i.e. a **500**-step window where `PREREG.md` declares **1 000**. On the same run
the gate quantity differs by ~40 % (C1 at 5 000 steps: 1.64e-4 over 500 steps vs
2.30e-4 over 1 000 steps). Fixed: `run_channel`'s default is now
`sample_every = 1000`, and the verdict records `drift_window_steps`.

### F5 (minor) — evidence pointers pointed at files that did not exist

`figures/figures_manifest.json` listed `fig1_profiles.png … fig4_convergence.png`
and `REPORT.md` §⑥ cites all four, but `figures/` contained **only the manifest**.
Regenerated from the committed artifact by the committed `make_figures.py`
(no number re-typed); the four PNGs are now in the tree. Similarly, `§⑦` records
that the suite "was still queued … so the reported numbers come from two local
chunks" — the artifact carries `chunked_local_run` and no cluster job id; the
chunking is honest but the reproduction command in §⑥ does not mention
`--cases coarse|fine --merge`, so the cited command does not reproduce the
published artifact.

### F6 (minor, not fixed) — `REPORT.md` §⑤ "wall offset" diagnostic

Offsets of −0.03 … +0.04 lattice units are reported as evidence that the residual
is not a wall-placement artefact. With `H = 17`–`33` and the residual at the 1 %
level, a two-point linear extrapolation from rows 1–2 of a *discrete* parabola has
an O(`h`) truncation that is the same order as the number it measures (the
parabola's own second difference over one row is ~0.4 % of `u_max`). The
conclusion (wall offset ≪ 1 lattice) is safe; the precision implied by quoting
4 significant digits is not. Non-blocking.

## 3. What is *not* claimed / not run

* **not_run** — steady-gate march for `C2_fine`, `C4_fine`, `P_inlet_profile`
  (host wall-clock cap; cluster job `cj3830` still queued). The fine-resolution
  numbers therefore inherit F1 unchanged.
* **not_run** — 3-D (D3Q19/D3Q27) Zou/He, MRT/TRT collision, OSS numeric diff
  (round 2's `OSS_DIFF.md` §① survey found no OSS reconstruction implementation;
  unchanged by this review).
* **not_checked** — north/south Zou/He layers: `boundaries.py` implements only the
  west (velocity, pressure) and east (pressure) columns; the claim in the legacy
  in-tree docstring that "this module gives the four walls" does not describe the
  live module.
* The independent solver checks *algebra and closure*, not physics; it cannot
  detect a systematically wrong-but-consistent closure assumption (Zou & He's
  non-equilibrium bounce-back is a modelling choice, not a theorem).

## 4. Overstated claims (as published)

1. "PASS … worst predeclared `dp_max` = 4.15 %" — 4.15 % belongs to an arm that is
   not predeclared; the predeclared worst is 2.45 %.
2. "steady-state gate satisfied" is implied by quoting `drift` in the metric table
   while every value is 67×–2 553× above the bound.
3. "the fine-grid numbers are therefore a conservative error estimate" — plausible
   but unsupported as published; after this review it is supported for C1/C3 only.

## 5. Remediation (all in this branch)

| item | file |
|---|---|
| verdict enforces both declared gates | `run_xval.py` (`decide_verdict`) |
| `drift` window matches the pre-registration (1 000 steps) | `run_xval.py` (default) |
| steady-gate march + drift history, per case | `run_xval_r3.py`, `artifacts/r3_steady_gate.json` |
| round-2 reproduction, programmatic diff | `repro_round2.py`, `artifacts/repro_round2.json` |
| independent re-derivation of the closure | `independent_zouhe.py` |
| missing figures | `figures/fig1…fig4`, `make_review_figures.py`, `figures/fig5,fig6` |
| regression pins | `tests/test_zou_he_independent_closure.py`, `tests/test_xval_zouhe_verdict_steady_gate.py` |

Recommended follow-up (not in this branch): raise the declared march budget to
`≥ 8 τ_diff` **or** replace the absolute `drift < 1e-6` gate by a residual-based
one (`‖Δu‖/‖u‖` per `τ_diff`), and register `P_inlet_profile` in `PREREG.md` if it
is to be part of the verdict.

## Reproduction

```bash
cd xval/XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTION
python3 independent_zouhe.py                 # 3 layers vs independent linear solve
python3 repro_round2.py                      # round-2 artifact, bitwise diff
python3 run_xval_r3.py --cases C1,C3 --steps 25000 --drift-tol 1e-6
python3 make_figures.py && python3 make_review_figures.py
python3 -m pytest tests/test_zou_he_channel_reconstruction.py \
                 tests/test_zou_he_independent_closure.py \
                 tests/test_xval_zouhe_verdict_steady_gate.py -q   # 12 passed
```