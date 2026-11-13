# XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR — Round 2 (open-source cross-benchmark, selection)

| item | value |
|---|---|
| mission | 18237 |
| anchor | `XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR` |
| repo arm | TensorLBM — `src/tensorlbm/les_graph_unroll/{solver2d,reference2d,selection}.py` |
| OSS target (round-1 lineage) | `loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow` — feature `graph-unroll-bptt-les-parameter`; provenance: `docs/XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER.md` §1 (source-line table), round-1 report id 17002 |
| prereg (written before any number) | `docs/XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR-PREREG.md` |
| driver | `scripts/xval_r2_selection_bptt_tgv.py` (stages: `ref / arm / controls / judge / all`) |
| figures | `scripts/xval_r2_selection_bptt_tgv_figures.py` |
| tests | `tests/test_les_graph_unroll_selection.py` (8 passed) |
| artefacts | `artifacts/xval_r2_selection_bptt_tgv/{results.json, reference.npz, fig1..fig4.png, figures.json}` |
| claim level | REAL — every number below is emitted by the driver into `results.json` |

---

## ① Problem and reference provenance

**Start-of-work triple check (mandated).**

1. *knowledge_base* — `/api/v1/tensor/knowledge/search` answers
   `{"available": false, "reason": "engine_offline"}` (knowledge engine
   `127.0.0.1:8021` connection refused) and the fallback store
   `knowledge/stats` reports `count: 0`. **No prior art retrieved** (friction
   F1). Nothing was silently skipped: the round-1 XVAL record in this repo was
   used instead.
2. *negative_knowledge_ledger* — `/tensor/negative-knowledge/search?q=selection+bptt+tgv`
   returned only process-level entries (top hit `negk:loop:18103`, a failed
   *audit* mission whose `failure_reason` is "result 未提供 checks.reason").
   No domain negative result to avoid; that entry's lesson (always emit
   `checks.reason`) is honoured in §⑥.
3. *oss_solutions* — no local OSS checkout and no outbound network, so the OSS
   practice was **not re-fetched**: it is taken from the repo's own recorded
   provenance (round-1 doc §1 source-line table: `LatFlow/Domain.py:134-140,
   192-196, 244-253` = `graph_unroll`/`Unroll()`, `Domain.py:122-128` =
   in-graph `tau`, `Domain.py:57` = `Sc = 0.17` default). Zero hand-copy on my
   side; the chain is round-1 doc → `results.json` → this report.

**The decision (prereg §1).** Round 1 proved the OSS *mechanism*
(`graph_unroll` + in-graph `τ` + BPTT) reproduces exactly, but that a
pointwise field-matching objective picks a physically wrong model. Round 2
turns that into a **selection** problem:

> Which (closure, objective) pair should a BPTT pipeline *select* on the
> Taylor–Green-vortex anchor at `Re = 500` so that the selected model meets the
> owner's 15 % bar — and does the pipeline actually pick it?

Two competing explanations were pre-declared: **A1** the closure dominates;
**A2** the objective dominates (round-1 lead).

## ② Method and implementation

* `selection.py` adds the missing layer on top of the round-1 differentiable
  kernel: a `SelectionCase` (trajectory + reference + train/hold-out split),
  three differentiable objectives (`field`, `energy`, `enstrophy`), BPTT
  selection (Adam on `log10 θ`), two autograd-free controls (25-point sweep,
  golden section) and the held-out evaluator (`field_err`, `ke_err`, `z_err`,
  `spec_err`).
* Objective definitions (all on the **training** snapshots only, 50/100/150):

  ```
  J_field = (1/K) Σ_k ||u(t_k) − ū_ref(t_k)||² / ||ū_ref(t_k)||²
  J_energy = Σ_k (K_k − K_ref,k)² / Σ_k K_ref,k² ,    K = ½⟨u²⟩
  J_enstrophy = Σ_k (Z_k − Z_ref,k)² / Σ_k Z_ref,k² ,  Z = ½⟨ω²⟩
  ```

  `ω` is the central-difference vorticity on the torus, so all three
  objectives are autograd-differentiable through the whole 200-step chain.
* **Selection rule (pre-declared, prereg §4)**: the winner is the pair with the
  smallest *training* loss; its quality is then judged on the **held-out**
  snapshot `t = 200`, which no objective sees. The no-model control
  (`θ = 0`) is present in every sweep, so the selector is able to say "no
  sub-grid model".
* Reference: independent numpy BGK run on 128² (same physical `ν`), 4×4 box
  filtered to the 32² LES grid; a 256² run gives the reference's own
  resolution error. Nothing differentiable touches the reference.
* No random seeds anywhere; fixed hyper-parameters; float64; staged execution.

## ③ Parameter table (all dimensionless)

| quantity | symbol | value |
|---|---|---|
| domain | — | `2π × 2π`, periodic |
| reference velocity | `U0` | 0.1 |
| Mach | `Ma = U0√3` | 0.1732 |
| integral length | `L` | 1.0 |
| physical viscosity | `ν = U0 L/Re`, `Re = 500` | 2.0e-4 |
| grids | `N_ref / N_conv / N_les` | 128 / 256 / 32 (`Δx = 0.049087 / 0.024544 / 0.19635`) |
| relaxation times | `τ_ref / τ_conv / τ0` | 0.749007 / 1.496029 / 0.515563 |
| LES filter | — | 4×4 box |
| window | `T` | 200 steps; train `t = 50, 100, 150`; hold-out `t = 200` |
| closures | — | `smagorinsky` (`Cs`), `oss_proxy` (`Sc`) |
| θ range / clamp | — | `[1e-3, 1.0]` |
| BPTT | — | Adam on `log10 θ`, `lr = 0.1`, 20 iters, starts `0.02`, `0.17` |
| derivative-free | — | 25-point sweep `θ ∈ [0, 0.6]` (`Δθ = 0.025`), golden section on `[0, 0.6]`, 40 iters |

## ④ Reference vs got (the comparison table)

Reference resolution (prereg §5): KE(200) = **2.12530e-3** (128² raw),
**2.13104e-3** (256² raw) → reference own error **0.269 %**; the filtered 128²
value used as truth is 2.11250e-3 (0.870 % from the 256² raw value, i.e. the
explicit filter, not the discretisation, is the larger of the two effects).

| id | got | reference | err | <15 % |
|---|---|---|---|---|
| P1 BPTT gradient vs central FD (worst of 12 arms, h=1e-3) | `∂J/∂θ` autograd | same loss, FD | **0.007 %** (max 7.35e-4) | ✅ |
| P2 selection agreement, `θ` (integral objectives) | θ = 1.63e-3 (clamp) | sweep argmin θ = 0 | bound hit, reported | ⚠️ (declared) |
| P2 selection agreement, `θ` (`smagorinsky`+`field`) | 0.08405 | 0.100 (sweep) / 0.09167 (golden) | 16.0 % / 8.3 % | ❌ / ✅ |
| P2 selection agreement, `θ` (`oss_proxy`+`field`) | 0.19665 | 0.200 (sweep) / 0.18962 (golden) | 1.7 % / 3.7 % | ✅ |
| P2 selection agreement, loss gap (all 6 combos) | J(θ_BPTT) | J(argmin) | ≤ 0.02 % | ✅ |
| P3 selected model vs held-out KE | 5.6098e-4 | 5.6092e-4 (best over the whole space) | **0.561 %** | ✅ |
| P4 regret of the training-loss selection | 5.6098e-4 | 5.6092e-4 | **5.3e-5 pp** (9.5e-5 rel) | ✅ |
| P5 reproducibility, `smagorinsky`+`field` | 0.08405 / 0.08213 | two starts | 2.3 % | ✅ |
| P5 reproducibility, `oss_proxy`+`field` | 0.16909 / 0.19665 | two starts | 16.3 % | ❌ |
| P5 reproducibility, integral objectives (4 combos) | 1.63e-3 vs 1.55–1.88e-2 | two starts | 850–1050 % (clamp) | ❌ |

Held-out metrics of the six selections (`t = 200`, `ke / z / field / spectrum` error):

| combo | θ selected | J_train | `ke_err` | `z_err` | `field_err` | `spec_err` |
|---|---|---|---|---|---|---|
| `smagorinsky`+`field` | 0.08405 | 1.2727e-2 | 1.155 % | 1.121 % | 14.60 % | 0.075 % |
| `smagorinsky`+`energy` | 1.63e-3 | 1.2661e-4 | 0.561 % | 0.528 % | 14.62 % | 0.073 % |
| `smagorinsky`+`enstrophy` | 1.63e-3 | 1.1278e-4 | 0.561 % | 0.528 % | 14.62 % | 0.073 % |
| `oss_proxy`+`field` | 0.19665 | 1.2727e-2 | 1.330 % | 1.295 % | 14.60 % | 0.075 % |
| `oss_proxy`+`energy` | 1.63e-3 | 1.2660e-4 | 0.561 % | 0.528 % | 14.62 % | 0.073 % |
| `oss_proxy`+`enstrophy` **(selected)** | 1.63e-3 | **1.1276e-4** | **0.561 %** | 0.528 % | 14.62 % | 0.073 % |
| *no sub-grid model* (`θ = 0`, control) | 0 | J_energy 1.2659e-4 | **0.561 %** | 0.528 % | 14.62 % | 0.073 % |
| *OSS as shipped* (`oss_proxy`, `Sc = 0.17`, no training) | 0.17 | — | 1.137 % | 1.103 % | 14.60 % | 0.075 % |

**Every column is inside the owner's 15 % bar**; the pointwise field error
(≈14.6 %) is the documented round-1 decorrelation floor and is *identical* for
all models, i.e. it carries no model information (which is exactly why it is
not the judge).

## ⑤ Convergence and parameter sensitivity (two axes, as required)

1. **Reference resolution** 128² → 256²: KE(200) 2.12530e-3 → 2.13104e-3,
   **0.269 %** — the energy integral is grid-converged; the explicit 4×4
   filter contributes 0.870 %, so the 0.561 % model error sits well above the
   reference's own uncertainty and is therefore resolvable.
2. **BPTT start point** (reproducibility, P5): `smagorinsky`+`field` 0.08405 vs
   0.08213 (**2.3 %**, well-conditioned); `oss_proxy`+`field` 0.16909 vs
   0.19665 (**16.3 %**); all four integral-objective arms differ by
   **850–1050 %** because the trajectory is pinned to the lower clamp
   `θ = 1e-3` — the *loss*, however, differs by only 0.02 %, i.e. the
   objective is flat, not the optimiser. `golden` (autograd-free) reproduces
   the field-objective optima at 0.09167 / 0.18962, confirming the BPTT
   values are optima of the same landscape.
3. **Not executed (honest)**: the secondary training-window study
   (`{50,100,150,200}`, in-sample for `energy`) — the driver stage `window`
   exists but was dropped for call budget. It is declared in prereg §5 and is
   *not* used by any verdict.

**Identifiability finding (the actual scientific content).** Between `θ = 0`
and the field-objective optimum the *training* loss changes by 0.03 %
(1.27313e-2 → 1.27274e-2), and between `θ = 0` and the integral optimum it
changes by 2.2 % (1.29441e-4 → 1.2660e-4) — both far below what any BPTT/Adam
tolerance can resolve. The consequence is visible in §④: the selected θ varies
by a factor 50 across objectives/starts while the held-out KE error varies by
only 0.6 pp. **A2 is supported (objective shifts the selected model from
`ke_err` 1.16 % to 0.561 %) but the effect is a *degenerate* one**: the honest
selection is `θ = 0`, i.e. "no sub-grid model is identifiable on 32² at
`Re = 500`", and only a candidate set that *contains* `θ = 0` can say so.

## ⑥ Failure modes, artefacts, friction

**Failure modes observed**

* **Clamp/bound hit masquerading as convergence**: the integral objectives
  drive θ to the lower clamp in ≤ 20 Adam steps; a pipeline that reports only
  "θ = 1.6e-3" without the `θ = 0` control would present an arbitrary number
  as a calibrated constant.
* **Objective flatness** (0.03–2 %) makes θ unidentifiable; the *physical*
  metric is nevertheless identifiable at 0.6 pp because the model error is
  above the reference's 0.27 % resolution error — the two sensitivities must be
  reported separately, which is why P2 (θ) fails while P4 (regret on the
  physical observable) passes.
* **Pointwise field error is a floor, not a metric** (14.60–14.62 % for every
  model including the no-model control) — reported, never used as judge.
* Deferred stage: training-window sensitivity (see ⑤).

**Artefacts** — `artifacts/xval_r2_selection_bptt_tgv/results.json` (all 12
arms with full BPTT histories, 6 controls, both sweep curves, verdict block),
`reference.npz`, `fig1_objective_landscapes.png` … `fig4_holdout_metrics.png`,
`figures.json` (captions + sha256 + base64).

**Ecosystem friction (recorded, not worked around silently)**

* **F1** knowledge engine offline (`8021` refused) and knowledge store empty
  (`count: 0`) → the mandated knowledge 三查 returned nothing.
* **F2** the named reference record `r2-selection-bptt-tgv-anchor` does not
  exist in TensorPapers (`papers/*`, `grep -ri bptt`) nor in the platform
  reports table (`scenario_key` exact match empty); the R2 practice was
  therefore reconstructed from the round-1 XVAL record (report 17002) + this
  round's prereg. A future feeder should publish the reference record as a
  report, not only as a mission name.
* **F3** no outbound network / no local OSS checkout → OSS source not
  re-fetched; provenance inherited from the repo's recorded source-line table.
* **F4** cluster queue depth **311 ahead** at submission (`cj1851` submitted
  with `stage_dir`); all stages were executed locally in < 100 s chunks instead
  of waiting. Total local compute ≈ 11 min.
* **F5** shell calls are capped at ~120 s, hence the stage registry.

## ⑦ TensorLBM vs OSS difference list and porting tradeoffs (required)

| aspect | OSS (`loliverhennigh`, TF1 graph mode) | TensorLBM (this round) | impact |
|---|---|---|---|
| sub-grid algebra | `Q_oss = √(2 Σ_i (f_i−f_i^eq)² EEk_i)`, `τ = ½(τ0+√(τ0²+6 Q Sc/ρ))` | literature form `Q = √(Π:Π)`, `τ = ½(τ0+√(τ0²+18√2 Cs² Q/ρ))`; `oss_proxy` kept as OSS-fidelity arm | one *numeric* constant is not transferable between the two forms (round 1: `Sc = 0.17` ≈ 25 % of the literature `Cs = 0.17` eddy viscosity) |
| default constant | `Sc = 0.17` (hard-coded) | none; θ selected, with `θ = 0` in the candidate set | the OSS default is close to its *own* field-objective optimum (0.17 vs golden 0.18962, 10 % low) but 2.0× worse than no model on the held-out KE integral (1.137 % vs 0.561 %) |
| objective | single terminal field loss inside the graph | three objectives; train snapshots 50/100/150, **held-out** 200 | the objective, not the closure, decides the physical outcome (1.330 % vs 0.561 % at fixed closure) |
| optimiser | training script unfinished upstream (`lr`/`total_loss` undefined in the inventory) → no reference optimum | Adam on `log10 θ`, `lr = 0.1`, 20 iters + sweep/golden controls | OSS supplies the *mechanism* only; the optimum had to be cross-validated here |
| unroll length / loss cadence | whole training loop in-graph | 200 steps, snapshot cadence 50 | flat objectives need ≥2 training snapshots + 1 hold-out, otherwise "converged" is meaningless |
| precision | TF1 (float32 default) | float64 | gradient check at 7e-5 relative is only possible in float64 |
| boundary conditions | also mask bounce-back / Zou-He (separate inventory features) | periodic torus only (this XVAL) | out of scope here; 3D differentiable path already covers walls in TensorLBM |

**Porting tradeoffs (recommended).**
1. **Keep** the OSS in-graph `τ` + `Unroll()` mechanism — it is exact (P1: FD
   agreement 7e-5 relative).
2. **Reject** the OSS implicit assumption that a field-matching objective is a
   sufficient selection criterion: on a broadband/coherent 2D flow the field
   metric is dominated by decorrelation (14.6 % for *every* model).
3. **Adopt** the literature closure for reporting and `oss_proxy` only as a
   fidelity arm; never carry `Sc = 0.17` across closure forms.
4. **Add** to the pipeline, as a hard rule: `θ = 0` in every candidate set, a
   held-out snapshot, and an identifiability margin (`ΔJ/J` over the θ range)
   printed next to any "calibrated" constant. Without it the pipeline reports
   arbitrary constants with 0.03 % loss differences.
5. **Do not** port upstream's single terminal loss; port its graph structure.

## Verdict

| id | quantity | err | pass |
|---|---|---|---|
| P1 | BPTT gradient vs central FD | 0.007 % | ✅ |
| P2 | selection agreement with derivative-free control | loss gap ≤ 0.02 % ✅ / θ 16.0 % ❌ (flat objective) | mixed |
| P3 | selected model vs held-out KE (<15 % owner bar) | 0.561 % | ✅ |
| P4 | regret of the training-loss selection | 5.3e-5 pp | ✅ |
| P5 | start-point reproducibility | 2.3 % (field, smag) ✅ / 16.3 % (field, oss) ❌ / 850–1050 % (integral, clamp) ❌ | mixed |

**Bottom line.** The BPTT *selection pipeline* is mechanically correct
(gradient, regret, physical accuracy all pass by large margins), but on this
anchor the sub-grid constant is **not identifiable**: the honest selection is
"no model" (`ke_err` 0.561 %, indistinguishable from the best point of the
whole candidate space to 5e-5 pp), and the choice of objective moves the
selected model between 0.561 % and 1.33 %. The decision this study supports is
therefore **A2-with-a-caveat**: the objective dominates the closure, but the
correct pipeline response is to select `θ = 0` and report the identifiability
margin — not to publish a calibrated constant. Coverage limits: two closures,
one flow, one `Re`, one grid pair; `θ` values are identified *for this case
only*.