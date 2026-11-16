# XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY — round 3: the unsteady-Stokes reference of round 2 was wrong by √2

*Repo: TensorLBM · anchor `XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY` · mission m23214 · parent: `r2/REPORT2.md` (mission m22617-era closure, claim "C_I within 6.0–14.6 %, PASS at the 15 % gate, quadrature WITHHELD at 29–38 % low")*

## ⓪ What changed and why it matters

Round 2 closed this anchor with the verdict *"the in-phase added-mass coefficient lands at
+6.0 / +10.7 / +14.6 % against the exact unsteady-Stokes reference, i.e. inside the owner
15 % gate; resolution (the Stokes layer), not the estimator, sets the error; the quadrature
channel is 29–38 % low and stays WITHHELD"*.

This round audited the **reference itself** and found it is wrong:

```
shipped:   delta/R = 2 sqrt(2/(pi beta))          k = 1/(delta/R)
derived:   delta = sqrt(2 nu/omega) => delta/R = 2/sqrt(pi beta),
           k = R sqrt(omega/nu) = sqrt(pi beta/2) = sqrt(2)/(delta/R)
```

With `beta = D^2/(nu T)` and `T = 2 pi/omega` both mappings are exact, and the shipped ones
are exactly **√2 off** in `delta/R` (layer quoted 41 % too thick) and **1/√2 off** in `k`.
Because `Gamma` is evaluated at the wrong `k`, the reference the round-2 verdict was scored
against overstates the in-phase coefficient by **+18.5 % at beta = 50** and the quadrature
coefficient by *2.1×*.

Consequences (all recomputed by `r3/analyze_r3.py` from the **committed** round-2 raw JSON —
no re-run involved, no number typed by hand):

| arm (D = 16) | β | C_I E1 | ref shipped | ref corrected | err vs shipped | err vs corrected |
|---|---|---|---|---|---|---|
| `r2l_b25_d16` | 25 | 1.6850 | 1.5890 | 1.2775 | +6.0 % | **+31.9 %** |
| `r2_b50_d16` | 50 | 1.5569 | 1.4062 | 1.1867 | +10.7 % | **+31.2 %** |
| `r2l_b100_d16` | 100 | 1.4636 | 1.2775 | 1.1227 | +14.6 % | **+30.4 %** |

The round-2 PASS lines were an artefact of the reference error: on the corrected reference
the same measurements are **~ +31 %**, i.e. **outside** the owner 15 % gate. The quadrature
channel does not merely sit low, it sits **+43 % high** against the corrected
`-Im Gamma(beta=50) = 0.2382` (round 2 compared against 0.5009). Both predeclared
hypotheses H1 and H2 of `r3/predeclared.json` are confirmed.

The other consequence is that round 2's *resolution axis* was mis-scaled: what it called a
Stokes layer of 2.55 / 1.81 / 1.28 cells at β = 25 / 50 / 100 is really **1.81 / 1.28 /
0.90** cells. The primary arm was below one cell in true thickness — which is exactly why
the "resolution, not physics" reading looked plausible.

## ① Case

Doubly periodic D2Q9 box of side `box = 4D`, fixed staircase circular cylinder of reference
diameter `D`, uniform harmonic body force `a(t) = a0 cos(omega t)` on fluid nodes (second-order
Guo source), no moving boundary, no pressure gradient. Similarity numbers
`KC = 2 pi u0/(omega D)`, `beta = D^2/(nu T)`, `Ma = u0 sqrt(3)`; free parameters of the
reference comparison are the **true** resolution parameters

```
delta = D/sqrt(pi beta)  [cells],     k = sqrt(pi beta/2)
```

Round-2 arms re-scored here: `D = 16`, `box = 4D` (`nx = 64`), `u0 = 0.01`, `omega = 2 pi/200`,
`beta = 25 / 50 / 100`, plus the half-drive (`KC` halved) and the 4-neighbour interior
controls. No `src/` physics was changed to obtain these numbers — only the analytic
reference (`stokes_cylinder_coefficients`) and its test.

## ② How

1. **Reference audit.** `delta = sqrt(2 nu/omega)` and `k = R sqrt(omega/nu)` are derived
   from `beta = D^2/(nu T)` and pinned as identities in
   `tests/test_stokes_layer_thickness.py::test_beta_to_k_and_delta_mappings_are_consistent`,
   including `k * delta/R = sqrt(2)` and the independent large-`k` asymptote
   `Re Gamma -> 1 + 2/k`, `-Im Gamma -> 2/k + 1/k^2` (from `K_1/K_0 = 1 + 1/(2z)`, *not*
   from the code under test).
2. **Independent physical measurement of the layer.** The same test file measures the decay
   length of the oscillating velocity next to a no-slip wall **with the LBM** (Stokes'
   second problem, `ny = 96`, `delta` target 6 cells): the complex first-harmonic deficit
   `D(y) = U(plateau) - U(y)` has modulus `e^{-y/delta}` and phase `-y/delta`, so both a
   logarithmic amplitude fit and a phase fit return `delta` without using any Bessel
   function. Measured **6.18 cells amplitude / 6.18 cells phase vs 6.00 exact** (2–3 %), and
   the corrected `delta_over_R(beta)` of the library reproduces that measurement for
   `D = 16 / 32 / 64` within 2 % — the shipped formula would have been 41 % off.
3. **Re-scoring, not re-running.** `r3/analyze_r3.py` rebuilds the reference from each raw
   file's own `case.beta` with the corrected mapping and reports the shipped value side by
   side (recovered from the raw JSON, so the audit is reproducible from committed evidence
   alone).
4. **Resolution rungs (submitted, see ⑤).** Same physics at fixed `beta = 50`, `KC`, `Ma`,
   `Re`, `box/D`, refined `D = 16 -> 32 -> 64` i.e. `delta_true = 1.28 -> 2.55 -> 5.11`
   cells, driving `r3/run_r3.py` (round-2 machinery reused verbatim: ramp-on, convergence
   gate instead of a fixed transient, three independent estimators E1/E2/E3). Submitted as
   cluster jobs **cj4691** (`r3_b50_d32`, `r3_b25_d32`, `r3_b100_d32`) and **cj4692**
   (`r3_b50_d64`); both were still `queued` (queue depth > 120) when this report was written.

## ③ Results

Full generated tables: `r3/summary_r3.md` (produced by `r3/analyze_r3.py`).

### 3.1 Reference audit (generated, T1)

| arm | β | δ true [cells] | δ shipped [cells] | k shipped | k corrected | Γ_re shipped | Γ_re corrected | C_a same-mask | ref shipped | ref corrected |
|---|---|---|---|---|---|---|---|---|---|---|
| `r2l_b25_d16` | 25 | 1.805 | 2.553 | 3.1333 | 6.2666 | 1.6414 | 1.3196 | 0.9681 | 1.5890 | 1.2775 |
| `r2_b50_d16` | 50 | 1.277 | 1.805 | 4.4311 | 8.8623 | 1.4525 | 1.2258 | 0.9681 | 1.4062 | 1.1867 |
| `r2l_b100_d16` | 100 | 0.903 | 1.277 | 6.2666 | 12.5331 | 1.3196 | 1.1596 | 0.9681 | 1.2775 | 1.1227 |

### 3.2 Re-scored in-phase error (generated, T2)

| arm | C_I E1 | C_I E2 | C_I E3 | err vs shipped (E1) | err vs corrected (E1) | E2 [%] | E3 [%] |
|---|---|---|---|---|---|---|---|
| `r2l_b25_d16` | 1.6850 | 1.8674 | 1.6860 | +6.0 % | **+31.9 %** | +46.2 % | +32.0 % |
| `r2_b50_d16` | 1.5569 | 1.7218 | 1.5578 | +10.7 % | **+31.2 %** | +45.1 % | +31.3 % |
| `r2l_b100_d16` | 1.4636 | 1.6248 | 1.4646 | +14.6 % | **+30.4 %** | +44.7 % | +30.5 % |
| `r2l_b50_d16_kc_half` | 1.5580 | 1.7232 | 1.5590 | +10.8 % | +31.3 % | +45.2 % | +31.4 % |
| `r2l_b50_d16_int4` (defect arm) | 1.6258 | 1.6857 | 1.5610 | +15.6 % | +37.0 % | +42.0 % | +31.5 % |

E1 (Ladd exchange) and E3 (global momentum balance, exact) agree to 1e-3 on every healthy
arm, so the **measured force** is not in doubt; the disagreement is with the reference.

### 3.3 Quadrature channel (generated, T3)

| arm | C_Q E1 | −ImΓ shipped | −ImΓ corrected | err vs shipped | err vs corrected |
|---|---|---|---|---|---|
| `r2_b50_d16` | 0.3403 | 0.5009 | 0.2382 | −32.1 % | **+42.8 %** |
| `r2l_b25_d16` | 0.5235 | 0.7363 | 0.3441 | −28.9 % | +52.1 % |
| `r2l_b100_d16` | 0.2142 | 0.3441 | 0.1659 | −37.8 % | +29.1 % |

The channel stays WITHHELD — but for the opposite reason than round 2 gave. The round-2
mechanistic story ("staircase tangential area *underestimates* the quadrature") is
falsified by its own data.

### 3.4 Where the residual actually comes from (generated, T5/T5b)

At **fixed D = 16** the corrected in-phase error is flat while the Stokes layer changes by
2×: **+31.9 % (δ = 1.805) / +31.2 % (δ = 1.277) / +30.4 % (δ = 0.903)** — spread **1.5 pp**
for a 2× change in δ. So the residual is **not** a Stokes-layer resolution effect, contrary
to round 2's central claim ("resolution, not the estimator, sets the error").

Inverting the corrected analytic curve (T5b) gives the numerical layer that would explain
the measured force: `k_implied = 3.30 / 3.92 / 2.72` against `k = 8.86 / 12.53 / 6.27`, i.e.
the LBM behaves like a Stokes layer **2.3–3.2× thicker** than the exact one, *without* the
ratio being constant — so it is not simply a rescaling of ν.

The standing candidate mechanism (predeclared H6, added after this re-scoring and before the
new arms returned) is the **staircase mask**: halfway bounce-back places the wall half a cell
outside the last solid centre (blocked area `n_solid + P/2`, +11–16 % at D = 16, decreasing
like `1/D`) while the same-mask potential reference uses `n_solid`, and a staircase surface
raises the added mass further. The D = 32 / 64 arms were submitted precisely to discriminate
this `~C/D` law from a flat offset. **That discrimination is not in this report** (see ⑤).

## ④ Figures

| figure | content |
|---|---|
| `r3/figures/fig1_reference_audit.png` | left: shipped (dashed, red) vs corrected (solid, blue) `Re Gamma(beta)` with the measured reference points of every healthy arm; right: the same for the quadrature reference with the **measured** `C_Q` (black triangles) — the measurement sits between the two curves, which is exactly why the sign of the discrepancy flipped |
| `r3/figures/fig2_error_rescored.png` | left: in-phase error vs the **true** Stokes layer with both references and the 15 % / 3 % gates; right: the same error vs β at fixed D = 16 — flat, which is the falsification of the Stokes-layer explanation |

## ⑤ Caveats (honest state)

* **The resolution rungs are not in this report.** `cj4691` / `cj4692` were submitted and
  still queued at report time (queue depth > 120; local heavy runs are forbidden by the
  capacity rule — `load15 ≈ 107` on a 128-core box). The generator is committed and runnable
  unchanged, and `r3/analyze_r3.py` picks up `r3/runs/*/result.json` automatically, so the
  missing rungs fold into the tables with no code change. Hypotheses H3/H4/H5 and the
  `~C/D` versus flat discrimination of H6 remain **not_run / WITHHELD**.
* **The mechanism is a hypothesis, not a verdict.** The 2.3–3.2× "effective layer" ratio is
  a diagnostic read-out (inversion of the analytic curve), not a proof of a staircase
  bias; the planned mask-normalisation A/B (reference built on `n_solid + P/2`) has *not*
  been run.
* **What the fix does not change.** Round 2's protocol work (convergence gate, ramp,
  three-estimator agreement, the 8-neighbour interior-zeroing defect) stands and is
  unaffected; those invariants still hold on every healthy arm (`|dM|/M ≲ 3e-7`,
  `max|rho-1| ≈ 1e-2` cold-start, `|E1-E3| = 1e-3`).
* **Downstream readers.** Any report that quoted `Gamma_re`, `C_Q_viscous` or a Stokes-layer
  thickness from `tensorlbm.oscillatory_cylinder_inertia` before this change should be
  re-scored: the reference is now 5–19 % smaller in phase and up to 2.1× smaller in
  quadrature at `beta = 25..100`, and every "resolution in cells" number doubles its
  honest value (it halves, i.e. the layer was thinner than quoted).
* **Caveat on the caveat.** The corrected reference is bounded by its own assumptions
  (unbounded-domain `Gamma` factorised against a confined potential solve); the
  mask-sensitivity of the potential part is 2.9 % (`reference_mask_sensitivity_pct` in the
  JSON), which does not change any verdict above.

## ⑥ Error judgement, item by item

| quantity | reference | value | error | < 3 %? | < 15 %? |
|---|---|---|---|---|---|
| `delta` (Stokes layer, LBM measurement, δ target 6 cells) | `sqrt(2 nu/omega)` | 6.18 cells (amplitude fit), 6.18 cells (phase fit) | 3 % | **no** | **yes** |
| library `delta_over_R(beta)` vs that measurement, D = 16/32/64 | measured | within 2 % | ≤ 2 % | **yes** | **yes** |
| `Re Gamma(beta=50)`, corrected mapping vs large-`k` asymptote `1+2/k` | asymptote | 1.2258 vs 1 + 2/8.862 = 1.2257 | 0.01 % | **yes** | **yes** |
| `C_I` (E1), β = 25, D = 16 | `C_a * Re Gamma` corrected | 1.6850 vs 1.2775 | **+31.9 %** | **no** | **no** |
| `C_I` (E1), β = 50, D = 16 (round-2 primary arm) | same | 1.5569 vs 1.1867 | **+31.2 %** | **no** | **no** |
| `C_I` (E1), β = 100, D = 16 | same | 1.4636 vs 1.1227 | **+30.4 %** | **no** | **no** |
| `C_Q` (E1), β = 50 | `-Im Gamma` corrected | 0.3403 vs 0.2382 | **+42.8 %** | **no** | **no** |
| fluid mass budget, healthy arms | exact `dM/M = 0` | ≤ 3e-7 | — | **yes** | **yes** |
| E1 vs E3 (estimator independence) | exact momentum balance | ≤ 1e-3 | — | **yes** | **yes** |

**Answer to "is it < 3 %?": no.** The corrected reference removes the round-2 PASS and puts
the in-phase coefficient at **+30.4 … +31.9 %** (owner 15 % gate: FAIL; 3 % caliber: FAIL) at
`D = 16`, and the quadrature channel at **+29 … +52 %** (WITHHELD). What *is* inside 3 % is
the reference chain itself: the β→δ→k→Γ mapping is now verified by an independent LBM
measurement of the Stokes layer (3 %) and by an independent asymptotic expansion (0.01 %).
The claim that a 3 % in-phase result is reachable by refinement is **not made** — the D = 32 /
64 arms that would test it are queued.

## ⑦ Reproduce

```bash
export PYTHONPATH=$PWD/src
python -m pytest tests/test_stokes_layer_thickness.py tests/test_oscillatory_cylinder_inertia.py -q
python xval/XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY/r3/analyze_r3.py   # re-scores committed r2 raw JSON
python xval/XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY/r3/make_figures_r3.py
# resolution rungs (cluster): 
python xval/XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY/r3/run_r3.py --arms r3_b50_d32,r3_b25_d32,r3_b100_d32 --threads 8
python xval/XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY/r3/run_r3.py --arms r3_b50_d64 --threads 8
```