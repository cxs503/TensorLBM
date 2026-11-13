# XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER

Cross-validation of the *graph-unroll / BPTT / LES-parameter* feature of
`loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow` against a
re-implementation in TensorLBM.

| item | value |
|---|---|
| mission | 17843 |
| anchor | XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER |
| OSS target | `loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow`, feature tag `graph-unroll-bptt-les-parameter` (OSS inventory id 14, report 9555; `oss_solutions` ids 37 & 39) |
| OSS source refs (zero transcription) | `LatFlow/Domain.py:134-140, 192-196, 244-253` (graph_unroll / `Unroll()`), `LatFlow/Domain.py:122-128` + `90-96` (tau from field + scalar `Sc`), `LatFlow/Domain.py:57` (`Sc = 0.17` default) |
| code | `src/tensorlbm/les_graph_unroll/` (solver2d, reference2d, calibrate) |
| driver | `scripts/xval_graph_unroll_bptt_les.py` (stage-chunked, reproducible) |
| figures | `scripts/xval_graph_unroll_bptt_les_figures.py` |
| tests | `tests/test_les_graph_unroll_bptt.py` (6 passed) |
| artefacts | `artifacts/xval_graph_unroll_bptt_les/{results.json, oss_source_notes.json, fig1..fig4}.png` |
| claim level | REAL — every number below is emitted by the driver into `results.json` |

---

## 1. Problem, OSS reference and what is actually being cross-validated

The upstream repository is the 2016-2017 TF1-graph-mode LBM that first put the
*whole time loop inside the computational graph*. Its genuinely novel idea (as
recorded in the platform inventory from the source) is narrow and precise:

* `collide`/`stream` have a `graph_unroll` branch that returns in-graph tensors
  instead of assigning into a TF `Variable`;
* `Unroll()` chains `N` steps on the Python side and hands back the state chain;
* `tau` is recomputed **inside the graph** from the local field and the scalar
  model constant `Sc`, therefore a field mismatch `N` steps downstream
  back-propagates into the model constant.

Upstream's own training script is unfinished (the inventory notes `lr` /
`total_loss` are undefined), so the *only* thing worth cross-validating is the
mechanism: **can a field-matching loss on an unrolled LBM trajectory train an
LES constant, and how accurate is the resulting parameter and the resulting
flow?**

Two questions are deliberately separated, because they have different answers:

* **method question** (Stage 1, identical twin): is the gradient/optimum right?
  The reference is a coarse run of the *same* kernel at a planted `Cs_truth`
  (`0.17`, `0.30`), so the recoverable answer is known exactly and the
  calibration is falsifiable.
* **physical question** (Stage 2): does a single Smoluchowski/Smagorinsky
  constant reproduce a fine-grid (DNS-proxy) solution within the owner's 15 %
  bar, starting from filtered DNS data?

### Workload and parameters (all dimensionless numbers listed)

| quantity | symbol | value |
|---|---|---|
| domain | — | `2π × 2π`, periodic (torus) |
| reference velocity | `U0` | 0.1 (lattice) |
| Mach number | `Ma = U0/c_s`, `c_s=1/√3` | 0.173 |
| integral length | `L` | 1.0 |
| physical viscosity | `ν = U0 L / Re` | `2.0e-4` (Re = 500) |
| coarse grid | `nx_c` | 32 (`Δx = 0.19635`) |
| coarse relaxation time | `τ0 = 0.5 + 3ν/Δx²` | **0.515563** |
| reference grid | `nx_r` | 128 (`Δx = 0.049087`), `τ_ref` = **0.749007** |
| convergence grid | `nx_conv` | 256, `τ = 1.4960` |
| LES filter | — | 4×4 box average of the reference solution |
| unroll window | `T` | 200 steps (≈ 0.63 large-eddy turnover = `2π/U0` in lattice units × `Δx`-scaled) |
| loss snapshots | `t_k` | 50, 100, 150, 200 |
| initial conditions | — | 2D Taylor-Green vortex; 6-mode analytic solenoidal field (fixed phases, **no random seed anywhere**) |
| closure constants | `Cs` | sweep 0.000…0.600 (25 pts), twin truths 0.17/0.30, upstream default 0.17 |
| BPTT optimiser | — | Adam on `log10 Cs`, 20 iterations, `lr = 0.1`, start `Cs = 0.02` (fixed, not retuned) |

Loss (relative-`L₂`, mean over snapshots):

```
J(Cs) = (1/K) Σ_k  ‖u_LES(t_k) − ū_ref(t_k)‖² / ‖ū_ref(t_k)‖²
```

## 2. Method and implementation

`src/tensorlbm/les_graph_unroll/solver2d.py` — differentiable D2Q9 BGK:
streaming is 9 `torch.roll` calls (periodic, autograd-native), collision is
`f − (f − f_eq)/τ` with `τ` a **graph function of the field and of `Cs`**.
Two closures:

* `smagorinsky` (used for everything that is judged): with
  `Π^neq_ab = Σ_i (f_i−f_i^eq)c_ia c_ib`, `Q = √(Π:Π)` and
  `Π^neq_ab = −(2/3) τ ρ S_ab` one gets `|S| = 3√2 Q/(2τρ)`, so
  `ν_t = Cs²|S|` and `τ = ½ + 3(ν0+ν_t)` close to

  ```
  τ = ½ ( τ0 + √( τ0² + 18√2 Cs² Q / ρ ) )
  ```

* `oss_proxy` (comparison arm only, reproduced from the OSS record):
  `Q = √(2 Σ_i (f_i−f_i^eq)² EEk_i)`, `EEk_i = Σ_{n,m}|c_in c_im|`,
  `τ = ½(τ0 + √(τ0² + 6 Q Sc/ρ))`.

`calibrate.py` — `unroll()` (the `Unroll()` analogue), `field_loss`,
`bptt_calibrate` (Adam on `log10 Cs`), and three controls that share **no
autograd code** with it: `fd_gradient` (central difference), `sweep_loss`,
`golden_minimize`; plus `measure_eddy_viscosity`, which measures the closure's
realised eddy viscosity from the linear decay of a single shear mode and
compares it with `Cs²|S|`.

`reference2d.py` — numpy (no autograd) BGK reference runs; all grids use the
**same physical** viscosity (`τ` per grid from `ν_lat = ν/Δx²`), so grid
refinement changes only the discretisation, never the problem.

Reproduce: `python3 scripts/xval_graph_unroll_bptt_les.py all` (or one stage at
a time: `closure | twin tgv 0.17 | phys tgv | convergence tgv | twin_start |
oss_diff`), then `python3 scripts/xval_graph_unroll_bptt_les_figures.py`.

## 3. Results (programmatic, from `results.json`)

### 3.1 X4 — the closure itself is right (analytic shear probe)

| `Cs` | `ν_t` measured | `ν_t = Cs²|S|` analytic | ratio | err |
|---|---|---|---|---|
| 0.10 | 1.030e-4 | 9.817e-5 | 1.047 | 4.7 % |
| 0.17 | 2.970e-4 | 2.837e-4 | 1.046 | 4.6 % |
| 0.30 | 9.190e-4 | 8.836e-4 | 1.040 | 4.0 % |
| 0.50 | 2.519e-3 | 2.454e-3 | 1.026 | 2.6 % |

The `18√2` prefactor is therefore verified to ≲5 % (the residual is the
finite-amplitude decay of the probe, which makes the running `|S|` slightly
smaller than its initial value).

### 3.2 Stage 1 — identical twin: gradient, optimum, recovery

| case / truth | autograd `dJ/dCs` vs central FD (rel. err at 4 probes) | BPTT `Cs*` | golden `Cs*` | recovery err (BPTT) | `J` init → final |
|---|---|---|---|---|---|
| tgv / 0.17 | 0.004 %, 0.005 %, 0.009 %, 0.004 % | 0.1564 | 0.1700 | **8.0 %** | 7.6e-5 → 2.7e-8 |
| tgv / 0.30 | 0.001 %, 0.002 %, 0.003 %, 0.001 % | 0.2726 | 0.3000 | **9.2 %** | 7.3e-4 → 4.7e-8 |
| fourier / 0.17 | 0.005 %, 0.005 %, 0.008 %, 0.002 % | 0.1607 | 0.1700 | **5.5 %** | 1.8e-3 → 1.1e-5 |

Golden-section recovers the planted constant to machine precision (0.00 %),
which confirms the twins are genuinely parameter-identifiable; the 5-9 %
BPTT residual is the fixed 20-iteration Adam budget, not a gradient error
(the gradient itself agrees with FD to 1e-4 relative — figure `fig4`).

### 3.3 Stage 2 — physical cases vs the fine-grid reference

Taylor-Green vortex (`fig2`, `fig3`, left):

| arm | `Cs` | terminal relative-`L₂` field err | KE terminal err | KE *decay* err |
|---|---|---|---|---|
| no model | 0 | 14.62 % | 1.16 % | 6.58 % |
| upstream default | 0.17 | 14.59 % | 3.53 % | 20.04 % |
| **BPTT** | 0.1099 | **14.60 %** | 2.17 % | 12.28 % |
| golden (independent) | 0.1241 | 14.59 % | 2.44 % | 13.84 % |

6-mode synthetic turbulence (`fig2`, `fig3`, right):

| arm | `Cs` | terminal field err | KE terminal err |
|---|---|---|---|
| no model | 0 | 87.51 % | 1.15 % |
| upstream default | 0.17 | 85.48 % | 5.94 % |
| BPTT | 1.000 (clamped at the declared upper bound) | 72.15 % | 70.43 % |
| golden | 0.600 (clamped at the interval end) | 75.95 % | 45.55 % |

### 3.4 OSS closure vs literature closure (the migration trade-off, quantified)

| quantity | literature closure | OSS proxy closure |
|---|---|---|
| `ν_t` at `Cs = 0.17` (shear probe) | 2.970e-4 | 7.371e-5 (**ratio 0.248**) |
| calibrated optimum on the tgv DNS loss | `Cs* = 0.1241` | `Sc* = 0.2558` |
| — ratio of calibrated constants | | **2.061** |

`√(18√2 / 6) = 2.060`, i.e. the two closures differ exactly by their declared
prefactor and the calibrated constants scale as its square root. This is a
clean, quantitative explanation of the inventory remark that the upstream
`6 Q Sc` form is *not* equivalent to `18√2 Cs² Q/ρ` and "needs recalibration":
a user porting upstream's `Sc = 0.17` into a literature-form Smagorinsky sees
only **25 %** of the intended eddy viscosity, and needs `Cs ≈ 0.085` (or `Sc ≈
0.26` in the upstream form) for the same SGS stress level.

## 4. Figures

* `fig1_twin_loss.png` — twin loss curves with BPTT iterates, golden optimum
  and the planted truth (`log J` vs `Cs`).
* `fig2_ke_dns.png` — kinetic-energy decay of every arm vs the filtered
  fine-grid reference, both cases.
* `fig3_identifiability.png` — physical loss curve `J(Cs)`; the title prints
  the sweep range as a percentage of `J(0)`, which is the identifiability
  diagnostic.
* `fig4_gradient.png` — `|dJ/dCs|` from autograd (BPTT) vs central FD on a log
  axis, all twin problems.

## 5. Attention points, limitations, and the honest failures

1. **The method cross-validates; the single constant does not.** For the TGV at
   `Re = 500` on 32² the loss improves by only **0.15 %** between `Cs = 0` and
   the optimum (`J(0) = 0.014893` vs `J_min = 0.014871` at `Cs = 0.125`), and
   the BPTT calibration buys a **0.17 %** reduction of the terminal field error.
   The coarse BGK chain already carries enough *implicit* numerical dissipation
   that an explicit SGS viscosity mostly adds to it: measured KE decay error is
   6.6 % with no model vs 12.3 % at the calibrated `Cs`. The predeclared
   "≥30 % improvement over the no-model baseline" gate **fails** — a one-constant
   Smagorinsky model is effectively *unidentifiable* in this configuration.
2. **Pointwise field matching breaks down for broadband 2D turbulence.** With
   the 6-mode initial condition the fine and coarse solutions decorrelate
   (relative-`L2` error 53 % already at `t = 50`, 87 % at `t = 200`), because 2D
   enstrophy-cascade dynamics amplify sub-grid differences. BPTT then slides to
   the declared bound (`Cs = 1.0`), "improves" the pointwise metric by 18 % while
   destroying the energy integral (70 % KE error). The gradient is correct; the
   *objective* is the wrong observable for this flow. Statistically robust
   observables (energy/enstrophy decay, spectra) are the honest alternative and
   are the natural next node.
3. **Upstream default `Sc = 0.17` is not transferable.** In its own closure form
   it delivers 25 % of the literature `Cs = 0.17` eddy viscosity, and in the
   literature form it is >50 % too dissipative for this case (KE decay error
   20.0 % vs the 15 % bar).
4. **Reference resolution is adequate for the integral quantity but is a proxy,
   not DNS**: 128 → 256 KE difference is 0.27 % (tgv) and 0.01 % (fourier), so
   the *energy* observable is grid converged; pointwise fields are not (that is
   the error floor of the 14.6 %/87 % figures above, not a model effect).
5. **Not executed** (predeclared as exploratory, dropped for wall-clock
   budget): unroll-horizon sensitivity `T = 100` vs `200`. The two sensitivity
   axes actually delivered are (i) reference-grid convergence 128 → 256 and
   (ii) BPTT start-point robustness (start 0.02 → `Cs = 0.1564`; start 0.45 →
   `Cs = 0.1472`; 5.9 % spread, both within 13.4 % of the truth).
6. **Scope of the port**: this XVAL covers periodic boxes only. Upstream's
   mask-blend bounce-back and Zou-He in/outlet reconstruction are *separate*
   inventory features (`zou-he-channel-inlet-outlet-reconstruction`) and were
   not re-derived here; TensorLBM's 3D differentiable path
   (`autograd_path.rollout` + `collide_smagorinsky_bgk3d`) already provides the
   wall/mask counterpart.

## 6. Verdict — does every key quantity meet the 15 % bar?

| id | quantity | got | reference | err | <15 % |
|---|---|---|---|---|---|
| X1 | autograd (BPTT) `dJ/dCs` | see §3.2 | central FD, `h = 1e-3` | 0.001–0.009 % | ✅ |
| X2 | BPTT optimum, twin (3 problems) | 0.1564 / 0.2726 / 0.1607 | 0.1700 / 0.3000 / 0.1700 (golden) | 5.5–9.2 % | ✅ |
| X2 | BPTT optimum, physical tgv | 0.1099 | 0.1241 (golden) | 11.5 % | ✅ |
| X2 | BPTT optimum, physical fourier | 1.000 | 0.600 (golden) | 66.7 % (both at declared bounds) | ❌ |
| X3 | planted-constant recovery | 0.1564 / 0.2726 / 0.1607 | 0.17 / 0.30 / 0.17 | 8.0 / 9.2 / 5.5 % | ✅ |
| X4 | closure eddy viscosity | 0.103–2.519e-3 | `Cs²|S|` | 2.6–4.7 % | ✅ |
| X5 | tgv terminal field err (BPTT) | 14.60 % | filtered 128² reference | 14.6 % | ✅ (marginal) |
| X5 | tgv KE decay err (BPTT) | 12.28 % | reference decay | 12.3 % | ✅ |
| X5 | tgv improvement over no model | 0.17 % | predeclared ≥30 % | — | ❌ |
| X5 | fourier terminal field err (BPTT) | 72.15 % | filtered 128² reference | 72.2 % | ❌ |

**Bottom line.** The OSS mechanism (`graph_unroll` + in-graph `tau` + BPTT) is
faithfully reproduced in TensorLBM and is *mechanically verified*: the
autograd gradient equals a central finite difference to `1e-4` relative, the
calibrated constant lands within 5.5–9.2 % of a planted truth and within 11.5 %
of an independent derivative-free optimum. What does **not** transfer is the
upstream parameter value and the "one constant fixes the flow" expectation: at
`Re = 500` on a 32² grid the single Smagorinsky constant is unidentifiable for
a coherent vortex (0.15 % loss improvement) and pathological for broadband
turbulence under a pointwise-`L₂` objective. Upstream's own `Sc = 0.17` is a
25 %-equivalent of the literature constant in its own closure form.

## 7. Ecosystem friction (recorded, not worked around silently)

1. **No outbound network from the shell** (`raw.githubusercontent.com` →
   `000`), so the OSS source could not be fetched live. Provenance came from the
   platform records instead (`oss_inventory` id 14, `oss_solutions` ids 37/39,
   `report 9555`, with source line numbers) — recorded in
   `artifacts/xval_graph_unroll_bptt_les/oss_source_notes.json`.
2. **`import tensorlbm` resolves to a foreign worktree**: an editable install
   points at `/root/tp-agent/work/m17719/src/...`, i.e. *another agent's* live
   checkout. All runs here force `PYTHONPATH=src` / `pythonpath=["src"]`
   (the repo's pytest config already does this, ad-hoc scripts do not).
3. **Cluster queue depth 312** at submission time (`cj1697` probe) → the work
   was chunked locally into < 100 s stages instead of waiting on the queue;
   the driver was therefore written stage-wise with incremental `results.json`
   merging (`all` still reproduces everything in one command).
4. **Knowledge base is empty** (0 entries, `knowledge/search` for
   `lbm les bptt`, `graph unroll`, `smagorinsky`, `differentiable lbm` returned
   0 hits; `knowledge/stats.count = 0`), so the mandated "开工三查" found no
   prior art to build on and no negative-knowledge entry to avoid.
5. Local shell calls are capped at ~120 s, which is why the first monolithic
   driver run was killed mid-flight; the stage registry is the direct
   consequence.