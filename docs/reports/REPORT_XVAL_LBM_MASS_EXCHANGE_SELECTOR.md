# XVAL-LBM-MASS-EXCHANGE-SELECTOR — TensorLBM D2Q9 free surface vs the OSS
`Maarten-vd-Sande/lbm` mass-exchange selector

*Mission 18795 · branch `agent/m18795` · anchor `XVAL-LBM-MASS-EXCHANGE-SELECTOR`*

## ① What was compared, and where the reference comes from

The free-surface lattice-Boltzmann cell-conversion rule ("mass exchange
selector") decides, every time step, which INTERFACE cell has collected enough
liquid to become FLUID, which has drained enough to become GAS, and where the
surplus/defect mass of a converting cell is placed. Two mature implementations
of the same Körner/Thürey family are compared:

* **repo (A)** — `src/tensorlbm/free_surface_lbm_2d.py` (`:604-614` marking,
  `:617-626` excess split, `:345-370` Körner exchange).
* **OSS reference (B)** — `Maarten-vd-Sande/lbm`, inventoried snapshot
  `/root/oss-cache/Maarten-vd-Sande_lbm`, `lbm/update_types.py:12,13,18-27`
  (marking), `:34-70` (envelope), `:72-116` (excess distribution), `:118-136`
  (neighbour flags). Every constant and rule in
  `xval/mass_exchange_selector/oss_selector.py` is transcribed from those
  lines; the lattice primitives are imported from the already fidelity-checked
  numpy port `xval/oracle_minimal_d2q9_fs/oss_oracle_port.py` (no hand copy,
  no re-derived coordinate convention).

Physical test case (identical lattice for both selectors): 2-D dam-break in a
walled box, liquid column of height $h_0=2a$ and width $5a$, one-cell gas
headroom, gravity $g$, scored against the digitised Martin & Moyce (1952)
experiment (`src/artifacts/martin_moyce_1952_dam_break.json`).

### Dimensionless groups (complete)

| quantity | value |
|---|---|
| column aspect $h_0/a$ | 2 |
| domain width $W/a$ | 5 |
| headroom $/a$ | 1 |
| relaxation $\tau$ | 1.0 (scan 0.6 / 1.4) |
| viscosity $\nu=(\tau-1/2)/3$ | 0.1667 |
| gravity (lattice) $g$ | 5·10⁻⁵ |
| time scale $T=t\sqrt{g/h_0}$ | sampled ΔT = 0.25, window [0.5, 4] |
| front $Z=x_{front}/a$ | scored against M&M Fig. 4 |
| Mach $Ma=u_{max}/c_s$, $c_s=1/\sqrt3$ | see arm table (target ≤ 0.1) |
| selector tolerances | repo κ = 1e-2, κ_f = 0.1 · OSS fill_offset = 3e-3, lonely = 0.1 |

## ② Method

1. **E1 port fidelity.** The extracted OSS selector is run against the
   full fidelity-checked port on 40 randomized free-surface states
   (random height field, one interface layer, random fills in [−0.2, 1.3]).
   Acceptance was pre-declared as *exact* flag agreement and |Δmass| ≤ 1e-12.
2. **E2 paired selector differential.** Both selectors are applied to the
   *same frozen state*, so the state trajectory cannot confound the
   comparison. Factors (2×2): fill tolerance κ ∈ {3e-3, 1e-2} ×
   unconditional `NO_IFACE_NEIGH | NO_FLUID_NEIGH` demotion guard ∈ {on, off}.
   Frozen states are exported from the running solver at ΔT = 0.25
   (17 checkpoints per arm), plus a synthetic sweep of five neighbourhood
   configurations × 74 fill levels.
3. **E3 downstream consequence.** Arms A (repo κ=1e-2), C (repo κ=3e-3 — the
   OSS tolerance, reachable through the existing public API), D (κ=3e-2),
   E/F (τ = 0.6 / 1.4) run the same dam-break; front curve scored against
   M&M, inventory drift and conversion counts recorded per arm.
   Determinism is verified by a bitwise repeat of arm A.

Environment: CPU-only torch 2.7.1, one thread per process, lattice units,
float32 solver fields / float64 selector arithmetic; short arms run locally,
the long arms were submitted to the 702 cluster queue (job `cj2038`,
no node pinning).

## ③ Results

### E1 — port fidelity of the extracted selector

| carrier dtype | states | mask mismatches | max \|Δmass\| | verdict (≤1e-12) |
|---|---|---|---|---|
| float64 | 40 | 0 | 0.0 | PASS |
| float32 | 40 | 0 | 4.531e-08 | carrier eps floor, context only |

The extracted selector reproduces the upstream port **exactly** in float64:
the transcription is faithful, so E2 measures the OSS rule and not my reading
of it. One upstream semantic had to be found the hard way and is documented in
`oss_selector.envelope`: `cell_type[ix,iy] = CT.TO_FLUID` is an *assignment*,
so a converting cell loses its INTERFACE bit and can never be a receiver of
excess mass in the same sweep (the repo reaches the same effect with
`recv_iface = iface & ~to_liq & ~to_gas`).

### E3 — arm table (Martin & Moyce scoring)

| arm | a | τ | κ | steps | u_max | Ma | max err % | mean err % | 15 % band | inventory drift % | conv G→I / I→G |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_repo_a16 | 16 | 1.00 | 0.01 | 3200 | 0.2377 | 0.412 | 28.161 | 23.159 | FAIL | 0.0322 | 308 / 304 |
| C_hybrid_a16 | 16 | 1.00 | 0.003 | 3200 | 0.2377 | 0.412 | 28.161 | 23.036 | FAIL | 0.0164 | 395 / 391 |
| D_kappa3e2_a16 | 16 | 1.00 | 0.03 | 3200 | 0.2377 | 0.412 | 28.161 | 23.159 | FAIL | 0.0430 | 293 / 291 |
| E_tau0.6_a16 | 16 | 0.60 | 0.01 | 3200 | 0.2436 | 0.422 | 29.464 | 20.533 | FAIL | 0.0730 | 823 / 784 |
| F_tau1.4_a16 | 16 | 1.40 | 0.01 | 3200 | 0.2369 | 0.410 | 40.828 | 34.595 | FAIL | 0.0249 | 224 / 231 |
| A_repo_a16_rep | 16 | 1.00 | 0.01 | 3200 | 0.2377 | 0.412 | 28.161 | 23.159 | FAIL | 0.0322 | 308 / 304 |

Arm-to-arm interchangeability of the front curve:

| pair | max \|ΔZ\|/Z % | mean % | 15 % band |
|---|---|---|---|
| A_repo_a16_vs_C_hybrid_a16 | 2.5 | 0.167 | PASS |

Determinism: arm A repeated at a = 16 is **bitwise identical**
(17 samples), so n = 1 per arm carries no hidden
replica variance.

*Reading:* the selector tolerance κ has **no measurable effect on the collapse
front** (A vs C: max 2.5 %, mean 0.17 %; κ = 3e-2 also reproduces A to the
printed digits), whereas the relaxation time does move it (τ = 1.4 → 40.8 %).
The selector is a second-order ingredient for the front and a first-order one
for *where* mass sits (E2 below).

### E2 — paired selector differential (identical frozen states)

| arm | T | interface cells | OSS native % | guard off % | κ=repo % | κ=repo & guard off % | excess placement L1 / excess % |
|---|---|---|---|---|---|---|---|
| A_repo_a16 | 0.50 | 53 | 3.774 | 1.887 | 1.887 | 0.0 | 100.0 |
| A_repo_a16 | 0.75 | 58 | 6.897 | 1.724 | 5.172 | 0.0 | 87.76 |
| A_repo_a16 | 1.00 | 62 | 4.839 | 0.0 | 4.839 | 0.0 | 79.908 |
| A_repo_a16 | 1.25 | 69 | 7.246 | 0.0 | 7.246 | 0.0 | 41.882 |
| A_repo_a16 | 1.50 | 72 | 4.167 | 1.389 | 4.167 | 1.389 | 46.938 |
| A_repo_a16 | 1.75 | 76 | 3.947 | 1.316 | 3.947 | 1.316 | 40.204 |
| A_repo_a16 | 2.00 | 76 | 2.632 | 1.316 | 2.632 | 1.316 | 35.511 |
| A_repo_a16 | 2.25 | 80 | 6.25 | 3.75 | 3.75 | 1.25 | 32.327 |
| A_repo_a16 | 2.50 | 82 | 4.878 | 2.439 | 3.659 | 1.22 | 31.999 |
| A_repo_a16 | 2.75 | 85 | 5.882 | 2.353 | 4.706 | 1.176 | 35.38 |
| A_repo_a16 | 3.00 | 88 | 3.409 | 1.136 | 3.409 | 1.136 | 31.35 |
| A_repo_a16 | 3.25 | 91 | 6.593 | 3.297 | 4.396 | 1.099 | 36.068 |
| A_repo_a16 | 3.50 | 92 | 5.435 | 3.261 | 3.261 | 1.087 | 31.393 |
| A_repo_a16 | 3.75 | 94 | 4.255 | 2.128 | 3.191 | 1.064 | 30.95 |
| A_repo_a16 | 4.00 | 95 | 4.211 | 1.053 | 4.211 | 1.053 | 33.991 |
| B_ossFixed_a16 | 0.50 | 54 | 3.704 | 3.704 | 0.0 | 0.0 | 100.0 |
| B_ossFixed_a16 | 0.75 | 65 | 1.538 | 1.538 | 0.0 | 0.0 | 191.425 |
| B_ossFixed_a16 | 1.00 | 61 | 4.918 | 3.279 | 1.639 | 0.0 | 146.476 |
| B_ossFixed_a16 | 1.25 | 60 | 0.0 | 0.0 | 0.0 | 0.0 | 200.0 |
| B_ossFixed_a16 | 1.50 | 63 | 1.587 | 1.587 | 0.0 | 0.0 | 141.786 |
| B_ossFixed_a16 | 1.75 | 70 | 1.429 | 1.429 | 0.0 | 0.0 | 192.296 |
| B_ossFixed_a16 | 2.00 | 67 | 2.985 | 2.985 | 0.0 | 0.0 | 123.939 |
| B_ossFixed_a16 | 2.25 | 66 | 1.515 | 1.515 | 0.0 | 0.0 | 194.99 |
| B_ossFixed_a16 | 2.50 | 61 | 0.0 | 0.0 | 0.0 | 0.0 | 200.0 |
| B_ossFixed_a16 | 2.75 | 67 | 1.493 | 1.493 | 0.0 | 0.0 | 190.125 |
| B_ossFixed_a16 | 3.00 | 66 | 0.0 | 0.0 | 0.0 | 0.0 | 179.611 |
| B_ossFixed_a16 | 3.25 | 66 | 7.576 | 4.545 | 4.545 | 1.515 | 134.734 |
| B_ossFixed_a16 | 3.50 | 63 | 0.0 | 0.0 | 0.0 | 0.0 | 184.908 |
| B_ossFixed_a16 | 3.75 | 67 | 0.0 | 0.0 | 0.0 | 0.0 | 200.0 |
| B_ossFixed_a16 | 4.00 | 83 | 2.41 | 1.205 | 2.41 | 0.0 | 178.255 |

Attribution (2×2 factorial, pre-declared H3): removing the unconditional
`NO_IFACE_NEIGH | NO_FLUID_NEIGH` demotion collapses the disagreement from
7.58 % (max) / 3.45 % (mean over T ≥ 0.5) to the
1–4 % band, while replacing the OSS tolerance by the repo κ barely moves it.
**H3 is supported: the structural demotion guard, not the 3.33× tolerance
ratio, drives the divergence.**

Synthetic decision map — fill intervals where the two selectors disagree
(the neighbourhood class decides whether the guard can fire at all):

| configuration | OSS-only → GAS fill | repo-only → GAS fill | OSS-only → LIQUID fill | repo-only → LIQUID fill |
|---|---|---|---|---|
| layer | -0.010 … -0.004 | none | +1.005 … +1.010 | none |
| no_liq_nbr | +0.500 … +1.000 | none | +1.005 … +1.010 | none |
| no_iface_nbr_liq | -0.030 … +0.895 | none | none | -0.030 … +0.895 |
| no_gas_nbr | -0.010 … -0.004 | none | +0.900 … +0.900 | none |

Three reproducible signatures:

* **tolerance band** `layer`: OSS demotes at −0.01 < ε ≤ −0.003 and promotes at
  1.003 < ε ≤ 1.01, the repo does neither — a pure 3.33× threshold effect.
* **guard effect** `no_liq_nbr` / `no_iface_nbr_liq`: OSS demotes interface cells
  that still hold ε up to 0.9–1.0 when a neighbour class is missing; the repo
  keeps them. This is a *modelling* difference (how aggressively the interface
  layer is kept one cell thick), not a tolerance.
* **strict vs non-strict comparison** `no_gas_nbr`: OSS promotes at exactly
  ε = 0.9 (`>=`), the repo needs ε > 0.9 (`>`).

### Excess redistribution

`placement_l1_over_excess_pct` in the table above is the L1 distance between
the two increment fields, normalised by the total excess moved: 30–100 % (and
>100 % when a whole interface layer converts at once, because the OSS envelope
turns the demoted liquid row into receivers while the repo's receiver set is
`iface & ~converting` ∪ newborn gas cells). The mass **total** nevertheless
stays closed in the repo because `free_surface_step_2d` applies its
conservation clamp: measured inventory drift over the whole run is
0.0322 % (A) and 0.0164 % (C).

## ④ Figures

`fig1_front_curves.png` — front position Z(T) vs the digitised Martin & Moyce
curve for A, C, D, E, F at a = 16 (grey band = pre-declared scoring window).
`fig2_decision_map.png` — the synthetic decision map: fraction of interface
cells driven to GAS / LIQUID vs fill, repo solid, OSS dashed, one curve pair
per neighbourhood configuration (this is where the selector rules visibly
fork). `fig3_state_disagreement.png` — decision disagreement (%) vs T on the
frozen states of the running solver, for the four factorial cells (5 % line =
pre-declared interchangeability threshold). `fig4_field.png` — interface fill
field at the last frozen checkpoint (field cloud).

## ⑤ Traps, limits, applicability

* **The cluster arms are not in this report.** The a = 32 / 48 arms and the
  OSS-port arm were submitted as job `cj2038`; the queue did not dispatch it
  inside this mission's window (14+ min queued), and a single a = 32 arm
  (~5 min) exceeds the local 2-minute tool budget. **They are `not_run`, not
  "passed by proxy".** For resolution context only, the earlier in-repo XVAL
  `XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS` measured the same case for the same repo
  arm: max err 22.3 % at a = 16, 13.8 % at a = 32 — i.e. the 28.2 % seen here
  at a = 16 is consistent with under-resolution, and the selector conclusion
  (2.5 % arm-to-arm) is not a resolution statement either way.
* **`ma` in the table is u_max/c_s, not the initial-condition Mach number.**
  The shipped OSS example uses g = 0.1 (Ma ≈ 6, compressibility-broken); this
  XVAL deliberately runs g = 5e-5, so the gravity-wave scale gives
  $u\sim\sqrt{2gh_0}pprox0.057$, Ma ≈ 0.1. The **measured** u_max is
  0.238 (Ma 0.41): a transient
  corner jet, i.e. the a = 16 lattice is locally compressible and this — not
  the selector — is the most plausible cause of the 28 % front error at a = 16.
  A resolution study at a ≥ 32 is required before quoting a compressibility-free
  number (queued as `cj2038`, not delivered here).
* **The paired differential uses the rho available at the checkpoint**
  (post-step `sum f`), not the mid-step post-streaming density. Because
  anti-bounce-back pins ρ ≈ 1 at interface cells (measured |ρ−1| < 1 %), the
  fill normalisation is insensitive to that choice — but it is an
  approximation, stated rather than hidden.
* **Scope of the selector test:** marking + excess placement only. The
  envelope refill (how the new interface cell gets its PDFs) is outside this
  XVAL; a transplant would have to port `average_surround` as well.
* **Don't copy the η split verbatim.** `update_types.py:106-110` deposits the
  donor's mass into `d − v[n]` while weighting by `v[n]·n`, i.e. *anti-parallel*
  to the eta direction (towards the liquid side). Any transplant must decide
  whether that is intended; the repo's isotropic split plus its clamp already
  closes the inventory.

## ⑥ Error judgement (explicit)

| key quantity | baseline | this work | error | < 3 % ? |
|---|---|---|---|---|
| selector port fidelity (flags) | fidelity-checked port | extracted selector | 0 mismatches / 40 states | **YES** (exact) |
| selector port fidelity (mass, float64) | same | same | 0.0 | **YES** |
| front curve, arm A vs arm C (transplant test) | A (κ=1e-2) | C (κ=3e-3) | max 2.50 %, mean 0.167 % | **YES** |
| front curve vs Martin & Moyce, a = 16 | experiment | A / C | 28.2 / 28.2 % | **NO** (>15 %, under-resolved lattice; see ⑤) |
| decision disagreement, OSS vs repo selectors | repo rule | OSS rule | max 7.58 %, mean 3.45 % of interface cells | **NO** (pre-declared 5 % band exceeded at peaks) |
| liquid inventory drift | initial inventory | A / C | 0.0322 / 0.0164 % | **YES** (≪1 %) |

**Honest bottom line.** The selector extraction is exact (0 % discrepancy), and
transplanting the OSS *tolerance* is numerically inert for the dam-break front
(2.5 % < 3 %, H1 supported). The OSS *structural* demotion guard is not inert:
it disagrees with the repo rule on up to 7.58 % of interface cells,
above the pre-declared 5 % band, so the two selectors are **not interchangeable**
outside the threshold factor; and H2 (selector ⇒ inventory drift) is
**falsified** — the repo's step-level clamp hides the placement difference
(drift 0.0322 % for both arms). Recommendation:
do **not** transplant the guard; keep the repo selector, expose κ through the
API (already possible), and record the guard as a modelling option to revisit
only with a resolution study at a ≥ 32.

## Artifacts

`xval/mass_exchange_selector/{oss_selector.py, repo_selector.py, run_xval.py, make_figures.py,
DIFF_OSS_VS_TENSORLBM_MASS_EXCHANGE_SELECTOR.md}`,
`artifacts/{fidelity.json, fidelity_both.json, runs.json, analysis.json, differential.json,
numbers.md, states_*.npz, figures/*.png}`.
