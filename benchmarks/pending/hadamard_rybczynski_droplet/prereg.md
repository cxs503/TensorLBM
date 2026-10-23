# W9-B Prereg — Hadamard–Rybczynski droplet terminal velocity ( creeping, Re≪1 )

Locked 2026-09-29, before any simulation run. No changes after formal runs start.
Probe runs (P0/P1) may inform feasibility go/no-go and the locked Re_t choice; any such
choice is recorded in the appendix BEFORE the first formal run and never changed after.

## 1. Reference formula lock (two independent sources + limit checks)

### Source 1 (drag form) — Pigeonneau 1998 thesis, via literature aggregate
F_D = 6π μ_c R U (2+3λ)/(3(1+λ)),  λ = μ_i/μ_o = μ_b/μ_0
(equivalently F_D = 2π μ_o a U (2+3λ)/(1+λ), a = R).

### Source 2 (terminal-velocity form, verbatim) — Wikipedia "Hadamard–Rybczynski equation",
### citing Clift, Grace & Weber (2005), Bubbles, Drops, and Particles, Dover
W_b = (2/3) · [R² g (ρ_b − ρ_0)/μ_0] · (μ_0 + μ_b)/(2 μ_0 + 3 μ_b)

### Consistency: force balance (4/3)πR³(ρ_o−ρ_i)g = F_D from Source 1 gives
U_t = (2/3) R² g (ρ_o − ρ_i) (1+λ)/(μ_o (2+3λ))   [droplet lighter → rises]
Substitute λ=μ_b/μ_0, ρ_i=ρ_b: identical to Source 2 (sign per buoyancy direction). CLOSED.

### Limit checks
- λ→∞ (rigid sphere): (1+λ)/(2+3λ)→1/3 → U_t=(2/9)a²gΔρ/μ_o = Stokes terminal velocity. PASS
- λ→∞ drag: 6πμ_o aU. PASS
- λ→0 (clean bubble): (1+λ)/(2+3λ)→1/2 → U_t=(1/3)a²gΔρ/μ_o = 1.5× rigid. PASS
- λ→0 drag: 2πμ_o aU·2/1 = 4πμ_o aU. PASS

### Lattice-unit form with library λ=ρ lock (single τ → uniform ν, μ=ρ_tot·ν; ρ_o=1, r=ρ_i/ρ_o=λ)
U_t^HR = (2/3) a² g (1 − r²) / (ν (2+3r))
Direction: r<1 rises (+y), r>1 sinks (−y). Observable compared by |U|.

### Hand-computed double-precision example (locked; script must reproduce ≤1e-12 rel)
r=λ=0.5, a=12, ν=1/6 (τ=1), g=8.1e-6, ρ_o=1:
U_t = (2/3)·144·8.1e-6·(1−0.25)/((1/6)·3.5)
numerator = 96 · 8.1e-6 · 0.75 = 5.832e-4
denominator = 0.5833333333333333
U_t = 9.997714285714286e-4  (upward, |U|)
Re = U·2a/ν = 9.997714285714286e-4 · 24 · 6 = 0.14396685714285715

## 2. Module inventory (complete, read from bm_w9 @ e717b464)

| module | λ control | ρ-ratio control | verdict |
|---|---|---|---|
| SCMP 3D (collide_sc_single_component) | none: single τ, λ fixed ≈ ρ_l/ρ_v = 12.3 (coexistence ρ_l=1.957/ρ_v=0.1596, laplace archive) | fixed by coexistence | DEAD: λ not controllable; spurious max\|u\|≈0.137–0.140 ≫ U_t~1e-3 |
| SC-MCMP 3D (collide_sc_two_component_3d) | per-τ exists BUT unequal τ structurally poisoned: u_eq="self" 26.8%/33.2% anti-convergent bias (W5-A two_phase_poiseuille FAIL), u_eq="mixture" NaN @step 29 for G_12≤−1.5 unequal τ | YES (SC segregation holds real ρ contrast; W4-B precedent) | VIABLE only at EQUAL τ → λ=ρ lock |
| Color-Gradient 3D (color_gradient_step_3d / collide_cg_mrt_3d) | λ = ρ ratio locked (single τ on f_total; μ=ρ_tot·ν) | free in init but hypothesized UNSUSTAINABLE: single-component-like EOS p=ρc_s² with capillary Δp=2σ/R ≈ 1.7e-3 ≪ Δρ/3 = 0.167 → ρ_tot contrast collapses to 1 → no buoyancy. Probe P0a decides. | probe |
| Free-Energy 2D (free_energy_step) | λ≡1 (Boussinesq ρ_eff) | no | DEAD (also 2D only) |
| free-surface modules | no viscous droplet interior | — | DEAD |

**Library-gap headline: no module separates λ from ρ with acceptable accuracy.**
Every viable configuration carries λ=ρ. The ladder {0.1, 1, 10} therefore degenerates:
λ=1 ⟺ ρ ratio 1 ⟺ zero buoyancy ⟺ U_t=0 (unmeasurable). Ladder replaced by
r ∈ {0.5, 2} primary (λ=ρ), {0.25, 4} secondary if stable. This substitution is itself
a preregistered consequence of the module constraint, not a post-hoc change.

### Equivalence-principle finding (preregistered design constraint)
Periodic domain + per-mass gravity on all components = free fall: no relative motion, U=0 forever.
Walls (momentum sink) are MANDATORY. Tank = make_channel_wall_mask_3d ∪ make_tank_wall_mask_3d
(all 6 faces) + bounce_back_cells_3d post-stream on both distributions (W4-A validated pattern,
creeping-flow analytics to 0.54%).

### Confinement error budget (preregistered expectation, rigid-sphere cylinder
### Ladenburg–Faxén factor 1 − 2.105(a/R_c) + 2.0865(a/R_c)³ as attribution model)
walls at 4a: ≈ −26%; 6a: ≈ −12.6%; 8a: ≈ −6.6% (upper bound magnitude; droplet internal
circulation reduces blockage). Direct-observable gate ≤3% vs UNBOUNDED HR is therefore likely
unreachable at feasible grid sizes — the run's value is the honest verdict + quantified
attribution ladder (confinement series at fixed D), consistent with campaign rules
(direct observables only, corrections are diagnostic evidence, never the gate).

### Finite-Re error budget
Oseen-type inertial correction to drag O(3Re/8) rigid → velocity shift ≈ −1.9% at Re=0.05
(smaller for circulating droplets). Locked Re_t = 0.05 for ALL formal runs
(U_t ∝ 1/D at fixed Re_t ⟹ g ∝ 1/D³·(2+3r)/(1−r²); computed per-run in script).
Memory cap: tank cubic L=4a up to D=64 (256³=16.8M cells ×2 dists ×19 ×fp32 ≈ 2.6 GiB live,
stream temporaries ~5 GiB — fits 32 GB).

## 3. Probes (Phase 1, feasibility — verdicts recorded, not gated)

- P0a CG static (no g): D=24 (a=12) in L=96 cube, r ∈ {0.5, 2}, 6000 steps.
  Measure: ρ_tot inside/outside bulk, R_eq, max|u|, survival of contrast.
  Feasible iff contrast retention >90% after 6000 steps AND max|u| < 0.3·U_t^target.
- P0b MCMP static (no g): same geometry, G_12 ∈ {+2.5, −2.5} (3D docstring says >0 separates;
  2D used −2.5 — sign resolved empirically), r=0.5 target via init ρ1_bulk/ρ2_bulk.
  Measure: segregation quality, mutual solubility (dissolved fraction), ρ_tot contrast,
  max|u|. Feasible iff steady segregated droplet + contrast retention >90% + max|u| < 0.3·U_t.
- P1 gravity probe (winner of P0a/P0b; both if both pass): D=24, r=0.5, Re_t=0.05,
  20k steps. Measure U(t)=d/dt φ-weighted centroid y; plateau quality (window drift),
  spurious-current contamination (RMS|u−U| inside droplet vs U), travel budget.
  Feasible iff: U plateau drift in last 20% < 3% of U AND droplet travels ≥1.5a before
  nearing wall (else reduce g / shorten run — recorded, not tuned per-point).

## 4. Formal ladder (Phase 2 — LOCKED)

Carrier module: whichever passes P0 (CG preferred if P0a passes since its momentum
coupling is the physical one-fluid mixture; MCMP equal-τ carries W5-A-style structural
bias risk — quantified either way).

Grids: D ∈ {32, 48, 64} at L=4a (tank cube 128/192/256).
r ∈ {0.5, 2.0} primary; {0.25, 4.0} secondary (run only if primary completes stable).
Confinement attribution series at D=32: L/a ∈ {4, 6, 8} (tank 128/192/256), r=0.5.
Re_t = 0.05 everywhere: g(r, a) = Re_t·ν²·(2+3r) / ((4/3)·a³·(1−r²)).
Run length: 20k steps D=32, 25k D=48, 30k D=64 (travel-bounded; re-locked from P1 if
travel budget violated — recorded in appendix before formal start).

Observable: U_sim = slope of φ-weighted droplet centroid y over last-20% window
(linear fit slope, φ = ρ_droplet fraction field; interior-mean u_y cross-check reported).
Measured inputs to reference (parameter measurement, NOT observable correction):
bulk ρ_tot inside/outside at plateau, R_eq = (3V_φ/4π)^(1/3) at plateau.
U_ref = (2/3)·R_eq²·g·|1−r_eff²|/(ν(2+3r_eff)), r_eff = ρ_in/ρ_out measured.
err = |U_sim|/U_ref − 1.

Steady gate (preregistered): window drift = |slope(first half of window) − slope(second
half)| / |slope(window)| < 0.10; centroid travel within window ≥ 1 R_eq.

PASS: err ≤ 3% at a grid point. Monotonic convergence: |err| non-increasing over
≥2 successive D at fixed r. Overall track verdict = gate result at best grid point
+ monotonicity; confinement series reported as attribution evidence (fit err vs a/L
against Ladenburg–Faxén form; extrapolated intercept reported as evidence ONLY).

Deformation check: aspect ratio (y-extent/x-extent of φ>0.5 region) reported per run;
Ca estimate = μ_o U_t/σ_est with σ_est from static probe Laplace-style Δp if obtainable
(diagnostic only).

## 5. Machine archives
Every run: out/<name>/result.json = full params, per-1000-step traces (centroid, U slope
rolling, R_eq, ρ bulk in/out, max|u|, RMS interior residual), final verdict numbers.
Raw f-fields NOT archived (size); final φ slice + centroid trace archived as .pt for one
representative run per module.

## 6. Grep self-check (library-only kernels)
Composition scripts must contain no hand-written collision/streaming/equilibrium kernels:
grep -c 'def collide\|def stream\|def equilibrium\|def bounce\|def zou_he\|def far_world' run scripts → 0.

## Appendix (added 2026-09-29 after Phase-1 probes, before any formal run) — outcome

Phase-1 feasibility probes (P0a/P0b/P1-AC, machine archives in probe/ and out/)
found the formal ladder UNRUNNABLE: no module can hold a buoyant viscous droplet.
Formal Phase-2 was therefore never started and no gate was evaluated. Full record:
NOTES.md. Corrections to section 2 of this prereg (inventory was incomplete at lock
time, corrected here BEFORE any formal run):

- allen_cahn_lbm.allen_cahn_step WAS missed at lock: it is the only module with
  independent (rho_h, rho_l) and (nu_h, nu_l) — the true lambda ladder carrier
  candidate. Probed: droplet dissolves <200 steps at all W in {2,4,8} x sigma in
  {0,0.01} (structural: hardcoded M=0.02, tau_phi=0.7, no anti-diffusion flux).
- multiphase3d_d3q27 (MCMP 27) and cg_advanced_collision (CG variants) were missed:
 前者 same raw-rho cross force => same inversion family; 后者 same lambda=rho lock
  + contrast collapse. Both probed or argued; see NOTES.md table rows 5-6.
- phasefield/ package was missed: double-well free energy, lambda==1 (Boussinesq).
- Section 4 ladder, window protocol and gates stand UNEXERCISED (no formal run).
  The lambda-ladder substitution rule (r in {0.5,2} etc.) is moot.

Verdict registered: FAIL — module capability (see NOTES.md for the nine-branch
exclusion table and five library gaps G1-G5).
