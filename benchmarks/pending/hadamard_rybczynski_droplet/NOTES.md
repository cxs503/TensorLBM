# W9-B Hadamard–Rybczynski droplet terminal velocity — verdict record

Track: bm_widen_w9_20260929/droplet_hb · baseline e717b464 (main)
Date: 2026-09-29 · GPU 6 · all runs fp32 torch 2.11+cu128

## Overall verdict: FAIL — module capability

The Hadamard–Rybczynski benchmark (buoyant viscous droplet, Re<<1, terminal velocity
vs analytic) is UNRUNNABLE on TensorLBM e717b464: no multiphase module can represent
a buoyant viscous droplet with any viscosity ratio lambda != 1. Formal Phase-2 ladder
never started (Phase-1 feasibility probes decided it); prereg gates therefore never
reached. Every branch below is machine-traced.

## Reference formula lock (Phase 0, CLOSED)

Two independent sources, limit checks, hand double example: see prereg.md section 1.
U_t = (2/3) a^2 g (rho_o - rho_i)(1+lambda)/(mu_o (2+3lambda)); lambda->inf -> Stokes
(2/9)a^2 g dr/mu_o; lambda->0 -> clean bubble (1/3)a^2 g dr/mu_o = 1.5x rigid.
Hand example r=lambda=0.5, a=12, nu=1/6, g=8.1e-6: U_t = 9.997714285714286e-4.

## Module-by-module exclusion (complete inventory, D3Q19/27 3D + 2D)

| # | module | exclusion | evidence (file:finding) |
|---|--------|-----------|--------------------------|
| 1 | SCMP 3D collide_sc_single_component (19/27) | lambda fixed ~= 12.3 by coexistence; spurious max u 0.137-0.140 >> U_t~1e-3 | verified/laplace_droplet archive (in-repo) |
| 2 | Color-Gradient 3D color_gradient_step_3d (+ cg_advanced_collision variants) | lambda=rho lock (single tau); rho_tot contrast COLLAPSES acoustically: r=0.5 init ratio 0.536 -> 1.0011 by step 1000 -> 1.0027 final (mass vents through interface; EOS p=rho c_s^2 cannot hold dr>>2 sigma/R); r=2 case R_eq 11->14.9 (expansion as excess density vents) | out/evidence.json E1; probe/p0_static.json cg_r0.5/cg_r2.0; probe/p0_static.log |
| 3 | SC-MCMP 3D collide_sc_two_component_3d, equal tau | pressure-crush inversion: ambient comp floods droplet core, r1_ctr 0.65->0.09 by step 500 (G=-2.5, contrast 1.8x); sustainable only at zero total contrast => zero buoyancy | out/evidence.json E2/E3; probe/p0b_envelope.json (15 cases, contrasts 1.05-2.1 x G in [-2.5,+1.5]: all invert or mix); probe/p_final_diligence.json d27 |
| 4 | SC-MCMP 3D, unequal tau (the only lambda!=rho path in SC family) | W5-A structural bias (u_eq=self 26.8%/33.2% anti-convergent) + NaN (mixture, G<=-1.5); and buoyancy still needs total contrast => same inversion | benchmarks/pending/two_phase_poiseuille (W5-A record); docstring of collide_sc_two_component_3d |
| 5 | Allen-Cahn 3D allen_cahn_lbm.allen_cahn_step | ONLY module with independent rho_h/rho_l AND nu_h/nu_l (lambda decoupled). BUT phase field cannot sustain a dispersed minority droplet: dissolves in <200 steps at ALL W in {2,4,8} x sigma in {0, 0.01} (phi<-0.5 volume 6272->~400@100->0@200; corrected light-droplet init). Structural: M_mobility=0.02, tau_phi=0.7 hardcoded -> reaction-diffusion interface width ~0.8 cell << init 4; no anti-diffusion flux (volume not conserved); observed phi peak decay 1.0->0.97 in 40 steps. Also: sigma force lacks curvature (F=(sigma/W)(1-phi^2) n_hat), wall cells are collided (no solid skip), g reset to equilibrium each step (hybrid). Zero tests/benchmarks reference this module (first exercise = this probe) | out/evidence.json E4; probe/p1_ac.log [AC1] R_eq=0 by step 250; probe/p_final_diligence.json ac (W/sigma sensitivity) |
| 6 | D3Q27 family multiphase3d_d3q27 (MCMP) | same raw-rho cross force (psi functions only feed SCMP), same inversion: G=+2.5 NaN@50, G=-2.5 r1_ctr 0.65->0.0896 @500, comp2 in core 1.06 | out/evidence.json E5/E6; probe/p_final_diligence.json d27 |
| 7 | phasefield/ package (double-well free energy, Korteweg -phi grad mu) | lambda==1 (Boussinesq, single density); no buoyancy contrast possible | src/tensorlbm/phasefield/free_energy.py |
| 8 | free-surface family (dam_break*, hull_free_surface, sloshing_tank, multiphase_water_entry, free_surface_application) | no viscous droplet interior (one-fluid + flag) | module docstrings |
| 9 | multiphase.py 2D family | 2D only; same SC/CG/FE locks | multiphase.py |

## Library gap list (new findings, for controller/owner)

G1. **G_12 sign bug, 3D MCMP (D3Q19 and D3Q27)**: docstring + StaticDroplet3DConfig
    validation say "G_12 > 0 for phase separation"; implementation (backward gather,
    psi(x-c) with +c weights => F = +(G/3) rho1 grad rho2) makes +G ATTRACTIVE
    (mixing/collapse; +2.5 NaNs) and segregation require G < 0 — same convention as
    2D (W4-B used -2.5). The library's own run_static_droplet_3d at validated default
    G=0.9 yields a MIXED state, so its sigma_eff measurement is vacuous. Same wording
    bug copied into multiphase3d_d3q27 docstring.
G2. **No module separates viscosity ratio from density ratio with acceptable accuracy**
    (headline): CG family locks lambda=rho; SC-MCMP unequal-tau is structurally biased
    (W5-A); Allen-Cahn has the knobs but the phase dynamics are non-functional
    (dissolution; hardcoded M/tau_phi; no volume conservation).
G3. **No 3D two-phase module sustains a total-density contrast droplet** (needed for
    ANY buoyancy-driven rise): CG collapses acoustically (EOS), SC-MCMP inverts by
    pressure crush at every probed G/contrast (envelope: contrast >= ~1.05 fails).
    This kills not just HR but every buoyant-drop/bubble benchmark class in 3D.
G4. **allen_cahn_lbm is an unvalidated orphan** (zero tests/benchmarks) with multiple
    implementation defects (curvature-less sigma force, no solid-skip in collision,
    non-conserving phase equation, hardcoded mobility). Fix path if owner wants HR:
    Fakhari/Geier full scheme (anti-diffusion flux n_hat·grad phi term + volume
    correction + mobility/tau_phi exposed + curvature sigma force).
G5. bounce_back_cells_3d hardcodes the 19-direction OPPOSITE (fails on 27-q fields;
    boundaries_d3q27.bounce_back_cells_27 exists — footgun for composition).

## Equivalence-principle finding (design constraint, documented)

Periodic domain + per-mass gravity on all components = free fall, no relative motion.
Walls (momentum sink) are mandatory for buoyancy benchmarks; tank composition
(channel|tank mask union + full-way bounce-back, W4-A-validated) is sound and was NOT
the failure cause (probe p0b_diag: G=0 + walls stable, spurious decays 1.9e-2->5e-4).

## Staging + reproduction

bm_widen_w9_20260929/droplet_hb/ (staging run root)
  prereg.md                  formula lock + ladders + gates (locked pre-run)
  NOTES.md                   this record
  out/evidence.json          canonical 6-case machine archive
  out/evidence.py            generator (run in place from this directory)
  probe/p0_static.{py,json,log}        CG collapse + first MCMP asym NaN scan
  probe/p0b_scan.{py,json,log}         G>=1 asym NaN matrix
  probe/p0b_diag.{py,json}             NaN isolation (walls innocent: G=0 stable)
  probe/p0b_gsweep.{py,json}           G window (sym <=1.2, asym <=0.6, all mixed states)
  probe/p0b_envelope.{py,json}         15-case contrast x G envelope (all invert/mix)
  probe/p1_ac.{py,log}                 Allen-Cahn probes (both init orientations dissolve;
                                       log head = corrected init R_eq=0 by 250; NOTE: log/json
                                       partially overwritten by concurrent process — canonical
                                       AC evidence is out/evidence.json E4 + p_final_diligence)
  probe/p_final_diligence.{py,json}    AC W/sigma sensitivity + D27 inversion
Grep self-check (no hand-written kernels) = 0 across all eight scripts.

Honesty note: probe/p1_ac.log and p1_ac.json were corrupted by an accidental double
launch (two processes, same log). The affected conclusions are independently
re-established in out/evidence.json (E4) and p_final_diligence.json, which are clean
single-process runs.
