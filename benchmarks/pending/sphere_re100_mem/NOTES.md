# W5-B — 球绕流直测力法修复（sphere Re=100 force-method repair）

Wave-5 benchmark campaign, track W5-B. Worktree (read-only): `/nfs/wangxi/worktrees/bm_w5`
(main @ cf5709db3c). Staging: `/nfs/wangxi/runs/bm_widen_w5_20260921/sphere_force/`.
GPU: cuda:6 exclusive. Python: `/nfs/wangxi/venvs/tensorlbm/bin/python`.

## 0. Blocked baseline (inherited, 2026-08-19)

GeneralSimEngine sphere Re=100: pressure+friction integration (extrap='none',
p0='near_wall') Cd −16.6% @D40; MEM `momentum_exchange_standard` (Ladd sum),
free-stream background subtracted, +264.6%. Reference Cd=1.09.

## 1. Locked references (pre-registered, before any formal run)

Primary anchor (the campaign convention, identical formula to
`benchmarks/general_sim_acceptance_sphere_re100.py::schiller_naumann_cd`):

    Cd_ref(Re=100) = 24/100 · (1 + 0.15·100^0.687) = 1.0917311…

Gate: |Cd − Cd_ref| / Cd_ref ≤ 3%  →  Cd ∈ [1.05898, 1.12448].

(rev-3: the locked FORMULA above is and always was the anchor; rev-1 rendered
its numeric value as 1.091566 — a transcription slip, −0.015% relative.
All errors in this record are computed against the formula value 1.0917311;
the §6 diagnostic tables quoted vs 1.091566, shifting every quoted error by
+0.015 pp — immaterial to every conclusion.)

Cross-references (independent sources, disclosure only — the 3% gate is vs the
primary anchor only):

    classic Schiller–Naumann (exponent 0.681):  24/100·(1+0.15·100^0.681) = 1.06852
    Clift–Gauvin: 24/100·(1+0.1315·100^(0.82−0.05·log10 100))          = 1.10923

Band across the three correlations: [1.069, 1.109] (spread 3.7%). Any judgment
within 3% of 1.0917 is inside the correlation spread.

## 2. Locked case definition (formal runs)

- Geometry: parametric sphere, radius 0.5, D = resolution cells (D40/D60;
  D80 contingency only if D40 sits on a resolution floor and D60+D80 pass).
- Re = 100 = u_phys·L/nu_phys with L = D_cells (reference length = diameter).
- Lattice: D3Q19, u_lb = 0.05 (Ma = 0.0866), nu_lb = u_lb·D/100,
  tau = 0.5 + 3·nu_lb.
- Collision: MRT (`collide_mrt3d_low_memory`, default rates s_e=1.19,
  s_eps=1.4, s_q=1.2, s_pi=s_e) — identical to engine AUTO at Re=100.
- Wall: half-way bounce-back via engine stepper order
  (collide → NoDynamics → BB at solid → stream → far-field BC), mass
  correction every 200 steps (`correct_mass3d`), engine initialization
  (uniform free-stream equilibrium; solid at rest).
- BC: far-field (free-stream equilibrium) on x−, y±, z±; zero-gradient x+
  (`far_field_bc_3d`, engine bc_config for 3D geometry).
- Domain: sphere centred; lateral half-width H and streamwise padding locked
  per §4 sensitivity outcome BEFORE formal runs (default candidate 3.0D
  lateral each side = engine auto-equivalent, 1.25D upstream + 2.25D
  downstream streamwise).
- Steps: run until the judging observable is steady; steady criterion: mean
  over last 20% window vs mean over preceding 20% window differ by <0.3%,
  minimum 8000 steps. Recorded in result.json.
- Normalisation: dpS = 0.5·u_lb²·π·R_lb² (engine `_compute_dpS` for spheres).
- Judging observable: the DIRECTLY measured integral force (see §3) divided
  by dpS, window-averaged over the steady window. No post-hoc corrections,
  no extrapolated values as judgment (extrap columns are diagnostics only).

## 3. Candidate direct force methods (choice locked after §4 diagnostics)

- M1 wet-node MEM: F = 2·Σ_links c_q f_q(x_solid), crossing = all fluid→solid
  links, sampled post-stream pre-BC (production SUBOFF convention).
- M2 pair MEM: F = Σ_links c_q·(f_q(x_s) + f_opp(x_f)), same phase — exact
  per-step momentum budget (steady state: equals M1).
- M3 pressure+friction integration with best-behaved p0 (direct wall
  integral; judgment channel only if it reaches the gate without extrap).
Historical broken estimators (std/galilean/bg_sub MEM, PF near_wall/none)
are diagnostic columns only.

## 4. Phase-1 diagnostics (pre-registered)

Run D40 (and D60 confirm) with phase-instrumented stepper replica that calls
the same library kernels as `GeneralSimEngine.run`, sampling:

1. MEM term decomposition: F_A = Σ c_q f_q(x_f) (current impl term 1),
   F_B = Σ c_q f_opp(x_s) (current impl term 2), M1, M2; crossing-set
   asymmetry Σ c_q w_q; near-restricted vs full fluid→solid link counts.
2. Background closure: all MEM variants on the t=0 uniform free-stream state
   (expected 0 for a correctly closed background).
3. PF: cd_p for p0 ∈ {near_wall, far_field, domain_avg, inlet} ×
   extrap ∈ {none, linear, quadratic}; friction ∈ {standard, lagrange,
   faces, mix50}; stagnation-line pressure profile (hole quantification).
4. Control-volume momentum-flux integral on a box 6 cells off the body
   (independent direct measurement, cross-check).
5. Sensitivity (D40): lateral domain 3.0D vs 4.0D vs 5.0D; u_lb 0.05 vs
   0.03 (Ma); steps/steady; mass drift; Cl symmetry; run-to-run determinism
   (no RNG — bitwise identical expected).

Error-budget table: each error source × magnitude × scaling with D, and the
path selection for §3.

## 5. Phase-2 deliverables

- Patch (staging `src_patched/`): new MEM variant(s) in momentum_exchange.py
  + engine wiring (`mem_variant` option), default behaviour unchanged.
- Discriminating tests (staging `tests/`): empty-domain zero force for old
  and new variants; uniform-background closure; sphere Cl≈0 symmetry;
  momentum-budget identity M2 vs CV integral on a small case.
- Full regression: `PYTHONPATH=staging/src_patched pytest tests -q` green on
  default paths.
- Formal grids D40+D60 with locked protocol; `result.json` + `verify.py`
  independent recomputation from raw force histories.
- Judgment: PASS if both grids ≤3% vs 1.0916 AND |err| monotonically
  decreasing from coarse to fine. Otherwise FAIL with the quantified error
  budget and scaling (definitive-source record).

## 6. Phase-1 error budget (diagnostic runs, locked before formal runs)

All numbers: window = last 20% of a 10000-step run (steady, drift < 0.35%),
error vs Cd_ref (quoted here vs the rev-1 rendering 1.091566; against the
formula value 1.0917311 every error shifts by +0.015 pp — see §1 rev-3).
`diag_*.json` / `bfl_*.json` in `diag/`.

### 6.1 Root cause of the historical +264.6% (estimator bug — FIXED)

`momentum_exchange_standard` pairs `f_q(x_f)` (population at the FLUID cell,
one step before crossing) with `f_opp(x_s)` (population left over at the SOLID
node from the previous step), summed over the near-wall (face-adjacent) link
subset only. Term decomposition on D40 lat1.25: termA = Σ c_q f_q(x_f) carries
a non-cancelling linear free-stream background (+43.64 Cd on the t=0 state),
termB ≈ 0; termA+termB ≡ `momentum_standard` exactly. The library's
`momentum_exchange_background_subtracted` subtracts the free-stream background
on a crossing set that is direction-symmetric — the closed-form background of
the Ladd sum is identical on both sides — hence it is a bitwise no-op
(mem_std ≡ mem_bgsub in every run). Measured: +260.2% (D40), +171.8% (D60) —
the inherited blocked state.

### 6.2 Repaired estimators and their validation

- M1 wet-node MEM `momentum_exchange_wet_node`: F = 2·Σ_links c_q·f_q(x_solid)
  over ALL fluid→solid links (face + staircase-corner; corner links are 13.8%
  of the D40 set). Exact per-step wall momentum budget of the production BB
  stepper. t=0 closure on the engine initial state: exactly 0.0 (float64;
  fp32 reduction noise 1.1e-5 Cd-units) vs +43.64 for the standard pairing.
- M2 pair MEM `momentum_exchange_pair` = M1 + unsteady link-storage term;
  equals M1 to 4 decimals at steady state in every run (measured identical
  window means).
- Independent control-volume momentum budget (exact In−Out link ledger of a
  box 6 cells off the body, evaluated on the pre-streaming state): at D40
  lat1.25 the BFL link ledger (M3 below) gives Cd=1.2753, the CV budget gives
  Cd=1.2768 — 0.12% agreement; t=0 baseline exactly [0,0,0]. The directly
  measured force is the true momentum extraction rate of the flow.

### 6.3 Remaining error is a SOLUTION error (blockage / far field), not measurement

| route / estimator | D40 lat1.25 (6.41%) | D40 lat2.0 (3.14%) | D40 lat4.0 (0.97%) | D60 lat1.25 (6.41%) |
|---|---|---|---|---|
| BB wet-node (M1)    | 1.2923 (+18.39%) | 1.2301 (+12.69%) | 1.2155 (+11.36%) | 1.2831 (+17.55%) |
| BFL link ledger (M3)| 1.2753 (+16.83%) | 1.2154 (+11.34%) | — | 1.2734 (+16.66%) |
| CV momentum budget  | 1.2768 (+16.97%) | — | — | — |
| MEM std (broken)    | 3.9320 (+260.2%) | 3.7949 (+247.7%) | 3.7629 (+244.7%) | 2.9666 (+171.8%) |
| PF integration      | 0.5689 (−47.9%)  | 0.5402 (−50.5%)  | 0.5335 (−51.1%)  | 0.2565 (−76.5%)  |

- Interpolated wall (BFL) removes the staircase error: at fixed blockage it
  sits 1.5 pp below the BB wet-node value, and both routes agree to 0.5 pp —
  the force measurement is no longer the bottleneck.
- The residual is D-flat at fixed lateral multiple (blockage % is
  D-invariant): −0.17 pp from D40→D60 on BFL, −0.84 pp on BB. Lateral scan
  (hard free-stream clamping on y±/z±): ≈ −1.8 pp per 1% blockage; linear
  zero-blockage extrapolation leaves ≈ +6…+11% at D40. The clamped lateral
  far field (kills the induced displacement flow) is the dominant error
  source.
- u_lb sensitivity (0.05→0.03, D40): wet_full +18.39%→+19.51% — Ma effects
  O(1%), not the bottleneck. Mass drift ≤ 0.3 ppm everywhere; Cl ≤ 5e-4.
- wet_near (face-adjacent subset only) crosses the reference by accident
  (+1.83% @D40lat1.25 → −4.03% @lat4.0) — staircase/blockage cancellation,
  unprincipled; diagnostic column only, NOT a candidate judgment route.

### 6.4 Sensitivity map of the residual (BFL route, hard far field, CV-validated)

| D40 probe | Cd | err |
|---|---|---|
| lat1.25 down2.25 (base)  | 1.2753 | +16.83% |
| lat1.25 down6.0          | 1.2864 | +17.85% |
| lat1.25 down10.0         | 1.2875 | +17.95% |
| lat2.0  down2.25         | 1.2154 | +11.34% |
| lat1.25 sponge/non-eq ('prod') | 1.4103 | +29.20% |

Streamwise length is ruled out (flat, slightly rising). The lateral scan
saturates: BB +11.65% @1.86% blockage vs +11.36% @0.97% — the residual is a
blockage-independent ≈ +11% floor at D40 on top of the blockage term. The
equilibrium-difference sponge makes it worse (stronger effective confinement
+ outflow damping of the wake).

### 6.5 Library production pipeline probe (context for §7.1)

`sphere_bfl_control_volume` at worktree HEAD (cf5709db3c) with its own
defaults (R=12, 192×96×96, u=0.06, `cumulant_d3q19_cs0`,
non-equilibrium far field + sponge) diverges at step 18 — reproduced on
cuda:6 and CPU, with `natural_kbc_d3q19` as well, and at u=0.03 (diverges at
step 22). The in-repo 5%-gate precedent for this case is not runnable at
HEAD; no external calibration point exists inside the codebase.

## 7. Formal protocol lock (rev-2, before any formal run)

- Formal runs use `run.py` only (no diagnostic replicas): route `bb` =
  GeneralSimEngine + patched package, force_method=MOMENTUM_EXCHANGE,
  mem_variant='wet_node' (M1, the repaired estimator); route `bfl` = library
  kernels + Bouzidi interpolated wall on the analytic sphere with the
  laboratory-frame link ledger (M3) — CV momentum budget recorded per sample
  as the independent cross-observable.
- Far-field treatment for both formal routes: `hard` (engine default,
  see §7.1).
- Formal set (exactly four runs):
  - `formal_bb_D40.json`, `formal_bb_D60.json`: lat 1.25 D (engine standard
    domain), up 1.25 D, down 2.25 D — the engine-integrated column with the
    repaired estimator as the engine's primary reported force.
  - `formal_bfl_D40.json`, `formal_bfl_D60.json`: lat 2.0 D (saturation
    point of the §6.4 lateral scan — blockage-independent floor), up 1.25 D,
    down 2.25 D — the judgment route.
- Steps 12000, sample 50, mass correction /200. Steady criterion per §2.
- Judging observable: window-mean (last 20%) Cd of the DIRECTLY measured
  force on route `bfl`; the `bb` route is recorded as the engine-integrated
  comparison column. PASS = both grids ≤ 3% vs 1.091566 AND |err| strictly
  decreasing from D40 to D60. No post-hoc corrections of any kind.
- `verify.py` recomputes every number independently from the raw histories
  and re-hashes NOTES.md against the sha256 chain.

### 7.1 Treatment selection record

- `hard`: free-stream equilibrium clamping (`far_field_bc_3d`, y±/z± feq,
  x− feq, x+ zero-gradient) — the engine default. Best-performing treatment
  in §6.4. Selected.
- `prod`: library production treatment for this benchmark class
  (`sphere_bfl_control_volume.py` defaults): incoming-only non-equilibrium
  far-field extrapolation on all six faces (`non_equilibrium_far_field_bc_3d`)
  + equilibrium-difference sponge (width 18, strength 0.2, faces
  x+,y±,z±). Rejected: +29.2% @D40 lat1.25 (CV-confirmed 1.4096), and the
  upstream pipeline itself diverges at step 18 at HEAD (§6.5).

## Revision chain

All revisions appended to `NOTES_sha256_chain.txt` BEFORE the affected run.
- rev-1 (pre-registration): §0–§5 as deployed.
- rev-2: §6 error budget + §7 formal protocol lock.
- rev-3 (record correction, after the formal runs, before result.json):
  §1 numeric rendering of the locked formula corrected (1.091566 → 1.0917311;
  formula, gate structure and all conclusions unchanged); §6 tables annotated.
