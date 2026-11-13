# Diff list (round 2): TensorLBM vs OSS `Maarten-vd-Sande/lbm`

Anchor `XVAL-LBM-ROUND2-METHOD-SELECTION`, mission 18241. Full narrative and numbers:
`docs/XVAL_LBM_ROUND2_METHOD_SELECTION.md`. OSS source read at
`/root/oss-cache/Maarten-vd-Sande_lbm @ 573a01c4` (sha256 in
`xval/benchmark_method_selection/oss_crosscheck.json`).

## Method selection (the decision this XVAL supports)

| aspect | OSS | TensorLBM | impact / porting |
|---|---|---|---|
| collision operator | single-relaxation BGK, one global `omega` (`examples/cylinder_flow.py:26`, omega=[1.8] -> tau=0.5556) | MRT (s_e=1.64, s_eps=1.54, s_q=1.7), shear rate 1/tau; BGK and TRT also available | matched viscosity is the contract; MRT adds stability dials, BGK is the degenerate case (arm A2) |
| TRT magic parameter | not implemented | lambda=3/16 transplant => tau_minus=25.5 at tau_plus=0.5075 -> divergence (A3, step 600/21259); re-derived tau_minus=2.0 -> divergence (A5, step 5800) | do not port literally; at this viscosity stay on MRT |
| sponge / absorbing layer | none (no hit in source) | 6D quadratic sponge, alpha=10, tau_eff = tau(1+alpha*sigma), MRT `tau_field` (round 2 added the same for BGK) | keep; removing it costs dCl_amp = -15.4 % at D=10 (A1 -> A4) |
| relaxation field | scalar only | per-cell `tau_field` (MRT before, BGK added this round) | needed for any sponge study; the BGK path is bit-identical when unused |

## Boundary conditions

| aspect | OSS | TensorLBM | impact |
|---|---|---|---|
| outlet | zero-gradient copy `fin[n,ix,iy]=fin[n,ix-1,iy]`, n=3..5, applied **pre-stream inside the collision loop** (`lbm/streamcollide.py:211-216`) | far-field outlet (`far_field_bc_2d`, post-stream) | K3 instrument (indicative, controls failed): R = 0.325 (OSS) vs 0.0025 (far-field) vs 0.0015 (far-field+sponge) |
| inlet | Zou-He density closure (`:219-234`) | far-field inlet, same routine as outlet | not compared numerically (different closure families); flagged gap |
| wall / body | `cell_type & CT.OBSTACLE`, bounce-back inside the streaming loop | half-way bounce-back on a boolean mask + post-stream momentum-exchange force | the force sampler cannot be ported: OSS has no force integration at all |
| lateral | `cell_type[:, (0,-1)] = CT.OBSTACLE` (walls in the example) | far-field sides | different case topology; not transferable |
| direction map | OSS dirs | TensorLBM dirs, permutation [0,6,2,3,1,8,5,4,7] | verified equal in round 1 (K0); OSS {3,4,5} == TensorLBM {3,6,7} |

## Parameters / workflow

| aspect | OSS | TensorLBM | impact |
|---|---|---|---|
| resolution | demo grid 300x130, body radius 10-24, 5 random obstacles | fixed Re=200 case, blockage 1/20, domain 20D x 20D, D = 10/14/20 ladder | OSS cannot serve as a resolution reference |
| reproducibility | random obstacles, no seed, no schedule | seeded inlet perturbation, predeclared step counts 21259/29763/42518 | the OSS example is a demo, not a benchmark |
| KPIs | none (no drag/lift/St anywhere) | Cd, Cl (amp + rms), St by two estimators (FFT + zero-crossing, 5 % cross-check) | the physics comparison must use a transplant or a kernel-level check, not an OSS number |
| verification in repo | none | round-1 kernel XVAL rel-L2 1.18e-07 / 2.71e-08; tests `tests/test_xval_benchmark_method_selection.py`, `tests/test_xval_round2_method_selection.py` | keep |

## Porting trade-offs (what to take, what not to)

1. **Take**: the OSS kernel is numerically identical to ours (K1/K2 < 1e-7), so the two
   codes differ only in method selection and boundaries - which is what makes the
   comparison meaningful at all.
2. **Do not take**: the zero-gradient outlet. Both instruments that touch it agree in
   direction (round-1 long-run behaviour and the K3 reflection ordering), even though
   K3's controls failed.
3. **Do not take**: TRT with a magic lambda transplanted from a tau ~ 1 textbook value;
   at tau_plus ~ 0.51 it is outside the stable tau_minus window (two measured
   divergences).
4. **Do not take**: the demo workflow (random obstacles, no KPI, no convergence
   criterion) - it cannot support a 15 % accuracy claim.
5. **Open**: whether MRT+sponge interacts with the collision knob (needs the A6 cell:
   predeclared and instrumented, not executed - see the report's section 6).
