# Benchmark status (generated 2026-10-06)

verified: **22** | pending: **21**

Acceptance: every checkpoint within 3% AND mesh-converged (two grids, span <=3%).

## verified/

| case | note |
|---|---|
| `cavity` | 2D lid-driven cavity (Ghia) |
| `cavity_3d_full` | 3D cavity |
| `couette_2d` | analytic |
| `cylinder` | 2D cylinder Re=100 (Braza Cd=1.35/St=0.1645) |
| `kovasznay_2d` | analytic |
| `laplace_droplet` | analytic |
| `poiseuille_2d` | analytic |
| `poiseuille_3d_pipe` | analytic |
| `shear_wave_decay` | analytic (nu to 0.05%) |
| `sod_shock_tube` | Riemann exact |
| `square_cylinder` | 2D square cyl Re=100: Cd +0.11%/+0.30%, St +0.82%/+0.68% (2 grids) |
| `startup_poiseuille` | analytic unsteady |
| `stokes_first_problem` | analytic |
| `stokes_second_problem` | analytic (Womersley family) |
| `suboff_re1000` | SUBOFF hull drag Re=1000: Cd -1.15% end-to-end (mix50 friction, 3.4h) |
| `cylinder_3d` | 3D extruded cyl Re=40: Cd +2.45%/−0.19% (surface-only Ladd MEM, 2 grids, span 2.64%) |
| `taylor_couette` | analytic |
| `taylor_green_2d` | analytic decay |
| `taylor_green_3d` | analytic decay |
| `womersley` | analytic pulsatile |
| `aircraft_icing_naca0012_rime` | NACA 0012 IRT rime: internal-consistency family all PASS (G1 closure 0.0, G7 bitwise 15/15, G9 LWC linearity 1.66%, G10 Richardson |err| <= 0.86%); no external anchor |

## pending/ (not yet accepted -- blockers documented)

| case | status / blocker |
|---|---|
| `backward_step` | classic (needs run) |
| `blasius_flat_plate` | Cf +20%: effective-scale excess 1.65x (blockage/leading-edge origin), long runs in flight |
| `bstep_3d` | 3D backward step |
| `cavity_natural_convection` | +2.57%/+6.64% -- not converged |
| `channel_turb` | turbulent channel |
| `couette_3d` | no result yet |
| `cylinder_re40_st` | Schafer-Turek channel Re=40; no official Re=40 reference |
| `dam_break_3d_mm` | STRUCTURAL: X/H mutually exclusive (same mass channel) -- docs/free_surface_architecture_gaps.md |
| `dam_break_sc` | STRUCTURAL: constitutive barrier (err(T=1)=0 needs ratio 12.7, err(T=2.96)=0 needs 6.9) |
| `dit_turbulence` | - |
| `droplet_oscillation` | SCMP -6~-8% |
| `naca_0012` | airfoil |
| `poiseuille_3d_annulus` | no result yet |
| `poiseuille_3d_ellipse` | no result yet |
| `rayleigh_benard` | thermal convection |
| `rayleigh_taylor` | STRUCTURAL: VOF has no free pressure field |
| `sphere_re100` | surface-only Ladd MEM: D=12 +9.15% / D=18 +8.92% -- mesh-converged (span 0.23%) TO THE WRONG VALUE; blockage lever dead (L32==L16 bit-identical), CV instrument agrees (+10%); staircase geometry bias, not caliber |
| `sphere_re100_d3q27` | D3Q27 lattice path (systematic offsets) |
| `sphere_re200` | reference corrected (SN 0.8056, not 0.769) |
| `stokes_sphere_dg` | DG force method: spurious (2.15% then 53% on refine) |
| `aircraft_icing_rg15_ipw2` | RG-15 IPW2 3 official cases: G3 FAIL 3.03e-02 (per-bin closure, warmup-window), G11a FAIL 10.1% (input reconstruction); MCCS D1 area ratio 0.14-0.32x model-class gap; per-bin beta curves empty (library asymmetry) |

## Notes

* **Force caliber matters**: the Ladd MEM sum must be restricted to the *surface* solid cells;
  the all-solid sum carries an interior pseudo-force (see `docs/mem_surface_caliber_finding.md`).
  This turned cylinder_3d from −8% into +2.45%/−0.19% (verified) on the same fields.
* **...but surface-only is not a blank cheque**: the sphere_re100 staircase still converges to
  Cd ≈ +9% under the SAME caliber (D=12 +9.15% / D=18 +8.92%, span 0.23%). The interior pseudo-term
  on the present BGK Ladd chain is only +0.02~0.06 (cylinder-like), so removing it trims all-solid
  from +12.7% only to +9.15%; the residual is a doubly-curved-staircase geometry bias (an independent
  control-volume instrument reads the same +10%), not a caliber bug. Blockage/domain is a dead lever
  (lateral 32 vs 16 bit-identical). See `benchmarks/pending/sphere_re100/result_mem_surface.json`.
* Free-surface / multiphase failures are structural (architecture), not tuning; see
  `docs/free_surface_architecture_gaps.md` (27 tried levers with failure reasons).
* Reference-caliber errors (not solver errors) accounted for two of the biggest false
  failures this cycle: the M&M dam-break table (axis x sqrt2 AND the wrong table) and
  the square-cylinder / cylinder-3d references (experiment/finite-span vs 2D numerical).
  Always run the `REFERENCE_AUDIT.md` procedure before believing an error percentage.
* `benchmarks/compile_route.py` auto-falls back to eager when the SDAA teco-inductor
  backend fails (5 known defects); results are bit-identical and `compile_mode_effective`
  is recorded in result.json.
