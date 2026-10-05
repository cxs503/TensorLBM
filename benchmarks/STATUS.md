# Benchmark status (generated 2026-10-05, after the 22-agent marine sweep)

verified: **30** | pending: **30**

Acceptance: every checkpoint within 3% AND mesh-converged (two grids, span <=3%).

## verified/

| case | note |
|---|---|
| `acoustics` | analytic / standard case |
| `cavity` | analytic / standard case |
| `cavity_3d_full` | analytic / standard case |
| `couette_2d` | analytic / standard case |
| `cylinder` | analytic / standard case |
| `cylinder_3d` | 3D extruded cyl Re=40: +2.45%/-0.19% (surface-only Ladd MEM, span 2.64%) |
| `droplet_oscillation` | analytic / standard case |
| `floating_body_free_decay` | 6DOF still-water free-decay eigenfrequency: heave +1.07%/+0.26%, roll +1.21%/+0.30% |
| `kovasznay_2d` | analytic / standard case |
| `laplace_droplet` | analytic / standard case |
| `maneuvering_steady_turn` | steady turning response: R +0.009%, beta -0.011%, span <0.001% (given-derivative closure) |
| `permeability` | analytic / standard case |
| `poiseuille_2d` | analytic / standard case |
| `poiseuille_3d_duct` | analytic / standard case |
| `poiseuille_3d_pipe` | analytic / standard case |
| `shear_wave_decay` | analytic / standard case |
| `sod_shock_tube` | analytic / standard case |
| `sphere_re100` | sphere Re=100: Cd +2.771%/+2.540% (BFL interpolated smooth wall + per-link momentum ledger; D40/D60, span 0.231%) |
| `sphere_re200` | sphere Re=200: Cd +1.866%/+0.216% (same caliber; D16/D20, span 1.650%, monotone) |
| `square_cylinder` | 2D square cyl Re=100: Cd +0.11%/+0.30%, St +0.82%/+0.68% |
| `startup_poiseuille` | analytic / standard case |
| `stokes_first_problem` | analytic / standard case |
| `stokes_second_problem` | analytic / standard case |
| `suboff_re1000` | SUBOFF hull drag Re=1000: Cd -1.15% end-to-end (mix50 friction, 3.4h) |
| `taylor_couette` | analytic / standard case |
| `taylor_green_2d` | analytic / standard case |
| `taylor_green_3d` | analytic / standard case |
| `thermal_cavity` | analytic / standard case |
| `wigley_hydrostatics` | Wigley hull form coefficients: max |err| 2.25%, max span 1.02% (L=640/960; statics only, NOT seakeeping) |
| `womersley` | analytic / standard case |

## pending/ (not yet accepted -- blockers documented)

| case | status / blocker |
|---|---|
| `backward_step` | mode K fix (pre-stream BB + non-eq sponge) kills the acoustic standing wave (drift 4e-2 -> 2e-7/step); near-wall travelling artefact remains -> X1 not plateaued |
| `blasius_flat_plate` | domain height / exit BC / virtual-LE all REFUTED; residual = leading-edge near-field acceleration + inlet transient (le=100 -> +12.1%) |
| `bstep_3d` | in progress |
| `capillary_invasion_washburn` | in progress |
| `cavity_3d_full` | in progress |
| `cavity_natural_convection` | in progress |
| `channel_turb` | in progress |
| `collision_kernels_3d_tg` | in progress |
| `couette_3d` | in progress |
| `cylinder_re40_st` | BLOCKED (new): exit-BC hypothesis REFUTED (2D/3D far_field_bc both zero-gradient); residual ~2.9% is D2Q9 vs D3Q19 lattice difference; surface-MEM requires solid freeze |
| `dam_break_3d_mm` | STRUCTURAL: X/H mutually exclusive (same mass channel) |
| `dam_break_sc` | STRUCTURAL: constitutive barrier (err(T=1)=0 needs density ratio 12.7, err(T=2.96)=0 needs 6.9) |
| `dit_turbulence` | in progress |
| `dtmb5415_resistance` | STRUCTURAL: real geometry ok (zenodo cloud -> STL, Cb 0.53) but single-phase LBM has no wave-making (-22% gap even with perfect Cf) |
| `forchheimer` | in progress |
| `kcs_resistance` | reference audited (MDPI 2020 KCS Cb 0.6505, Fr=0.26 EFT Ct 3.55e-3); caliber = Blasius friction low-Re; runner ready, converged run pending |
| `naca_0012` | BLOCKED: perf fixed via fast_kernel (288->36 ms/step); CD_REF corrected to 0.12 (Kurtulus + Di Ilio 2020 0.119; old 0.105 rejected); surface-MEM +82% on a friction-dominated airfoil |
| `permeability3d` | in progress |
| `poiseuille_3d_annulus` | in progress |
| `poiseuille_3d_ellipse` | in progress |
| `propeller_openwater` | STRUCTURAL x3: CAD chord/area -32%/-24%; Re mismatch ~600x; low-tip-Mach gate vs mask time-resolution mutually exclusive |
| `rayleigh_benard` | in progress |
| `rayleigh_taylor` | STRUCTURAL: VOF has no free pressure field |
| `series60_resistance` | BLOCKED: parametric hull Cb phase-oscillates with grid (no stable pair); ALSO exposed a library bug -- GeometrySource.PARAMETRIC_HULL returns an EMPTY mask in GeneralSimEngine |
| `sphere_re100_d3q27` | in progress |
| `stokes_second_problem` | in progress |
| `stokes_sphere_dg` | BLOCKED: Stokes flow is linear yet Cd moves 133->67 as u_in 0.02->0.05 at Re=0.1 => staircase sphere is non-physical there |
| `suboff_sail_resistance` | STRUCTURAL: real sail half-width 0.0077L is sub-grid; L<=96 sail absent or 1-cell. Needs n>=256 (~180M cells, >=60GB) |
| `taylor_aris_dispersion` | in progress |
| `two_phase_poiseuille` | in progress |

## Key calibers (this cycle)

* **surface-only Ladd MEM**: sum the momentum exchange over the *surface* solid cells only; on a curved voxel body the interior cells carry a non-degenerate pseudo-force.
* **BFL interpolated smooth wall**: for doubly-curved bodies (sphere) a smooth boundary representation is what removes the staircase geometric bias.
* **Reference audit first**: several 'failures' were caliber errors (square_cylinder's Okajima 1.6; cylinder_3d's experimental 1.54; M&M's axis factor; backward_step's factor-2).
* **Structural blocks are documented, not tuned**: free-surface系列 needs a free pressure field + independent gas EOS; ship resistance needs wave-making; propeller needs real section tables + Re matching.
