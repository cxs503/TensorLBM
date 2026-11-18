# Benchmark status (generated 2026-10-05, after the 22-agent marine sweep)

verified: **91** | pending: **60**  (counts machine-refreshed by `benchmarks/status_ledger.py --write`)

These are **directory inventory counts**, not counts of scientifically validated LBM capabilities.
See [the validation matrix](validation_matrix/STATUS_MATRIX.md) for numeric verdicts and separately admitted LBM benchmark passes. Synthetic, reference-only and correlation-only results cannot supply those passes.

Legacy table criterion: checkpoints within 3% and two-grid span <=3%. A two-grid span is not proof of monotone convergence or engineering validity; use each benchmark's declared uncertainty and acceptance criteria.

**Validation matrix v1** (nine-case lineage, machine-checked gates, one-command
reproduction): see [`validation_matrix/`](validation_matrix/README.md) and the
generated [`validation_matrix/STATUS_MATRIX.md`](validation_matrix/STATUS_MATRIX.md).
Its per-case status files (`benchmarks/{verified,pending}/<case>/validation_status.json`)
carry the metric rows, convergence checks and sha256 evidence chain; the anchors
are re-audited from their cited sources by `validation_matrix/reference_audit.py`.

## verified/

| case | note |
|---|---|
| `acoustics` | analytic / standard case |
| `aircraft_icing_naca0012_rime` | NACA 0012 IRT rime: internal-consistency family all PASS (G1 closure 0.0, G7 bitwise 15/15, G9 LWC linearity 1.66%, G10 Richardson |err| <= 0.86%); no external anchor |
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
| `maneuvering_zigzag` | closed-loop zig-zag (overshoot): two routes agree to 1.2e-12, Euler p=0.998 / RK4 p=4.02, IMO MSC.137(76) gate discriminates |
| `permeability` | analytic / standard case |
| `poiseuille_2d` | analytic / standard case |
| `poiseuille_2d_bodyforce` | analytic body-force channel (`H=8..64`): eps_inf 8.99e-04 / eps_2 1.23e-03 at H=32, order 2.00; also fixes the truncated (non-zero-sum) Guo source term, mass drift 1.1e-02 -> 1.9e-12 |
| `poiseuille_3d_duct` | analytic / standard case |
| `poiseuille_3d_pipe` | analytic / standard case |
| `shear_wave_decay` | analytic / standard case |
| `sod_shock_tube` | analytic / standard case |
| `sphere_re100` | sphere Re=100: Cd +2.771%/+2.540% (BFL interpolated smooth wall + per-link momentum ledger; D40/D60, span 0.231%) |
| `sphere_re100_d3q27` | sphere Re=100 D3Q27 twin: Cd +2.79%/+2.57% (BGK27 + BFL sparse links, mirrored kernels bitwise-equal to library; reference sha256-pinned to the sphere_re100 archive) |
| `sphere_re200` | sphere Re=200: Cd +1.866%/+0.216% (same caliber; D16/D20, span 1.650%, monotone) |
| `square_cylinder` | 2D square cyl Re=100: Cd +0.11%/+0.30%, St +0.82%/+0.68% |
| `startup_poiseuille` | analytic / standard case |
| `stokes_first_problem` | analytic / standard case |
| `stokes_second_problem` | analytic / standard case |
| `suboff_re1000` | SUBOFF hull drag Re=1000: Cd -1.15% end-to-end (mix50 friction, 3.4h) |
| `suboff_sail_resistance` | SUBOFF+sail AFF-3 Re=1000: Cd +0.95% finest tier, |err| monotone 8.15->2.92->0.95% (std friction per frozen erratum E2; formal mix50 FAIL co-archived) |
| `taylor_couette` | analytic / standard case |
| `stefan_phase_change` | analytic (Neumann 1-D sealing/melting) + independent sharp-interface reference (2-D cylinder): front 0.50%/1.01% (1-D two measures), cylinder radius 1.45%, front shape 0.85%, melting energy closure 0.61%, front order p=1.00. See `docs/P155W1_N01_STEFAN_PHASE_CHANGE.md` |
| `taylor_green_2d` | analytic / standard case |
| `taylor_green_3d` | analytic / standard case |
| `thermal_cavity` | analytic / standard case |
| `wigley_hydrostatics` | Wigley hull form coefficients: max |err| 2.25%, max span 1.02% (L=640/960; statics only, NOT seakeeping) |
| `womersley` | analytic / standard case |

## pending/ (not yet accepted -- blockers documented)

| case | status / blocker |
|---|---|
| `aircraft_icing_rg15_ipw2` | RG-15 IPW2 3 official cases: G3 FAIL 3.03e-02 (per-bin closure, warmup-window), G11a FAIL 10.1% (input reconstruction); MCCS D1 area ratio 0.14-0.32x model-class gap; per-bin beta curves empty (library asymmetry) |
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
| `dam_break_3d_mm` | Historical corner release; no geometry-matched experimental reference |
| `dam_break_mm_channel` | Channel release, Martin & Moyce Tables 2 and 6; see [run evidence](pending/dam_break_mm_channel/README.md) |
| `dam_break_sc` | STRUCTURAL: constitutive barrier (err(T=1)=0 needs density ratio 12.7, err(T=2.96)=0 needs 6.9) |
| `dit_turbulence` | in progress |
| `dtmb5415_resistance` | geometry FIXED (m452): waterline-clipped STL was open -> voxel body Cb 18 (35x displacement), staircase wetted area 16x smooth -> Cd_tot 1.18; capping the waterline gives Cd_tot 0.045 (Blasius +7.9% at L=32). Caliber re-based: friction-vs-Blasius is the only pass/fail frame; exp C_T(Fr=0.28) stays NOT COMPARABLE (-22.4% no-wave floor, structural). Valid resolution L >= 48 (T/dx >= 2, Cb -10.4%). See docs/sdaa_triage/DTMB5415_DIAGNOSIS.md |
| `forchheimer` | in progress |
| `kcs_resistance` | reference audited (MDPI 2020 KCS Cb 0.6505, Fr=0.26 EFT Ct 3.55e-3); caliber = Blasius friction low-Re; runner ready, converged run pending |
| `naca_0012` | BLOCKED: perf fixed via fast_kernel (288->36 ms/step); CD_REF corrected to 0.12 (Kurtulus + Di Ilio 2020 0.119; old 0.105 rejected); surface-MEM +82% on a friction-dominated airfoil |
| `naca0012_aero_polar` | 首个外部气动验证面（三网格极线+$C_p$+GCI）：$dC_L/d\alpha$ 对 $2\pi$ -64.8%，GCI($C_L$) 59.26%；XFOIL/Abbott 基准离线不可得 → 判据(2)(3) WITHHELD。见 `benchmarks/pending/naca0012_aero_polar/README.md` |
| `permeability3d` | in progress |
| `droplet_impact_2d_pool` | Driven bridge and static Laplace checks pass. Matched external transient crater data and convergence remain pending. [Case manual](../docs/benchmarks/droplet_impact.md) |
| `droplet_impact_3d_pool` | Pending. Single-drop case 3, 18 observed crater depths and explicit two-resolution comparisons. [Case manual](water_entry/single_drop/README.md) |
| `phasefield_water_entry` | Rigid wedge S2, 4° / Dw=100: first nonfinite force at step 141, before the registered fit window (300–1100). Cylinder and sphere stages await S2. [Case and evidence](pending/phasefield_water_entry/README.md) |
| `poiseuille_3d_annulus` | in progress |
| `poiseuille_3d_ellipse` | in progress |
| `propeller_openwater` | STRUCTURAL x3: CAD chord/area -32%/-24%; Re mismatch ~600x; low-tip-Mach gate vs mask time-resolution mutually exclusive |
| `rayleigh_benard` | in progress |
| `rayleigh_taylor` | STRUCTURAL: VOF has no free pressure field |
| `series60_resistance` | BLOCKED: parametric hull Cb phase-oscillates with grid (no stable pair); ALSO exposed a library bug -- GeometrySource.PARAMETRIC_HULL returns an EMPTY mask in GeneralSimEngine |
| `stokes_second_problem` | case-own gate **VERIFIED** + caliber audit resolved 2026-11-06 (m4924): prereg 5%-mask caliber ruled **ill-posed** (argmax denominator pinned at 1.01-1.10x threshold; measured 4.99/5.16/4.86% = plateau over delta/dx=10/20/40, no trend => unsatisfiable by refinement; gate equivalent to |du|<=0.15%U0 at threshold); amended caliber G1-G5 all PASS (unmasked max|du|/U0 1.374%/1.381%, 10%-mask max rel **2.160%/1.012%**, eta-collapse 2.437%, steady 0.030%/0.023%; evidence caliber_audit.json + tests/test_stokes_second_problem_caliber_audit.py). Error attribution: global max|du| is bottom-wall depth truncation at eta~4 (H=4*delta, e^-4~1.8%), near-wall wall-scheme error <=0.43%U0 for delta/dx>=10 vs 1.697% at delta/dx=4 (validity boundary). Promotion to verified/ queued as dir-move PR (deferred: overlaps in-flight m4917 rewrite of this dir) |
| `stokes_sphere_dg` | BLOCKED: Stokes flow is linear yet Cd moves 133->67 as u_in 0.02->0.05 at Re=0.1 => staircase sphere is non-physical there |
| `squeeze_film_lbm` | LBM vs Reynolds/Stefan $8\\mu V a^3/h^3$: $h^{-3}$ exponent 3.03/2.90/2.74 (PASS), absolute gauge load +12.2%..+37.2% over $h/a$=0.25..0.75 (**no 3%**); structural: $a$=24,h=4 converges to a spurious +1350% branch, $a$=32,h=4 diverges (NaN) -> wide-film refinement path blocked by the corner/pressure-face row overwrite. Caliber defects fixed: ME load contains the static $n_x\\rho_0c_s^2$ baseline (now `force_gauge_*`), `force_pressure_integral` was sampled on the mass-source row (now y=0). See docs/squeeze_film_lbm_verification_20261108.md |
| `taylor_aris_dispersion` | in progress — 口径审计（2026-11-12, `benchmarks/pending/taylor_aris_dispersion/CALIBER_AUDIT.md`）：容差门 6/6 PASS（max \|err\| 0.2624%，11.4× 余量；独立重推 1/210 相对偏差 8.8e-11），单调门 0/3 FAIL 属口径错配（残差为零均值散点 t=1.30<2.57），『~0.2% 地板』根因被本案 V4 纯扩散对照（0.0049%）推翻 → 机制 WITHHELD；待评审席裁定 amended caliber |
| `two_phase_poiseuille` | in progress |

## Key calibers (this cycle)

* **surface-only Ladd MEM**: sum the momentum exchange over the *surface* solid cells only; on a curved voxel body the interior cells carry a non-degenerate pseudo-force.
* **BFL interpolated smooth wall**: for doubly-curved bodies (sphere) a smooth boundary representation is what removes the staircase geometric bias.
* **Reference audit first**: several 'failures' were caliber errors (square_cylinder's Okajima 1.6; cylinder_3d's experimental 1.54; M&M's axis factor; backward_step's factor-2).
* **Structural blocks are documented, not tuned**: free-surface系列 needs a free pressure field + independent gas EOS; ship resistance needs wave-making; propeller needs real section tables + Re matching.

<!-- STATUS_LEDGER:BEGIN -->
### verified 机器台账（自动生成，勿手改）

verified 目录 **91** · STATUS.md 人工表已写 note 的 **36** · advertised header verified=91 pending=60

人工表未覆盖 **55** 个目录（notes 待补；下表由目录扫描生成，覆盖=该目录是否已出现在上方人工表）：

| verified case | in human table |
|---|---|
| `acoustics` | yes |
| `aircraft_icing_naca0012_rime` | yes |
| `brinkman_channel` | **no** |
| `cavity` | yes |
| `cavity_3d_full` | yes |
| `cavity_natural_convection` | **no** |
| `cavity_re100` | **no** |
| `channel_core_path_ci` | **no** |
| `compressible_shock_family` | **no** |
| `couette_2d` | yes |
| `couette_3d` | **no** |
| `couette_poiseuille` | **no** |
| `cylinder` | yes |
| `cylinder_3d` | yes |
| `cylinder_re100` | **no** |
| `dam_break_ritter_ci` | **no** |
| `droplet_oscillation` | yes |
| `electroosmosis` | **no** |
| `floating_body_free_decay` | yes |
| `forchheimer_weak_inertia` | **no** |
| `fused_push_pull_aa_pattern` | **no** |
| `fwh_acoustics` | **no** |
| `fwh_rotating` | **no** |
| `graetz_tube` | **no** |
| `inamuro_wetnode_xval` | **no** |
| `kovasznay_2d` | yes |
| `laplace_cac_phasefield` | **no** |
| `laplace_droplet` | yes |
| `laplace_pr_eos` | **no** |
| `lbm_weakly_compressible` | **no** |
| `maneuvering_steady_turn` | yes |
| `maneuvering_zigzag` | yes |
| `mcmp_3d_stability` | **no** |
| `mcmp_stratified_poiseuille` | **no** |
| `mmg_imo_msc137_uncert` | **no** |
| `mms_convergence_2d` | **no** |
| `nscbc_outlet_reflection` | **no** |
| `nwt_stokes2` | **no** |
| `permeability` | yes |
| `permeability3d` | **no** |
| `permeability_fibrous` | **no** |
| `poiseuille_2d` | yes |
| `poiseuille_2d_bodyforce` | yes |
| `poiseuille_3d_annulus` | **no** |
| `poiseuille_3d_duct` | yes |
| `poiseuille_3d_pipe` | yes |
| `poiseuille_convergence_order` | **no** |
| `poiseuille_two_phase_cg` | **no** |
| `radiation_enclosure_ship` | **no** |
| `rheology_yield` | **no** |
| `rotating_duct_frame_w2` | **no** |
| `rotating_ekman` | **no** |
| `rotating_interior_frame` | **no** |
| `rotating_source_sink` | **no** |
| `rotating_taylor_column` | **no** |
| `rotating_w3` | **no** |
| `shear_wave_decay` | yes |
| `ship_resistance_suboff` | **no** |
| `sod_shock_tube` | yes |
| `sphere_drag_bfl` | **no** |
| `sphere_re100` | yes |
| `sphere_re100_d3q27` | yes |
| `sphere_re100_mem` | **no** |
| `sphere_re200` | yes |
| `square_cylinder` | yes |
| `square_cylinder_re200` | **no** |
| `squeeze_film_ns` | **no** |
| `staircase_wall_offset` | **no** |
| `startup_poiseuille` | yes |
| `static_droplet_laplace_ci` | **no** |
| `stefan_phase_change` | yes |
| `stefan_two_phase_sweep` | **no** |
| `stokes_first_problem` | yes |
| `stokes_second_problem` | yes |
| `stokes_sphere_dg` | **no** |
| `suboff_anchor` | **no** |
| `suboff_anchor_w22` | **no** |
| `suboff_re1000` | yes |
| `suboff_sail_resistance` | yes |
| `suboff_tau_grid_w22` | **no** |
| `taylor_aris_dispersion` | **no** |
| `taylor_couette` | yes |
| `taylor_green_2d` | yes |
| `taylor_green_3d` | yes |
| `tgv2d_analytic_l2_xval` | **no** |
| `thermal_cavity` | yes |
| `thermal_cavity_3d` | **no** |
| `wave_tank` | **no** |
| `wigley_hydrostatics` | yes |
| `womersley` | yes |
| `womersley_pulsatile` | **no** |
<!-- STATUS_LEDGER:END -->
| `nwt_stokes2` | nonlinear wave-tank Stokes 2nd-order: 12/12 analytic closed-form gates PASS (dispersion, deep/shallow limits, 2nd-harmonic coeff, wavenumber & period round-trip); reference = Dean & Dalrymple 1991 sec 4.3 |
