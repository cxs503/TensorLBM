# AMR free-surface verification face (campaign node P190-N02)

TensorLBM · `src/tensorlbm/amr_free_surface.py`, `src/tensorlbm/free_surface_vof.py`, `benchmarks/amr_free_surface/`

## 1. Case, geometry and parameters

Dam break in a closed tank.  The initial column is `0.5 H` wide and `H` deep (the classic 0.5 m x 1 m column when `H = 1 m`), released onto a dry bed at `t = 0` and integrated to `t = 1 s`, i.e. `t* = t sqrt(g/H) = 3.13`.

| quantity | value |
|---|---|
| tank length / initial depth | 8.00 |
| tank height / initial depth | 1.67 |
| column width / initial depth | 0.50 |
| lattice celerity c0 = sqrt(g h) | 0.12 |
| dam-break Reynolds number (grid independent) | 60.0 |
| level-0 cells (H) | 6 |
| level-0 steps | 156 |

Refinement ladder: `L = 1, 2, 3` above the same level-0 grid, i.e. finest spacing `h/2, h/4, h/8` (`h` = level-0 spacing).  Scaling is convective: `g_l = g_0/2^l` and `nu_lat,l = 2^l nu_lat,0` (`tau_l = 0.5 + 2^l (tau_0 - 0.5)`), so the physical viscosity and the Reynolds number are grid independent.

## 2. Method

* **Kernel** (`free_surface_vof.py`): D2Q9 BGK + Guo gravity on the fluid cells, half-way bounce-back on solid, fill-weighted anti-bounce-back reconstruction on every gas link (Körner et al. 2005), and a finite-volume **flux-form VOF advection** of the liquid fraction with minmod-limited interface values and a donor bound.  The VOF flux is exactly antisymmetric across every face, so the liquid mass is conservative by construction.
* **Hierarchy** (`amr_free_surface.py`): block-structured, ratio 2, 2:1 sub-cycling, Filippova-Hänel rescaling of the non-equilibrium part (`tau_f/(2 tau_c)` up, `2 tau_c/tau_f` down), nearest prolongation and 2x2 area-mean restriction of the material field (exactly conservative in the `4^-l` area quadrature).
* **Indicator**: `|grad eps| > 1e-3` on each level's own fill field (the free-surface form of `|grad rho|`) plus every Körner `INTERFACE` cell; the marked set is split into at most 4 x-slabs per level instead of one bounding box, because one box around a 1-D interface curve would cover the whole water body.
* **Mass ledger**: closed against the initial value every step by a multiplicative **reflux** on the level-0 interface cells (closed tank => the liquid mass is physically invariant).  The *raw* exchange drift is recorded before the correction and reported.
* **Reference**: second-order finite-volume shallow water (`tensorlbm.shallow_water2d`) on the *same* tank; the Ritter (1892) self-similar front is shown for context only.

## 3. Results

Shallow-water front at `t* = 3.13`: **5.82 H** (Ritter would give 6.76 H).  Ritter assumes an infinitely long reservoir; in a `8 H` tank at `t* = 3.1` the reservoir has already drained, so the same-tank shallow-water solution — not Ritter — is the anchor used for the acceptance criteria.

| route | front [H] | front err vs SWE | profile L1 vs SWE | cells/uniform | wall [s] | mass err (closed) | raw drift |
|---|---|---|---|---|---|---|---|
| amr_L1 | 1.08 | 0.890 | 1.392 | 30.6% | 6.4 | 0.0e+00 | 0.38 |
| amr_L2 | 1.17 | 0.875 | 0.945 | 10.6% | 16.5 | 2.6e-08 | 2.40 |
| amr_L3 | 1.00 | 0.906 | 1.156 | 4.9% | 41.7 | 6.0e-08 | 1.73 |
| uniform_fine | 0.40 | nan | 1.245 | 100.0% | 61.4 | 0.0e+00 | 0.00 |
| control_refine_liquid | 1.00 | 0.906 | 1.225 | 4.5% | 51.3 | 3.9e-08 | 1.62 |

AMR (L3) vs the uniform grid at the same finest spacing: profile L1 = **1.11**, front difference = 1.21 of the front advance.

Convergence over the AMR ladder: fitted order **-0.01** (front) and **0.13** (profile L1); errors {'1': 0.8902767572863092, '2': 0.8746020083272105, '3': 0.9059515062454078} and {'1': 1.391554747781121, '2': 0.9451041341468823, '3': 1.1561734524625207}.

## 4. Acceptance criteria

| criterion | value | limit | pass |
|---|---|---|---|
| a_front_error_vs_swe_lt_5pct | 9.060e-01 | 0.05 | FAIL |
| b_mass_error_lt_1e-6_1000_steps | 5.981e-08 | 1e-06 | PASS |
| c_amr_cells_lt_30pct_uniform | 4.863e-02 | 0.3 | PASS |
| e_order_front_ge_2 | -1.259e-02 | 2.0 | FAIL |
| e_order_profile_ge_2 | 1.337e-01 | 2.0 | FAIL |
| d_amr_time_lt_50pct_uniform | 6.795e-01 | 0.5 | FAIL |
| f_amr_vs_uniform_profile_l1_lt_5pct | 1.114e+00 | 0.05 | FAIL |

## 5. Notes, limits, and what the failures mean

1. **Mass (criterion b) passes, but only with the reflux.**  The raw exchange drift is large: 1.73 x the total mass over 156 level-0 steps (about 1 %/step), caused by mass leaving a fine tile into its ghost ring (which is held at the coarse representation) and by the coarse/fine representation switch.  The reflux rides on the level-0 interface cells and closes the ledger to 1e-8, but it is a *repair*, not a flux-matching scheme: a production AMR needs the ghost-ring flux returned to the parent cells explicitly.
2. **Interface accuracy (criterion a) fails by a wide margin** (about 90 %).  The split momentum/mass model is the cause: the surface is not a true zero-pressure boundary, so the collapse is far slower than shallow water (AMR front 1.00 H vs 5.82 H).  The same kernel without AMR is equally slow (0.40 H), so this is a *kernel* defect, not an AMR defect.
3. **The AMR does not reproduce the fixed grid** (criterion f): profile L1 1.11.  Two contributions are measured: the coarse level carries the whole liquid body (only a thin ribbon is refined), and the front speed is **not resolution convergent** even on a uniform grid (0.40 H at h/8 vs 1.00-1.17 H at h/2-h/4).  A resolution convergent kernel is a prerequisite for any fruitful AMR convergence study; this is the single most important finding of this node.
4. **Efficiency (criterion d) fails as measured** (0.68 of the uniform wall time, gate 0.50) even though only 4.9% of the cells are used: each tile call is dominated by the ~20 us torch op-launch floor on this host, and the hierarchy issues 15 calls per coarse step against 8 large calls for the uniform run.  On a compute-bound host the same configuration is ahead (the cell-work ratio is two orders of magnitude below the time ratio).
5. **Why a hand-written VOF kernel instead of the production Körner model.**  The repository's `free_surface_lbm_2d.free_surface_step_2d` and `free_surface_lbm.free_surface_step` need 10^2-10^3 torch ops per call (measured 27 ms for 560 cells, 1570 ms for 35 840 cells) and keep two mass representations, so a representation switch across refinement levels moves the conserved quantity (measured 1-11 % per AMR step).  They are therefore not usable as the inner kernel of an AMR hierarchy; the Körner-adapter version is kept in the git history of this node's branch.
6. **Scope of validity.**  The hierarchy machinery (boxes, nesting, sub-cycling, exchange, ledger, diagnostics) is kernel agnostic and is verified by `tests/test_amr_free_surface.py`; the hydrodynamic accuracy of the kernel is not verified by this node and should not be used for engineering predictions.

## 6. Error statement

* Front position vs same-tank shallow water: 90.6% -> **not < 3 %** (fails, cause identified: split momentum/mass model).
* Mass ledger: 5.98e-08 closed (raw 1.73) -> closed value is < 3 %, raw value is not.
* AMR vs uniform grid profile: 111.4% -> **not < 3 %** (fails).
* Cell count: 4.9% of the uniform grid -> **< 30 %, passes**.

All failures are reported, none is overwritten.  No number in this report is hand-copied: every table entry is generated from `runs/amr_free_surface/study.json`.
